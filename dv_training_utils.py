#!/usr/bin/env python3
"""
dv_training_utils.py

Shared utilities for DisplacedVertex graph-level binary classification.

This classifier version consumes HDF5 event groups with:
  x [N, 7], edge_index [2, E], edge_attr [E, 5], y [1] or labels [1].
It keeps the same GNN backbone/layer choices as the displaced-vertex regressor,
but replaces the 3-target regression heads with one graph-level logit trained
with BCEWithLogitsLoss.
"""

import argparse
import json
import math
import glob
import os
import random
import time
import faulthandler
import atexit
import signal
from pathlib import Path
from collections import Counter, OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import Dataset, DataLoader, Sampler

try:
    import wandb
except ImportError:
    wandb = None


NODE_FEATURE_NAMES = [
    "r_pos", "theta_pos", "phi_pos", "theta_dir", "phi_dir", "energy_like", "nCells_or_DoF"
]

EDGE_FEATURE_NAMES = [
    "d_energy_like", "d_phi", "d_eta", "cos_angle", "same_sector"
]

# Optional track block of the classification graphs (DisplacedVertex_converter.py with a track tree):
#   track_x [T, 11], track_edge_index [2, Et] (global indices, track t = x.shape[0] + t), track_edge_attr [Et, 5]
# Optional segment blocks (DisplacedVertex_converter.py --segment-edges / --tracklets):
#   seg_edge_index [2, Es] (muon-node indices), seg_edge_attr [Es, 7]
#   tracklet_x [K, 11], tracklet_edge_index [2, Ek] (muon-node indices and tracklet k = x.shape[0] + k, i.e. counted
#   without the track nodes; the model places the tracklet nodes after the track nodes), tracklet_edge_attr [Ek, 4]
from dv_converter_utils import (  # noqa: E402
    DV_NODE_FEATURE_NAMES,
    SEGMENT_EDGE_FEATURE_NAMES,
    TRACKLET_EDGE_FEATURE_NAMES,
    TRACKLET_FEATURE_NAMES,
    TRACK_FEATURE_NAMES,
    track_feature_asinh_scale_vector,
    DV_PACKED_BLOCKS,
    DV_PACKED_FORMAT,
    DV_PACKED_OPTIONAL,
    DV_PACKED_SUFFIX,
    PackedPartWriter,
    data_part_stem,
    is_packed_part,
    read_packed_meta,
)

# Node types of the merged graph (muon | calo | track | tracklet), used by the node-type embedding
NODE_TYPE_MUON, NODE_TYPE_CALO, NODE_TYPE_TRACK, NODE_TYPE_TRACKLET = 0, 1, 2, 3

# |x| cap of the track-feature asinh (asinh(1e6) = 14.5); same value as in dv_converter_utils
_ASINH_ABS_MAX = 1.0e6


def onnx_safe_asinh(x: torch.Tensor) -> torch.Tensor:
    """asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1)) with |x| capped at 1e6, written with ONNX-exportable
    operators (the TorchScript ONNX exporter has no aten::asinh). Identical to the numpy version in
    dv_converter_utils.apply_track_feature_transform, which the preprocessing uses for the track stats."""
    z = x.abs().clamp(max=_ASINH_ABS_MAX)
    return torch.sign(x) * torch.log(z + torch.sqrt(z * z + 1.0))


def _ensure_parent_dir(path: str) -> None:
    try:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _torch_save_atomic_with_retries(
    obj: Dict[str, Any],
    final_path: str,
    *,
    retries: int = 6,
    base_sleep_s: float = 1.0,
) -> None:
    final_p = Path(final_path).expanduser().resolve()
    parent = final_p.parent
    tmp_p = parent / (final_p.name + ".tmp")

    last_err: Optional[Exception] = None
    for i in range(int(retries)):
        try:
            parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj, str(tmp_p))
            os.replace(str(tmp_p), str(final_p))
            return
        except Exception as e:
            last_err = e
            try:
                if tmp_p.exists():
                    tmp_p.unlink()
            except Exception:
                pass
            sleep_s = base_sleep_s * (2 ** i)
            print(
                f"[ckpt] WARN: save failed (attempt {i+1}/{retries}) "
                f"to {final_p}: {type(e).__name__}: {e}. "
                f"Retrying in {sleep_s:.1f}s",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"[ckpt] Failed to save checkpoint to {final_p} after {retries} attempts: {last_err!r}"
    ) from last_err


def _build_save_path(args, run_id: str) -> str:
    base = os.path.basename(args.save)
    stem, ext = os.path.splitext(base)
    ext = ext if ext else ".pt"
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        return os.path.join(args.save_dir, f"{stem}_{run_id}{ext}")
    return os.path.join(os.path.dirname(args.save) or ".", f"{stem}_{run_id}{ext}")


def _normalize_path_str(p: str) -> str:
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return os.path.abspath(os.path.expanduser(str(p)))


def _check_split_paths_compatible(split_npz, current_paths, *, strict: bool = False):
    if "h5_paths" not in split_npz.files:
        return

    saved_paths = [str(p) for p in split_npz["h5_paths"].tolist()]
    cur_norm = [_normalize_path_str(p) for p in current_paths]
    saved_norm = [_normalize_path_str(p) for p in saved_paths]

    if saved_norm == cur_norm:
        return

    # part stems: a split made on the H5 parts also fits their packed copies (<stem>.packed)
    saved_base = [data_part_stem(p) for p in saved_norm]
    cur_base = [data_part_stem(p) for p in cur_norm]

    if saved_base == cur_base:
        if ddp_is_main():
            print(
                "[warn] Split-file paths differ from the current parts, but the ordered part names (H5 or packed) match. "
                "Proceeding with the split.",
                flush=True,
            )
        return

    if Counter(saved_base) == Counter(cur_base):
        if ddp_is_main():
            print(
                "[warn] Split-file H5 paths/order differ from current paths, but the basename multiset matches. "
                "Proceeding with the split.",
                flush=True,
            )
        return

    msg = (
        "Split file appears incompatible with the current H5 files.\n"
        f"  split has {len(saved_paths)} files, current glob resolved {len(current_paths)} files.\n"
        "  The normalized paths did not match, and neither did the file basenames.\n"
        "  Regenerate the split for this dataset."
    )
    if strict:
        raise RuntimeError(msg)
    if ddp_is_main():
        print(f"[warn] {msg}", flush=True)


@contextmanager
def timed_section(name: str, device: torch.device, enabled: bool = True):
    t = {"seconds": 0.0}
    if not enabled:
        yield t
        return

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t["seconds"] = time.perf_counter() - t0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def ddp_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1


def ddp_is_main() -> bool:
    return ddp_rank() == 0


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if not torch.cuda.is_available():
            raise SystemExit("[ddp] CUDA is not available but torchrun/DDP was requested.")

        n_visible = torch.cuda.device_count()
        if n_visible <= 0:
            raise SystemExit("[ddp] No CUDA devices visible.")

        if local_rank < 0 or local_rank >= n_visible:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
            raise SystemExit(
                f"[ddp] LOCAL_RANK={local_rank} but only {n_visible} CUDA device(s) are visible. "
                f"CUDA_VISIBLE_DEVICES={cvd}"
            )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")


def ddp_cleanup():
    if ddp_is_initialized():
        dist.destroy_process_group()


def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if ddp_is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def ddp_barrier():
    if ddp_is_initialized():
        dist.barrier()


def _stats_dict_to_arrays(stats_dict: Dict[str, Any], feature_names: List[str], *, kind: str):
    center = []
    scale = []

    if kind == "standard":
        ckey, skey = "mean", "std"
    elif kind == "robust":
        ckey, skey = "median", "iqr"
    else:
        raise ValueError(f"Unsupported normalization kind: {kind}")

    for name in feature_names:
        if name not in stats_dict:
            raise KeyError(f"Missing feature {name!r} in stats file.")
        row = stats_dict[name]
        if ckey not in row or skey not in row:
            raise KeyError(
                f"Feature {name!r} missing required keys {ckey!r}/{skey!r} for kind={kind!r}."
            )
        center.append(float(row[ckey]))
        scale.append(max(float(row[skey]), 1e-12))

    return (
        np.asarray(center, dtype=np.float32),
        np.asarray(scale, dtype=np.float32),
    )


def _pick_first_present(payload: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in payload:
            return k
    return None


def load_feature_stats_json(stats_path: str, norm_kind: str = "standard") -> Dict[str, np.ndarray]:
    with open(stats_path, "r") as f:
        payload = json.load(f)

    if norm_kind == "standard":
        mu_key = "mu_standard"
        ca_key = "ca_standard"
        edge_key = _pick_first_present(payload, ["edge_standard", "ed_standard"])
    elif norm_kind == "robust":
        mu_key = "mu_robust"
        ca_key = "ca_robust"
        edge_key = _pick_first_present(payload, ["edge_robust", "ed_robust"])
    else:
        raise ValueError(f"Unsupported norm_kind={norm_kind!r}")

    if mu_key not in payload:
        raise KeyError(f"Stats file missing top-level key {mu_key!r}")
    if ca_key not in payload:
        raise KeyError(f"Stats file missing top-level key {ca_key!r}")

    mu_center, mu_scale = _stats_dict_to_arrays(payload[mu_key], NODE_FEATURE_NAMES, kind=norm_kind)
    ca_center, ca_scale = _stats_dict_to_arrays(payload[ca_key], NODE_FEATURE_NAMES, kind=norm_kind)

    out = {
        "mu_center": mu_center,
        "mu_scale": mu_scale,
        "ca_center": ca_center,
        "ca_scale": ca_scale,
        "meta": {
            "mu_key": mu_key,
            "ca_key": ca_key,
            "edge_key": edge_key,
            "norm_kind": norm_kind,
            "stats_path": stats_path,
            "has_edge_stats": edge_key is not None,
        },
    }
    if edge_key is not None:
        edge_center, edge_scale = _stats_dict_to_arrays(payload[edge_key], EDGE_FEATURE_NAMES, kind=norm_kind)
        out["edge_center"] = edge_center
        out["edge_scale"] = edge_scale

    # Track block stats (DisplacedVertex_preproc.py on graphs with tracks). The track node stats describe the
    # features after the model-side asinh transform, whose scales are recorded in the meta block.
    tr_key = "tr_standard" if norm_kind == "standard" else "tr_robust"
    tr_edge_key = "track_edge_standard" if norm_kind == "standard" else "track_edge_robust"
    meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
    out["meta"]["has_track_stats"] = tr_key in payload
    out["meta"]["has_track_edge_stats"] = tr_edge_key in payload
    if tr_key in payload:
        if not meta.get("tr_stats_after_asinh", False):
            raise RuntimeError(
                f"{stats_path}: the track stats were computed without the asinh transform of the current "
                "DisplacedVertex_preproc.py. Rerun the preprocessing to use these stats with --use-tracks."
            )
        names = list(meta.get("track_feature_names", TRACK_FEATURE_NAMES))
        if names != list(TRACK_FEATURE_NAMES):
            raise RuntimeError(f"{stats_path}: track_feature_names {names} differ from {TRACK_FEATURE_NAMES}")
        out["tr_center"], out["tr_scale"] = _stats_dict_to_arrays(payload[tr_key], TRACK_FEATURE_NAMES, kind=norm_kind)
        out["tr_asinh_scale"] = np.asarray(meta["track_feature_asinh_scales"], dtype=np.float32)
    if tr_edge_key in payload:
        out["track_edge_center"], out["track_edge_scale"] = _stats_dict_to_arrays(
            payload[tr_edge_key], EDGE_FEATURE_NAMES, kind=norm_kind
        )

    # Optional segment blocks: segment-segment edges, tracklet nodes and tracklet edges
    for prefix, names in (("seg_edge", SEGMENT_EDGE_FEATURE_NAMES), ("tl", TRACKLET_FEATURE_NAMES),
                          ("tracklet_edge", TRACKLET_EDGE_FEATURE_NAMES)):
        key = f"{prefix}_standard" if norm_kind == "standard" else f"{prefix}_robust"
        out["meta"][f"has_{prefix}_stats"] = key in payload
        if key in payload:
            out[f"{prefix}_center"], out[f"{prefix}_scale"] = _stats_dict_to_arrays(payload[key], names, kind=norm_kind)
    # Meaning of the muon-node columns the mu_* stats were computed on (column 5: energy_like or chi2_ndof)
    out["meta"]["muon_node_feature_names"] = list(meta.get("muon_node_feature_names", DV_NODE_FEATURE_NAMES))

    return out


def _apply_feature_norm_np(arr: np.ndarray, center: np.ndarray, scale: np.ndarray, clip: float = -1.0) -> np.ndarray:
    out = (arr - center) / scale
    if clip is not None and clip > 0:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32, copy=False)


def _require_feature_stats(feature_stats, what: str):
    if feature_stats is None:
        raise RuntimeError(
            f"{what} normalization was requested but no feature stats were loaded. "
            f"Pass --feature-stats-json <path>."
        )


class H5EventDataset(Dataset):
    def __init__(
        self,
        h5_paths,
        *,
        normalize_node_features: bool = False,
        normalize_edge_features: bool = False,
        feature_stats: Optional[Dict[str, np.ndarray]] = None,
        feature_norm_clip: float = -1.0,
    ):
        self.h5_paths = list(h5_paths)
        self.normalize_node_features = bool(normalize_node_features)
        self.normalize_edge_features = bool(normalize_edge_features)
        self.feature_stats = feature_stats
        self.feature_norm_clip = float(feature_norm_clip)

        if self.normalize_node_features:
            _require_feature_stats(self.feature_stats, "Node-feature")
        if self.normalize_edge_features:
            _require_feature_stats(self.feature_stats, "Edge-feature")

        if not self.h5_paths:
            raise ValueError("No H5 files provided.")

        self.index = []
        for fi, p in enumerate(self.h5_paths):
            with h5py.File(p, "r") as f:
                if "events" not in f:
                    continue
                keys = sorted(list(f["events"].keys()))
                for k in keys:
                    self.index.append((fi, k))

        if not self.index:
            raise ValueError("No events found in provided H5 files.")

        self._files = None
        self._pid = None

    def __len__(self):
        return len(self.index)

    def _close_files(self):
        if self._files is None:
            return
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files = None

    def _ensure_open(self):
        pid = os.getpid()
        if self._files is not None and self._pid == pid:
            return
        self._close_files()
        self._pid = pid
        self._files = [h5py.File(p, "r") for p in self.h5_paths]
        atexit.register(self._close_files)

    def __getitem__(self, idx):
        self._ensure_open()
        assert self._files is not None

        fi, k = self.index[idx]
        f = self._files[fi]
        g = f["events"][k]

        x = torch.from_numpy(g["x"][...]).float()
        edge_index = torch.from_numpy(g["edge_index"][...]).long()
        edge_attr = torch.from_numpy(g["edge_attr"][...]).float()

        if "y_vertex" not in g:
            raise RuntimeError(f"Missing 'y_vertex' in {self.h5_paths[fi]} /events/{k}")

        y_vertex = torch.from_numpy(g["y_vertex"][...]).float()

        if self.normalize_node_features:
            if "n_muon_nodes" not in g.attrs:
                raise RuntimeError(
                    f"Missing attribute 'n_muon_nodes' in {self.h5_paths[fi]} /events/{k}; "
                    "cannot split muon/calo nodes for feature normalization."
                )
            n_mu = int(g.attrs["n_muon_nodes"])
            x_np = x.numpy()
            if n_mu > 0:
                x_np[:n_mu] = _apply_feature_norm_np(
                    x_np[:n_mu], self.feature_stats["mu_center"], self.feature_stats["mu_scale"],
                    clip=self.feature_norm_clip,
                )
            if n_mu < x_np.shape[0]:
                x_np[n_mu:] = _apply_feature_norm_np(
                    x_np[n_mu:], self.feature_stats["ca_center"], self.feature_stats["ca_scale"],
                    clip=self.feature_norm_clip,
                )

        if self.normalize_edge_features:
            edge_attr_np = edge_attr.numpy()
            edge_attr_np[:] = _apply_feature_norm_np(
                edge_attr_np, self.feature_stats["edge_center"], self.feature_stats["edge_scale"],
                clip=self.feature_norm_clip,
            )

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y_vertex": y_vertex,
        }


def collate_one(batch):
    assert len(batch) == 1
    return batch[0]


@dataclass
class EMA:
    decay: float
    shadow: dict

    @staticmethod
    def create(model: nn.Module, decay: float):
        raw = model.module if hasattr(model, "module") else model
        shadow = {k: v.detach().clone() for k, v in raw.state_dict().items()}
        return EMA(decay=decay, shadow=shadow)

    @torch.no_grad()
    def update(self, model: nn.Module):
        raw = model.module if hasattr(model, "module") else model
        msd = raw.state_dict()
        for k, v in msd.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def apply_to(self, model: nn.Module):
        raw = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            cur = raw.state_dict()
            backup = {}
            for k, v in cur.items():
                backup[k] = v.detach().clone()
                if k in self.shadow:
                    v.copy_(self.shadow[k])
        try:
            yield
        finally:
            with torch.no_grad():
                cur2 = raw.state_dict()
                for k, v in cur2.items():
                    if k in backup:
                        v.copy_(backup[k])


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, n_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CostumEdgeConvLayer(nn.Module):
    def __init__(self, nn_module: nn.Module, aggregation: str = "mean", add_self_loops: bool = True):
        super().__init__()
        self.nn = nn_module
        self.aggregation = aggregation
        self.add_self_loops = add_self_loops

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        if self.add_self_loops:
            self_loops = torch.arange(x.size(0), device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            src = edge_index[0]
            dst = edge_index[1]

        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_out = self.nn(edge_features)
        aggregated_out = torch.zeros((x.size(0), edge_out.size(1)), device=x.device, dtype=edge_out.dtype)

        if self.aggregation == "mean":
            index = dst.unsqueeze(-1).expand_as(edge_out)
            aggregated_out.scatter_add_(0, index, edge_out)
            counts = torch.zeros(x.size(0), device=x.device, dtype=edge_out.dtype)
            counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=edge_out.dtype))
            aggregated_out = aggregated_out / counts.clamp(min=1).unsqueeze(-1)
        elif self.aggregation == "max":
            aggregated_out = torch.full_like(aggregated_out, float("-inf"))
            index = dst.unsqueeze(-1).expand_as(edge_out)
            if hasattr(aggregated_out, "scatter_reduce_"):
                aggregated_out.scatter_reduce_(0, index, edge_out, reduce="amax", include_self=True)
            else:
                aggregated_out.scatter_(0, index, edge_out)
            aggregated_out = torch.where(torch.isfinite(aggregated_out), aggregated_out, torch.zeros_like(aggregated_out))
        elif self.aggregation == "sum":
            index = dst.unsqueeze(-1).expand_as(edge_out)
            aggregated_out.scatter_add_(0, index, edge_out)
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggregation}")

        return aggregated_out


class CustomGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: int = 0,
        edge_attention: bool = False,
        gatv2_edge_attention: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.edge_dim = int(edge_dim or 0)
        self.edge_attention = bool(edge_attention)
        self.gatv2_edge_attention = bool(gatv2_edge_attention)
        if self.edge_attention and self.gatv2_edge_attention:
            raise ValueError(
                "GAT edge-bias attention and GATv2 source-destination-edge attention "
                "are mutually exclusive."
            )
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        if self.gatv2_edge_attention:
            # These parameters belong to the baseline/edge-bias GAT scorer and
            # would be unused in DDP for the GATv2 scorer.
            self.register_parameter("attn_l", None)
            self.register_parameter("attn_r", None)
        else:
            self.attn_l = nn.Parameter(torch.empty(1, heads, out_channels))
            self.attn_r = nn.Parameter(torch.empty(1, heads, out_channels))
            nn.init.xavier_uniform_(self.attn_l)
            nn.init.xavier_uniform_(self.attn_r)

        if self.edge_attention:
            if self.edge_dim <= 0:
                raise ValueError("edge_dim must be positive when GAT edge attention is enabled")
            edge_hidden = max(8, 2 * self.edge_dim)
            self.edge_attn_encoder = nn.Sequential(
                nn.LayerNorm(self.edge_dim),
                nn.Linear(self.edge_dim, edge_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(edge_hidden, heads, bias=False),
            )
        else:
            self.edge_attn_encoder = None

        if self.gatv2_edge_attention:
            if self.edge_dim <= 0:
                raise ValueError(
                    "edge_dim must be positive when GATv2 source-destination-edge "
                    "attention is enabled"
                )
            # This is an Edge-GATv2-style attention MLP:
            #
            #   z_ij = W_src h_j + W_dst h_i + W_edge e_ij
            #   score_ij = a^T SiLU(z_ij)
            #
            # The three separately projected terms are equivalent to one linear
            # layer applied to concat([h_j, h_i, e_ij]), while avoiding a large
            # explicitly materialized concatenation for every edge and head.
            self.gatv2_attn_src = nn.Linear(
                in_channels, heads * out_channels, bias=False
            )
            self.gatv2_attn_dst = nn.Linear(
                in_channels, heads * out_channels, bias=False
            )
            self.gatv2_attn_edge = nn.Linear(
                self.edge_dim, heads * out_channels, bias=False
            )
            self.gatv2_attn_score = nn.Parameter(
                torch.empty(1, heads, out_channels)
            )
            # Keep the value/message path independent from the attention path.
            # Edge attributes therefore control both which neighbours matter
            # and what vector each neighbour transmits:
            #
            #   m_ij = SiLU(V_src h_j + V_dst h_i + V_edge e_ij)
            #
            # Each projection is equivalent to one linear layer on
            # concat([h_j, h_i, e_ij]), without explicitly materializing that
            # large per-edge tensor.
            self.gatv2_msg_dst = nn.Linear(
                in_channels, heads * out_channels, bias=False
            )
            self.gatv2_msg_edge = nn.Linear(
                self.edge_dim, heads * out_channels, bias=False
            )
            nn.init.xavier_uniform_(self.gatv2_attn_src.weight)
            nn.init.xavier_uniform_(self.gatv2_attn_dst.weight)
            nn.init.xavier_uniform_(self.gatv2_attn_edge.weight)
            nn.init.xavier_uniform_(self.gatv2_attn_score)
            nn.init.xavier_uniform_(self.gatv2_msg_dst.weight)
            nn.init.xavier_uniform_(self.gatv2_msg_edge.weight)
        else:
            self.gatv2_attn_src = None
            self.gatv2_attn_dst = None
            self.gatv2_attn_edge = None
            self.register_parameter("gatv2_attn_score", None)
            self.gatv2_msg_dst = None
            self.gatv2_msg_edge = None

        if not concat:
            self.out_proj = nn.Linear(heads * out_channels, out_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_dropout_p: float = 0.0,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        node_input = x
        x = self.linear(node_input).view(num_nodes, self.heads, self.out_channels)
        out_dtype = x.dtype
        x_f = x.float()

        if self.gatv2_edge_attention:
            if (
                self.gatv2_attn_src is None
                or self.gatv2_attn_dst is None
                or self.gatv2_attn_edge is None
                or self.gatv2_attn_score is None
                or self.gatv2_msg_dst is None
                or self.gatv2_msg_edge is None
            ):
                raise RuntimeError(
                    "GATv2 attention/message modules were not initialized."
                )
            gatv2_src_f = self.gatv2_attn_src(node_input).view(
                num_nodes, self.heads, self.out_channels
            ).float()
            gatv2_dst_f = self.gatv2_attn_dst(node_input).view(
                num_nodes, self.heads, self.out_channels
            ).float()
            gatv2_msg_dst_f = self.gatv2_msg_dst(node_input).view(
                num_nodes, self.heads, self.out_channels
            ).float()
        else:
            gatv2_src_f = None
            gatv2_dst_f = None
            gatv2_msg_dst_f = None

        if self.edge_attention or self.gatv2_edge_attention:
            if edge_attr is None:
                raise RuntimeError("Edge-aware GAT attention requires edge_attr.")
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise RuntimeError(
                    f"edge_attr rows ({edge_attr.shape[0]}) do not match edges ({edge_index.shape[1]})."
                )
            if edge_attr.shape[1] != self.edge_dim:
                raise RuntimeError(
                    f"edge_attr dim ({edge_attr.shape[1]}) does not match configured edge_dim ({self.edge_dim})."
                )

        if self.training and edge_dropout_p > 0.0 and edge_index.shape[1] > 0:
            keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= float(edge_dropout_p)
            if not bool(keep.any()):
                keep[torch.randint(0, edge_index.shape[1], (1,), device=edge_index.device)] = True
            edge_index = edge_index[:, keep]
            if edge_attr is not None:
                edge_attr = edge_attr[keep]

        if self.add_self_loops:
            self_loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            if self.edge_attention or self.gatv2_edge_attention:
                if edge_attr is None:
                    raise RuntimeError("Edge-aware GAT attention requires edge_attr.")
                loop_attr = torch.zeros(
                    (num_nodes, edge_attr.shape[1]),
                    device=edge_attr.device,
                    dtype=edge_attr.dtype,
                )
                edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        if self.edge_attention or self.gatv2_edge_attention:
            if edge_attr is None:
                raise RuntimeError("Edge-aware GAT attention requires edge_attr.")
            if edge_attr.shape[0] != edge_index.shape[1]:
                raise RuntimeError(
                    f"edge_attr rows ({edge_attr.shape[0]}) do not match edges ({edge_index.shape[1]})."
                )
            if edge_attr.shape[1] != self.edge_dim:
                raise RuntimeError(
                    f"edge_attr dim ({edge_attr.shape[1]}) does not match configured edge_dim ({self.edge_dim})."
                )

        src = edge_index[0]
        dst = edge_index[1]

        if self.gatv2_edge_attention:
            if (
                edge_attr is None
                or gatv2_src_f is None
                or gatv2_dst_f is None
                or self.gatv2_attn_edge is None
                or self.gatv2_attn_score is None
                or gatv2_msg_dst_f is None
                or self.gatv2_msg_edge is None
            ):
                raise RuntimeError(
                    "GATv2 attention/message inputs are unavailable."
                )
            edge_f = self.gatv2_attn_edge(edge_attr.float()).view(
                edge_attr.shape[0], self.heads, self.out_channels
            ).float()
            alpha_hidden = gatv2_src_f[src] + gatv2_dst_f[dst] + edge_f
            alpha = (
                F.silu(alpha_hidden) * self.gatv2_attn_score.float()
            ).sum(dim=-1)
            msg_edge_f = self.gatv2_msg_edge(edge_attr.float()).view(
                edge_attr.shape[0], self.heads, self.out_channels
            ).float()
            messages = F.silu(
                x_f[src] + gatv2_msg_dst_f[dst] + msg_edge_f
            )
        else:
            if self.attn_l is None or self.attn_r is None:
                raise RuntimeError("Baseline GAT attention parameters are unavailable.")
            alpha_l = (x_f[src] * self.attn_l).sum(dim=-1)
            alpha_r = (x_f[dst] * self.attn_r).sum(dim=-1)
            alpha_raw = alpha_l + alpha_r
            if self.edge_attn_encoder is not None:
                alpha_raw = alpha_raw + self.edge_attn_encoder(edge_attr.float())
            alpha = F.leaky_relu(alpha_raw, negative_slope=0.2)
            messages = x_f[src]

        # Stable segment softmax: every destination node/head needs its own
        # maximum. A single global maximum makes otherwise-disjoint graphs in
        # a batch affect one another through floating-point underflow.
        alpha_max = torch.full(
            (num_nodes, self.heads),
            -torch.inf,
            device=x.device,
            dtype=torch.float32,
        )
        alpha_max.scatter_reduce_(
            0,
            dst.unsqueeze(-1).expand_as(alpha),
            alpha,
            reduce="amax",
            include_self=True,
        )
        alpha = torch.exp(alpha - alpha_max[dst])
        alpha_sum = torch.zeros((num_nodes, self.heads), device=x.device, dtype=torch.float32)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / alpha_sum[dst].clamp(min=1e-6)
        alpha = self.dropout(alpha)

        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=torch.float32)
        for h in range(self.heads):
            out[:, h].scatter_add_(
                0,
                dst.unsqueeze(-1).expand_as(messages[:, h]),
                alpha[:, h].unsqueeze(-1) * messages[:, h],
            )

        if self.concat:
            return out.view(num_nodes, self.heads * self.out_channels).to(out_dtype)
        out = out.mean(dim=1)
        return self.out_proj(out.to(out_dtype))


class CustomSAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean", normalize: bool = True, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]
        agg_features = torch.zeros((num_nodes, self.in_channels), device=x.device, dtype=x.dtype)

        if self.aggr == "mean":
            agg_features.scatter_add_(0, dst.unsqueeze(-1).expand_as(x[src]), x[src])
            counts = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            agg_features = agg_features / counts.clamp(min=1).unsqueeze(-1)
        elif self.aggr == "sum":
            agg_features.scatter_add_(0, dst.unsqueeze(-1).expand_as(x[src]), x[src])
        elif self.aggr == "max":
            agg_features = torch.full((num_nodes, self.in_channels), float("-inf"), device=x.device, dtype=x.dtype)
            index = dst.unsqueeze(-1).expand_as(x[src])
            if hasattr(agg_features, "scatter_reduce_"):
                agg_features.scatter_reduce_(0, index, x[src], reduce="amax", include_self=True)
            else:
                agg_features.scatter_(0, index, x[src])
            agg_features = torch.where(torch.isfinite(agg_features), agg_features, torch.zeros_like(agg_features))
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggr}")

        h_neigh = self.lin_neigh(agg_features)
        h_self = self.lin_self(x)
        h = h_self + h_neigh
        if self.bias is not None:
            h = h + self.bias
        if self.normalize:
            h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))
            h = F.normalize(h, p=2.0, dim=-1)
        return h


class EdgeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float, aggr: str = "mean", add_self_loops: bool = True):
        super().__init__()
        self.project = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None
        self.edge_conv1 = CostumEdgeConvLayer(
            nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            aggregation=aggr, add_self_loops=add_self_loops,
        )
        self.edge_conv2 = CostumEdgeConvLayer(
            nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            aggregation=aggr, add_self_loops=add_self_loops,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.project is not None:
            x = self.project(x)
        identity = x
        x = F.relu(self.edge_conv1(x, edge_index))
        x = self.edge_conv2(x, edge_index)
        x = self.dropout(x)
        return identity + x


class GATResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: int = 0,
        edge_attention: bool = False,
        gatv2_edge_attention: bool = False,
    ):
        super().__init__()
        self.project = nn.Linear(in_channels, out_channels * heads) if in_channels != out_channels * heads else None
        self.gat = CustomGAT(
            in_channels=(out_channels * heads if self.project else in_channels),
            out_channels=out_channels,
            heads=heads, dropout=dropout, add_self_loops=True, concat=True,
            edge_dim=edge_dim, edge_attention=edge_attention,
            gatv2_edge_attention=gatv2_edge_attention,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_dropout_p: float = 0.0,
    ) -> torch.Tensor:
        if self.project is not None:
            x = self.project(x)
        identity = x
        x = F.relu(
            self.gat(
                x,
                edge_index,
                edge_attr=edge_attr,
                edge_dropout_p=edge_dropout_p,
            )
        )
        x = self.dropout(x)
        return identity + x


class SAGEResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean", dropout: float = 0.2, normalize: bool = True):
        super().__init__()
        self.project = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.conv = CustomSAGEConv(in_channels=out_channels, out_channels=out_channels, aggr=aggr, normalize=normalize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.project is not None:
            x = self.project(x)
        identity = x
        x = F.relu(self.conv(x, edge_index))
        x = self.dropout(x)
        return identity + x


class FourierEncoder(nn.Module):
    def __init__(self, xdim: int, base: float = 3.0, min_exp: int = -6, max_exp: int = 6):
        super().__init__()
        self.exps = list(range(int(min_exp), int(max_exp) + 1))
        divs = torch.tensor([float(base) ** e for e in self.exps], dtype=torch.float32)
        self.register_buffer("divs", divs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.unsqueeze(-1) / self.divs.view(1, 1, -1)
        out = torch.cat([torch.sin(x3), torch.cos(x3)], dim=-1)
        # flatten(1) == reshape(N, -1), but also valid for N = 0 (e.g. an event without track nodes)
        return out.flatten(1)


class EdgeMPNNLayer(nn.Module):
    def __init__(self, hdim, edim, msg_hidden=128, upd_hidden=128, dropout=0.0):
        super().__init__()
        self.edge_mlp = MLP(in_dim=2 * hdim + edim, out_dim=hdim, hidden_dim=msg_hidden, n_layers=2, dropout=dropout)
        self.node_mlp = MLP(in_dim=2 * hdim, out_dim=hdim, hidden_dim=upd_hidden, n_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hdim)

    def _ddp_touch_edge_mlp(self) -> torch.Tensor:
        z = None
        for p in self.edge_mlp.parameters():
            z = (p.sum() * 0.0) if z is None else (z + p.sum() * 0.0)
        return z if z is not None else torch.tensor(0.0)

    def forward(self, h, edge_index, edge_attr, edge_dropout_p: float = 0.0):
        E = edge_attr.size(0)

        if E == 0:
            agg = torch.zeros_like(h)
            h_upd = self.node_mlp(torch.cat([h, agg], dim=1))
            touch = self._ddp_touch_edge_mlp().to(h.device)
            return self.norm(h + h_upd) + touch

        src = edge_index[0]
        dst = edge_index[1]

        if self.training and edge_dropout_p > 0.0:
            keep = (torch.rand(E, device=edge_attr.device) >= edge_dropout_p)
            if keep.sum().item() == 0:
                j = torch.randint(0, E, (1,), device=keep.device).item()
                keep[j] = True
            src = src[keep]
            dst = dst[keep]
            edge_attr = edge_attr[keep]

        if edge_attr.size(0) == 0:
            agg = torch.zeros_like(h)
            h_upd = self.node_mlp(torch.cat([h, agg], dim=1))
            return self.norm(h + h_upd)

        h_src = h[src]
        h_dst = h[dst]
        m_in = torch.cat([h_src, h_dst, edge_attr], dim=1)
        m = self.edge_mlp(m_in)

        agg = torch.zeros((h.size(0), m.size(1)), device=h.device, dtype=m.dtype)
        agg.index_add_(0, dst, m.to(agg.dtype))
        if agg.dtype != h.dtype:
            agg = agg.to(h.dtype)

        h_upd = self.node_mlp(torch.cat([h, agg], dim=1))
        return self.norm(h + h_upd)


def global_pool(h: torch.Tensor, mode: str = "meanmax") -> torch.Tensor:
    if h.ndim != 2:
        raise ValueError(f"Expected h to have shape [N, H], got {tuple(h.shape)}")

    if mode == "mean":
        return h.mean(dim=0, keepdim=True)
    elif mode == "max":
        return h.max(dim=0, keepdim=True).values
    elif mode == "sum":
        return h.sum(dim=0, keepdim=True)
    elif mode == "meanmax":
        g_mean = h.mean(dim=0, keepdim=True)
        g_max = h.max(dim=0, keepdim=True).values
        return torch.cat([g_mean, g_max], dim=-1)
    else:
        raise ValueError(f"Unknown pool mode: {mode}")


def global_pool_batched(
    h: torch.Tensor,
    batch: Optional[torch.Tensor],
    mode: str = "meanmax",
) -> torch.Tensor:
    """Pool a concatenated graph batch without requiring torch-geometric."""
    if batch is None:
        return global_pool(h, mode=mode)
    batch = batch.to(device=h.device, dtype=torch.long).reshape(-1)
    if batch.numel() != h.shape[0]:
        raise ValueError(f"batch has {batch.numel()} entries for {h.shape[0]} nodes")
    if batch.numel() == 0:
        raise ValueError("Cannot pool an empty graph batch")
    n_graphs = int(batch.max().item()) + 1
    sums = h.new_zeros((n_graphs, h.shape[1]))
    sums.index_add_(0, batch, h)
    counts = h.new_zeros((n_graphs, 1))
    counts.index_add_(0, batch, h.new_ones((h.shape[0], 1)))
    means = sums / counts.clamp_min(1.0)
    if mode == "sum":
        return sums
    if mode == "mean":
        return means
    if mode in ("max", "meanmax"):
        maxima = h.new_full((n_graphs, h.shape[1]), -torch.inf)
        maxima.scatter_reduce_(0, batch[:, None].expand(-1, h.shape[1]), h, reduce="amax", include_self=True)
        return maxima if mode == "max" else torch.cat([means, maxima], dim=-1)
    raise ValueError(f"Unknown pool mode: {mode}")


class DisplacedVertexGNN(nn.Module):
    def __init__(
        self,
        xdim,
        edim,
        hdim=128,
        n_layers=5,
        dropout=0.020720971979495448,
        layer_type: str = "gat_residual",
        gat_heads: int = 2,
        gat_edge_attn: bool = False,
        gatv2_edge_attn: bool = True,
        sage_aggr: str = "mean",
        edgeconv_aggr: str = "mean",
        pool: str = "meanmax",
        use_fourier=False,
        fourier_base=3.0,
        fourier_min_exp=-6,
        fourier_max_exp=6,
        phi_mode: str = "sincos",
        phi_index: int = 1,
    ):
        super().__init__()

        self.fourier = None
        self.pool = pool
        self.phi_mode = phi_mode
        self.phi_index = int(phi_index)

        if use_fourier:
            self.fourier = FourierEncoder(
                xdim, base=fourier_base, min_exp=fourier_min_exp, max_exp=fourier_max_exp,
            )
            xdim_in = xdim * 2 * (fourier_max_exp - fourier_min_exp + 1)
        else:
            xdim_in = xdim

        self.node_enc = MLP(xdim_in, hdim, hidden_dim=hdim, n_layers=2, dropout=dropout)

        if layer_type == "mpnn":
            self.layers = nn.ModuleList([EdgeMPNNLayer(hdim, edim, dropout=dropout) for _ in range(n_layers)])
            self._uses_edge_attr = True
        elif layer_type == "edge_residual":
            self.layers = nn.ModuleList([
                EdgeResidualBlock(in_channels=hdim, hidden_channels=hdim, dropout=dropout, aggr=edgeconv_aggr)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
        elif layer_type == "sage_residual":
            self.layers = nn.ModuleList([
                SAGEResidualBlock(in_channels=hdim, out_channels=hdim, aggr=sage_aggr, dropout=dropout)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
        elif layer_type == "gat_residual":
            if gat_edge_attn and gatv2_edge_attn:
                raise ValueError(
                    "--gat-edge-attn and --gatv2-edge-attn are mutually exclusive"
                )
            if gat_heads <= 0:
                raise ValueError("--gat-heads must be >= 1")
            if hdim % gat_heads != 0:
                raise ValueError(f"hidden_dim={hdim} must be divisible by gat_heads={gat_heads}")
            per_head = hdim // gat_heads
            self.layers = nn.ModuleList([
                GATResidualBlock(
                    in_channels=hdim, out_channels=per_head, heads=gat_heads, dropout=dropout,
                    edge_dim=edim, edge_attention=gat_edge_attn,
                    gatv2_edge_attention=gatv2_edge_attn,
                )
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = bool(gat_edge_attn or gatv2_edge_attn)
        else:
            raise ValueError(f"Unknown layer_type={layer_type}")

        graph_dim = hdim * 2 if pool == "meanmax" else hdim
        self.heads = nn.ModuleList()
        for i in range(3):
            out_dim = 2 if (i == self.phi_index and self.phi_mode == "sincos") else 1
            self.heads.append(MLP(in_dim=graph_dim, out_dim=out_dim, hidden_dim=hdim, n_layers=3, dropout=dropout))

    def forward(self, x, edge_index, edge_attr, edge_dropout_p: float = 0.0):
        if self.fourier is not None:
            x = self.fourier(x)

        h = self.node_enc(x)

        for layer in self.layers:
            if self._uses_edge_attr:
                h = layer(h, edge_index, edge_attr, edge_dropout_p=edge_dropout_p)
            elif isinstance(layer, GATResidualBlock):
                h = layer(h, edge_index, edge_dropout_p=edge_dropout_p)
            else:
                h = layer(h, edge_index)

        g = global_pool(h, mode=self.pool)
        out = []
        for head in self.heads:
            v = head(g).squeeze(0)
            out.append(v)
        return out




# ============================================================
# Classification dataset / model / training
# ============================================================

class H5EventDataset(Dataset):
    """Lazy HDF5 dataset for raw DisplacedVertex graph-level classification inputs."""

    def __init__(
        self,
        h5_paths,
        *,
        event_refs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        dataset_names: Optional[np.ndarray] = None,
        root_files: Optional[np.ndarray] = None,
        max_open_h5_files: int = 16,
        use_tracks: bool = False,
        use_segment_edges: bool = False,
        use_tracklets: bool = False,
    ):
        self.h5_paths = list(h5_paths)
        self.max_open_h5_files = max(1, int(max_open_h5_files))
        # Also return the track block (track_x, track_edge_index, track_edge_attr)
        self.use_tracks = bool(use_tracks)
        # Also return the segment-segment edges (seg_edge_index, seg_edge_attr)
        self.use_segment_edges = bool(use_segment_edges)
        # Also return the tracklet block (tracklet_x, tracklet_edge_index, tracklet_edge_attr)
        self.use_tracklets = bool(use_tracklets)

        if not self.h5_paths:
            raise ValueError("No H5 files provided.")

        self.index: List[Tuple[int, str]] = []

        if event_refs is not None:
            refs = np.asarray(event_refs, dtype=object)
            for ref in refs:
                if len(ref) != 2:
                    raise ValueError(f"Expected event_ref=(file_idx, event_key), got {ref!r}")
                fi = int(ref[0])
                if fi < 0 or fi >= len(self.h5_paths):
                    raise ValueError(f"Split event_ref file index {fi} is out of range for {len(self.h5_paths)} H5 files.")
                self.index.append((fi, str(ref[1])))

            if labels is None:
                raise ValueError("labels must be provided when event_refs are used.")
            labels_arr = np.asarray(labels, dtype=np.float32).reshape(-1)
            if labels_arr.shape[0] != len(self.index):
                raise ValueError(f"labels length {labels_arr.shape[0]} does not match event_refs length {len(self.index)}")

            if dataset_names is None:
                dataset_names_arr = np.asarray(["unknown"] * len(self.index), dtype=object)
            else:
                dataset_names_arr = np.asarray(dataset_names, dtype=object).reshape(-1)
            if root_files is None:
                root_files_arr = np.asarray(["unknown"] * len(self.index), dtype=object)
            else:
                root_files_arr = np.asarray(root_files, dtype=object).reshape(-1)

            if dataset_names_arr.shape[0] != len(self.index):
                raise ValueError("dataset_names length does not match event_refs length")
            if root_files_arr.shape[0] != len(self.index):
                raise ValueError("root_files length does not match event_refs length")
        else:
            labels_list: List[float] = []
            dataset_names_list: List[str] = []
            root_files_list: List[str] = []

            for fi, p in enumerate(self.h5_paths):
                with h5py.File(p, "r") as f:
                    if "events" not in f:
                        continue
                    for k in sorted(list(f["events"].keys())):
                        g = f["events"][k]
                        y_np = self._read_label_from_group(g, file_path=p, event_key=k)
                        self.index.append((fi, k))
                        labels_list.append(float(y_np.reshape(-1)[0]))
                        dataset_names_list.append(_decode_attr_to_str(g.attrs.get("dataset_name", "unknown")))
                        root_files_list.append(_decode_attr_to_str(g.attrs.get("root_file", "unknown")))

            labels_arr = np.asarray(labels_list, dtype=np.float32)
            dataset_names_arr = np.asarray(dataset_names_list, dtype=object)
            root_files_arr = np.asarray(root_files_list, dtype=object)

        if not self.index:
            raise ValueError("No events found in provided H5 files.")

        self.labels_np = labels_arr
        self.dataset_names = dataset_names_arr
        self.root_files = root_files_arr
        self._files = OrderedDict()
        self._pid = None

    @staticmethod
    def _read_label_from_group(g, *, file_path: str, event_key: str) -> np.ndarray:
        if "y" in g:
            y = g["y"][...]
        elif "labels" in g:
            y = g["labels"][...]
        elif "label" in g.attrs:
            y = np.asarray([g.attrs["label"]], dtype=np.float32)
        else:
            raise RuntimeError(
                f"Missing classifier label 'y'/'labels' in {file_path} /events/{event_key}. "
                "This classifier trainer expects graph_classification H5 files."
            )
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size != 1:
            raise RuntimeError(f"Expected scalar binary label in {file_path} /events/{event_key}, got shape {y.shape}.")
        if not np.isfinite(y[0]) or y[0] not in (0.0, 1.0):
            raise RuntimeError(f"Expected label 0/1 in {file_path} /events/{event_key}, got {y[0]!r}.")
        return y.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.index)

    def _close_files(self):
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()

    def _ensure_process(self):
        pid = os.getpid()
        if self._pid == pid:
            return
        self._close_files()
        self._pid = pid
        atexit.register(self._close_files)

    def _get_file(self, fi: int):
        self._ensure_process()
        if fi in self._files:
            self._files.move_to_end(fi)
            return self._files[fi]
        while len(self._files) >= self.max_open_h5_files:
            _, old_file = self._files.popitem(last=False)
            old_file.close()
        f = h5py.File(self.h5_paths[fi], "r")
        self._files[fi] = f
        return f

    def get_label(self, idx: int) -> float:
        return float(self.labels_np[int(idx)])

    def __getitem__(self, idx):
        fi, k = self.index[idx]
        f = self._get_file(fi)
        g = f["events"][k]

        x = torch.from_numpy(g["x"][...]).float()
        edge_index = torch.from_numpy(g["edge_index"][...]).long()
        edge_attr = torch.from_numpy(g["edge_attr"][...]).float()
        y = torch.from_numpy(self._read_label_from_group(g, file_path=self.h5_paths[fi], event_key=k)).float()

        if edge_index.ndim == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        if "n_muon_nodes" not in g.attrs:
            raise RuntimeError(
                f"Missing attribute 'n_muon_nodes' in {self.h5_paths[fi]} /events/{k}; "
                "cannot split muon/calo nodes for model-side feature normalization."
            )
        n_muon_nodes = torch.tensor(int(g.attrs["n_muon_nodes"]), dtype=torch.long)

        item = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": y,
            "labels": y,
            "n_muon_nodes": n_muon_nodes,
        }
        if self.use_tracks:
            missing = [name for name in ("track_x", "track_edge_index", "track_edge_attr") if name not in g]
            if missing:
                raise RuntimeError(
                    f"{self.h5_paths[fi]} /events/{k} has no track block ({missing}); convert the inputs with "
                    "DisplacedVertex_converter.py --track-tree-name TrigFSTrackDump, or train without --use-tracks."
                )
            item["track_x"] = torch.from_numpy(g["track_x"][...]).float().reshape(-1, len(TRACK_FEATURE_NAMES))
            item["track_edge_index"] = torch.from_numpy(g["track_edge_index"][...]).long().reshape(2, -1)
            item["track_edge_attr"] = torch.from_numpy(g["track_edge_attr"][...]).float().reshape(-1, len(EDGE_FEATURE_NAMES))
        if self.use_segment_edges:
            missing = [name for name in ("seg_edge_index", "seg_edge_attr") if name not in g]
            if missing:
                raise RuntimeError(
                    f"{self.h5_paths[fi]} /events/{k} has no segment-segment edges ({missing}); convert the inputs "
                    "with DisplacedVertex_converter.py --segment-edges, or train without --use-segment-edges."
                )
            item["seg_edge_index"] = torch.from_numpy(g["seg_edge_index"][...]).long().reshape(2, -1)
            item["seg_edge_attr"] = torch.from_numpy(g["seg_edge_attr"][...]).float().reshape(-1, len(SEGMENT_EDGE_FEATURE_NAMES))
        if self.use_tracklets:
            missing = [name for name in ("tracklet_x", "tracklet_edge_index", "tracklet_edge_attr") if name not in g]
            if missing:
                raise RuntimeError(
                    f"{self.h5_paths[fi]} /events/{k} has no tracklet block ({missing}); convert the inputs with "
                    "DisplacedVertex_converter.py --tracklets, or train without --use-tracklets."
                )
            item["tracklet_x"] = torch.from_numpy(g["tracklet_x"][...]).float().reshape(-1, len(TRACKLET_FEATURE_NAMES))
            item["tracklet_edge_index"] = torch.from_numpy(g["tracklet_edge_index"][...]).long().reshape(2, -1)
            item["tracklet_edge_attr"] = torch.from_numpy(g["tracklet_edge_attr"][...]).float().reshape(
                -1, len(TRACKLET_EDGE_FEATURE_NAMES))
        return item


# ---------------------------------------------------------------------------------------------------------------------
# Packed parts (DisplacedVertex_converter.py --output-format packed, the default, or pack_DisplacedVertex_dataset.py
# for H5 parts; format in dv_converter_utils): memory-mapped by the DataLoader workers (an event = a few array slices
# instead of ~12 gzip H5 datasets, ~20x faster; the processes of a node share the page cache).
# ---------------------------------------------------------------------------------------------------------------------
class PackedEventDataset(Dataset):
    """Training feed from packed parts: same interface and the same tensors as H5EventDataset over the H5 parts they
    were packed from (index, labels_np, dataset_names, root_files, get_label, __getitem__)."""

    def __init__(
        self,
        part_dirs,
        *,
        event_refs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        dataset_names: Optional[np.ndarray] = None,
        root_files: Optional[np.ndarray] = None,
        use_tracks: bool = False,
        use_segment_edges: bool = False,
        use_tracklets: bool = False,
        **_ignored,
    ):
        self.part_dirs = [str(p) for p in part_dirs]
        self.h5_paths = self.part_dirs  # name used by the callers of H5EventDataset
        if not self.part_dirs:
            raise ValueError("No packed parts provided.")
        self.blocks = ["x", "edge_index", "edge_attr"]
        wanted = {"use_tracks": use_tracks, "use_segment_edges": use_segment_edges, "use_tracklets": use_tracklets}
        for flag, names in DV_PACKED_OPTIONAL.items():
            if wanted[flag]:
                self.blocks += list(names)
        keys_per_part = []
        for pd in self.part_dirs:
            meta = read_packed_meta(pd)
            if meta.get("format") != DV_PACKED_FORMAT:
                raise RuntimeError(f"{pd}: not a {DV_PACKED_FORMAT} part (format={meta.get('format')!r})")
            missing = [b for b in self.blocks if b not in meta["blocks"]]
            if missing:
                raise RuntimeError(f"{pd} has no {missing}: pack H5 parts converted with these blocks, or train "
                                   "without the corresponding --use-* option.")
            keys_per_part.append(np.load(os.path.join(pd, "event_key.npy")))

        if event_refs is not None:
            rows = [{str(k): j for j, k in enumerate(keys)} for keys in keys_per_part]
            refs = np.asarray(event_refs, dtype=object)
            self.index = []
            self._rows = np.empty(len(refs), dtype=np.int64)
            for i, ref in enumerate(refs):
                fi, key = int(ref[0]), str(ref[1])
                if fi < 0 or fi >= len(self.part_dirs):
                    raise ValueError(f"Split event_ref file index {fi} is out of range for {len(self.part_dirs)} parts.")
                if key not in rows[fi]:
                    raise ValueError(f"Split event {key} is not in {self.part_dirs[fi]}: the split was made on "
                                     "other parts.")
                self.index.append((fi, key))
                self._rows[i] = rows[fi][key]
            if labels is None:
                raise ValueError("labels must be provided when event_refs are used.")
            self.labels_np = np.asarray(labels, dtype=np.float32).reshape(-1)
            n = len(self.index)
            self.dataset_names = (np.asarray(dataset_names, dtype=object).reshape(-1) if dataset_names is not None
                                  else np.asarray(["unknown"] * n, dtype=object))
            self.root_files = (np.asarray(root_files, dtype=object).reshape(-1) if root_files is not None
                               else np.asarray(["unknown"] * n, dtype=object))
            if not (self.labels_np.shape[0] == len(self.dataset_names) == len(self.root_files) == n):
                raise ValueError("labels / dataset_names / root_files lengths do not match event_refs")
        else:
            self.index = [(fi, str(k)) for fi, keys in enumerate(keys_per_part) for k in keys]
            self._rows = np.concatenate([np.arange(len(k), dtype=np.int64) for k in keys_per_part])
            self.labels_np = np.concatenate([np.load(os.path.join(pd, "y.npy")) for pd in self.part_dirs])
            self.dataset_names = np.concatenate(
                [np.load(os.path.join(pd, "dataset_name.npy")).astype(object) for pd in self.part_dirs])
            self.root_files = np.concatenate(
                [np.load(os.path.join(pd, "root_file.npy")).astype(object) for pd in self.part_dirs])
        if not self.index:
            raise ValueError("No events found in the packed parts.")
        self._parts: Dict[int, Dict[str, np.ndarray]] = {}
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parts"] = {}
        state["_pid"] = None
        return state

    def __len__(self):
        return len(self.index)

    def _close_files(self):
        self._parts = {}

    def get_label(self, idx: int) -> float:
        return float(self.labels_np[int(idx)])

    def _part(self, fi: int) -> Dict[str, np.ndarray]:
        if self._pid != os.getpid():
            self._parts = {}
            self._pid = os.getpid()
        p = self._parts.get(fi)
        if p is None:
            pd = self.part_dirs[fi]
            p = {"y": np.load(os.path.join(pd, "y.npy")), "n_muon_nodes": np.load(os.path.join(pd, "n_muon_nodes.npy"))}
            for b in self.blocks:
                p[b] = np.load(os.path.join(pd, f"{b}.npy"), mmap_mode="r")
                p[b + "_ptr"] = np.load(os.path.join(pd, f"{b}_ptr.npy"))
            self._parts[fi] = p
        return p

    def __getitem__(self, idx):
        fi = self.index[idx][0]
        j = int(self._rows[idx])
        p = self._part(fi)
        y = torch.tensor([float(p["y"][j])], dtype=torch.float32)
        item = {"y": y, "labels": y, "n_muon_nodes": torch.tensor(int(p["n_muon_nodes"][j]), dtype=torch.long)}
        for b in self.blocks:
            s, e = int(p[b + "_ptr"][j]), int(p[b + "_ptr"][j + 1])
            if b.endswith("edge_index"):
                item[b] = torch.from_numpy(np.array(p[b][:, s:e], dtype=np.int64))
            else:
                item[b] = torch.from_numpy(np.array(p[b][s:e], dtype=np.float32))
        return item


def make_event_dataset(paths, **kwargs):
    """PackedEventDataset when all paths are packed parts, H5EventDataset when none is."""
    packed = [is_packed_part(p) for p in paths]
    if all(packed):
        return PackedEventDataset(paths, **kwargs)
    if any(packed):
        raise SystemExit("--data-glob matches both packed parts and H5 files: point it at one kind.")
    return H5EventDataset(paths, **kwargs)


def collate_graphs(items):
    """Concatenate variable-size graphs and retain their graph membership.

    With a track block, the merged node order of the batch is [all muon+calo nodes | all track nodes]:
    track node t of graph g has the index N_mc_total + track_offset[g] + t, where N_mc_total is the number
    of muon+calo nodes of the whole batch. For a single graph this is x.shape[0] + t, the H5/ONNX convention.
    """
    if not items:
        raise ValueError("Cannot collate an empty batch")
    xs, edges, edge_attrs, ys = [], [], [], []
    graph_ids, mu_masks, n_muons = [], [], []
    node_offset = 0
    for graph_id, item in enumerate(items):
        x = item["x"]
        edge_index = item["edge_index"]
        n_nodes = int(x.shape[0])
        n_mu = int(item["n_muon_nodes"].item())
        xs.append(x)
        edges.append(edge_index + node_offset)
        edge_attrs.append(item["edge_attr"])
        ys.append(item["y"].reshape(-1))
        graph_ids.append(torch.full((n_nodes,), graph_id, dtype=torch.long))
        mu_masks.append(torch.arange(n_nodes, dtype=torch.long) < n_mu)
        n_muons.append(item["n_muon_nodes"].reshape(()))
        node_offset += n_nodes
    y = torch.cat(ys, dim=0)
    out = {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "y": y,
        "labels": y,
        "n_muon_nodes": torch.stack(n_muons),
        "batch": torch.cat(graph_ids, dim=0),
        "node_is_muon": torch.cat(mu_masks, dim=0),
    }

    if "track_x" in items[0]:
        n_mc_total = node_offset
        track_xs, track_edges, track_edge_attrs, track_ids = [], [], [], []
        mc_offset = 0
        track_offset = 0
        for graph_id, item in enumerate(items):
            n_mc = int(item["x"].shape[0])
            track_x = item["track_x"]
            n_trk = int(track_x.shape[0])
            tei = item["track_edge_index"]
            is_track = tei >= n_mc
            # muon/calo endpoints shift with the graph offset; track endpoints move behind all muon+calo nodes
            tei = torch.where(is_track, tei - n_mc + n_mc_total + track_offset, tei + mc_offset)
            track_xs.append(track_x)
            track_edges.append(tei)
            track_edge_attrs.append(item["track_edge_attr"])
            track_ids.append(torch.full((n_trk,), graph_id, dtype=torch.long))
            mc_offset += n_mc
            track_offset += n_trk
        out["track_x"] = torch.cat(track_xs, dim=0)
        out["track_edge_index"] = torch.cat(track_edges, dim=1)
        out["track_edge_attr"] = torch.cat(track_edge_attrs, dim=0)
        out["track_batch"] = torch.cat(track_ids, dim=0)

    if "seg_edge_index" in items[0]:
        # muon-node endpoints only: shift with the graph offset
        seg_edges, seg_attrs = [], []
        mc_offset = 0
        for item in items:
            seg_edges.append(item["seg_edge_index"] + mc_offset)
            seg_attrs.append(item["seg_edge_attr"])
            mc_offset += int(item["x"].shape[0])
        out["seg_edge_index"] = torch.cat(seg_edges, dim=1)
        out["seg_edge_attr"] = torch.cat(seg_attrs, dim=0)

    if "tracklet_x" in items[0]:
        # Same convention as a single graph: tracklet k of graph g -> N_mc_total + tracklet_offset[g] + k
        # (counted without the track nodes; the model moves the tracklet nodes behind the track nodes)
        n_mc_total = node_offset
        tl_xs, tl_edges, tl_edge_attrs, tl_ids = [], [], [], []
        mc_offset = 0
        tl_offset = 0
        for graph_id, item in enumerate(items):
            n_mc = int(item["x"].shape[0])
            tl_x = item["tracklet_x"]
            n_tl = int(tl_x.shape[0])
            tei = item["tracklet_edge_index"]
            is_tl = tei >= n_mc
            tei = torch.where(is_tl, tei - n_mc + n_mc_total + tl_offset, tei + mc_offset)
            tl_xs.append(tl_x)
            tl_edges.append(tei)
            tl_edge_attrs.append(item["tracklet_edge_attr"])
            tl_ids.append(torch.full((n_tl,), graph_id, dtype=torch.long))
            mc_offset += n_mc
            tl_offset += n_tl
        out["tracklet_x"] = torch.cat(tl_xs, dim=0)
        out["tracklet_edge_index"] = torch.cat(tl_edges, dim=1)
        out["tracklet_edge_attr"] = torch.cat(tl_edge_attrs, dim=0)
        out["tracklet_batch"] = torch.cat(tl_ids, dim=0)
    return out


def batch_model_inputs(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[tuple, Dict[str, Any]]:
    """Move a collated batch to ``device`` and return (positional args, keyword args) for DisplacedVertexGNN."""
    args = (
        batch["x"].to(device, non_blocking=True).float(),
        batch["edge_index"].to(device, non_blocking=True).long(),
        batch["edge_attr"].to(device, non_blocking=True).float(),
    )
    kwargs = {
        "n_muon_nodes": batch["n_muon_nodes"].to(device, non_blocking=True).long(),
        "batch": batch["batch"].to(device, non_blocking=True).long(),
        "node_is_muon": batch["node_is_muon"].to(device, non_blocking=True).bool(),
    }
    if "track_x" in batch:
        kwargs["track_x"] = batch["track_x"].to(device, non_blocking=True).float()
        kwargs["track_edge_index"] = batch["track_edge_index"].to(device, non_blocking=True).long()
        kwargs["track_edge_attr"] = batch["track_edge_attr"].to(device, non_blocking=True).float()
        kwargs["track_batch"] = batch["track_batch"].to(device, non_blocking=True).long()
    if "seg_edge_index" in batch:
        kwargs["seg_edge_index"] = batch["seg_edge_index"].to(device, non_blocking=True).long()
        kwargs["seg_edge_attr"] = batch["seg_edge_attr"].to(device, non_blocking=True).float()
    if "tracklet_x" in batch:
        kwargs["tracklet_x"] = batch["tracklet_x"].to(device, non_blocking=True).float()
        kwargs["tracklet_edge_index"] = batch["tracklet_edge_index"].to(device, non_blocking=True).long()
        kwargs["tracklet_edge_attr"] = batch["tracklet_edge_attr"].to(device, non_blocking=True).float()
        kwargs["tracklet_batch"] = batch["tracklet_batch"].to(device, non_blocking=True).long()
    return args, kwargs


class FileLocalDistributedSampler(Sampler[int]):
    """Shuffle at file granularity while giving each DDP rank different HDF5 files; with mix_files != 1 the events of
    mix_files consecutive files are shuffled together (mixed-class batches, see __iter__).

    Dataset indices returned here are positions inside the Subset, not indices in
    the underlying H5EventDataset. Training drops at most ``world_size - 1``
    tail events so optimizer-step counts stay synchronized. Validation never
    pads or truncates: every held-out event is emitted exactly once.
    """

    def __init__(
        self,
        dataset: H5EventDataset,
        subset_indices: np.ndarray,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        mix_files: int = 1,
    ):
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.num_replicas < 1 or not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"Invalid rank {self.rank} for {self.num_replicas} replicas")

        base_indices = np.asarray(subset_indices, dtype=np.int64).reshape(-1)
        file_ids = np.fromiter(
            (dataset.index[int(i)][0] for i in base_indices),
            dtype=np.int32,
            count=base_indices.size,
        )
        positions = np.arange(base_indices.size, dtype=np.int64)
        file_order = np.argsort(file_ids, kind="stable")
        sorted_files = file_ids[file_order]
        boundaries = np.flatnonzero(np.diff(sorted_files)) + 1
        self.file_blocks = [x for x in np.split(positions[file_order], boundaries) if x.size]
        # files whose events are mixed together (shuffle only): 1 = one file at a time (single-file batches),
        # <= 0 = all files (global shuffle)
        self.mix_files = int(mix_files)
        self.n = int(base_indices.size)
        if self.drop_last:
            self.num_samples = self.n // self.num_replicas
        else:
            quotient, remainder = divmod(self.n, self.num_replicas)
            self.num_samples = quotient + int(self.rank < remainder)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        block_order = np.arange(len(self.file_blocks))
        if self.shuffle:
            rng.shuffle(block_order)

        ordered_blocks = []
        for block_id in block_order:
            block = self.file_blocks[int(block_id)]
            if self.shuffle:
                block = block.copy()
                rng.shuffle(block)
            ordered_blocks.append(block)

        if self.shuffle and self.mix_files != 1 and ordered_blocks:
            # Every file holds one sample (one class): shuffle the events of each group of mix_files consecutive
            # (shuffled) files together, so a batch mixes ~mix_files files while a DataLoader worker still reads
            # only ~mix_files files at a time (H5: keep mix_files <= --max-open-h5-files)
            group = len(ordered_blocks) if self.mix_files <= 0 else self.mix_files
            mixed = []
            for g0 in range(0, len(ordered_blocks), group):
                events = np.concatenate(ordered_blocks[g0:g0 + group])
                mixed.append(events[rng.permutation(events.size)])
            ordered_blocks = mixed
        ordered = np.concatenate(ordered_blocks) if ordered_blocks else np.empty(0, dtype=np.int64)
        if self.drop_last:
            usable = (ordered.size // self.num_replicas) * self.num_replicas
            ordered = ordered[:usable]
            start = self.rank * self.num_samples
            stop = start + self.num_samples
        else:
            quotient, remainder = divmod(ordered.size, self.num_replicas)
            start = self.rank * quotient + min(self.rank, remainder)
            stop = start + quotient + int(self.rank < remainder)
        own = ordered[start:stop]
        return iter(own.tolist())


def _decode_attr_to_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    return str(v)


# ---------------------------------------------------------------------------------------------------------------------
# Optional architecture blocks of DisplacedVertexGNN. All are off by default: the defaults build the legacy model.
#   block="prenorm"     pre-LayerNorm residual blocks h += Drop(MP(LN(h))); h += Drop(FFN(LN(h))), final LayerNorm
#   relation_mp=True    relation-typed message passing: one GATv2 attention with its own projections and its own softmax
#                       per relation dst <- src (muon <- calo, calo <- muon, muon <- track, track <- muon,
#                       muon <- muon, muon <- tracklet, tracklet <- muon, tracklet <- tracklet), a self term per node
#                       type and separate muon / calo encoders
#   edge_update         hidden states on the segment-segment (and tracklet) edges, updated in every layer
#   readout="typed"     per-node-type mean+max pooling; "typed_attn" adds attention pooling over muons and tracklets
# ---------------------------------------------------------------------------------------------------------------------
BLOCK_TYPES = ("legacy", "prenorm")
READOUT_TYPES = ("global", "typed", "typed_attn")
RELATION_AGGRS = ("attn", "attn_max")
CALO_CONTEXTS = ("two_way", "one_way")
EDGE_UPDATE_MODES = ("none", "seg", "seg_tracklet")


def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_segments) -> torch.Tensor:
    """Softmax of logits [E, H] over the rows with the same index [E] (per-segment maximum for stability)."""
    logits = logits.float()
    idx = index.unsqueeze(-1).expand_as(logits)
    seg_max = torch.full((num_segments, logits.shape[1]), -torch.inf, device=logits.device, dtype=torch.float32)
    seg_max.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)
    w = torch.exp(logits - seg_max[index])
    seg_sum = torch.zeros((num_segments, logits.shape[1]), device=logits.device, dtype=torch.float32)
    seg_sum.scatter_add_(0, idx, w)
    return w / seg_sum[index].clamp(min=1e-6)


def _segment_max(values: torch.Tensor, index: torch.Tensor, num_segments) -> torch.Tensor:
    """Per-segment maximum of values [E, C] -> [num_segments, C]; 0 for segments without rows."""
    values = values.float()
    out = torch.full((num_segments, values.shape[1]), -torch.inf, device=values.device, dtype=torch.float32)
    out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(values), values, reduce="amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(mult * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNormGATBlock(nn.Module):
    """Pre-LayerNorm residual block around the legacy GAT message passing (one softmax over all incoming edges):
    h += Drop(GAT(LN(h))), then h += Drop(FFN(LN(h))) when ffn_mult > 0. The update can have either sign and every
    message passing sees a normalized input (GATResidualBlock adds ReLU(GAT(h)) >= 0 to an unnormalized stream).
    The residual stream is kept in float32 (also under bf16 autocast)."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        edge_dim: int = 0,
        edge_attention: bool = False,
        gatv2_edge_attention: bool = False,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gat = CustomGAT(
            in_channels=dim, out_channels=dim // heads, heads=heads, dropout=dropout, add_self_loops=True, concat=True,
            edge_dim=edge_dim, edge_attention=edge_attention, gatv2_edge_attention=gatv2_edge_attention,
        )
        self.ffn_norm = nn.LayerNorm(dim) if ffn_mult > 0 else None
        self.ffn = FeedForward(dim, ffn_mult, dropout) if ffn_mult > 0 else None
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_dropout_p: float = 0.0,
    ) -> torch.Tensor:
        x = x.float()
        x = x + self.dropout(
            self.gat(self.norm(x), edge_index, edge_attr=edge_attr, edge_dropout_p=edge_dropout_p)
        ).float()
        if self.ffn is not None:
            x = x + self.dropout(self.ffn(self.ffn_norm(x))).float()
        return x


class RelationGATv2(nn.Module):
    """GATv2 attention of one relation (source node type -> destination node type) with its own projections and its
    own softmax over the incoming edges of this relation, so e.g. ~25 calo towers cannot drown the few segment or
    tracklet messages of a segment:
        score_ij = a^T SiLU(W_src h_j + W_dst h_i + W_edge e_ij)   (per head)
        m_ij     = SiLU(V_src h_j + V_dst h_i + V_edge e_ij)
        out_i    = O [ sum_j softmax_j(score_ij) m_ij  |  max_j m_ij ]   (max only with aggr="attn_max")
    A node without incoming edges of the relation gets O(0), i.e. the bias of O: a learned "no neighbour" vector.
    h_src / h_dst hold only the nodes of the source / destination type; src / dst are indices into them."""

    def __init__(self, dim: int, heads: int, edge_dim: int, aggr: str = "attn_max", dropout: float = 0.0):
        super().__init__()
        if heads <= 0 or dim % heads != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by gat_heads={heads}")
        if aggr not in RELATION_AGGRS:
            raise ValueError(f"Unknown relation aggregation {aggr!r}; choices: {RELATION_AGGRS}")
        if edge_dim <= 0:
            raise ValueError("RelationGATv2 needs edge features (edge_dim > 0)")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.use_max = aggr == "attn_max"
        self.src_proj = nn.Linear(dim, 2 * dim, bias=False)  # [W_src | V_src]
        self.dst_proj = nn.Linear(dim, 2 * dim, bias=False)  # [W_dst | V_dst]
        self.edge_proj = nn.Linear(edge_dim, 2 * dim, bias=False)  # [W_edge | V_edge]
        self.att = nn.Parameter(torch.empty(1, self.heads, self.head_dim))
        self.out = nn.Linear(2 * dim if self.use_max else dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        for lin in (self.src_proj, self.dst_proj, self.edge_proj):
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_dropout_p: float = 0.0,
    ) -> torch.Tensor:
        if self.training and edge_dropout_p > 0.0:
            keep = torch.rand(src.shape[0], device=src.device) >= float(edge_dropout_p)
            src, dst, edge_attr = src[keep], dst[keep], edge_attr[keep]
        n_dst = h_dst.shape[0]
        a_src, v_src = self.src_proj(h_src).float().chunk(2, dim=-1)
        a_dst, v_dst = self.dst_proj(h_dst).float().chunk(2, dim=-1)
        a_edge, v_edge = self.edge_proj(edge_attr).float().chunk(2, dim=-1)
        z = F.silu(a_src[src] + a_dst[dst] + a_edge)  # [E, D]
        # a_h^T z_h of every head as one MatMul with the block-diagonal [D, H] matrix of the score vectors, in float32:
        # a relation can have no edges in an event, and onnxruntime (1.22) does not reduce the last axis of an empty
        # [0, H, C] tensor in ReduceSum (it returns [0, H, C]), which a sum(dim=-1) would export to.
        score = (self.att.float().reshape(self.heads, self.head_dim, 1)
                 * torch.eye(self.heads, device=z.device, dtype=torch.float32).reshape(self.heads, 1, self.heads))
        with torch.amp.autocast(z.device.type, enabled=False):
            logits = torch.matmul(z, score.reshape(self.dim, self.heads))  # [E, H]
        alpha = _segment_softmax(logits, dst, n_dst)
        alpha = self.attn_dropout(alpha)
        msg = F.silu(v_src[src] + v_dst[dst] + v_edge)  # [E, D]
        weighted = (alpha.unsqueeze(-1) * msg.reshape(-1, self.heads, self.head_dim)).reshape(-1, self.dim)
        agg = torch.zeros((n_dst, self.dim), device=msg.device, dtype=torch.float32)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)
        if self.use_max:
            agg = torch.cat([agg, _segment_max(msg, dst, n_dst)], dim=-1)
        return self.out(agg)


class EdgeStateUpdate(nn.Module):
    """Interaction-network update of hidden edge states: e_ij += Drop(MLP([h_j, h_i, LN(e_ij)]))."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, e: torch.Tensor, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        e = e.float()
        upd = self.mlp(torch.cat([h_src.float(), h_dst.float(), self.norm(e)], dim=-1))
        return e + self.dropout(upd).float()


class _RelationGraph:
    """Type-local edge lists of one forward pass of the relation-typed DisplacedVertexGNN.
    relations: name -> (source index, destination index, rows of the edge block or None = all rows);
    endpoints: edge block with hidden states -> (node table, source index, destination index), node table "muon" or
    "muon_tracklet" (the muon states followed by the tracklet states)."""

    def __init__(self):
        self.relations: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}
        self.endpoints: Dict[str, Tuple[str, torch.Tensor, torch.Tensor]] = {}

    def edge_endpoint_states(self, kind: str, h: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        table_name, src, dst = self.endpoints[kind]
        table = h["muon"] if table_name == "muon" else torch.cat([h["muon"], h["tracklet"]], dim=0)
        return table[src], table[dst]


class RelationalMPLayer(nn.Module):
    """One relation-typed message-passing layer (DisplacedVertexGNN relation_mp=True).

    relations:         name -> (source type, destination type, edge block, edge dim)
    update_types:      node types whose state is updated (all but the calo towers with calo_context="one_way")
    edge_update_kinds: edge blocks with hidden states ("seg", "tracklet"), updated before the messages
    Node update of type t (legacy block):  h_t += Drop(ReLU(S_t h_t + sum_{r -> t} R_r))
                       (prenorm block): h_t += Drop(S_t n_t + sum_{r -> t} R_r(n)), n = LN(h); h_t += Drop(FFN(LN(h_t)))
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        relations: Dict[str, Tuple[str, str, str, int]],
        node_types: List[str],
        update_types: List[str],
        block: str = "legacy",
        ffn_mult: int = 2,
        aggr: str = "attn_max",
        edge_update_kinds: Tuple[str, ...] = (),
    ):
        super().__init__()
        self.relations = {name: (s, d, kind) for name, (s, d, kind, _) in relations.items()}
        self.rel = nn.ModuleDict({
            name: RelationGATv2(dim, heads, edge_dim, aggr=aggr, dropout=dropout)
            for name, (_, _, _, edge_dim) in relations.items()
        })
        self.update_types = list(update_types)
        self.self_lin = nn.ModuleDict({t: nn.Linear(dim, dim) for t in self.update_types})
        self.prenorm = block == "prenorm"
        self.norm = nn.ModuleDict({t: nn.LayerNorm(dim) for t in node_types}) if self.prenorm else None
        use_ffn = self.prenorm and ffn_mult > 0
        self.ffn_norm = nn.ModuleDict({t: nn.LayerNorm(dim) for t in self.update_types}) if use_ffn else None
        self.ffn = nn.ModuleDict({t: FeedForward(dim, ffn_mult, dropout) for t in self.update_types}) if use_ffn else None
        self.edge_update = nn.ModuleDict({kind: EdgeStateUpdate(dim, dropout) for kind in edge_update_kinds})
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Dict[str, torch.Tensor],
        graph: _RelationGraph,
        edge_feats: Dict[str, torch.Tensor],
        edge_dropout_p: float = 0.0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        n = {t: self.norm[t](v) for t, v in h.items()} if self.norm is not None else h
        edge_feats = dict(edge_feats)
        for kind, upd in self.edge_update.items():
            src_states, dst_states = graph.edge_endpoint_states(kind, n)
            edge_feats[kind] = upd(edge_feats[kind], src_states, dst_states)
        msg = {t: self.self_lin[t](n[t]).float() for t in self.update_types}
        for name, mod in self.rel.items():
            src_type, dst_type, kind = self.relations[name]
            src, dst, rows = graph.relations[name]
            e = edge_feats[kind] if rows is None else edge_feats[kind].index_select(0, rows)
            msg[dst_type] = msg[dst_type] + mod(
                n[src_type], n[dst_type], src, dst, e, edge_dropout_p=edge_dropout_p,
            ).float()
        out = dict(h)
        for t in self.update_types:
            x = h[t] + self.dropout(msg[t] if self.prenorm else F.relu(msg[t]))
            if self.ffn is not None:
                x = x + self.dropout(self.ffn[t](self.ffn_norm[t](x))).float()
            out[t] = x
        return out, edge_feats


class TypedReadout(nn.Module):
    """Graph readout per node type: mean+max pooling of every type (a learned "empty" vector replaces the pooling of a
    type without nodes in a graph) and, for attn_types, multi-head softmax attention pooling (a gate MLP scores every
    node, the softmax runs over the nodes of the type in the graph, each head pools its slice of a value projection).
    No sum pooling: the multiplicities differ between samples for reasons unrelated to the physics."""

    def __init__(self, dim: int, node_types: List[str], attn_types: List[str], heads: int = 4):
        super().__init__()
        if attn_types and (heads <= 0 or dim % heads != 0):
            raise ValueError(f"hidden_dim={dim} must be divisible by readout_heads={heads}")
        self.dim = int(dim)
        self.node_types = list(node_types)
        self.attn_types = list(attn_types)
        self.heads = int(heads)
        self.empty = nn.ParameterDict({t: nn.Parameter(torch.zeros(1, 2 * dim)) for t in self.node_types})
        self.attn_gate = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, heads)) for t in self.attn_types
        })
        self.attn_value = nn.ModuleDict({t: nn.Linear(dim, dim) for t in self.attn_types})
        self.attn_empty = nn.ParameterDict({t: nn.Parameter(torch.zeros(1, dim)) for t in self.attn_types})
        self.out_dim = 2 * self.dim * len(self.node_types) + self.dim * len(self.attn_types)

    def forward(self, h: Dict[str, torch.Tensor], batches: Dict[str, torch.Tensor], num_graphs) -> torch.Tensor:
        parts = []
        counts = {}
        for t in self.node_types:
            x = h[t].float()
            b = batches[t]
            idx = b.unsqueeze(-1).expand_as(x)
            sums = torch.zeros((num_graphs, x.shape[1]), device=x.device, dtype=torch.float32).scatter_add_(0, idx, x)
            cnt = torch.zeros((num_graphs, 1), device=x.device, dtype=torch.float32).scatter_add_(
                0, b.unsqueeze(-1), torch.ones_like(x[:, :1]))
            pooled = torch.cat([sums / cnt.clamp_min(1.0), _segment_max(x, b, num_graphs)], dim=-1)
            parts.append(torch.where(cnt > 0, pooled, self.empty[t].float()))
            counts[t] = cnt
        for t in self.attn_types:
            x = h[t]
            b = batches[t]
            alpha = _segment_softmax(self.attn_gate[t](x), b, num_graphs)  # [n, heads]
            v = self.attn_value[t](x).float().reshape(-1, self.heads, self.dim // self.heads)
            weighted = (alpha.unsqueeze(-1) * v).reshape(-1, self.dim)
            pooled = torch.zeros((num_graphs, self.dim), device=weighted.device, dtype=torch.float32)
            pooled.scatter_add_(0, b.unsqueeze(-1).expand_as(weighted), weighted)
            parts.append(torch.where(counts[t] > 0, pooled, self.attn_empty[t].float()))
        return torch.cat(parts, dim=-1)


class DisplacedVertexGNN(nn.Module):
    """Single-logit graph classifier using raw inputs and optional model-side normalization."""

    def __init__(
        self,
        xdim,
        edim,
        hdim=128,
        n_layers=5,
        dropout=0.020720971979495448,
        layer_type: str = "gat_residual",
        gat_heads: int = 2,
        gat_edge_attn: bool = False,
        gatv2_edge_attn: bool = True,
        sage_aggr: str = "mean",
        edgeconv_aggr: str = "mean",
        pool: str = "meanmax",
        use_fourier=False,
        fourier_base=3.0,
        fourier_min_exp=-6,
        fourier_max_exp=6,
        normalize_node_features: bool = False,
        normalize_edge_features: bool = False,
        feature_stats: Optional[Dict[str, np.ndarray]] = None,
        feature_norm_clip: float = -1.0,
        track_xdim: int = 0,
        node_type_embedding: bool = False,
        track_asinh_scales: Optional[Any] = None,
        seg_edim: int = 0,
        tracklet_xdim: int = 0,
        tracklet_edim: int = 0,
        block: str = "legacy",
        ffn_mult: int = 2,
        relation_mp: bool = False,
        relation_aggr: str = "attn_max",
        calo_context: str = "two_way",
        edge_update: str = "none",
        readout: str = "global",
        readout_heads: int = 4,
    ):
        """
        Architecture options (gat_residual only; the defaults give the legacy model, bit-identical):
          block="prenorm"     PreNormGATBlock layers (h += Drop(GAT(LN(h))); h += Drop(FFN(LN(h))) with ffn_mult > 0)
                              and a final LayerNorm before the readout
          relation_mp=True    RelationalMPLayer layers: one GATv2 attention per relation dst <- src with its own
                              projections and softmax (relation_aggr="attn_max": attention mean and max of the messages,
                              "attn": attention mean only), a self term per node type, separate muon / calo encoders;
                              gat_heads heads per relation. calo_context="one_way" drops the calo <- muon relation and
                              keeps the calo towers at their encoder output (fixed context, less compute).
                              The block option applies to these layers as well.
          edge_update         (relation_mp) "seg": hidden states on the segment-segment edges, "seg_tracklet": also on
                              the tracklet edges, updated in every layer from both end nodes and used by the attention
          readout="typed"     TypedReadout: mean+max per node type; "typed_attn" adds readout_heads-head attention pooling
                              over the muon segments and the tracklets. "global": mean+max over all nodes (pool).

        track_xdim > 0 adds the track block: track_x [T, track_xdim] is transformed (asinh of the heavy-tailed
        columns), normalized with its own stats, encoded by its own MLP and appended after the muon+calo nodes;
        the muon-track edges (track_edge_index, global indices, track t = x.shape[0] + t) are merged with the
        muon-calo edges, and every edge gets a flag column (0 = calo edge, 1 = track edge).
        seg_edim > 0 adds the segment-segment edges (seg_edge_index [2, Es] muon-node indices, seg_edge_attr [Es, seg_edim]).
        tracklet_xdim > 0 adds the tracklet block: tracklet_x [K, tracklet_xdim] (own normalization and MLP, appended
        after the track nodes) and tracklet_edge_index [2, Ek] (tracklet k = x.shape[0] + k, counted without the track
        nodes) with tracklet_edge_attr [Ek, tracklet_edim].
        The edge features seen by the message passing are [edim shared muon-calo/muon-track columns | track flag |
        segment-edge columns | segment flag | tracklet-edge columns | tracklet flag], each block only when used, so
        every edge type has its own columns.
        node_type_embedding adds a learned embedding of the node type (muon / calo / track / tracklet) to the encoded nodes.
        With track_xdim = seg_edim = tracklet_xdim = 0 and node_type_embedding = False the model is identical to the
        muon+calo classifier, and with only track_xdim > 0 to the track classifier.
        """
        super().__init__()
        self.fourier = None
        self.pool = pool
        self.normalize_node_features = bool(normalize_node_features)
        self.normalize_edge_features = bool(normalize_edge_features)
        self.feature_norm_clip = float(feature_norm_clip)
        self.track_xdim = int(track_xdim or 0)
        self.use_tracks = self.track_xdim > 0
        self.seg_edim = int(seg_edim or 0)
        self.use_segment_edges = self.seg_edim > 0
        self.tracklet_xdim = int(tracklet_xdim or 0)
        self.tracklet_edim = int(tracklet_edim or 0)
        self.use_tracklets = self.tracklet_xdim > 0
        if self.use_tracklets and self.tracklet_edim <= 0:
            raise ValueError("tracklet_xdim > 0 needs tracklet_edim > 0")
        self.edim_in = int(edim)
        # Edge-feature layout of the message-passing layers (see the docstring); with tracks only: edim + 1 flag column
        col = int(edim)
        self._track_flag_col = None
        if self.use_tracks:
            self._track_flag_col = col
            col += 1
        self._seg_cols = None  # (first column, flag column)
        if self.use_segment_edges:
            self._seg_cols = (col, col + self.seg_edim)
            col += self.seg_edim + 1
        self._tracklet_cols = None
        if self.use_tracklets:
            self._tracklet_cols = (col, col + self.tracklet_edim)
            col += self.tracklet_edim + 1
        edim_model = col
        self.edim_model = int(edim_model)

        track_scales = None
        if self.use_tracks:
            stats_scales = None
            if feature_stats is not None and "tr_asinh_scale" in feature_stats:
                stats_scales = np.asarray(feature_stats["tr_asinh_scale"], dtype=np.float32).reshape(-1)
            if track_asinh_scales is not None:
                track_scales = np.asarray(track_asinh_scales, dtype=np.float32).reshape(-1)
            elif stats_scales is not None:
                track_scales = stats_scales
            elif self.track_xdim == len(TRACK_FEATURE_NAMES):
                track_scales = track_feature_asinh_scale_vector(TRACK_FEATURE_NAMES)
            else:
                track_scales = np.zeros(self.track_xdim, dtype=np.float32)
            if track_scales.shape[0] != self.track_xdim:
                raise ValueError(f"track_asinh_scales has {track_scales.shape[0]} entries for track_xdim={self.track_xdim}")
            if stats_scales is not None and not np.allclose(stats_scales, track_scales):
                raise ValueError(
                    f"Track stats were computed with asinh scales {stats_scales.tolist()}, but the model uses "
                    f"{track_scales.tolist()}"
                )

        self._register_feature_normalization_buffers(
            xdim=xdim,
            edim=edim,
            feature_stats=feature_stats,
            track_xdim=self.track_xdim,
            track_scales=track_scales,
            seg_edim=self.seg_edim,
            tracklet_xdim=self.tracklet_xdim,
            tracklet_edim=self.tracklet_edim,
        )

        if use_fourier:
            self.fourier = FourierEncoder(
                xdim, base=fourier_base, min_exp=fourier_min_exp, max_exp=fourier_max_exp,
            )
            xdim_in = xdim * 2 * (fourier_max_exp - fourier_min_exp + 1)
        else:
            xdim_in = xdim

        self.node_enc = MLP(xdim_in, hdim, hidden_dim=hdim, n_layers=2, dropout=dropout)

        self.track_fourier = None
        self.track_enc = None
        if self.use_tracks:
            if use_fourier:
                self.track_fourier = FourierEncoder(
                    self.track_xdim, base=fourier_base, min_exp=fourier_min_exp, max_exp=fourier_max_exp,
                )
                track_xdim_in = self.track_xdim * 2 * (fourier_max_exp - fourier_min_exp + 1)
            else:
                track_xdim_in = self.track_xdim
            self.track_enc = MLP(track_xdim_in, hdim, hidden_dim=hdim, n_layers=2, dropout=dropout)

        self.tracklet_fourier = None
        self.tracklet_enc = None
        if self.use_tracklets:
            if use_fourier:
                self.tracklet_fourier = FourierEncoder(
                    self.tracklet_xdim, base=fourier_base, min_exp=fourier_min_exp, max_exp=fourier_max_exp,
                )
                tracklet_xdim_in = self.tracklet_xdim * 2 * (fourier_max_exp - fourier_min_exp + 1)
            else:
                tracklet_xdim_in = self.tracklet_xdim
            self.tracklet_enc = MLP(tracklet_xdim_in, hdim, hidden_dim=hdim, n_layers=2, dropout=dropout)
        # muon / calo / track (+ tracklet)
        self.type_emb = nn.Embedding(4 if self.use_tracklets else 3, hdim) if node_type_embedding else None

        # Architecture options (see the docstring); nothing below is created with the defaults
        self.block = str(block)
        self.ffn_mult = int(ffn_mult)
        self.relation_mp = bool(relation_mp)
        self.relation_aggr = str(relation_aggr)
        self.calo_context = str(calo_context)
        self.edge_update = str(edge_update)
        self.readout = str(readout)
        self.readout_heads = int(readout_heads)
        for name, value, choices in (("block", self.block, BLOCK_TYPES), ("readout", self.readout, READOUT_TYPES),
                                     ("relation_aggr", self.relation_aggr, RELATION_AGGRS),
                                     ("calo_context", self.calo_context, CALO_CONTEXTS),
                                     ("edge_update", self.edge_update, EDGE_UPDATE_MODES)):
            if value not in choices:
                raise ValueError(f"Unknown {name}={value!r}; choices: {choices}")
        if self.ffn_mult < 0:
            raise ValueError(f"ffn_mult must be >= 0, got {self.ffn_mult}")
        if (self.block != "legacy" or self.relation_mp) and layer_type != "gat_residual":
            raise ValueError("--block prenorm and --relation-mp need --layer-type gat_residual")
        if self.relation_mp and gat_edge_attn:
            raise ValueError("--relation-mp uses its own GATv2 attention; drop --gat-edge-attn")
        if not self.relation_mp and (self.edge_update != "none" or self.calo_context != "two_way"):
            raise ValueError("--edge-update and --calo-context one_way need --relation-mp")
        self.node_types = ["muon", "calo"] + (["track"] if self.use_tracks else []) + (
            ["tracklet"] if self.use_tracklets else [])
        self.edge_update_kinds: Tuple[str, ...] = ()
        if self.edge_update in ("seg", "seg_tracklet"):
            if not self.use_segment_edges:
                raise ValueError(f"--edge-update {self.edge_update} needs the segment-segment edges")
            self.edge_update_kinds += ("seg",)
        if self.edge_update == "seg_tracklet":
            if not self.use_tracklets:
                raise ValueError("--edge-update seg_tracklet needs the tracklets")
            self.edge_update_kinds += ("tracklet",)

        edim = edim_model
        if self.relation_mp:
            if gat_heads <= 0 or hdim % gat_heads != 0:
                raise ValueError(f"hidden_dim={hdim} must be divisible by gat_heads={gat_heads}")
            self.calo_enc = MLP(xdim_in, hdim, hidden_dim=hdim, n_layers=2, dropout=dropout)
            self.seg_edge_enc = (
                MLP(self.seg_edim, hdim, hidden_dim=hdim, n_layers=2, dropout=dropout)
                if "seg" in self.edge_update_kinds else None
            )
            self.tracklet_edge_enc = (
                MLP(self.tracklet_edim, hdim, hidden_dim=hdim, n_layers=2, dropout=dropout)
                if "tracklet" in self.edge_update_kinds else None
            )
            rels: Dict[str, Tuple[str, str, str, int]] = {"mu_from_calo": ("calo", "muon", "calo", self.edim_in)}
            if self.calo_context == "two_way":
                rels["calo_from_mu"] = ("muon", "calo", "calo", self.edim_in)
            if self.use_tracks:
                rels["mu_from_trk"] = ("track", "muon", "track", self.edim_in)
                rels["trk_from_mu"] = ("muon", "track", "track", self.edim_in)
            if self.use_segment_edges:
                rels["mu_from_mu"] = ("muon", "muon", "seg", hdim if "seg" in self.edge_update_kinds else self.seg_edim)
            if self.use_tracklets:
                tl_edim = hdim if "tracklet" in self.edge_update_kinds else self.tracklet_edim
                rels["mu_from_tl"] = ("tracklet", "muon", "tracklet", tl_edim)
                rels["tl_from_mu"] = ("muon", "tracklet", "tracklet", tl_edim)
                rels["tl_from_tl"] = ("tracklet", "tracklet", "tracklet", tl_edim)
            update_types = [t for t in self.node_types if not (t == "calo" and self.calo_context == "one_way")]
            self.layers = nn.ModuleList([
                RelationalMPLayer(
                    hdim, gat_heads, dropout, rels, self.node_types, update_types, block=self.block,
                    ffn_mult=self.ffn_mult, aggr=self.relation_aggr, edge_update_kinds=self.edge_update_kinds,
                )
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = True
        elif layer_type == "gat_residual" and self.block == "prenorm":
            if gat_edge_attn and gatv2_edge_attn:
                raise ValueError("--gat-edge-attn and --gatv2-edge-attn are mutually exclusive")
            if gat_heads <= 0 or hdim % gat_heads != 0:
                raise ValueError(f"hidden_dim={hdim} must be divisible by gat_heads={gat_heads}")
            self.layers = nn.ModuleList([
                PreNormGATBlock(
                    hdim, gat_heads, dropout=dropout, edge_dim=edim, edge_attention=gat_edge_attn,
                    gatv2_edge_attention=gatv2_edge_attn, ffn_mult=self.ffn_mult,
                )
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = bool(gat_edge_attn or gatv2_edge_attn)
        elif layer_type == "mpnn":
            self.layers = nn.ModuleList([EdgeMPNNLayer(hdim, edim, dropout=dropout) for _ in range(n_layers)])
            self._uses_edge_attr = True
        elif layer_type == "edge_residual":
            self.layers = nn.ModuleList([
                EdgeResidualBlock(in_channels=hdim, hidden_channels=hdim, dropout=dropout, aggr=edgeconv_aggr)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
        elif layer_type == "sage_residual":
            self.layers = nn.ModuleList([
                SAGEResidualBlock(in_channels=hdim, out_channels=hdim, aggr=sage_aggr, dropout=dropout)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
        elif layer_type == "gat_residual":
            if gat_edge_attn and gatv2_edge_attn:
                raise ValueError(
                    "--gat-edge-attn and --gatv2-edge-attn are mutually exclusive"
                )
            if gat_heads <= 0:
                raise ValueError("--gat-heads must be >= 1")
            if hdim % gat_heads != 0:
                raise ValueError(f"hidden_dim={hdim} must be divisible by gat_heads={gat_heads}")
            per_head = hdim // gat_heads
            self.layers = nn.ModuleList([
                GATResidualBlock(
                    in_channels=hdim, out_channels=per_head, heads=gat_heads, dropout=dropout,
                    edge_dim=edim, edge_attention=gat_edge_attn,
                    gatv2_edge_attention=gatv2_edge_attn,
                )
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = bool(gat_edge_attn or gatv2_edge_attn)
        else:
            raise ValueError(f"Unknown layer_type={layer_type}")

        self.final_norm = None
        if self.block == "prenorm":
            self.final_norm = (
                nn.ModuleDict({t: nn.LayerNorm(hdim) for t in self.node_types}) if self.relation_mp
                else nn.LayerNorm(hdim)
            )
        self.typed_readout = None
        if self.readout != "global":
            attn_types = [t for t in ("muon", "tracklet") if t in self.node_types] if self.readout == "typed_attn" else []
            self.typed_readout = TypedReadout(hdim, self.node_types, attn_types, heads=self.readout_heads)
            graph_dim = self.typed_readout.out_dim
        else:
            graph_dim = hdim * 2 if pool == "meanmax" else hdim
        self.head = MLP(in_dim=graph_dim, out_dim=1, hidden_dim=hdim, n_layers=3, dropout=dropout)

    @staticmethod
    def _num_graphs(n_muon_nodes: Optional[Any], batch: Optional[torch.Tensor]) -> int:
        """Graphs in the input: 1 without batch, else one n_muon_nodes entry per graph (collate_graphs)."""
        if batch is None:
            # 1, taken from the input shape so that the readout shapes stay symbolic in the ONNX export (a constant 1
            # gives the If branches of the exported scatter_reduce mismatching output shapes: onnxruntime warnings)
            return n_muon_nodes.reshape(-1).shape[0] if torch.is_tensor(n_muon_nodes) else 1
        if torch.is_tensor(n_muon_nodes) and n_muon_nodes.numel() > 0:
            return int(n_muon_nodes.numel())
        return int(batch.max().item()) + 1

    def _register_feature_normalization_buffers(
        self,
        *,
        xdim: int,
        edim: int,
        feature_stats: Optional[Dict[str, np.ndarray]],
        track_xdim: int = 0,
        track_scales: Optional[np.ndarray] = None,
        seg_edim: int = 0,
        tracklet_xdim: int = 0,
        tracklet_edim: int = 0,
    ) -> None:
        def _buffer(name: str, default: np.ndarray) -> None:
            value = default
            if feature_stats is not None and name in feature_stats:
                value = np.asarray(feature_stats[name], dtype=np.float32)
            value = np.asarray(value, dtype=np.float32).reshape(1, -1)
            self.register_buffer(name, torch.from_numpy(value), persistent=True)

        if self.normalize_node_features:
            _buffer("mu_center", np.zeros(xdim, dtype=np.float32))
            _buffer("mu_scale", np.ones(xdim, dtype=np.float32))
            _buffer("ca_center", np.zeros(xdim, dtype=np.float32))
            _buffer("ca_scale", np.ones(xdim, dtype=np.float32))
            for name in ("mu_center", "mu_scale", "ca_center", "ca_scale"):
                if getattr(self, name).shape[1] != int(xdim):
                    raise ValueError(f"Normalization buffer {name} has incompatible shape {tuple(getattr(self, name).shape)} for xdim={xdim}")

        if self.normalize_edge_features:
            _buffer("edge_center", np.zeros(edim, dtype=np.float32))
            _buffer("edge_scale", np.ones(edim, dtype=np.float32))
            for name in ("edge_center", "edge_scale"):
                if getattr(self, name).shape[1] != int(edim):
                    raise ValueError(f"Normalization buffer {name} has incompatible shape {tuple(getattr(self, name).shape)} for edim={edim}")

        if track_xdim > 0:
            # asinh(x / scale) for the columns with scale > 0, applied before the track normalization
            scales = np.asarray(track_scales, dtype=np.float32).reshape(1, -1)
            self.register_buffer("tr_asinh_scale", torch.from_numpy(scales), persistent=True)
            if self.normalize_node_features:
                _buffer("tr_center", np.zeros(track_xdim, dtype=np.float32))
                _buffer("tr_scale", np.ones(track_xdim, dtype=np.float32))
                for name in ("tr_center", "tr_scale"):
                    if getattr(self, name).shape[1] != int(track_xdim):
                        raise ValueError(f"Normalization buffer {name} has incompatible shape {tuple(getattr(self, name).shape)} for track_xdim={track_xdim}")
            if self.normalize_edge_features:
                _buffer("track_edge_center", np.zeros(edim, dtype=np.float32))
                _buffer("track_edge_scale", np.ones(edim, dtype=np.float32))
                for name in ("track_edge_center", "track_edge_scale"):
                    if getattr(self, name).shape[1] != int(edim):
                        raise ValueError(f"Normalization buffer {name} has incompatible shape {tuple(getattr(self, name).shape)} for edim={edim}")

        def _block_buffers(prefix: str, dim: int) -> None:
            _buffer(f"{prefix}_center", np.zeros(dim, dtype=np.float32))
            _buffer(f"{prefix}_scale", np.ones(dim, dtype=np.float32))
            for name in (f"{prefix}_center", f"{prefix}_scale"):
                if getattr(self, name).shape[1] != int(dim):
                    raise ValueError(f"Normalization buffer {name} has incompatible shape {tuple(getattr(self, name).shape)} for dim={dim}")

        if seg_edim > 0 and self.normalize_edge_features:
            _block_buffers("seg_edge", seg_edim)
        if tracklet_xdim > 0:
            if self.normalize_node_features:
                _block_buffers("tl", tracklet_xdim)
            if self.normalize_edge_features:
                _block_buffers("tracklet_edge", tracklet_edim)

    def _apply_feature_norm_torch(
        self,
        values: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        out = (values - center.to(device=values.device, dtype=values.dtype)) / scale.to(device=values.device, dtype=values.dtype).clamp_min(1e-12)
        if self.feature_norm_clip > 0:
            out = out.clamp(-self.feature_norm_clip, self.feature_norm_clip)
        return out

    @staticmethod
    def _muon_mask(x: torch.Tensor, n_muon_nodes: Optional[Any], node_is_muon: Optional[torch.Tensor]) -> torch.Tensor:
        """Boolean [N] mask of the muon nodes of x (batched: node_is_muon; single graph: the first n_muon_nodes)."""
        if node_is_muon is None:
            if n_muon_nodes is None:
                raise RuntimeError("Model-side node normalization / node types require n_muon_nodes.")
            if not torch.is_tensor(n_muon_nodes):
                n_muon_nodes = torch.tensor([n_muon_nodes], device=x.device, dtype=torch.long)
            n_mu = n_muon_nodes.to(device=x.device, dtype=torch.long).reshape(-1)[0]
            node_ids = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            node_is_muon = node_ids < n_mu
        return node_is_muon.to(device=x.device, dtype=torch.bool).reshape(-1)

    def _normalize_features(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        n_muon_nodes: Optional[Any],
        node_is_muon: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_node_features:
            mu_mask = self._muon_mask(x, n_muon_nodes, node_is_muon).reshape(-1, 1)

            x_mu = self._apply_feature_norm_torch(x, self.mu_center, self.mu_scale)
            x_ca = self._apply_feature_norm_torch(x, self.ca_center, self.ca_scale)
            x = torch.where(mu_mask, x_mu, x_ca)

        if self.normalize_edge_features:
            edge_attr = self._apply_feature_norm_torch(edge_attr, self.edge_center, self.edge_scale)

        return x, edge_attr

    def _prepare_track_inputs(
        self,
        track_x: torch.Tensor,
        track_edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw track block -> transformed + normalized node features and normalized edge features."""
        scale = self.tr_asinh_scale.to(device=track_x.device, dtype=track_x.dtype)
        use_asinh = scale > 0
        safe_scale = torch.where(use_asinh, scale, torch.ones_like(scale))
        track_x = torch.where(use_asinh, onnx_safe_asinh(track_x / safe_scale), track_x)
        if self.normalize_node_features:
            track_x = self._apply_feature_norm_torch(track_x, self.tr_center, self.tr_scale)
        if self.normalize_edge_features:
            track_edge_attr = self._apply_feature_norm_torch(track_edge_attr, self.track_edge_center, self.track_edge_scale)
        return track_x, track_edge_attr

    def _edge_rows(self, attr: torch.Tensor, start: int, flag_col: Optional[int]) -> torch.Tensor:
        """Place an edge block at columns [start, start + width) of the message-passing edge features, with a 1 in
        flag_col (None: no flag), zeros elsewhere (torch.cat only, for the ONNX export)."""
        n = attr.shape[0]
        parts = []
        if start > 0:
            parts.append(attr.new_zeros((n, start)))
        parts.append(attr)
        pos = start + attr.shape[1]
        if flag_col is not None:
            if flag_col > pos:
                parts.append(attr.new_zeros((n, flag_col - pos)))
            parts.append(attr.new_ones((n, 1)))
            pos = flag_col + 1
        if self.edim_model > pos:
            parts.append(attr.new_zeros((n, self.edim_model - pos)))
        return torch.cat(parts, dim=1) if len(parts) > 1 else attr

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        *,
        n_muon_nodes: Optional[Any] = None,
        batch: Optional[torch.Tensor] = None,
        node_is_muon: Optional[torch.Tensor] = None,
        edge_dropout_p: float = 0.0,
        feature_noise_std: float = 0.0,
        track_x: Optional[torch.Tensor] = None,
        track_edge_index: Optional[torch.Tensor] = None,
        track_edge_attr: Optional[torch.Tensor] = None,
        track_batch: Optional[torch.Tensor] = None,
        seg_edge_index: Optional[torch.Tensor] = None,
        seg_edge_attr: Optional[torch.Tensor] = None,
        tracklet_x: Optional[torch.Tensor] = None,
        tracklet_edge_index: Optional[torch.Tensor] = None,
        tracklet_edge_attr: Optional[torch.Tensor] = None,
        tracklet_batch: Optional[torch.Tensor] = None,
    ):
        if self.relation_mp:
            return self._forward_relational(
                x, edge_index, edge_attr, n_muon_nodes=n_muon_nodes, batch=batch, node_is_muon=node_is_muon,
                edge_dropout_p=edge_dropout_p, feature_noise_std=feature_noise_std,
                track_x=track_x, track_edge_index=track_edge_index, track_edge_attr=track_edge_attr,
                track_batch=track_batch, seg_edge_index=seg_edge_index, seg_edge_attr=seg_edge_attr,
                tracklet_x=tracklet_x, tracklet_edge_index=tracklet_edge_index,
                tracklet_edge_attr=tracklet_edge_attr, tracklet_batch=tracklet_batch,
            )
        need_mu_mask = self.type_emb is not None or self.typed_readout is not None
        mu_mask = self._muon_mask(x, n_muon_nodes, node_is_muon) if need_mu_mask else None
        n_x = x.shape[0]
        batch_x = batch
        x, edge_attr = self._normalize_features(x, edge_attr, n_muon_nodes, node_is_muon)

        if self.training and feature_noise_std > 0:
            x = x + torch.randn_like(x) * float(feature_noise_std)

        if self.fourier is not None:
            x = self.fourier(x)

        h = self.node_enc(x)
        if self.type_emb is not None:
            h = h + self.type_emb((~mu_mask).long())  # 0 = muon, 1 = calo

        if self.use_tracks:
            if track_x is None or track_edge_index is None or track_edge_attr is None:
                raise RuntimeError("This model needs the track block inputs: track_x, track_edge_index, track_edge_attr.")
            if batch is not None and track_batch is None:
                raise RuntimeError("Batched inputs with tracks need track_batch.")
            t, track_edge_attr = self._prepare_track_inputs(track_x, track_edge_attr)
            if self.training and feature_noise_std > 0:
                t = t + torch.randn_like(t) * float(feature_noise_std)
            if self.track_fourier is not None:
                t = self.track_fourier(t)
            h_trk = self.track_enc(t)
            if self.type_emb is not None:
                h_trk = h_trk + self.type_emb.weight[NODE_TYPE_TRACK]
            h = torch.cat([h, h_trk.to(dtype=h.dtype)], dim=0)
            edge_index = torch.cat([edge_index, track_edge_index], dim=1)
            edge_attr = torch.cat(
                [
                    self._edge_rows(edge_attr, 0, None),
                    self._edge_rows(track_edge_attr, 0, self._track_flag_col),
                ],
                dim=0,
            )
            if batch is not None:
                batch = torch.cat([batch, track_batch], dim=0)
        elif self.edim_model > self.edim_in:
            edge_attr = self._edge_rows(edge_attr, 0, None)

        if self.use_segment_edges:
            if seg_edge_index is None or seg_edge_attr is None:
                raise RuntimeError("This model needs the segment-segment edge inputs: seg_edge_index, seg_edge_attr.")
            if self.normalize_edge_features:
                seg_edge_attr = self._apply_feature_norm_torch(seg_edge_attr, self.seg_edge_center, self.seg_edge_scale)
            edge_index = torch.cat([edge_index, seg_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, self._edge_rows(seg_edge_attr, self._seg_cols[0], self._seg_cols[1])], dim=0)

        if self.use_tracklets:
            if tracklet_x is None or tracklet_edge_index is None or tracklet_edge_attr is None:
                raise RuntimeError("This model needs the tracklet inputs: tracklet_x, tracklet_edge_index, tracklet_edge_attr.")
            if batch is not None and tracklet_batch is None:
                raise RuntimeError("Batched inputs with tracklets need tracklet_batch.")
            tl = tracklet_x
            if self.normalize_node_features:
                tl = self._apply_feature_norm_torch(tl, self.tl_center, self.tl_scale)
            if self.training and feature_noise_std > 0:
                tl = tl + torch.randn_like(tl) * float(feature_noise_std)
            if self.tracklet_fourier is not None:
                tl = self.tracklet_fourier(tl)
            h_tl = self.tracklet_enc(tl)
            if self.type_emb is not None:
                h_tl = h_tl + self.type_emb.weight[NODE_TYPE_TRACKLET]
            # tracklet node k arrives as n_x + k; the track nodes (if any) sit in between
            n_trk = track_x.shape[0] if self.use_tracks else 0
            tracklet_edge_index = torch.where(tracklet_edge_index >= n_x, tracklet_edge_index + n_trk, tracklet_edge_index)
            if self.normalize_edge_features:
                tracklet_edge_attr = self._apply_feature_norm_torch(
                    tracklet_edge_attr, self.tracklet_edge_center, self.tracklet_edge_scale)
            h = torch.cat([h, h_tl.to(dtype=h.dtype)], dim=0)
            edge_index = torch.cat([edge_index, tracklet_edge_index], dim=1)
            edge_attr = torch.cat(
                [edge_attr, self._edge_rows(tracklet_edge_attr, self._tracklet_cols[0], self._tracklet_cols[1])], dim=0)
            if batch is not None:
                batch = torch.cat([batch, tracklet_batch], dim=0)

        for layer in self.layers:
            if self._uses_edge_attr:
                h = layer(h, edge_index, edge_attr, edge_dropout_p=edge_dropout_p)
            elif isinstance(layer, (GATResidualBlock, PreNormGATBlock)):
                h = layer(h, edge_index, edge_dropout_p=edge_dropout_p)
            else:
                h = layer(h, edge_index)

        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.typed_readout is not None:
            # node order of h: [muon+calo nodes of x | track nodes | tracklet nodes]
            num_graphs = self._num_graphs(n_muon_nodes, batch_x)
            if batch_x is None:
                batch_x = torch.zeros_like(mu_mask, dtype=torch.long)
            mu_idx = torch.nonzero(mu_mask).reshape(-1)
            ca_idx = torch.nonzero(~mu_mask).reshape(-1)
            h_x = h[:n_x]
            parts = {"muon": h_x.index_select(0, mu_idx), "calo": h_x.index_select(0, ca_idx)}
            batches = {"muon": batch_x.index_select(0, mu_idx), "calo": batch_x.index_select(0, ca_idx)}
            n_trk = track_x.shape[0] if self.use_tracks else 0
            if self.use_tracks:
                parts["track"] = h[n_x:n_x + n_trk]
                batches["track"] = track_batch if batch is not None else torch.zeros_like(parts["track"][:, 0], dtype=torch.long)
            if self.use_tracklets:
                parts["tracklet"] = h[n_x + n_trk:]
                batches["tracklet"] = (
                    tracklet_batch if batch is not None else torch.zeros_like(parts["tracklet"][:, 0], dtype=torch.long)
                )
            g = self.typed_readout(parts, batches, num_graphs)
        else:
            g = global_pool_batched(h, batch=batch, mode=self.pool)
        return self.head(g).view(-1)

    def _forward_relational(
        self,
        x,
        edge_index,
        edge_attr,
        *,
        n_muon_nodes,
        batch,
        node_is_muon,
        edge_dropout_p,
        feature_noise_std,
        track_x,
        track_edge_index,
        track_edge_attr,
        track_batch,
        seg_edge_index,
        seg_edge_attr,
        tracklet_x,
        tracklet_edge_index,
        tracklet_edge_attr,
        tracklet_batch,
    ):
        """relation_mp=True: per-type node states, type-local edge lists per relation (same inputs as forward)."""
        dev = x.device
        mu_mask = self._muon_mask(x, n_muon_nodes, node_is_muon)
        n_x = x.shape[0]
        num_graphs = self._num_graphs(n_muon_nodes, batch)
        x, edge_attr = self._normalize_features(x, edge_attr, n_muon_nodes, node_is_muon)
        if self.training and feature_noise_std > 0:
            x = x + torch.randn_like(x) * float(feature_noise_std)
        if self.fourier is not None:
            x = self.fourier(x)

        batch_x = batch if batch is not None else torch.zeros_like(mu_mask, dtype=torch.long)
        mu_idx = torch.nonzero(mu_mask).reshape(-1)
        ca_idx = torch.nonzero(~mu_mask).reshape(-1)
        h = {
            "muon": self.node_enc(x.index_select(0, mu_idx)).float(),
            "calo": self.calo_enc(x.index_select(0, ca_idx)).float(),
        }
        if self.type_emb is not None:
            h["muon"] = h["muon"] + self.type_emb.weight[NODE_TYPE_MUON].float()
            h["calo"] = h["calo"] + self.type_emb.weight[NODE_TYPE_CALO].float()
        batches = {"muon": batch_x.index_select(0, mu_idx), "calo": batch_x.index_select(0, ca_idx)}
        # index of every muon / calo node of x inside its own type
        mu_l = mu_mask.long()
        loc_x = torch.where(mu_mask, torch.cumsum(mu_l, dim=0) - 1, torch.cumsum(1 - mu_l, dim=0) - 1)

        graph = _RelationGraph()
        edge_feats: Dict[str, torch.Tensor] = {"calo": edge_attr}
        src, dst = edge_index[0], edge_index[1]
        src_mu, dst_mu = mu_mask[src], mu_mask[dst]
        directions = [("mu_from_calo", dst_mu & ~src_mu)]
        if self.calo_context == "two_way":
            directions.append(("calo_from_mu", src_mu & ~dst_mu))
        for name, sel_mask in directions:
            rows = torch.nonzero(sel_mask).reshape(-1)
            graph.relations[name] = (loc_x[src.index_select(0, rows)], loc_x[dst.index_select(0, rows)], rows)

        if self.use_tracks:
            if track_x is None or track_edge_index is None or track_edge_attr is None:
                raise RuntimeError("This model needs the track block inputs: track_x, track_edge_index, track_edge_attr.")
            if batch is not None and track_batch is None:
                raise RuntimeError("Batched inputs with tracks need track_batch.")
            t, track_edge_attr = self._prepare_track_inputs(track_x, track_edge_attr)
            if self.training and feature_noise_std > 0:
                t = t + torch.randn_like(t) * float(feature_noise_std)
            if self.track_fourier is not None:
                t = self.track_fourier(t)
            h["track"] = self.track_enc(t).float()
            if self.type_emb is not None:
                h["track"] = h["track"] + self.type_emb.weight[NODE_TYPE_TRACK].float()
            batches["track"] = track_batch if batch is not None else torch.zeros_like(h["track"][:, 0], dtype=torch.long)
            edge_feats["track"] = track_edge_attr
            # track node t arrives as n_x + t
            loc_trk = torch.cat([loc_x, torch.arange(t.shape[0], device=dev, dtype=torch.long)])
            ts, td = track_edge_index[0], track_edge_index[1]
            to_trk = td >= n_x
            for name, sel_mask in (("mu_from_trk", ~to_trk), ("trk_from_mu", to_trk)):
                rows = torch.nonzero(sel_mask).reshape(-1)
                graph.relations[name] = (loc_trk[ts.index_select(0, rows)], loc_trk[td.index_select(0, rows)], rows)

        if self.use_segment_edges:
            if seg_edge_index is None or seg_edge_attr is None:
                raise RuntimeError("This model needs the segment-segment edge inputs: seg_edge_index, seg_edge_attr.")
            if self.normalize_edge_features:
                seg_edge_attr = self._apply_feature_norm_torch(seg_edge_attr, self.seg_edge_center, self.seg_edge_scale)
            ss, sd = loc_x[seg_edge_index[0]], loc_x[seg_edge_index[1]]
            graph.relations["mu_from_mu"] = (ss, sd, None)
            if self.seg_edge_enc is not None:
                edge_feats["seg"] = self.seg_edge_enc(seg_edge_attr).float()
                graph.endpoints["seg"] = ("muon", ss, sd)
            else:
                edge_feats["seg"] = seg_edge_attr

        if self.use_tracklets:
            if tracklet_x is None or tracklet_edge_index is None or tracklet_edge_attr is None:
                raise RuntimeError("This model needs the tracklet inputs: tracklet_x, tracklet_edge_index, tracklet_edge_attr.")
            if batch is not None and tracklet_batch is None:
                raise RuntimeError("Batched inputs with tracklets need tracklet_batch.")
            tl = tracklet_x
            if self.normalize_node_features:
                tl = self._apply_feature_norm_torch(tl, self.tl_center, self.tl_scale)
            if self.training and feature_noise_std > 0:
                tl = tl + torch.randn_like(tl) * float(feature_noise_std)
            if self.tracklet_fourier is not None:
                tl = self.tracklet_fourier(tl)
            h["tracklet"] = self.tracklet_enc(tl).float()
            if self.type_emb is not None:
                h["tracklet"] = h["tracklet"] + self.type_emb.weight[NODE_TYPE_TRACKLET].float()
            batches["tracklet"] = (
                tracklet_batch if batch is not None else torch.zeros_like(h["tracklet"][:, 0], dtype=torch.long)
            )
            if self.normalize_edge_features:
                tracklet_edge_attr = self._apply_feature_norm_torch(
                    tracklet_edge_attr, self.tracklet_edge_center, self.tracklet_edge_scale)
            # tracklet node k arrives as n_x + k (the tracks do not shift it here)
            loc_tl = torch.cat([loc_x, torch.arange(tl.shape[0], device=dev, dtype=torch.long)])
            ks, kd = tracklet_edge_index[0], tracklet_edge_index[1]
            s_tl, d_tl = ks >= n_x, kd >= n_x
            ks_l, kd_l = loc_tl[ks], loc_tl[kd]
            for name, sel_mask in (("mu_from_tl", s_tl & ~d_tl), ("tl_from_mu", d_tl & ~s_tl), ("tl_from_tl", s_tl & d_tl)):
                rows = torch.nonzero(sel_mask).reshape(-1)
                graph.relations[name] = (ks_l.index_select(0, rows), kd_l.index_select(0, rows), rows)
            if self.tracklet_edge_enc is not None:
                edge_feats["tracklet"] = self.tracklet_edge_enc(tracklet_edge_attr).float()
                n_mu = h["muon"].shape[0]
                graph.endpoints["tracklet"] = (
                    "muon_tracklet", torch.where(s_tl, ks_l + n_mu, ks_l), torch.where(d_tl, kd_l + n_mu, kd_l))
            else:
                edge_feats["tracklet"] = tracklet_edge_attr

        for layer in self.layers:
            h, edge_feats = layer(h, graph, edge_feats, edge_dropout_p=edge_dropout_p)

        if self.final_norm is not None:
            h = {t: self.final_norm[t](v) for t, v in h.items()}
        if self.typed_readout is not None:
            g = self.typed_readout(h, batches, num_graphs)
        else:
            h_all = torch.cat([h[t] for t in self.node_types], dim=0)
            if batch is None:
                g = global_pool(h_all, mode=self.pool)
            else:
                g = global_pool_batched(h_all, batch=torch.cat([batches[t] for t in self.node_types], dim=0),
                                        mode=self.pool)
        return self.head(g).view(-1)


def _binary_counts_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits.detach())
    pred = (probs >= float(threshold)).to(torch.int64)
    y = labels.detach().to(torch.int64)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    return torch.stack([tp, fp, tn, fn]).to(torch.float64)


def _metrics_from_counts(counts: torch.Tensor) -> Dict[str, float]:
    tp, fp, tn, fn = [float(x) for x in counts.detach().cpu().tolist()]
    total = max(tp + fp + tn + fn, 1.0)
    acc = (tp + tn) / total
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_acc = 0.5 * (recall + specificity)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _average_ranks_for_auc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    i = 0
    while i < scores.shape[0]:
        j = i + 1
        while j < scores.shape[0] and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def binary_auc_np(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores).astype(np.float64).reshape(-1)
    finite = np.isfinite(scores) & np.isfinite(labels)
    labels = labels[finite]
    scores = scores[finite]
    n_pos = int(np.count_nonzero(labels == 1))
    n_neg = int(np.count_nonzero(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks_for_auc(scores)
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def tpr_at_fpr_np(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> Dict[str, float]:
    """
    Return the best achievable TPR with FPR <= target_fpr.

    Scores are interpreted as signal scores: larger score -> more signal-like.
    The threshold is chosen from the validation scores. If several thresholds satisfy
    the FPR constraint, the one with the highest TPR is used; ties prefer lower FPR.
    """
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores).astype(np.float64).reshape(-1)
    finite = np.isfinite(scores) & np.isfinite(labels)
    labels = labels[finite]
    scores = scores[finite]

    n_pos = int(np.count_nonzero(labels == 1))
    n_neg = int(np.count_nonzero(labels == 0))
    if n_pos == 0 or n_neg == 0 or scores.size == 0:
        return {
            "tpr": float("nan"),
            "fpr": float("nan"),
            "threshold": float("nan"),
            "n_pos": float(n_pos),
            "n_neg": float(n_neg),
        }

    target_fpr = float(target_fpr)
    target_fpr = min(max(target_fpr, 0.0), 1.0)

    # Candidate 0: threshold above all scores, accepts no events.
    best_tpr = 0.0
    best_fpr = 0.0
    best_thr = float(np.nextafter(np.max(scores), np.inf))

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = labels[order]
    s_sorted = scores[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    # Evaluate only after the last item for each unique score value.
    end_of_block = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    idxs = np.nonzero(end_of_block)[0]
    tpr = tp_cum[idxs].astype(np.float64) / float(n_pos)
    fpr = fp_cum[idxs].astype(np.float64) / float(n_neg)
    thr = s_sorted[idxs].astype(np.float64)

    ok = fpr <= (target_fpr + 1e-15)
    if np.any(ok):
        cand_idxs = np.nonzero(ok)[0]
        cand_tpr = tpr[cand_idxs]
        max_tpr = np.max(cand_tpr)
        tied = cand_idxs[np.isclose(cand_tpr, max_tpr, rtol=0.0, atol=1e-15)]
        # Among equal-TPR candidates, choose the lowest FPR; then the lowest threshold
        # so the point is reproducible.
        tied_fpr = fpr[tied]
        min_fpr = np.min(tied_fpr)
        tied2 = tied[np.isclose(tied_fpr, min_fpr, rtol=0.0, atol=1e-15)]
        j = tied2[-1]
        best_tpr = float(tpr[j])
        best_fpr = float(fpr[j])
        best_thr = float(thr[j])

    return {
        "tpr": best_tpr,
        "fpr": best_fpr,
        "threshold": best_thr,
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
    }


class BCEWithLogitsLabelSmoothingLoss(nn.Module):
    def __init__(self, *, pos_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.detach().clone() if pos_weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.float()
        eps = min(max(self.label_smoothing, 0.0), 0.499)
        if eps > 0.0:
            y = y * (1.0 - eps) + 0.5 * eps
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight, reduction="mean")


class FocalLossWithLogits(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None or float(alpha) < 0 else float(alpha)
        self.register_buffer("pos_weight", pos_weight.detach().clone() if pos_weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y_hard = targets.float()
        y = y_hard
        eps = min(max(self.label_smoothing, 0.0), 0.499)
        if eps > 0.0:
            y = y_hard * (1.0 - eps) + 0.5 * eps
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * y_hard + (1.0 - p) * (1.0 - y_hard)
        mod = torch.pow((1.0 - p_t).clamp(min=1e-8), self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * y_hard + (1.0 - self.alpha) * (1.0 - y_hard)
            mod = mod * alpha_t
        return (mod * bce).mean()


class AsymmetricFocalLossWithLogits(nn.Module):
    def __init__(
        self,
        *,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        alpha: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.alpha = None if alpha is None or float(alpha) < 0 else float(alpha)
        self.register_buffer("pos_weight", pos_weight.detach().clone() if pos_weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y_hard = targets.float()
        y = y_hard
        eps = min(max(self.label_smoothing, 0.0), 0.499)
        if eps > 0.0:
            y = y_hard * (1.0 - eps) + 0.5 * eps
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pos_mod = torch.pow((1.0 - p).clamp(min=1e-8), self.gamma_pos)
        neg_mod = torch.pow(p.clamp(min=1e-8), self.gamma_neg)
        mod = y_hard * pos_mod + (1.0 - y_hard) * neg_mod
        if self.alpha is not None:
            alpha_t = self.alpha * y_hard + (1.0 - self.alpha) * (1.0 - y_hard)
            mod = mod * alpha_t
        return (mod * bce).mean()


def _parse_optional_float(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if s in ("none", "off", "false", "no", "-1", ""):
        return None
    return float(s)


def build_classification_loss(args, *, pos_weight_tensor: Optional[torch.Tensor]) -> nn.Module:
    loss_type = str(args.loss_type).strip().lower()
    alpha = _parse_optional_float(args.focal_alpha)
    label_smoothing = float(args.label_smoothing)

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if loss_type == "bce_smooth":
        return BCEWithLogitsLabelSmoothingLoss(
            pos_weight=pos_weight_tensor,
            label_smoothing=label_smoothing,
        )
    if loss_type == "focal":
        return FocalLossWithLogits(
            gamma=float(args.focal_gamma),
            alpha=alpha,
            pos_weight=pos_weight_tensor,
            label_smoothing=label_smoothing,
        )
    if loss_type == "asymmetric_focal":
        return AsymmetricFocalLossWithLogits(
            gamma_pos=float(args.asym_gamma_pos),
            gamma_neg=float(args.asym_gamma_neg),
            alpha=alpha,
            pos_weight=pos_weight_tensor,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unknown --loss-type {args.loss_type!r}")


def _ddp_gather_1d(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().view(-1)
    if not ddp_is_initialized():
        return t
    device = t.device
    local_n = torch.tensor([t.numel()], dtype=torch.long, device=device)
    sizes = [torch.zeros_like(local_n) for _ in range(ddp_world_size())]
    dist.all_gather(sizes, local_n)
    max_n = int(torch.stack(sizes).max().item())
    if t.numel() < max_n:
        pad = torch.zeros(max_n - t.numel(), dtype=t.dtype, device=device)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t
    gathered = [torch.zeros(max_n, dtype=t.dtype, device=device) for _ in range(ddp_world_size())]
    dist.all_gather(gathered, t_pad)
    pieces = [g[: int(s.item())] for g, s in zip(gathered, sizes)]
    return torch.cat(pieces, dim=0)


def _count_labels(labels: np.ndarray, indices: np.ndarray) -> Tuple[int, int]:
    vals = labels[np.asarray(indices, dtype=np.int64)]
    n_pos = int(np.count_nonzero(vals == 1.0))
    n_neg = int(np.count_nonzero(vals == 0.0))
    return n_neg, n_pos


def _parse_pos_weight(text: str, n_neg: int, n_pos: int) -> Optional[float]:
    s = str(text).strip().lower()
    if s in ("none", "off", "false", "0"):
        return None
    if s == "auto":
        if n_pos <= 0:
            return None
        return float(n_neg) / max(float(n_pos), 1.0)
    v = float(s)
    return v if v > 0 else None


def _monitor_is_better(name: str, new_value: float, best_value: Optional[float], min_delta: float) -> bool:
    if best_value is None:
        return True
    if name == "val_loss":
        return new_value < (best_value - min_delta)
    return new_value > (best_value + min_delta)


@torch.no_grad()
def _evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    threshold: float,
    target_fpr: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    edge_dropout_p: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    # Validation shards deliberately contain every event exactly once and can
    # therefore differ in length by one. Bypass the DDP wrapper during forward
    # passes so ranks with unequal final batch counts do not wait on DDP's
    # forward-time collectives. Metrics are synchronized explicitly below.
    eval_model = model.module if ddp_is_initialized() and hasattr(model, "module") else model
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
    total_n = torch.tensor(0.0, device=device, dtype=torch.float64)
    counts = torch.zeros(4, device=device, dtype=torch.float64)
    logits_parts = []
    labels_parts = []

    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp) if device.type == "cuda" else nullcontext()

    for batch in loader:
        model_args, model_kwargs = batch_model_inputs(batch, device)
        y = batch["y"].to(device, non_blocking=True).float().view(-1)
        with autocast_ctx:
            logits = eval_model(
                *model_args,
                **model_kwargs,
                edge_dropout_p=edge_dropout_p,
            ).view_as(y)
            loss = criterion(logits, y)
        total_loss += loss.detach().to(torch.float64) * y.numel()
        total_n += float(y.numel())
        counts += _binary_counts_from_logits(logits, y, threshold=threshold)
        logits_parts.append(logits.detach().float().view(-1))
        labels_parts.append(y.detach().float().view(-1))

    if ddp_is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_n, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    loss_mean = (total_loss / total_n.clamp(min=1.0)).item()
    metrics = _metrics_from_counts(counts)
    metrics["loss"] = loss_mean

    if logits_parts:
        logits_all = _ddp_gather_1d(torch.cat(logits_parts).to(device))
        labels_all = _ddp_gather_1d(torch.cat(labels_parts).to(device))
        if ddp_is_main():
            labels_np = labels_all.detach().cpu().numpy()
            logits_np = logits_all.detach().cpu().numpy()
            auc = binary_auc_np(labels_np, logits_np)
            fixed = tpr_at_fpr_np(labels_np, logits_np, target_fpr=target_fpr)
            fixed_vec = [
                float(fixed["tpr"]),
                float(fixed["fpr"]),
                float(fixed["threshold"]),
                float(fixed["n_pos"]),
                float(fixed["n_neg"]),
            ]
        else:
            auc = float("nan")
            fixed_vec = [float("nan")] * 5
        if ddp_is_initialized():
            metric_t = torch.tensor([auc] + fixed_vec, dtype=torch.float64, device=device)
            dist.broadcast(metric_t, src=0)
            vals = [float(x) for x in metric_t.detach().cpu().tolist()]
            auc = vals[0]
            fixed_vec = vals[1:]
        metrics["auc"] = auc
        metrics["tpr_at_target_fpr"] = fixed_vec[0]
        metrics["fpr_at_target_fpr"] = fixed_vec[1]
        metrics["threshold_at_target_fpr"] = fixed_vec[2]
        metrics["n_pos_for_curve"] = fixed_vec[3]
        metrics["n_neg_for_curve"] = fixed_vec[4]
    else:
        metrics["auc"] = float("nan")
        metrics["tpr_at_target_fpr"] = float("nan")
        metrics["fpr_at_target_fpr"] = float("nan")
        metrics["threshold_at_target_fpr"] = float("nan")
        metrics["n_pos_for_curve"] = 0.0
        metrics["n_neg_for_curve"] = 0.0

    return metrics


def build_scheduler(opt, args, steps_per_epoch: int):
    if args.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience, min_lr=args.lr_plateau_min_lr,
        )

    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    total_steps = max(1, int(args.epochs * steps_per_epoch))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def reset_optimizer_lr(opt, lr: float):
    for pg in opt.param_groups:
        pg["lr"] = lr


def load_best_checkpoint_into_model(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state"], strict=True)
    raw_model.to(device)


def _try_resume_from_checkpoint(
    *,
    ckpt_path: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[EMA],
    device: torch.device,
):
    p = Path(ckpt_path)
    if not p.exists():
        return 1, None, 0, None, False

    ckpt = torch.load(str(p), map_location="cpu")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state"], strict=True)
    raw_model.to(device)

    if "optimizer_state" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt and scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
    if "scaler_state" in ckpt and scaler is not None and scaler.is_enabled():
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            pass
    if ema is not None and "ema_shadow" in ckpt and isinstance(ckpt["ema_shadow"], dict):
        raw_state = raw_model.state_dict()
        ema.shadow = {
            k: v.detach().to(device=raw_state[k].device, dtype=raw_state[k].dtype).clone()
            if k in raw_state and torch.is_tensor(v)
            else v.clone()
            for k, v in ckpt["ema_shadow"].items()
            if torch.is_tensor(v)
        }

    last_epoch = int(ckpt.get("epoch", 0))
    best_monitor = ckpt.get("best_monitor", None)
    bad_epochs = int(ckpt.get("bad_epochs", 0))
    best_ckpt_epoch = ckpt.get("best_ckpt_epoch", None)
    start_epoch = max(1, last_epoch + 1)
    return start_epoch, best_monitor, bad_epochs, best_ckpt_epoch, True




def add_training_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--strict-split-check", action="store_true", default=False,
                    help="Require exact split-file path compatibility.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.00034882957103492035)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.020720971979495448)

    ap.add_argument("--layer-type", default="gat_residual",
                    choices=["mpnn", "edge_residual", "sage_residual", "gat_residual"])
    ap.add_argument("--gat-heads", type=int, default=2)
    ap.add_argument("--gat-edge-attn", action="store_true", default=False,
                    help="Use edge_attr through a normalized edge encoder in GAT attention logits.")
    ap.add_argument("--no-gat-edge-attn", dest="gat_edge_attn", action="store_false")
    ap.add_argument(
        "--gatv2-edge-attn",
        action="store_true",
        default=True,
        help=(
            "Use nonlinear source-destination-edge GATv2 attention and "
            "edge-conditioned messages: score=a^T SiLU(W_src h_src + "
            "W_dst h_dst + W_edge e), message=SiLU(V_src h_src + "
            "V_dst h_dst + V_edge e)."
        ),
    )
    ap.add_argument(
        "--no-gatv2-edge-attn",
        dest="gatv2_edge_attn",
        action="store_false",
    )
    ap.add_argument("--sage-aggr", default="mean", choices=["mean", "sum", "max"])
    ap.add_argument("--edgeconv-aggr", default="mean", choices=["mean", "sum", "max"])
    ap.add_argument("--pool", default="meanmax", choices=["mean", "max", "sum", "meanmax"])

    ap.add_argument("--feature-stats-json", default=None,
                    help="JSON file with precomputed node/edge feature normalization stats.")
    ap.add_argument("--feature-norm-kind", default="standard", choices=["standard", "robust"],
                    help="Which stats block to use from --feature-stats-json.")
    ap.add_argument("--normalize-node-features", action="store_true", default=True,
                    help="Apply precomputed normalization to node features x.")
    ap.add_argument("--no-normalize-node-features", dest="normalize_node_features", action="store_false")
    ap.add_argument("--normalize-edge-features", action="store_true", default=True,
                    help="Apply precomputed normalization to edge_attr.")
    ap.add_argument("--no-normalize-edge-features", dest="normalize_edge_features", action="store_false")
    ap.add_argument("--feature-norm-clip", type=float, default=-1.0,
                    help="Optional absolute clip after feature normalization. <=0 disables clipping.")

    ap.add_argument("--pos-weight", default="auto",
                    help="Positive-class weight: 'auto', 'none', or a positive float. Auto uses n_negative/n_positive in the train split.")
    ap.add_argument("--loss-type", default="focal",
                    choices=["bce", "bce_smooth", "focal", "asymmetric_focal"],
                    help="Binary classification loss function.")
    ap.add_argument("--label-smoothing", type=float, default=0.013390460005485128,
                    help="Binary-label smoothing used by bce_smooth/focal/asymmetric_focal. 0 disables it.")
    ap.add_argument("--focal-gamma", type=float, default=0.6358567127518362,
                    help="Gamma for focal loss.")
    ap.add_argument("--focal-alpha", default="0.8839913658720648",
                    help="Optional positive-class alpha for focal/asymmetric_focal: 'none' or float in (0,1).")
    ap.add_argument("--asym-gamma-pos", type=float, default=0.0,
                    help="Positive-class gamma for asymmetric focal loss.")
    ap.add_argument("--asym-gamma-neg", type=float, default=4.0,
                    help="Negative-class gamma for asymmetric focal loss.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for class metrics.")
    ap.add_argument("--target-fpr", type=float, default=0.01,
                    help="FPR working point for tpr_at_target_fpr metric. Default 0.01 = 1%% FPR.")
    ap.add_argument("--max-train-events", type=int, default=-1)

    ap.add_argument("--save", default="displaced_vertex_classifier_gnn.pt")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", dest="resume", action="store_true", default=False)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--code-version", default=None)

    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--time", action="store_true", default=True)
    ap.add_argument("--no-time", dest="time", action="store_false")

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Graphs per optimizer step on each rank. Larger values reduce DDP synchronization overhead.")
    ap.add_argument("--sampler-mix-files", type=int, default=-1,
                    help="Training batches mix the events of this many files (every file holds one sample, i.e. "
                         "one class). -1 (default): --max-open-h5-files for H5 parts, all files for packed parts; "
                         "0: all files; 1: the former single-file batches.")
    ap.add_argument("--max-open-h5-files", type=int, default=16,
                    help="Maximum HDF5 files kept open by each dataset process.")
    ap.add_argument("--pin-memory", action="store_true", default=True)
    ap.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=True)
    ap.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    ap.add_argument("--worker-start-method", type=str, default="fork",
                    choices=["fork", "forkserver", "spawn"])

    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="DisplacedVertex")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-dir", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-key", default=None)

    ap.add_argument("--early-stop", dest="early_stop", action="store_true", default=True)
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    ap.add_argument("--early-stop-patience", type=int, default=20)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--early-stop-monitor", choices=["val_loss", "val_auc", "val_acc", "val_f1", "val_balanced_acc", "val_tpr_at_target_fpr"], default="val_tpr_at_target_fpr")

    ap.add_argument("--lr-schedule", choices=["plateau", "cosine"], default="plateau")
    ap.add_argument("--lr-plateau-factor", type=float, default=0.5)
    ap.add_argument("--lr-plateau-patience", type=int, default=5)
    ap.add_argument("--lr-plateau-min-lr", type=float, default=0.0)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)

    ap.add_argument("--reload-best-half-patience", dest="reload_best_half_patience",
                    action="store_true", default=False)

    ap.add_argument("--fourier", dest="fourier", action="store_true", default=False)
    ap.add_argument("--no-fourier", dest="fourier", action="store_false")
    ap.add_argument("--fourier-base", type=float, default=3.0)
    ap.add_argument("--fourier-min-exp", type=int, default=-6)
    ap.add_argument("--fourier-max-exp", type=int, default=6)

    ap.add_argument("--weight-decay", type=float, default=2.8684229445023318e-06)
    ap.add_argument("--no-decay-norm-bias", action="store_true", default=True)
    ap.add_argument("--decay-norm-bias", dest="no_decay_norm_bias", action="store_false")

    ap.add_argument("--edge-dropout", type=float, default=0.033758773173194756)
    ap.add_argument("--feat-noise-std", type=float, default=0.0018684258382055787)

    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--ema-decay", type=float, default=0.999)

    ap.add_argument("--use-tracks", action="store_true", default=False,
                    help="Use the track block of the graphs (track_x, track_edge_index, track_edge_attr; "
                         "DisplacedVertex_converter.py with a track tree) as a third node type. The feature stats "
                         "JSON must then contain the track stats of the current DisplacedVertex_preproc.py.")
    ap.add_argument("--use-segment-edges", action="store_true", default=True,
                    help="Use the segment-segment edges of the graphs (seg_edge_index, seg_edge_attr; "
                         "DisplacedVertex_converter.py --segment-edges). With edge normalization the stats JSON must "
                         "contain the seg_edge_* stats. Default: on.")
    ap.add_argument("--no-use-segment-edges", dest="use_segment_edges", action="store_false")
    ap.add_argument("--use-tracklets", action="store_true", default=True,
                    help="Use the MS tracklet nodes of the graphs (tracklet_x, tracklet_edge_index, tracklet_edge_attr; "
                         "DisplacedVertex_converter.py --tracklets) as a fourth node type. With normalization the stats "
                         "JSON must contain the tl_* / tracklet_edge_* stats. Default: on.")
    ap.add_argument("--no-use-tracklets", dest="use_tracklets", action="store_false")
    ap.add_argument("--node-type-embedding", dest="node_type_embedding", action="store_true", default=None,
                    help="Add a learned node-type embedding (muon / calo / track / tracklet). "
                         "Default: on with --use-tracks or --use-tracklets.")
    ap.add_argument("--no-node-type-embedding", dest="node_type_embedding", action="store_false")

    # Architecture options (gat_residual; the defaults give the legacy model)
    ap.add_argument("--block", default="legacy", choices=list(BLOCK_TYPES),
                    help="legacy: h + ReLU(GAT(h)) residual blocks. prenorm: h += Drop(MP(LN(h))), "
                         "h += Drop(FFN(LN(h))) (--ffn-mult), final LayerNorm before the readout.")
    ap.add_argument("--ffn-mult", type=int, default=2,
                    help="Hidden width of the feed-forward sublayer of --block prenorm, in units of --hidden-dim "
                         "(0: no feed-forward sublayer).")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="Clip the global gradient norm to this value (<= 0: off). With clipping on, the epoch log "
                         "shows the largest gradient norm and the fraction of clipped steps.")
    ap.add_argument("--relation-mp", action="store_true", default=False,
                    help="Relation-typed message passing: one GATv2 attention (own projections and softmax, "
                         "--gat-heads heads) per relation dst <- src (muon <- calo, calo <- muon, muon <- track, "
                         "track <- muon, muon <- muon, muon <- tracklet, tracklet <- muon, tracklet <- tracklet), "
                         "a self term per node type, separate muon / calo encoders.")
    ap.add_argument("--no-relation-mp", dest="relation_mp", action="store_false")
    ap.add_argument("--relation-aggr", default="attn_max", choices=list(RELATION_AGGRS),
                    help="--relation-mp output of each relation: attention mean of the messages (attn) or attention "
                         "mean and elementwise max (attn_max).")
    ap.add_argument("--calo-context", default="two_way", choices=list(CALO_CONTEXTS),
                    help="--relation-mp: two_way also updates the calo towers from the muon segments; one_way keeps the "
                         "towers at their encoder output (fixed context for the segments, less compute).")
    ap.add_argument("--edge-update", default="none", choices=list(EDGE_UPDATE_MODES),
                    help="--relation-mp: hidden edge states updated in every layer from both end nodes, on the "
                         "segment-segment edges (seg) or also on the tracklet edges (seg_tracklet).")
    ap.add_argument("--readout", default="global", choices=list(READOUT_TYPES),
                    help="global: --pool over all nodes. typed: mean+max per node type. typed_attn: typed plus "
                         "--readout-heads attention pooling over the muon segments and the tracklets.")
    ap.add_argument("--readout-heads", type=int, default=4,
                    help="Attention-pooling heads of --readout typed_attn (must divide --hidden-dim).")
    return ap


def h5_converter_config(h5_path: str) -> Dict[str, Any]:
    """Effective DisplacedVertex_converter.py settings recorded in a converted file ({} for older files)."""
    if is_packed_part(h5_path):
        return dict(read_packed_meta(h5_path).get("converter_config") or {})
    with h5py.File(h5_path, "r") as f:
        if "converter_config" in f.attrs:
            try:
                return dict(json.loads(_decode_attr_to_str(f.attrs["converter_config"])))
            except Exception:
                return {}
    return {}


def h5_muon_node_feature_names(h5_path: str) -> List[str]:
    """Muon-node feature names of a converted file (column 5: energy_like, or chi2_ndof with --segment-chi2-feature)."""
    if is_packed_part(h5_path):
        return list(read_packed_meta(h5_path).get("muon_node_feature_names") or DV_NODE_FEATURE_NAMES)
    with h5py.File(h5_path, "r") as f:
        if "muon_node_feature_names" in f.attrs:
            return list(json.loads(_decode_attr_to_str(f.attrs["muon_node_feature_names"])))
    return list(DV_NODE_FEATURE_NAMES)


def _fmt_metric(v: float) -> str:
    return "nan" if v is None or not np.isfinite(v) else f"{float(v):.5f}"


def run_training(args, *, task_name: str = "displaced_vertex_classification"):
    if args.gat_edge_attn and args.gatv2_edge_attn:
        raise ValueError(
            "--gat-edge-attn and --gatv2-edge-attn select different GAT "
            "architectures and cannot be enabled together."
        )
    if (args.gat_edge_attn or args.gatv2_edge_attn) and args.layer_type != "gat_residual":
        raise ValueError(
            "Edge-aware GAT attention requires --layer-type gat_residual."
        )

    try:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGBUS, all_threads=True, chain=True)
    except Exception:
        pass

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_interop_threads(int(os.environ.get("TORCH_INTEROP_THREADS", "1")))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    try:
        mp.set_start_method(args.worker_start_method, force=True)
    except RuntimeError:
        pass
    ctx = mp.get_context(args.worker_start_method)

    ddp_setup()
    seed_all(args.seed + 1000 * ddp_rank())

    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if torch.cuda.is_available() and "LOCAL_RANK" in os.environ
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if ddp_is_main():
        print(f"[i] world_size={ddp_world_size()} device={device}", flush=True)

    run_id = args.run_id
    if ddp_is_main() and run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    if ddp_is_initialized():
        obj = [run_id]
        dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]

    save_path = _build_save_path(args, run_id)
    if ddp_is_main():
        print(f"[i] checkpoint path: {save_path}", flush=True)

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"No H5 files matched: {args.data_glob}")

    feature_stats = None
    if args.normalize_node_features or args.normalize_edge_features:
        if args.feature_stats_json is None:
            default_stats = os.path.join(
                os.path.dirname(os.path.abspath(args.split_file)),
                "normalization_stats_raw.json",
            )
            if os.path.isfile(default_stats):
                args.feature_stats_json = default_stats
                if ddp_is_main():
                    print(
                        f"[i] using default feature stats beside split file: {default_stats}",
                        flush=True,
                    )
            else:
                raise SystemExit(
                    "Feature normalization is enabled by default, but "
                    f"{default_stats} was not found. Pass --feature-stats-json PATH "
                    "or explicitly disable normalization with "
                    "--no-normalize-node-features --no-normalize-edge-features."
                )
        feature_stats = load_feature_stats_json(args.feature_stats_json, norm_kind=args.feature_norm_kind)
        if args.normalize_edge_features and not feature_stats["meta"]["has_edge_stats"]:
            raise SystemExit("Edge normalization requested, but the stats JSON does not contain edge stats.")
        if args.use_tracks and args.normalize_node_features and not feature_stats["meta"]["has_track_stats"]:
            raise SystemExit(
                "--use-tracks with node normalization needs the track stats (tr_*) in the stats JSON: rerun "
                "DisplacedVertex_preproc.py on the graphs with the track block."
            )
        if args.use_tracks and args.normalize_edge_features and not feature_stats["meta"]["has_track_edge_stats"]:
            raise SystemExit("--use-tracks with edge normalization needs the track_edge_* stats in the stats JSON.")
        if args.use_segment_edges and args.normalize_edge_features and not feature_stats["meta"]["has_seg_edge_stats"]:
            raise SystemExit("--use-segment-edges with edge normalization needs the seg_edge_* stats in the stats JSON: "
                             "rerun DisplacedVertex_preproc.py on the graphs with the segment-segment edges.")
        if args.use_tracklets and args.normalize_node_features and not feature_stats["meta"]["has_tl_stats"]:
            raise SystemExit("--use-tracklets with node normalization needs the tl_* stats in the stats JSON: rerun "
                             "DisplacedVertex_preproc.py on the graphs with the tracklet block.")
        if args.use_tracklets and args.normalize_edge_features and not feature_stats["meta"]["has_tracklet_edge_stats"]:
            raise SystemExit("--use-tracklets with edge normalization needs the tracklet_edge_* stats in the stats JSON.")
    # Muon-node column 5 (energy_like or chi2_ndof): the same convention in all graphs and in the stats
    muon_node_feature_names = h5_muon_node_feature_names(paths[0])
    for p_other in paths[1:]:
        if h5_muon_node_feature_names(p_other) != muon_node_feature_names:
            raise SystemExit(f"{p_other}: muon_node_feature_names differ from {paths[0]} "
                             "(mixed DisplacedVertex_converter.py --segment-chi2-feature settings).")
    if (feature_stats is not None and args.normalize_node_features
            and feature_stats["meta"]["muon_node_feature_names"] != muon_node_feature_names):
        raise SystemExit(f"The stats JSON was computed on muon nodes {feature_stats['meta']['muon_node_feature_names']}, "
                         f"but the graphs have {muon_node_feature_names}: rerun DisplacedVertex_preproc.py.")
    # Converter settings of the training graphs (segment-edge / tracklet parameters go into the ONNX metadata)
    converter_config = h5_converter_config(paths[0])
    if args.node_type_embedding is None:
        args.node_type_embedding = bool(args.use_tracks or args.use_tracklets)
        if ddp_is_main():
            print(
                f"[i] loaded feature stats from {args.feature_stats_json} "
                f"(kind={args.feature_norm_kind}, normalize_node_features={args.normalize_node_features}, "
                f"normalize_edge_features={args.normalize_edge_features}, clip={args.feature_norm_clip})",
                flush=True,
            )

    split     = np.load(args.split_file, allow_pickle=True)
    train_idx = split["train_idx"].astype(np.int64)
    val_idx   = split["val_idx"].astype(np.int64)
    _check_split_paths_compatible(split, paths, strict=args.strict_split_check)

    if "task" in split.files:
        task = str(split["task"].tolist())
        if task not in ("graph_classification", "displaced_vertex_classification"):
            raise RuntimeError(f"Split file task={task!r} is not a classifier split.")

    split_has_index = "event_refs" in split.files and "labels" in split.files
    with timed_section("dataset_index", device=device, enabled=args.time) as tt:
        if split_has_index:
            ds = make_event_dataset(
                paths,
                event_refs=split["event_refs"],
                labels=split["labels"],
                dataset_names=split["dataset_names"] if "dataset_names" in split.files else None,
                root_files=split["root_files"] if "root_files" in split.files else None,
                max_open_h5_files=args.max_open_h5_files,
                use_tracks=args.use_tracks,
                use_segment_edges=args.use_segment_edges,
                use_tracklets=args.use_tracklets,
            )
        else:
            ds = make_event_dataset(paths, max_open_h5_files=args.max_open_h5_files, use_tracks=args.use_tracks,
                                use_segment_edges=args.use_segment_edges, use_tracklets=args.use_tracklets)
    if ddp_is_main() and args.time:
        source = "split_file_event_refs" if split_has_index else "h5_scan"
        print(f"[time] dataset indexing: {tt['seconds']:.3f}s ({source})", flush=True)

    n = len(ds)
    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError(f"Split file has empty train/val: train={train_idx.size} val={val_idx.size}")
    if train_idx.min() < 0 or train_idx.max() >= n or val_idx.min() < 0 or val_idx.max() >= n:
        raise RuntimeError("Split indices out of range for current dataset.")

    if args.max_train_events > 0:
        train_idx = train_idx[:args.max_train_events]

    n_neg, n_pos = _count_labels(ds.labels_np, train_idx)
    v_neg, v_pos = _count_labels(ds.labels_np, val_idx)
    pos_weight_value = _parse_pos_weight(args.pos_weight, n_neg=n_neg, n_pos=n_pos)

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())

    packed_data = isinstance(ds, PackedEventDataset)
    mix_files = int(args.sampler_mix_files)
    if mix_files < 0:  # auto: H5 -> as many files as each worker keeps open, packed (cheap random access) -> all
        mix_files = 0 if packed_data else int(args.max_open_h5_files)
    train_sampler = FileLocalDistributedSampler(
        ds, train_idx,
        num_replicas=ddp_world_size(), rank=ddp_rank(),
        shuffle=True, seed=args.seed, drop_last=ddp_is_initialized(), mix_files=mix_files,
    )
    val_sampler = FileLocalDistributedSampler(
        ds, val_idx,
        num_replicas=ddp_world_size(), rank=ddp_rank(),
        shuffle=False, seed=args.seed, drop_last=False,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.pin_memory and torch.cuda.is_available():
        loader_kwargs["pin_memory_device"] = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    if args.num_workers > 0:
        loader_kwargs.update(
            multiprocessing_context=ctx,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
    train_loader = DataLoader(train_ds, shuffle=False, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **loader_kwargs)

    if ddp_is_initialized() and len(train_loader) == 0:
        raise RuntimeError(f"DDP training has 0 batches per rank (train_ds={len(train_ds)} world_size={ddp_world_size()}).")

    if ddp_is_main():
        print(
            f"[i] train={len(train_ds)} (pos={n_pos}, neg={n_neg}) "
            f"val={len(val_ds)} (pos={v_pos}, neg={v_neg}) total={len(ds)}",
            flush=True,
        )
        print(
            f"[i] loss_type={args.loss_type} pos_weight={pos_weight_value if pos_weight_value is not None else 'none'} "
            f"threshold={args.threshold} target_fpr={args.target_fpr}",
            flush=True,
        )
        print(
            f"[i] batch_size_per_rank={args.batch_size} effective_global_batch_size="
            f"{args.batch_size * ddp_world_size()} max_open_h5_files={args.max_open_h5_files}",
            flush=True,
        )
        mix_txt = ("single file per batch (legacy)" if mix_files == 1 else
                   "all files (global shuffle)" if mix_files == 0 else f"events of {mix_files} files shuffled together")
        print(f"[i] data={'packed parts' if packed_data else 'H5'} sampler: {mix_txt}", flush=True)

    sample = next(iter(train_loader))
    xdim = sample["x"].shape[1]
    edim = sample["edge_attr"].shape[1] if sample["edge_attr"].ndim == 2 else 5
    track_xdim = int(sample["track_x"].shape[1]) if args.use_tracks else 0
    seg_edim = int(sample["seg_edge_attr"].shape[1]) if args.use_segment_edges else 0
    tracklet_xdim = int(sample["tracklet_x"].shape[1]) if args.use_tracklets else 0
    tracklet_edim = int(sample["tracklet_edge_attr"].shape[1]) if args.use_tracklets else 0
    if ddp_is_main():
        print(f"[i] xdim={xdim} edim={edim} track_xdim={track_xdim} seg_edim={seg_edim} "
              f"tracklet_xdim={tracklet_xdim} tracklet_edim={tracklet_edim} "
              f"node_type_embedding={args.node_type_embedding} muon slot 5={muon_node_feature_names[5]}", flush=True)

    model = DisplacedVertexGNN(
        xdim=xdim, edim=edim, hdim=args.hidden_dim, n_layers=args.layers,
        dropout=args.dropout, layer_type=args.layer_type, gat_heads=args.gat_heads,
        gat_edge_attn=args.gat_edge_attn, gatv2_edge_attn=args.gatv2_edge_attn,
        sage_aggr=args.sage_aggr, edgeconv_aggr=args.edgeconv_aggr, pool=args.pool,
        use_fourier=args.fourier, fourier_base=args.fourier_base,
        fourier_min_exp=args.fourier_min_exp, fourier_max_exp=args.fourier_max_exp,
        normalize_node_features=args.normalize_node_features,
        normalize_edge_features=args.normalize_edge_features,
        feature_stats=feature_stats,
        feature_norm_clip=args.feature_norm_clip,
        track_xdim=track_xdim,
        node_type_embedding=args.node_type_embedding,
        seg_edim=seg_edim,
        tracklet_xdim=tracklet_xdim,
        tracklet_edim=tracklet_edim,
        block=args.block,
        ffn_mult=args.ffn_mult,
        relation_mp=args.relation_mp,
        relation_aggr=args.relation_aggr,
        calo_context=args.calo_context,
        edge_update=args.edge_update,
        readout=args.readout,
        readout_heads=args.readout_heads,
    ).to(device)
    if ddp_is_main():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[i] architecture: block={args.block} ffn_mult={args.ffn_mult} relation_mp={args.relation_mp} "
              f"relation_aggr={args.relation_aggr} calo_context={args.calo_context} edge_update={args.edge_update} "
              f"readout={args.readout} readout_heads={args.readout_heads} grad_clip={args.grad_clip} "
              f"trainable_params={n_params}", flush=True)
    track_asinh_scales = (
        model.tr_asinh_scale.detach().cpu().numpy().reshape(-1).tolist() if track_xdim > 0 else None
    )

    if ddp_is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False, find_unused_parameters=False,
        )

    raw_model = model.module if hasattr(model, "module") else model
    if args.no_decay_norm_bias:
        decay, no_decay = [], []
        for n_name, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            if n_name.endswith(".bias") or ("norm" in n_name.lower()) or ("layernorm" in n_name.lower()):
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:
        param_groups = [{"params": raw_model.parameters(), "weight_decay": args.weight_decay}]

    try:
        opt = torch.optim.AdamW(param_groups, lr=args.lr, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(param_groups, lr=args.lr)

    pos_weight_tensor = None
    if pos_weight_value is not None:
        pos_weight_tensor = torch.tensor([float(pos_weight_value)], dtype=torch.float32, device=device)
    criterion = build_classification_loss(args, pos_weight_tensor=pos_weight_tensor)

    scheduler = build_scheduler(opt, args, steps_per_epoch=len(train_loader))
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    ema = EMA.create(model, decay=args.ema_decay) if args.ema else None

    start_epoch = 1
    best_monitor = None
    bad_epochs = 0
    best_ckpt_epoch = None
    if args.resume:
        start_epoch, best_monitor, bad_epochs, best_ckpt_epoch, did_resume = _try_resume_from_checkpoint(
            ckpt_path=save_path, model=model, opt=opt, scheduler=scheduler, scaler=scaler,
            ema=ema, device=device,
        )
        if did_resume and ddp_is_main():
            print(f"[resume] resumed from {save_path} at epoch {start_epoch}", flush=True)

    if start_epoch > int(args.epochs):
        if ddp_is_main():
            print(
                f"[resume] Checkpoint {save_path} already reached epoch {start_epoch - 1}; "
                f"requested epochs={args.epochs}. Nothing to train.",
                flush=True,
            )
        return

    wandb_run = None
    if args.wandb and wandb is None and ddp_is_main():
        print("[wandb] wandb is not installed; disabling logging", flush=True)
    if args.wandb and wandb is not None and ddp_is_main():
        if args.wandb_key:
            try:
                wandb.login(key=args.wandb_key)
            except Exception as e:
                print(f"[wandb] login failed: {e}", flush=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir=args.wandb_dir,
            mode=args.wandb_mode,
            config={
                "task": task_name,
                "data_glob": args.data_glob,
                "split_file": args.split_file,
                "epochs": args.epochs,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "dropout": args.dropout,
                "layer_type": args.layer_type,
                "gat_edge_attn": args.gat_edge_attn,
                "gatv2_edge_attn": args.gatv2_edge_attn,
                "pool": args.pool,
                "fourier": args.fourier,
                "weight_decay": args.weight_decay,
                "edge_dropout": args.edge_dropout,
                "feat_noise_std": args.feat_noise_std,
                "pos_weight": pos_weight_value,
                "loss_type": args.loss_type,
                "label_smoothing": args.label_smoothing,
                "focal_gamma": args.focal_gamma,
                "focal_alpha": args.focal_alpha,
                "asym_gamma_pos": args.asym_gamma_pos,
                "asym_gamma_neg": args.asym_gamma_neg,
                "threshold": args.threshold,
                "target_fpr": args.target_fpr,
                "train_pos": n_pos,
                "train_neg": n_neg,
                "val_pos": v_pos,
                "val_neg": v_neg,
                "normalize_node_features": args.normalize_node_features,
                "normalize_edge_features": args.normalize_edge_features,
                "feature_norm_kind": args.feature_norm_kind,
                "feature_norm_clip": args.feature_norm_clip,
                "feature_norm_in_model": True,
                "use_tracks": bool(args.use_tracks),
                "use_segment_edges": bool(args.use_segment_edges),
                "use_tracklets": bool(args.use_tracklets),
                "muon_node_feature_names": muon_node_feature_names,
                "node_type_embedding": bool(args.node_type_embedding),
                "block": args.block,
                "ffn_mult": args.ffn_mult,
                "grad_clip": args.grad_clip,
                "relation_mp": bool(args.relation_mp),
                "relation_aggr": args.relation_aggr,
                "calo_context": args.calo_context,
                "edge_update": args.edge_update,
                "readout": args.readout,
                "readout_heads": args.readout_heads,
                "sampler_mix_files": mix_files,
                "packed_data": packed_data,
            },
        )

    half_pat = max(1, args.early_stop_patience // 2)
    reloaded_this_plateau = False

    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp) if device.type == "cuda" else nullcontext()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        train_n = torch.tensor(0.0, device=device, dtype=torch.float64)
        train_counts = torch.zeros(4, device=device, dtype=torch.float64)
        grad_clip = float(args.grad_clip)
        # gradient-norm stats (on the device, read once per epoch): max, sum, steps, clipped steps
        grad_stats = torch.zeros(4, device=device, dtype=torch.float64) if grad_clip > 0 else None

        for batch in train_loader:
            model_args, model_kwargs = batch_model_inputs(batch, device)
            y = batch["y"].to(device, non_blocking=True).float().view(-1)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(
                    *model_args,
                    **model_kwargs,
                    edge_dropout_p=float(args.edge_dropout),
                    feature_noise_std=float(args.feat_noise_std),
                ).view_as(y)
                loss = criterion(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_stats is not None:
                    scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).detach().double()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_stats is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).detach().double()
                opt.step()
            if grad_stats is not None:
                grad_stats[0] = torch.maximum(grad_stats[0], grad_norm)
                grad_stats[1] += grad_norm
                grad_stats[2] += 1.0
                grad_stats[3] += (grad_norm > grad_clip).double()

            if scheduler is not None and args.lr_schedule == "cosine":
                scheduler.step()
            if ema is not None:
                ema.update(model)

            train_loss_sum += loss.detach().to(torch.float64) * y.numel()
            train_n += float(y.numel())
            train_counts += _binary_counts_from_logits(logits, y, threshold=args.threshold)

        if ddp_is_initialized():
            dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_n, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_counts, op=dist.ReduceOp.SUM)

        train_metrics = _metrics_from_counts(train_counts)
        train_metrics["loss"] = (train_loss_sum / train_n.clamp(min=1.0)).item()
        grad_txt = ""
        if grad_stats is not None:
            # the gradients are already averaged over the DDP ranks: the same norms on every rank
            g_max, g_sum, g_steps, g_clipped = grad_stats.tolist()
            train_metrics["grad_norm_max"] = g_max
            train_metrics["grad_norm_mean"] = g_sum / max(g_steps, 1.0)
            train_metrics["clipped_frac"] = g_clipped / max(g_steps, 1.0)
            grad_txt = (f"grad_norm max={g_max:.3g} mean={train_metrics['grad_norm_mean']:.3g} "
                        f"clipped={train_metrics['clipped_frac']:.3f} | ")

        eval_ctx = ema.apply_to(model) if ema is not None else nullcontext()
        with eval_ctx:
            val_metrics = _evaluate_classifier(
                model, val_loader, criterion, device,
                threshold=args.threshold, target_fpr=args.target_fpr,
                use_amp=use_amp, amp_dtype=amp_dtype,
                edge_dropout_p=0.0,
            )

        if scheduler is not None and args.lr_schedule == "plateau":
            scheduler.step(val_metrics["loss"])

        monitor_key = args.early_stop_monitor.replace("val_", "")
        monitor_val = val_metrics[monitor_key]
        improved = np.isfinite(monitor_val) and _monitor_is_better(
            args.early_stop_monitor, float(monitor_val), best_monitor, float(args.early_stop_min_delta)
        )
        current_lr = opt.param_groups[0]["lr"]

        if ddp_is_main():
            print(
                f"[epoch {epoch:03d}] "
                f"train loss={train_metrics['loss']:.5f} acc={train_metrics['acc']:.5f} "
                f"precision={train_metrics['precision']:.5f} recall={train_metrics['recall']:.5f} "
                f"f1={train_metrics['f1']:.5f} bacc={train_metrics['balanced_acc']:.5f} | "
                f"val loss={val_metrics['loss']:.5f} acc={val_metrics['acc']:.5f} "
                f"precision={val_metrics['precision']:.5f} recall={val_metrics['recall']:.5f} "
                f"f1={val_metrics['f1']:.5f} bacc={val_metrics['balanced_acc']:.5f} auc={_fmt_metric(val_metrics['auc'])} "
                f"tpr_at_target_fpr={_fmt_metric(val_metrics['tpr_at_target_fpr'])} "
                f"fpr_at_target_fpr={_fmt_metric(val_metrics['fpr_at_target_fpr'])} "
                f"thr_at_target_fpr={_fmt_metric(val_metrics['threshold_at_target_fpr'])} | "
                f"{grad_txt}lr={current_lr:.3e} | {args.early_stop_monitor}={_fmt_metric(monitor_val)} "
                f"{'(best)' if improved else ''}",
                flush=True,
            )

        if wandb_run is not None and ddp_is_main():
            log_payload = {"epoch": epoch, "lr": current_lr, args.early_stop_monitor: monitor_val}
            for prefix_name, metrics in (("train", train_metrics), ("val", val_metrics)):
                for k, v in metrics.items():
                    log_payload[f"{prefix_name}/{k}"] = v
            wandb.log(log_payload, step=epoch)

        if ddp_is_main() and improved:
            best_monitor = float(monitor_val)
            best_ckpt_epoch = epoch
            save_ctx = ema.apply_to(model) if ema is not None else nullcontext()
            _ensure_parent_dir(str(save_path))
            with save_ctx:
                raw_model = model.module if hasattr(model, "module") else model
                ckpt_obj = {
                    "model_state": raw_model.state_dict(),
                    "task": task_name,
                    "model_type": "binary_graph_classifier",
                    "xdim": xdim,
                    "edim": edim,
                    "use_tracks": bool(args.use_tracks),
                    "track_xdim": int(track_xdim),
                    "track_feature_names": list(TRACK_FEATURE_NAMES) if track_xdim > 0 else None,
                    "track_asinh_scales": track_asinh_scales,
                    "node_type_embedding": bool(args.node_type_embedding),
                    "use_segment_edges": bool(args.use_segment_edges),
                    "seg_edim": int(seg_edim),
                    "segment_edge_feature_names": list(SEGMENT_EDGE_FEATURE_NAMES) if seg_edim > 0 else None,
                    "use_tracklets": bool(args.use_tracklets),
                    "tracklet_xdim": int(tracklet_xdim),
                    "tracklet_edim": int(tracklet_edim),
                    "tracklet_feature_names": list(TRACKLET_FEATURE_NAMES) if tracklet_xdim > 0 else None,
                    "tracklet_edge_feature_names": list(TRACKLET_EDGE_FEATURE_NAMES) if tracklet_xdim > 0 else None,
                    "muon_node_feature_names": muon_node_feature_names,
                    "converter_config": converter_config,
                    "block": args.block,
                    "ffn_mult": int(args.ffn_mult),
                    "grad_clip": float(args.grad_clip),
                    "relation_mp": bool(args.relation_mp),
                    "relation_aggr": args.relation_aggr,
                    "calo_context": args.calo_context,
                    "edge_update": args.edge_update,
                    "readout": args.readout,
                    "readout_heads": int(args.readout_heads),
                    "sampler_mix_files": int(mix_files),
                    "packed_data": bool(packed_data),
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "layer_type": args.layer_type,
                    "gat_heads": args.gat_heads,
                    "gat_edge_attn": args.gat_edge_attn,
                    "gatv2_edge_attn": args.gatv2_edge_attn,
                    "sage_aggr": args.sage_aggr,
                    "edgeconv_aggr": args.edgeconv_aggr,
                    "pool": args.pool,
                    "fourier": args.fourier,
                    "fourier_base": args.fourier_base,
                    "fourier_min_exp": args.fourier_min_exp,
                    "fourier_max_exp": args.fourier_max_exp,
                    "normalize_node_features": args.normalize_node_features,
                    "normalize_edge_features": args.normalize_edge_features,
                    "feature_stats_json": args.feature_stats_json,
                    "feature_stats": feature_stats,
                    "feature_norm_kind": args.feature_norm_kind,
                    "feature_norm_clip": args.feature_norm_clip,
                    "feature_norm_in_model": True,
                    "pos_weight": pos_weight_value,
                    "loss_type": args.loss_type,
                    "label_smoothing": args.label_smoothing,
                    "focal_gamma": args.focal_gamma,
                    "focal_alpha": args.focal_alpha,
                    "asym_gamma_pos": args.asym_gamma_pos,
                    "asym_gamma_neg": args.asym_gamma_neg,
                    "threshold": args.threshold,
                    "target_fpr": args.target_fpr,
                    "best_monitor": best_monitor,
                    "early_stop_monitor": args.early_stop_monitor,
                    "best_ckpt_epoch": best_ckpt_epoch,
                    "val_loss": float(val_metrics["loss"]),
                    "val_acc": float(val_metrics["acc"]),
                    "val_precision": float(val_metrics["precision"]),
                    "val_recall": float(val_metrics["recall"]),
                    "val_f1": float(val_metrics["f1"]),
                    "val_balanced_acc": float(val_metrics["balanced_acc"]),
                    "val_auc": float(val_metrics["auc"]),
                    "val_tpr_at_target_fpr": float(val_metrics["tpr_at_target_fpr"]),
                    "val_fpr_at_target_fpr": float(val_metrics["fpr_at_target_fpr"]),
                    "val_threshold_at_target_fpr": float(val_metrics["threshold_at_target_fpr"]),
                    "run_id": run_id,
                    "weight_decay": args.weight_decay,
                    "edge_dropout": args.edge_dropout,
                    "feat_noise_std": args.feat_noise_std,
                    "ema": bool(ema is not None),
                    "ema_decay": args.ema_decay,
                    "lr_schedule": args.lr_schedule,
                    "warmup_epochs": args.warmup_epochs,
                    "min_lr_ratio": args.min_lr_ratio,
                    "code_version": args.code_version,
                    "epoch": int(epoch),
                    "bad_epochs": int(bad_epochs),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
                    "scaler_state": (scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None),
                    "ema_shadow": (ema.shadow if ema is not None else None),
                }
                _torch_save_atomic_with_retries(ckpt_obj, save_path)
            print(
                f"  [*] saved best checkpoint to {save_path} "
                f"({args.early_stop_monitor}={_fmt_metric(best_monitor)}, epoch={epoch})",
                flush=True,
            )
            reloaded_this_plateau = False

        stop_now = False
        reload_now = False
        if args.early_stop and ddp_is_main():
            if improved:
                bad_epochs = 0
                reloaded_this_plateau = False
            else:
                bad_epochs += 1

            if (
                args.reload_best_half_patience and (half_pat > 0) and
                (bad_epochs >= half_pat) and (not reloaded_this_plateau)
            ):
                if os.path.exists(save_path):
                    reload_now = True
                    reloaded_this_plateau = True
                    print(
                        f"[reload-best] No improvement for {bad_epochs} epochs. "
                        f"Reloading best checkpoint (epoch={best_ckpt_epoch}) and resetting lr -> {args.lr:.3e}.",
                        flush=True,
                    )

            if bad_epochs >= args.early_stop_patience:
                stop_now = True
                print(f"[early-stop] Triggered at epoch {epoch}.", flush=True)

        reload_t = torch.tensor([1 if reload_now else 0], device=device, dtype=torch.int32)
        if ddp_is_initialized():
            dist.broadcast(reload_t, src=0)
        if bool(reload_t.item()):
            ddp_barrier()
            load_best_checkpoint_into_model(model, save_path, device)
            reset_optimizer_lr(opt, args.lr)
            scheduler = build_scheduler(opt, args, steps_per_epoch=len(train_loader))
            if ema is not None:
                ema = EMA.create(model, decay=args.ema_decay)
            if scaler.is_enabled():
                scaler = torch.amp.GradScaler("cuda", enabled=True)
            if args.early_stop:
                bad_epochs = 0
            ddp_barrier()

        stop_t = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int32)
        if ddp_is_initialized():
            dist.broadcast(stop_t, src=0)
        if bool(stop_t.item()):
            break

    if wandb_run is not None and ddp_is_main():
        wandb.finish(quiet=True)
