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
from collections import Counter
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

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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

    saved_base = [os.path.basename(p) for p in saved_norm]
    cur_base = [os.path.basename(p) for p in cur_norm]

    if saved_base == cur_base:
        if ddp_is_main():
            print(
                "[warn] Split-file H5 paths differ from current paths, but ordered basenames match. "
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

        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)


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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, heads, out_channels))
        self.attn_r = nn.Parameter(torch.empty(1, heads, out_channels))
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

        if not concat:
            self.out_proj = nn.Linear(heads * out_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        x = self.linear(x).view(num_nodes, self.heads, self.out_channels)
        out_dtype = x.dtype
        x_f = x.float()

        if self.add_self_loops:
            self_loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        src = edge_index[0]
        dst = edge_index[1]

        alpha_l = (x_f[src] * self.attn_l).sum(dim=-1)
        alpha_r = (x_f[dst] * self.attn_r).sum(dim=-1)
        alpha = F.leaky_relu(alpha_l + alpha_r, negative_slope=0.2)

        alpha = torch.exp(alpha - alpha.max(dim=0, keepdim=True)[0])
        alpha_sum = torch.zeros((num_nodes, self.heads), device=x.device, dtype=torch.float32)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / alpha_sum[dst].clamp(min=1e-6)
        alpha = self.dropout(alpha)

        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=torch.float32)
        for h in range(self.heads):
            out[:, h].scatter_add_(
                0,
                dst.unsqueeze(-1).expand_as(x_f[src, h]),
                alpha[:, h].unsqueeze(-1) * x_f[src, h],
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
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.project = nn.Linear(in_channels, out_channels * heads) if in_channels != out_channels * heads else None
        self.gat = CustomGAT(
            in_channels=(out_channels * heads if self.project else in_channels),
            out_channels=out_channels,
            heads=heads, dropout=dropout, add_self_loops=True, concat=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.project is not None:
            x = self.project(x)
        identity = x
        x = F.relu(self.gat(x, edge_index))
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
        return out.reshape(x.size(0), -1)


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


class DisplacedVertexGNN(nn.Module):
    def __init__(
        self,
        xdim,
        edim,
        hdim=128,
        n_layers=4,
        dropout=0.1,
        layer_type: str = "mpnn",
        gat_heads: int = 4,
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
            if gat_heads <= 0:
                raise ValueError("--gat-heads must be >= 1")
            if hdim % gat_heads != 0:
                raise ValueError(f"hidden_dim={hdim} must be divisible by gat_heads={gat_heads}")
            per_head = hdim // gat_heads
            self.layers = nn.ModuleList([
                GATResidualBlock(in_channels=hdim, out_channels=per_head, heads=gat_heads, dropout=dropout)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
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
    ):
        self.h5_paths = list(h5_paths)

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
        self._files = None
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

    def get_label(self, idx: int) -> float:
        return float(self.labels_np[int(idx)])

    def __getitem__(self, idx):
        self._ensure_open()
        assert self._files is not None

        fi, k = self.index[idx]
        f = self._files[fi]
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

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": y,
            "labels": y,
            "n_muon_nodes": n_muon_nodes,
        }


def _decode_attr_to_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    return str(v)


class DisplacedVertexGNN(nn.Module):
    """Single-logit graph classifier using raw inputs and optional model-side normalization."""

    def __init__(
        self,
        xdim,
        edim,
        hdim=128,
        n_layers=4,
        dropout=0.1,
        layer_type: str = "mpnn",
        gat_heads: int = 4,
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
    ):
        super().__init__()
        self.fourier = None
        self.pool = pool
        self.normalize_node_features = bool(normalize_node_features)
        self.normalize_edge_features = bool(normalize_edge_features)
        self.feature_norm_clip = float(feature_norm_clip)

        self._register_feature_normalization_buffers(
            xdim=xdim,
            edim=edim,
            feature_stats=feature_stats,
        )

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
            if gat_heads <= 0:
                raise ValueError("--gat-heads must be >= 1")
            if hdim % gat_heads != 0:
                raise ValueError(f"hidden_dim={hdim} must be divisible by gat_heads={gat_heads}")
            per_head = hdim // gat_heads
            self.layers = nn.ModuleList([
                GATResidualBlock(in_channels=hdim, out_channels=per_head, heads=gat_heads, dropout=dropout)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
        else:
            raise ValueError(f"Unknown layer_type={layer_type}")

        graph_dim = hdim * 2 if pool == "meanmax" else hdim
        self.head = MLP(in_dim=graph_dim, out_dim=1, hidden_dim=hdim, n_layers=3, dropout=dropout)

    def _register_feature_normalization_buffers(
        self,
        *,
        xdim: int,
        edim: int,
        feature_stats: Optional[Dict[str, np.ndarray]],
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

    def _normalize_features(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        n_muon_nodes: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_node_features:
            if n_muon_nodes is None:
                raise RuntimeError("Model-side node normalization requires n_muon_nodes.")
            if not torch.is_tensor(n_muon_nodes):
                n_muon_nodes = torch.tensor([n_muon_nodes], device=x.device, dtype=torch.long)
            n_mu = n_muon_nodes.to(device=x.device, dtype=torch.long).reshape(-1)[0]
 
            node_ids = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
            mu_mask = (node_ids < n_mu).unsqueeze(-1)

            x_mu = self._apply_feature_norm_torch(x, self.mu_center, self.mu_scale)
            x_ca = self._apply_feature_norm_torch(x, self.ca_center, self.ca_scale)
            x = torch.where(mu_mask, x_mu, x_ca)

        if self.normalize_edge_features:
            edge_attr = self._apply_feature_norm_torch(edge_attr, self.edge_center, self.edge_scale)

        return x, edge_attr

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        *,
        n_muon_nodes: Optional[Any] = None,
        edge_dropout_p: float = 0.0,
        feature_noise_std: float = 0.0,
    ):
        x, edge_attr = self._normalize_features(x, edge_attr, n_muon_nodes)

        if self.training and feature_noise_std > 0:
            x = x + torch.randn_like(x) * float(feature_noise_std)

        if self.fourier is not None:
            x = self.fourier(x)

        h = self.node_enc(x)
        for layer in self.layers:
            if self._uses_edge_attr:
                h = layer(h, edge_index, edge_attr, edge_dropout_p=edge_dropout_p)
            else:
                h = layer(h, edge_index)

        g = global_pool(h, mode=self.pool)
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
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
    total_n = torch.tensor(0.0, device=device, dtype=torch.float64)
    counts = torch.zeros(4, device=device, dtype=torch.float64)
    logits_parts = []
    labels_parts = []

    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp) if device.type == "cuda" else nullcontext()

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        edge_index = batch["edge_index"].to(device, non_blocking=True).long()
        edge_attr = batch["edge_attr"].to(device, non_blocking=True).float()
        n_muon_nodes = batch["n_muon_nodes"].to(device, non_blocking=True).long()
        y = batch["y"].to(device, non_blocking=True).float().view(-1)
        with autocast_ctx:
            logits = model(
                x, edge_index, edge_attr,
                n_muon_nodes=n_muon_nodes,
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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--layer-type", default="mpnn",
                    choices=["mpnn", "edge_residual", "sage_residual", "gat_residual"])
    ap.add_argument("--gat-heads", type=int, default=4)
    ap.add_argument("--sage-aggr", default="mean", choices=["mean", "sum", "max"])
    ap.add_argument("--edgeconv-aggr", default="mean", choices=["mean", "sum", "max"])
    ap.add_argument("--pool", default="meanmax", choices=["mean", "max", "sum", "meanmax"])

    ap.add_argument("--feature-stats-json", default=None,
                    help="JSON file with precomputed node/edge feature normalization stats.")
    ap.add_argument("--feature-norm-kind", default="standard", choices=["standard", "robust"],
                    help="Which stats block to use from --feature-stats-json.")
    ap.add_argument("--normalize-node-features", action="store_true", default=False,
                    help="Apply precomputed normalization to node features x.")
    ap.add_argument("--no-normalize-node-features", dest="normalize_node_features", action="store_false")
    ap.add_argument("--normalize-edge-features", action="store_true", default=False,
                    help="Apply precomputed normalization to edge_attr.")
    ap.add_argument("--no-normalize-edge-features", dest="normalize_edge_features", action="store_false")
    ap.add_argument("--feature-norm-clip", type=float, default=-1.0,
                    help="Optional absolute clip after feature normalization. <=0 disables clipping.")

    ap.add_argument("--pos-weight", default="auto",
                    help="Positive-class weight: 'auto', 'none', or a positive float. Auto uses n_negative/n_positive in the train split.")
    ap.add_argument("--loss-type", default="bce",
                    choices=["bce", "bce_smooth", "focal", "asymmetric_focal"],
                    help="Binary classification loss function.")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="Binary-label smoothing used by bce_smooth/focal/asymmetric_focal. 0 disables it.")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="Gamma for focal loss.")
    ap.add_argument("--focal-alpha", default="none",
                    help="Optional positive-class alpha for focal/asymmetric_focal: 'none' or float in (0,1).")
    ap.add_argument("--asym-gamma-pos", type=float, default=0.0,
                    help="Positive-class gamma for asymmetric focal loss.")
    ap.add_argument("--asym-gamma-neg", type=float, default=4.0,
                    help="Negative-class gamma for asymmetric focal loss.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for class metrics.")
    ap.add_argument("--target-fpr", type=float, default=0.01,
                    help="FPR working point for tpr_at_target_fpr metric. Default 0.01 = 1% FPR.")
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
    ap.add_argument("--early-stop-patience", type=int, default=25)
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

    ap.add_argument("--fourier", dest="fourier", action="store_true", default=True)
    ap.add_argument("--no-fourier", dest="fourier", action="store_false")
    ap.add_argument("--fourier-base", type=float, default=3.0)
    ap.add_argument("--fourier-min-exp", type=int, default=-6)
    ap.add_argument("--fourier-max-exp", type=int, default=6)

    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--no-decay-norm-bias", action="store_true", default=True)
    ap.add_argument("--decay-norm-bias", dest="no_decay_norm_bias", action="store_false")

    ap.add_argument("--edge-dropout", type=float, default=0.0)
    ap.add_argument("--feat-noise-std", type=float, default=0.0)

    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    return ap


def _fmt_metric(v: float) -> str:
    return "nan" if v is None or not np.isfinite(v) else f"{float(v):.5f}"


def run_training(args, *, task_name: str = "displaced_vertex_classification"):
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
            raise SystemExit("Feature normalization was requested but --feature-stats-json was not provided.")
        feature_stats = load_feature_stats_json(args.feature_stats_json, norm_kind=args.feature_norm_kind)
        if args.normalize_edge_features and not feature_stats["meta"]["has_edge_stats"]:
            raise SystemExit("Edge normalization requested, but the stats JSON does not contain edge stats.")
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
            ds = H5EventDataset(
                paths,
                event_refs=split["event_refs"],
                labels=split["labels"],
                dataset_names=split["dataset_names"] if "dataset_names" in split.files else None,
                root_files=split["root_files"] if "root_files" in split.files else None,
            )
        else:
            ds = H5EventDataset(paths)
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

    train_sampler = DistributedSampler(
        train_ds, num_replicas=ddp_world_size(), rank=ddp_rank(), shuffle=True, drop_last=True,
    ) if ddp_is_initialized() else None
    val_sampler = DistributedSampler(
        val_ds, num_replicas=ddp_world_size(), rank=ddp_rank(), shuffle=False, drop_last=False,
    ) if ddp_is_initialized() else None

    pin_device = f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}" if torch.cuda.is_available() else ""
    loader_kwargs = dict(
        batch_size=1,
        collate_fn=collate_one,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        pin_memory_device=pin_device,
    )
    if args.num_workers > 0:
        loader_kwargs.update(
            multiprocessing_context=ctx,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
    train_loader = DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler, **loader_kwargs)
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

    sample = next(iter(train_loader))
    xdim = sample["x"].shape[1]
    edim = sample["edge_attr"].shape[1] if sample["edge_attr"].ndim == 2 else 5
    if ddp_is_main():
        print(f"[i] xdim={xdim} edim={edim}", flush=True)

    model = DisplacedVertexGNN(
        xdim=xdim, edim=edim, hdim=args.hidden_dim, n_layers=args.layers,
        dropout=args.dropout, layer_type=args.layer_type, gat_heads=args.gat_heads,
        sage_aggr=args.sage_aggr, edgeconv_aggr=args.edgeconv_aggr, pool=args.pool,
        use_fourier=args.fourier, fourier_base=args.fourier_base,
        fourier_min_exp=args.fourier_min_exp, fourier_max_exp=args.fourier_max_exp,
        normalize_node_features=args.normalize_node_features,
        normalize_edge_features=args.normalize_edge_features,
        feature_stats=feature_stats,
        feature_norm_clip=args.feature_norm_clip,
    ).to(device)

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

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True).float()
            edge_index = batch["edge_index"].to(device, non_blocking=True).long()
            edge_attr = batch["edge_attr"].to(device, non_blocking=True).float()
            n_muon_nodes = batch["n_muon_nodes"].to(device, non_blocking=True).long()
            y = batch["y"].to(device, non_blocking=True).float().view(-1)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(
                    x, edge_index, edge_attr,
                    n_muon_nodes=n_muon_nodes,
                    edge_dropout_p=float(args.edge_dropout),
                    feature_noise_std=float(args.feat_noise_std),
                ).view_as(y)
                loss = criterion(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

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
                f"lr={current_lr:.3e} | {args.early_stop_monitor}={_fmt_metric(monitor_val)} "
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
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "layer_type": args.layer_type,
                    "gat_heads": args.gat_heads,
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
