#!/usr/bin/env python3
"""
dv_training_utils.py

Shared utilities for DisplacedVertex training scripts:
  train_DisplacedVertex_position.py
  train_DisplacedVertex_cylindrical_position.py
  train_DisplacedVertex_polar_position.py

Contains:
  - Utility and DDP helpers
  - H5EventDataset (unified)
  - EMA
  - All GNN model building blocks and DisplacedVertexGNN
  - Training utility functions
  - Shared argparse setup
  - run_training(): the full training loop parameterised by coordinate system
"""

import argparse
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

import wandb


# ============================================================
# Utility helpers
# ============================================================

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
    """Normalize a path for robust comparisons across different launch directories."""
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return os.path.abspath(os.path.expanduser(str(p)))


def _check_split_paths_compatible(split_npz, current_paths, *, strict: bool = False):
    """
    Validate that the split file was produced from the same H5 parts.
    Compatibility policy:
      1) exact normalized absolute paths match -> OK
      2) same ordered basenames match          -> OK with warning
      3) same basename multiset matches        -> OK with warning
      4) otherwise                             -> raise (or warn if not strict)
    """
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


# ============================================================
# DDP helpers
# ============================================================

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


# ============================================================
# Dataset
# ============================================================

class H5EventDataset(Dataset):
    """
    Loads events from multiple H5 part files.

    Expected structure:
      /events/<id>/x, edge_index, edge_attr, y_vertex
    """

    def __init__(self, h5_paths):
        self.h5_paths = list(h5_paths)
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
            raise RuntimeError(
                f"Missing 'y_vertex' in {self.h5_paths[fi]} /events/{k}"
            )

        y_vertex = torch.from_numpy(g["y_vertex"][...]).float()

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y_vertex": y_vertex,
        }


def collate_one(batch):
    assert len(batch) == 1
    return batch[0]


# ============================================================
# EMA
# ============================================================

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


# ============================================================
# Model building blocks
# ============================================================

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
    """Pool node embeddings of a single graph into one graph embedding. h: [N, H] -> [1, P]"""
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
    ):
        super().__init__()

        self.fourier = None
        self.pool = pool

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
        self.graph_head = MLP(in_dim=graph_dim, out_dim=3, hidden_dim=hdim, n_layers=3, dropout=dropout)

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
        out = self.graph_head(g).squeeze(0)
        return out


# ============================================================
# Training utilities
# ============================================================

@torch.no_grad()
def estimate_target_stats(train_ds, device, max_events: int = -1):
    """Estimate mean/std of y_vertex. Rank0 computes and broadcasts."""
    if ddp_is_main():
        loader = DataLoader(
            train_ds, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=False, collate_fn=collate_one,
        )
        ys = []
        for i, batch in enumerate(loader):
            if max_events > 0 and i >= max_events:
                break
            ys.append(batch["y_vertex"].float().view(1, 3))

        if len(ys) == 0:
            mean = torch.zeros(3, dtype=torch.float32, device=device)
            std = torch.ones(3, dtype=torch.float32, device=device)
        else:
            y = torch.cat(ys, dim=0).to(device)
            mean = y.mean(dim=0)
            std = y.std(dim=0, unbiased=False).clamp(min=1e-6)
    else:
        mean = torch.zeros(3, dtype=torch.float32, device=device)
        std = torch.ones(3, dtype=torch.float32, device=device)

    if ddp_is_initialized():
        dist.broadcast(mean, src=0)
        dist.broadcast(std, src=0)

    return mean, std


def build_regression_loss(name: str):
    name = name.lower()
    if name == "smoothl1":
        return nn.SmoothL1Loss(reduction="mean")
    if name == "mse":
        return nn.MSELoss(reduction="mean")
    if name == "l1":
        return nn.L1Loss(reduction="mean")
    raise ValueError(f"Unknown loss: {name}")


@torch.no_grad()
def regression_stats(pred: torch.Tensor, target: torch.Tensor, mape_eps: float = 1e-6) -> Tuple:
    """
    Compute regression statistics for a 3-component prediction.

    Returns:
        (mae, rmse, mape_pct, coord0_mae, coord1_mae, coord2_mae,
         coord0_mape_pct, coord1_mape_pct, coord2_mape_pct)
    """
    diff = pred - target
    abs_diff = diff.abs()
    sq_diff = diff.pow(2)
    denom = target.abs().clamp(min=float(mape_eps))
    ape = abs_diff / denom
    mape_pct = 100.0 * ape.mean()
    mae = abs_diff.mean()
    rmse = torch.sqrt(sq_diff.mean())
    return (
        mae,
        rmse,
        mape_pct,
        abs_diff[0],
        abs_diff[1],
        abs_diff[2],
        100.0 * ape[0],
        100.0 * ape[1],
        100.0 * ape[2],
    )


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
        ema.shadow = {k: v.clone() for k, v in ckpt["ema_shadow"].items()}

    last_epoch = int(ckpt.get("epoch", 0))
    best_monitor = ckpt.get("best_monitor", None)
    bad_epochs = int(ckpt.get("bad_epochs", 0))
    best_ckpt_epoch = ckpt.get("best_ckpt_epoch", None)
    start_epoch = max(1, last_epoch + 1)
    return start_epoch, best_monitor, bad_epochs, best_ckpt_epoch, True


# ============================================================
# Shared argparse setup
# ============================================================

def add_training_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all shared training arguments to an ArgumentParser."""
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--strict-split-check", action="store_true", default=False,
                    help="Require exact split-file path compatibility.")
    ap.add_argument("--epochs", type=int, default=200)
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

    ap.add_argument("--loss", default="smoothl1", choices=["smoothl1", "mse", "l1"])

    ap.add_argument("--normalize-target", action="store_true", default=True,
                    help="Train on normalized y_vertex and unnormalize for metrics.")
    ap.add_argument("--no-normalize-target", dest="normalize_target", action="store_false")
    ap.add_argument("--target-stats-max-events", type=int, default=-1)

    ap.add_argument("--max-train-events", type=int, default=-1)

    ap.add_argument("--save", default="displaced_vertex_gnn.pt")
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
    ap.add_argument("--early-stop-patience", type=int, default=30)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--early-stop-monitor", choices=["val_mae", "val_rmse", "val_loss"], default="val_mae")

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
    ap.add_argument("--mape-eps", type=float, default=1e-6,
                    help="Minimum |target| in MAPE denominator to avoid division by zero.")

    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--ema-decay", type=float, default=0.999)

    return ap


# ============================================================
# Full training loop
# ============================================================

def run_training(args, *, coordinate_system: str, target_labels: List[str]):
    """
    Full DDP training loop for DisplacedVertex graph regression.

    Args:
        args: parsed argparse namespace (from add_training_args)
        coordinate_system: string identifying the target convention
            (e.g. "cartesian", "cylindrical", "polar")
        target_labels: list of 3 strings naming the target coordinates
            (e.g. ["x","y","z"], ["rho","phi","z"], ["r","theta","phi"])
    """
    assert len(target_labels) == 3, "target_labels must have exactly 3 elements"

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

    with timed_section("dataset_index", device=device, enabled=args.time) as tt:
        ds = H5EventDataset(paths)
    if ddp_is_main() and args.time:
        print(f"[time] dataset indexing: {tt['seconds']:.3f}s", flush=True)

    split = np.load(args.split_file, allow_pickle=True)
    train_idx = split["train_idx"].astype(np.int64)
    val_idx = split["val_idx"].astype(np.int64)

    _check_split_paths_compatible(split, paths, strict=args.strict_split_check)

    if "coordinate_system" in split.files:
        coord = str(split["coordinate_system"].tolist())
        if coord.lower() != coordinate_system.lower():
            raise RuntimeError(
                f"Split file indicates coordinate_system={coord!r}, expected {coordinate_system!r}."
            )

    n = len(ds)
    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError(f"Split file has empty train/val: train={train_idx.size} val={val_idx.size}")
    if train_idx.min() < 0 or train_idx.max() >= n or val_idx.min() < 0 or val_idx.max() >= n:
        raise RuntimeError("Split indices out of range for current dataset.")

    if args.max_train_events > 0:
        train_idx = train_idx[:args.max_train_events]

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())

    train_sampler = DistributedSampler(
        train_ds, num_replicas=ddp_world_size(), rank=ddp_rank(), shuffle=True, drop_last=True,
    ) if ddp_is_initialized() else None

    val_sampler = DistributedSampler(
        val_ds, num_replicas=ddp_world_size(), rank=ddp_rank(), shuffle=False, drop_last=True,
    ) if ddp_is_initialized() else None

    pin_device = f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}" if torch.cuda.is_available() else ""

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=collate_one, num_workers=args.num_workers,
        pin_memory=args.pin_memory, multiprocessing_context=ctx,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory_device=pin_device,
    )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, sampler=val_sampler,
        collate_fn=collate_one, num_workers=args.num_workers, pin_memory=args.pin_memory,
        multiprocessing_context=ctx,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory_device=pin_device,
    )

    if ddp_is_initialized():
        if len(train_loader) == 0:
            raise RuntimeError(
                f"DDP training has 0 batches per rank "
                f"(train_ds={len(train_ds)} world_size={ddp_world_size()})."
            )
        if len(val_loader) == 0:
            raise RuntimeError(
                f"DDP validation has 0 batches per rank "
                f"(val_ds={len(val_ds)} world_size={ddp_world_size()})."
            )

    if ddp_is_main():
        print(f"[i] train={len(train_ds)} val={len(val_ds)} total={len(ds)}", flush=True)

    sample = next(iter(train_loader))
    xdim = sample["x"].shape[1]
    edim = sample["edge_attr"].shape[1]

    if ddp_is_main():
        print(f"[i] xdim={xdim} edim={edim}", flush=True)

    model = DisplacedVertexGNN(
        xdim=xdim, edim=edim, hdim=args.hidden_dim, n_layers=args.layers,
        dropout=args.dropout, layer_type=args.layer_type, gat_heads=args.gat_heads,
        sage_aggr=args.sage_aggr, edgeconv_aggr=args.edgeconv_aggr, pool=args.pool,
        use_fourier=args.fourier, fourier_base=args.fourier_base,
        fourier_min_exp=args.fourier_min_exp, fourier_max_exp=args.fourier_max_exp,
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

    loss_fn = build_regression_loss(args.loss)

    if args.normalize_target:
        target_mean, target_std = estimate_target_stats(
            train_ds, device=device, max_events=args.target_stats_max_events,
        )
    else:
        target_mean = torch.zeros(3, dtype=torch.float32, device=device)
        target_std = torch.ones(3, dtype=torch.float32, device=device)

    if ddp_is_main():
        labels_str = ", ".join(target_labels)
        print(f"[i] target_mean={target_mean.detach().cpu().numpy()}  # [{labels_str}]", flush=True)
        print(f"[i] target_std ={target_std.detach().cpu().numpy()}  # [{labels_str}]", flush=True)

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(opt, args, steps_per_epoch=steps_per_epoch)

    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    ema = EMA.create(model, decay=args.ema_decay) if args.ema else None

    wandb_run = None
    wandb_enabled = bool(args.wandb and ddp_is_main() and args.wandb_mode != "disabled")
    if wandb_enabled:
        if args.wandb_key is not None:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_MODE"] = args.wandb_mode

        config = {
            "epochs": args.epochs, "lr": args.lr, "hidden_dim": args.hidden_dim,
            "layers": args.layers, "dropout": args.dropout, "layer_type": args.layer_type,
            "pool": args.pool, "loss": args.loss, "fourier": args.fourier,
            "weight_decay": args.weight_decay, "edge_dropout": args.edge_dropout,
            "feat_noise_std": args.feat_noise_std, "mape_eps": args.mape_eps,
            "ema": args.ema, "ema_decay": args.ema_decay,
            "normalize_target": args.normalize_target,
            "target_mean": target_mean.detach().cpu().tolist(),
            "target_std": target_std.detach().cpu().tolist(),
            "coordinate_system": coordinate_system,
            "target_labels": target_labels,
        }

        try:
            wandb_run = wandb.init(
                project=args.wandb_project, name=args.wandb_name,
                dir=args.wandb_dir, config=config,
            )
        except Exception as e:
            print(f"[wandb] init failed ({type(e).__name__}: {e}). Falling back to offline.", flush=True)
            os.environ["WANDB_MODE"] = "offline"
            wandb_run = wandb.init(
                project=args.wandb_project, name=args.wandb_name,
                dir=args.wandb_dir, config=config,
            )

    best_monitor: Optional[float] = None
    bad_epochs: int = 0
    best_ckpt_epoch: Optional[int] = None
    start_epoch: int = 1

    if ddp_is_main() and args.resume:
        start_epoch, best_monitor, bad_epochs, best_ckpt_epoch, resumed = _try_resume_from_checkpoint(
            ckpt_path=save_path, model=model, opt=opt, scheduler=scheduler,
            scaler=scaler, ema=ema, device=device,
        )
        if resumed:
            print(
                f"[resume] Resumed from {save_path}: start_epoch={start_epoch} "
                f"best_monitor={best_monitor} bad_epochs={bad_epochs} best_ckpt_epoch={best_ckpt_epoch}",
                flush=True,
            )
        else:
            print(f"[resume] No checkpoint found at {save_path}; starting fresh.", flush=True)

    if ddp_is_initialized():
        payload = [
            int(start_epoch), best_monitor, int(bad_epochs),
            (int(best_ckpt_epoch) if best_ckpt_epoch is not None else -1),
        ]
        dist.broadcast_object_list(payload, src=0)
        start_epoch = int(payload[0])
        best_monitor = payload[1]
        bad_epochs = int(payload[2])
        best_ckpt_epoch = int(payload[3]) if int(payload[3]) >= 0 else None

    half_pat = (
        max(1, args.early_stop_patience // 2)
        if (args.early_stop and args.reload_best_half_patience)
        else 0
    )
    reloaded_this_plateau = False
    c0, c1, c2 = target_labels  # coordinate names for logging

    for epoch in range(int(start_epoch), args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # -------------------------
        # Train
        # -------------------------
        model.train()

        train_loss = torch.tensor(0.0, device=device)
        train_steps = torch.tensor(0.0, device=device)
        train_mae = torch.tensor(0.0, device=device)
        train_rmse = torch.tensor(0.0, device=device)
        train_mape = torch.tensor(0.0, device=device)
        train_c0 = torch.tensor(0.0, device=device)
        train_c1 = torch.tensor(0.0, device=device)
        train_c2 = torch.tensor(0.0, device=device)
        train_mape_c0 = torch.tensor(0.0, device=device)
        train_mape_c1 = torch.tensor(0.0, device=device)
        train_mape_c2 = torch.tensor(0.0, device=device)

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            edge_attr = batch["edge_attr"].to(device, non_blocking=True)
            y = batch["y_vertex"].to(device, non_blocking=True).float()

            y_train = (y - target_mean) / target_std if args.normalize_target else y

            if args.feat_noise_std > 0.0 and model.training:
                x = x + args.feat_noise_std * torch.randn_like(x)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred_train = model(x, edge_index, edge_attr, edge_dropout_p=args.edge_dropout)
                loss = loss_fn(pred_train, y_train)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

            if ema is not None:
                ema.update(model)

            if args.lr_schedule == "cosine":
                scheduler.step()

            with torch.no_grad():
                pred_metric = pred_train * target_std + target_mean if args.normalize_target else pred_train
                mae, rmse, mape, mc0, mc1, mc2, mm0, mm1, mm2 = regression_stats(
                    pred_metric, y, args.mape_eps
                )

            train_loss += loss.detach()
            train_steps += 1.0
            train_mae += mae
            train_rmse += rmse
            train_mape += mape
            train_c0 += mc0
            train_c1 += mc1
            train_c2 += mc2
            train_mape_c0 += mm0
            train_mape_c1 += mm1
            train_mape_c2 += mm2

        for t in (
            train_loss, train_steps, train_mae, train_rmse, train_mape,
            train_c0, train_c1, train_c2, train_mape_c0, train_mape_c1, train_mape_c2,
        ):
            ddp_all_reduce_sum(t)

        n_steps = torch.clamp(train_steps, min=1.0)
        train_loss_mean = (train_loss / n_steps).item()
        train_mae_mean = (train_mae / n_steps).item()
        train_rmse_mean = (train_rmse / n_steps).item()
        train_mape_mean = (train_mape / n_steps).item()
        train_c0_mean = (train_c0 / n_steps).item()
        train_c1_mean = (train_c1 / n_steps).item()
        train_c2_mean = (train_c2 / n_steps).item()
        train_mape_c0_mean = (train_mape_c0 / n_steps).item()
        train_mape_c1_mean = (train_mape_c1 / n_steps).item()
        train_mape_c2_mean = (train_mape_c2 / n_steps).item()

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_loss = torch.tensor(0.0, device=device)
        val_steps = torch.tensor(0.0, device=device)
        val_mae = torch.tensor(0.0, device=device)
        val_rmse = torch.tensor(0.0, device=device)
        val_mape = torch.tensor(0.0, device=device)
        val_c0 = torch.tensor(0.0, device=device)
        val_c1 = torch.tensor(0.0, device=device)
        val_c2 = torch.tensor(0.0, device=device)
        val_mape_c0 = torch.tensor(0.0, device=device)
        val_mape_c1 = torch.tensor(0.0, device=device)
        val_mape_c2 = torch.tensor(0.0, device=device)

        eval_ctx = ema.apply_to(model) if ema is not None else nullcontext()
        with eval_ctx:
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device, non_blocking=True)
                    edge_index = batch["edge_index"].to(device, non_blocking=True)
                    edge_attr = batch["edge_attr"].to(device, non_blocking=True)
                    y = batch["y_vertex"].to(device, non_blocking=True).float()

                    y_eval = (y - target_mean) / target_std if args.normalize_target else y

                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        pred_eval = model(x, edge_index, edge_attr, edge_dropout_p=0.0)
                        loss = loss_fn(pred_eval, y_eval)

                    pred_metric = pred_eval * target_std + target_mean if args.normalize_target else pred_eval
                    mae, rmse, mape, mc0, mc1, mc2, mm0, mm1, mm2 = regression_stats(
                        pred_metric, y, args.mape_eps
                    )

                    val_loss += loss
                    val_steps += 1.0
                    val_mae += mae
                    val_rmse += rmse
                    val_mape += mape
                    val_c0 += mc0
                    val_c1 += mc1
                    val_c2 += mc2
                    val_mape_c0 += mm0
                    val_mape_c1 += mm1
                    val_mape_c2 += mm2

        for t in (
            val_loss, val_steps, val_mae, val_rmse, val_mape,
            val_c0, val_c1, val_c2, val_mape_c0, val_mape_c1, val_mape_c2,
        ):
            ddp_all_reduce_sum(t)

        n_val_steps = torch.clamp(val_steps, min=1.0)
        val_loss_mean = (val_loss / n_val_steps).item()
        val_mae_mean = (val_mae / n_val_steps).item()
        val_rmse_mean = (val_rmse / n_val_steps).item()
        val_mape_mean = (val_mape / n_val_steps).item()
        val_c0_mean = (val_c0 / n_val_steps).item()
        val_c1_mean = (val_c1 / n_val_steps).item()
        val_c2_mean = (val_c2 / n_val_steps).item()
        val_mape_c0_mean = (val_mape_c0 / n_val_steps).item()
        val_mape_c1_mean = (val_mape_c1 / n_val_steps).item()
        val_mape_c2_mean = (val_mape_c2 / n_val_steps).item()

        if args.lr_schedule == "plateau":
            scheduler.step(val_loss_mean)
        current_lr = opt.param_groups[0]["lr"]

        if args.early_stop_monitor == "val_loss":
            monitor_val = val_loss_mean
        elif args.early_stop_monitor == "val_rmse":
            monitor_val = val_rmse_mean
        else:
            monitor_val = val_mae_mean

        improved = (
            best_monitor is None or
            monitor_val < best_monitor - args.early_stop_min_delta
        )

        if ddp_is_main():
            print(
                f"[epoch {epoch:03d}] "
                f"train loss={train_loss_mean:.5f} mae={train_mae_mean:.5f} rmse={train_rmse_mean:.5f} "
                f"mape={train_mape_mean:.2f}% "
                f"({c0}=mae:{train_c0_mean:.5f}/mape:{train_mape_c0_mean:.2f}%, "
                f"{c1}=mae:{train_c1_mean:.5f}/mape:{train_mape_c1_mean:.2f}%, "
                f"{c2}=mae:{train_c2_mean:.5f}/mape:{train_mape_c2_mean:.2f}%) | "
                f"val loss={val_loss_mean:.5f} mae={val_mae_mean:.5f} rmse={val_rmse_mean:.5f} "
                f"mape={val_mape_mean:.2f}% "
                f"({c0}=mae:{val_c0_mean:.5f}/mape:{val_mape_c0_mean:.2f}%, "
                f"{c1}=mae:{val_c1_mean:.5f}/mape:{val_mape_c1_mean:.2f}%, "
                f"{c2}=mae:{val_c2_mean:.5f}/mape:{val_mape_c2_mean:.2f}%) | "
                f"lr={current_lr:.3e} | "
                f"{args.early_stop_monitor}={monitor_val:.6f} {'(best)' if improved else ''}",
                flush=True,
            )

        if wandb_run is not None and ddp_is_main():
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss_mean,
                    "train/mae": train_mae_mean,
                    "train/rmse": train_rmse_mean,
                    "train/mape": train_mape_mean,
                    f"train/mae_{c0}": train_c0_mean,
                    f"train/mae_{c1}": train_c1_mean,
                    f"train/mae_{c2}": train_c2_mean,
                    f"train/mape_{c0}": train_mape_c0_mean,
                    f"train/mape_{c1}": train_mape_c1_mean,
                    f"train/mape_{c2}": train_mape_c2_mean,
                    "val/loss": val_loss_mean,
                    "val/mae": val_mae_mean,
                    "val/rmse": val_rmse_mean,
                    "val/mape": val_mape_mean,
                    f"val/mae_{c0}": val_c0_mean,
                    f"val/mae_{c1}": val_c1_mean,
                    f"val/mae_{c2}": val_c2_mean,
                    f"val/mape_{c0}": val_mape_c0_mean,
                    f"val/mape_{c1}": val_mape_c1_mean,
                    f"val/mape_{c2}": val_mape_c2_mean,
                    "lr": current_lr,
                    args.early_stop_monitor: monitor_val,
                },
                step=epoch,
            )

        if ddp_is_main() and improved:
            best_monitor = monitor_val
            best_ckpt_epoch = epoch

            save_ctx = ema.apply_to(model) if ema is not None else nullcontext()
            _ensure_parent_dir(str(save_path))
            with save_ctx:
                raw_model = model.module if hasattr(model, "module") else model
                ckpt_obj = {
                    "model_state": raw_model.state_dict(),
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
                    "loss": args.loss,
                    "fourier": args.fourier,
                    "fourier_base": args.fourier_base,
                    "fourier_min_exp": args.fourier_min_exp,
                    "fourier_max_exp": args.fourier_max_exp,
                    "normalize_target": args.normalize_target,
                    "target_mean": target_mean.detach().cpu(),
                    "target_std": target_std.detach().cpu(),
                    "best_monitor": best_monitor,
                    "early_stop_monitor": args.early_stop_monitor,
                    "best_ckpt_epoch": best_ckpt_epoch,
                    "run_id": run_id,
                    "weight_decay": args.weight_decay,
                    "edge_dropout": args.edge_dropout,
                    "feat_noise_std": args.feat_noise_std,
                    "mape_eps": args.mape_eps,
                    "ema": bool(ema is not None),
                    "ema_decay": args.ema_decay,
                    "lr_schedule": args.lr_schedule,
                    "warmup_epochs": args.warmup_epochs,
                    "min_lr_ratio": args.min_lr_ratio,
                    "code_version": args.code_version,
                    "epoch": int(epoch),
                    "bad_epochs": int(bad_epochs),
                    "coordinate_system": coordinate_system,
                    "target_order": target_labels,
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
                    "scaler_state": (scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None),
                    "ema_shadow": (ema.shadow if ema is not None else None),
                }
                _torch_save_atomic_with_retries(ckpt_obj, save_path)

            print(
                f"  [*] saved best checkpoint to {save_path} "
                f"({args.early_stop_monitor}={best_monitor:.6f}, epoch={epoch})",
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
                        f"[reload-best] No improvement for {bad_epochs} epochs (half_pat={half_pat}). "
                        f"Reloading best checkpoint (epoch={best_ckpt_epoch}) and resetting lr -> {args.lr:.3e}.",
                        flush=True,
                    )

            if bad_epochs >= args.early_stop_patience:
                stop_now = True
                print(
                    f"[early-stop] Triggered at epoch {epoch} (monitor={args.early_stop_monitor}).",
                    flush=True,
                )

        reload_t = torch.tensor([1 if reload_now else 0], device=device, dtype=torch.int32)
        if ddp_is_initialized():
            dist.broadcast(reload_t, src=0)
        reload_now_all = bool(reload_t.item())

        if reload_now_all:
            ddp_barrier()
            load_best_checkpoint_into_model(model, save_path, device)
            reset_optimizer_lr(opt, args.lr)
            scheduler = build_scheduler(opt, args, steps_per_epoch=len(train_loader))
            if ema is not None:
                ema = EMA.create(model, decay=args.ema_decay)
            if scaler.is_enabled():
                scaler = torch.cuda.amp.GradScaler(enabled=True)
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
