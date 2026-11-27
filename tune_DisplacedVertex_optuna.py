#!/usr/bin/env python3
"""
Optuna tuner for DisplacedVertex graph-level binary classifier training.

Default objective:
  maximize val_tpr_at_target_fpr, with --target-fpr 0.001 by default.
That means trials are ranked by the highest validation TPR achievable while the
validation FPR is constrained to be <= 0.1% (before 2026-11-24: 1%).

Trial score (--objective-window W): the best mean of the objective over W consecutive epochs (default 1 = the
single best epoch, as before). The single best epoch of a noisy metric is biased upwards by the epoch-to-epoch noise;
W=3 compares the trials on a plateau instead.

Protocol flags shared by all phases (stored in the study at its first run; a later run with other values stops):
  --global-batch-size N  graphs per optimizer step summed over the GPUs of a run: every phase (fast, confirm, refit)
                         gets N / its GPU count per GPU, so the learning rate found in the fast trials sees the same
                         batch in the refit (default: --batch-size per GPU in every phase, as before).
  --objective-window W, --compare-metric, --target-fpr, --fast-epochs, --fast-max-train-events.

Batch per architecture (--batch-size auto, as in tune_SegmentClassifier_optuna.py): every searched fast trial lets
the trainer probe --auto-batch-candidates per GPU for its own model (the fastest that fits --auto-batch-mem-frac of
the GPU memory), --lr is defined at the global batch --lr-ref-batch and scaled to the trial's global batch
(--lr-batch-scaling, default sqrt; the EMA decay too). The trial's confirm runs and refit keep its global batch,
split over their GPUs; trial 0 runs at --lr-ref-batch. Excludes --global-batch-size.

Pruning (--prune): every epoch line of a fast trial reports its running score to Optuna; after
--prune-warmup-epochs a trial whose score is outside the best --prune-percentile % of the completed trials at the
same epoch is stopped (process tree terminated) and marked PRUNED. Trial 0 is never pruned; pruning starts after
--prune-startup-trials completed trials.

Leaderboard (always): <out-dir>/leaderboard.md and leaderboard.csv are rewritten after every trial, confirm run and
refit: best configuration, refits, confirmed ranking, per-architecture summary, fast trials, parameter importances.
--report-only rewrites them without running anything (also while a study runs).

With the patched trainer, --normalize-node-features / --normalize-edge-features
make the trainer store normalization statistics as model buffers and apply them
inside DisplacedVertexGNN.forward().  The tuner should still pass the same flags;
no H5-level normalization is performed by the DataLoader.

The trainer is expected to print lines like:
[epoch 001] ... | val loss=... acc=... precision=... recall=... f1=... bacc=... auc=... tpr_at_target_fpr=... fpr_at_target_fpr=... thr_at_target_fpr=... | ...

Search space (defaults) = the 80 completed fast trials of tuning_dv_classifier_edge,
narrowed around their top 20% and around the best model (edge_gatv2.onnx,
gat_residual + GATv2 edge attention, ROC AUC 0.977 vs 0.958 for the best refit):
  fixed   : layer_type=gat_residual, gatv2_edge_attn, pool=meanmax, fourier=False,
            pos_weight=auto (16/16 of the top-20% trials), normalized node/edge features
  dropped : loss bce_smooth (0/16 in the top 20%), hidden_dim 64 (0/16)
  narrowed: lr, weight_decay, dropout, feat_noise_std, label_smoothing, focal_gamma,
            asym_gamma_pos/neg; layers 3-6, hidden_dim 128/256, gat_heads 2/4/8
Pass "any" to --fixed-layer-type/--fixed-pool/--fixed-fourier/--fixed-pos-weight to
search a knob again; the old wide ranges are in each flag's help.

Trial 0 is a baseline: train_DisplacedVertex.py's own defaults (the best model). It is
added on top of --n-trials, and --n-startup-trials random trials follow it.

Optional confirm phase (--confirm-top-k K --confirm-seeds N, between the fast phase and the refit): the top K fast
trials are re-run with N new seeds each and ranked by the mean over the seeds; the refit takes its --refit-top-k
from this ranking. With the fast protocol (default) the fast trial counts as one seed. With a higher fidelity
(--confirm-max-train-events / --confirm-epochs / --confirm-gpus-per-trial different from the fast trials) the
ranking uses only the N confirm runs (seeds: the trainer seed + 100*s, s = 0..N-1). Runs: <out-dir>/confirm.jsonl
(a restarted tuner reuses them), ranking: <out-dir>/confirm_summary.json (rewritten after every run). Also with
--refit-only (confirms the finished study).

Architecture options of the trainer (GAT trials; the defaults keep the legacy model, the search space above was
narrowed on it): --fixed-block legacy|prenorm, --fixed-relation-mp false|true, --fixed-edge-update
none|seg|seg_tracklet (relation-mp trials), --fixed-readout global|typed|typed_attn ('any' searches one of them),
plus the fixed knobs --ffn-mult, --grad-clip, --relation-aggr, --calo-context, --readout-heads. For a comparison of
the options fix one at a time; deeper prenorm stacks: --layers-max 8. The baseline trial keeps the trainer defaults
(it lies outside a study with other fixed options and then runs without Optuna params).

Example (the full command list: README.md, "Displaced-vertex classifier"):
TUNE_FLAGS=(
  --train-script ./train_DisplacedVertex.py
  --data-glob "/shared/wp2p5/data/displacedVtx_classifier_mu200/*.packed"
  --split-file "/shared/wp2p5/data/displacedVtx_classifier_mu200/split_displaced_vertex_seed12345.npz"
  --feature-stats-json "/shared/wp2p5/data/displacedVtx_classifier_mu200/normalization_stats_raw.json"
  --out-dir "/shared/wp2p5/models/tuning_dv_classifier_gatv2"
  --study-name dv_classifier_gatv2
  --compare-metric val_tpr_at_target_fpr --target-fpr 0.001
  --num-workers 4 --pin-memory --wandb-mode disabled
  --fixed-layer-type gat_residual --gatv2-edge-attn
  --fixed-block any --fixed-relation-mp any --fixed-edge-update any --fixed-readout any --layers-max 8
  --fast-epochs 40 --fast-max-train-events 100000 --fast-gpus-per-trial 1 --n-jobs 8
  --batch-size auto --objective-window 3
)
python -u tune_DisplacedVertex_optuna.py "${TUNE_FLAGS[@]}" \
  --n-trials 160 --n-startup-trials 50 --prune \
  --refit-top-k 0 2>&1 | tee -a log_tune_dv_classifier_gatv2.txt

# later: confirm the best 10 on 4x the training events, refit the best 5 on the full train split (same flags)
python -u tune_DisplacedVertex_optuna.py "${TUNE_FLAGS[@]}" \
  --refit-only --confirm-top-k 10 --confirm-seeds 2 --confirm-max-train-events 400000 \
  --refit-top-k 5 --refit-epochs 200 --refit-gpus-per-trial 8 2>&1 | tee -a log_tune_dv_classifier_gatv2_refit.txt
"""

import argparse
import ast
import csv
from datetime import datetime
import glob
import json
import math
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
import optuna.logging
import torch

# ── notebook-friendly monitoring ───────────────────────────────────────────
# Force unbuffered stdout so `!python ...` in Jupyter shows output in real time.
# Without this, Jupyter's pipe-captured stdout buffers everything.
import sys as _sys
try:
    _sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ["PYTHONUNBUFFERED"] = "1"  # also affects subprocesses

optuna.logging.set_verbosity(optuna.logging.INFO)

SEP  = "=" * 72
SEP2 = "-" * 72

def _log(msg: str = "") -> None:
    """Write a line to stdout with immediate flush — works in notebooks."""
    _sys.stdout.write(str(msg) + "\n")
    _sys.stdout.flush()

def _banner(title: str, char: str = "=") -> None:
    line = char * 72
    _log(f"\n{line}")
    _log(f"  {title}")
    _log(line)


EPOCH_LINE_RE = re.compile(
    r"^\[epoch\s+(?P<epoch>\d+)\].*?\|\s+val\s+"
    r"loss=(?P<val_loss>[-+0-9.eE]+)\s+"
    r"acc=(?P<val_acc>[-+0-9.eE]+)\s+"
    r"precision=(?P<val_precision>[-+0-9.eE]+)\s+"
    r"recall=(?P<val_recall>[-+0-9.eE]+)\s+"
    r"f1=(?P<val_f1>[-+0-9.eE]+)\s+"
    r"bacc=(?P<val_balanced_acc>[-+0-9.eE]+)\s+"
    r"auc=(?P<val_auc>[-+0-9.eE]+|nan)"
    r"(?:\s+tpr_at_target_fpr=(?P<val_tpr_at_target_fpr>[-+0-9.eE]+|nan)"
    r"\s+fpr_at_target_fpr=(?P<val_fpr_at_target_fpr>[-+0-9.eE]+|nan)"
    r"\s+thr_at_target_fpr=(?P<val_threshold_at_target_fpr>[-+0-9.eE]+|nan))?",
    re.IGNORECASE,
)

METRICS = (
    "val_loss",
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_balanced_acc",
    "val_auc",
    "val_tpr_at_target_fpr",
    "val_fpr_at_target_fpr",
    "val_threshold_at_target_fpr",
)

LOSS_TYPES = ("bce", "bce_smooth", "focal", "asymmetric_focal")

# Architecture options of the trainer (dv_training_utils BLOCK_TYPES / EDGE_UPDATE_MODES / READOUT_TYPES ...)
BLOCK_CHOICES = ("legacy", "prenorm")
EDGE_UPDATE_CHOICES = ("none", "seg", "seg_tracklet")
READOUT_CHOICES = ("global", "typed", "typed_attn")
# trainer defaults = the legacy model (used when the train script predates these options)
ARCH_LEGACY_DEFAULTS = {
    "block": "legacy", "ffn_mult": 2, "grad_clip": 0.0, "relation_mp": False, "relation_aggr": "attn_max",
    "calo_context": "two_way", "edge_update": "none", "readout": "global", "readout_heads": 4,
}


def now_utc_compact() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def metric_direction(metric: str) -> str:
    return "minimize" if metric == "val_loss" else "maximize"


def is_better(metric: str, new: float, best: Optional[float]) -> bool:
    if best is None:
        return True
    if metric_direction(metric) == "minimize":
        return new < best
    return new > best


def epoch_row_from_line(line: str) -> Optional[Dict[str, Any]]:
    """Metrics of a trainer '[epoch NNN] ...' line, or None."""
    m = EPOCH_LINE_RE.search(line.strip())
    if not m:
        return None
    row: Dict[str, Any] = {"epoch": int(m.group("epoch"))}
    for key in METRICS:
        row[key] = safe_float(m.groupdict().get(key))
    return row


def windowed_best(values: List[Optional[float]], window: int, maximize: bool) -> Tuple[Optional[float], Optional[int]]:
    """Best mean of `window` consecutive per-epoch values and the index of the window's last epoch (the first one on
    ties). Fewer epochs than the window: the mean of all of them. Windows with a missing value are skipped. window=1
    is the single best epoch."""
    w = max(1, int(window))
    if not values:
        return None, None
    if len(values) < w:
        if any(v is None for v in values):
            return None, None
        return sum(values) / len(values), len(values) - 1
    best: Optional[float] = None
    best_end: Optional[int] = None
    for end in range(w - 1, len(values)):
        win = values[end - w + 1:end + 1]
        if any(v is None for v in win):
            continue
        mean = sum(win) / w
        if best is None or (mean > best if maximize else mean < best):
            best, best_end = mean, end
    return best, best_end


# trainer lines that describe the run (stored with each trial: data format, sampler, batch, model size)
RUN_INFO_RES = {
    "trainable_params": (re.compile(r"trainable_params=(\d+)"), int),
    "indexing_seconds": (re.compile(r"^\[time\] dataset indexing: ([0-9.]+)s"), float),
    "data_format": (re.compile(r"^\[i\] data=(packed parts|H5) "), str),
    "sampler": (re.compile(r"^\[i\] data=(?:packed parts|H5) sampler: (.*)$"), str),
    "batch_size_per_rank": (re.compile(r"batch_size_per_rank=(\d+)"), int),
    "global_batch_size": (re.compile(r"effective_global_batch_size=(\d+)"), int),
}


def parse_best_metrics_from_log(log_path: Path, compare_metric: str, window: int = 1) -> Dict[str, Any]:
    """objective_value = windowed_best(...) of compare_metric over the epoch lines (window 1: the single best epoch);
    best_row = the single best epoch, window_row = the metrics averaged over the best window (+ window_epochs)."""
    empty = {"objective_value": None, "best_row": {}, "last_row": {}, "n_epochs_seen": 0, "best_epoch_value": None,
             "window_row": {}, "window_epochs": None, "objective_window": int(window), "run_info": {}}
    if not log_path.exists():
        return empty
    maximize = metric_direction(compare_metric) == "maximize"

    last_error: Optional[OSError] = None
    for attempt in range(6):
        rows: List[Dict[str, Any]] = []
        info: Dict[str, Any] = {}
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    row = epoch_row_from_line(line)
                    if row is not None:
                        rows.append(row)
                        continue
                    for key, (rx, cast) in RUN_INFO_RES.items():
                        if key not in info:
                            m = rx.search(line.rstrip("\n"))
                            if m:
                                info[key] = cast(m.group(1))
            out = dict(empty, run_info=info, n_epochs_seen=len(rows), last_row=rows[-1] if rows else {})
            values = [r.get(compare_metric) for r in rows]
            single, i_single = windowed_best(values, 1, maximize)
            if single is not None:
                out["best_epoch_value"] = single
                out["best_row"] = rows[i_single]
            value, end = windowed_best(values, window, maximize)
            if value is not None:
                win_rows = rows[max(0, end - max(1, int(window)) + 1):end + 1]
                out["objective_value"] = value
                out["window_epochs"] = [win_rows[0]["epoch"], win_rows[-1]["epoch"]]
                out["window_row"] = {k: (sum(r[k] for r in win_rows) / len(win_rows)
                                         if all(r.get(k) is not None for r in win_rows) else None) for k in METRICS}
            return out
        except OSError as exc:
            last_error = exc
            if attempt == 5:
                break
            delay = min(2 ** attempt, 16)
            _log(f"[warn] EOS log read failed for {log_path}: {exc}; retrying in {delay}s")
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def detect_gpus() -> List[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [x.strip() for x in cvd.split(",") if x.strip()]
    try:
        n = int(torch.cuda.device_count())
        if n > 0:
            return [str(i) for i in range(n)]
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        ids = []
        for line in out.splitlines():
            m = re.match(r"^\s*GPU\s+(\d+)\s*:", line)
            if m:
                ids.append(m.group(1))
        if ids:
            return ids
    except Exception:
        pass
    return []


class GPUAllocator:
    def __init__(self, devices: List[str]):
        self._all = list(devices)
        self._free = list(devices)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    @contextmanager
    def acquire(self, n: int):
        if not self._all:
            yield []
            return
        n = max(1, min(int(n), len(self._all)))
        with self._cv:
            while len(self._free) < n:
                self._cv.wait()
            got = self._free[:n]
            self._free = self._free[n:]
        try:
            yield got
        finally:
            with self._cv:
                self._free = got + self._free
                self._cv.notify_all()


def pick_master_port(base: int, trial_number: int, *, span: int = 2000) -> int:
    """
    Pick a stable per-trial torchrun port.
    """
    span = max(128, int(span))
    return int(base) + (int(trial_number) % span)


def torchrun_nproc(gpus_per_trial: int, devices: List[str]) -> int:
    """Use torchrun/DDP only when a trial really uses >1 GPU."""
    n_visible_for_trial = len(devices)
    if n_visible_for_trial <= 0:
        return 0
    n = max(1, min(int(gpus_per_trial), n_visible_for_trial))
    return n if n > 1 else 0


def phase_world(gpus_per_trial: int, n_devices: int) -> int:
    """Number of training processes of a run with gpus_per_trial GPUs (what the GPUAllocator hands out)."""
    if n_devices <= 0:
        return 1
    return max(1, min(int(gpus_per_trial), int(n_devices)))


def tuner_batch_arg(value):
    """--batch-size of the tuner: graphs per GPU, or 'auto' (the trainer probes every trial's model on its GPU)."""
    if str(value).lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"must be an integer or 'auto', got {value!r}")
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def auto_batch(args: argparse.Namespace) -> bool:
    return getattr(args, "batch_size", None) == "auto"


def per_rank_batch_size(args: argparse.Namespace, world: int, global_batch: Optional[int] = None):
    """Trainer --batch-size of a run with `world` processes:
    - global_batch given (a trial's own batch: --batch-size auto, its confirm runs and refit; trial 0): global / world;
    - --batch-size auto (searched fast trials): 'auto';
    - --global-batch-size: global / world;
    - else the tuner's --batch-size (per GPU in every phase, the former behaviour)."""
    world = max(1, int(world))
    if global_batch is not None:
        if int(global_batch) % world:
            raise SystemExit(f"global batch {global_batch} is not divisible by the {world} GPUs of a run")
        return int(global_batch) // world
    if auto_batch(args):
        return "auto"
    gbs = getattr(args, "global_batch_size", None)
    if gbs is None:
        return int(args.batch_size)
    if int(gbs) % world:
        raise SystemExit(f"--global-batch-size {gbs} is not divisible by the {world} GPUs of a run")
    return int(gbs) // world


def effective_global_batch(args: argparse.Namespace, world: int):
    b = per_rank_batch_size(args, world)
    return "auto" if b == "auto" else b * max(1, int(world))


def batch_protocol(args: argparse.Namespace):
    """The batch part of the study protocol: the fast trials' global batch, or the auto-batch settings."""
    if auto_batch(args):
        cands = ",".join(str(c) for c in sorted(set(args.auto_batch_candidates)))
        return (f"auto(candidates={cands};gpus={args.fast_gpus_per_trial};mem_frac={args.auto_batch_mem_frac:g};"
                f"tolerance={args.auto_batch_tolerance:g};lr={args.lr_batch_scaling}@{args.lr_ref_batch};"
                f"lr_scaled_max={args.lr_scaled_max})")
    return int(run_protocol(args, "fast")["global_batch"])


def _proc_start_time(pid: int) -> Optional[str]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2:].split()[19]  # field 22: start time
    except (OSError, ValueError, IndexError):
        return None


def _descendants(pid: int) -> List[int]:
    """All descendant pids of pid (Linux /proc)."""
    children: Dict[int, List[int]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def stop_process_tree(proc: subprocess.Popen, grace_seconds: float = 30.0) -> None:
    """SIGTERM to a training run (torchrun or python) and all its descendants (DDP ranks, DataLoader workers), SIGKILL
    to those still alive after grace_seconds. Does not block: the caller keeps reading the output until EOF."""
    pids = [proc.pid] + _descendants(proc.pid)
    started = {p: _proc_start_time(p) for p in pids}
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass

    def kill_survivors():
        time.sleep(grace_seconds)
        for p, t0 in started.items():
            if t0 is not None and _proc_start_time(p) == t0:  # same process, not a reused pid
                try:
                    os.kill(p, signal.SIGKILL)
                except OSError:
                    pass

    threading.Thread(target=kill_survivors, daemon=True).start()


def build_trial_ckpt_path(ckpt_dir: Path, save_base: str, run_id: str) -> Path:
    """Mirror dv_training_utils._build_save_path(...)."""
    base = os.path.basename(str(save_base))
    stem, ext = os.path.splitext(base)
    ext = ext if ext else ".pt"
    return Path(ckpt_dir) / f"{stem}_{run_id}{ext}"


def legacy_duplicated_ckpt_path(save_path: Path, run_id: str) -> Path:
    """Path produced by the pre-patch tuner, which passed a run-id-expanded --save."""
    save_path = Path(save_path)
    return save_path.with_name(f"{save_path.stem}_{run_id}{save_path.suffix}")


def read_ckpt_metrics(ckpt_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ckpt_path": str(ckpt_path), "ckpt_exists": False}
    if not ckpt_path.exists():
        return out
    out["ckpt_exists"] = True
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    for k in [
        "best_monitor", "early_stop_monitor", "best_ckpt_epoch", "epoch",
        "val_loss", "val_acc", "val_precision", "val_recall", "val_f1",
        "val_balanced_acc", "val_auc", "val_tpr_at_target_fpr",
        "val_fpr_at_target_fpr", "val_threshold_at_target_fpr",
    ]:
        if k in ckpt:
            out[k] = ckpt.get(k)
    return out


def extract_objective_from_ckpt(ckpt_info: Dict[str, Any], compare_metric: str) -> Optional[float]:
    v = safe_float(ckpt_info.get(compare_metric))
    if v is not None:
        return v
    # Older checkpoints only stored the monitored objective as best_monitor.
    if ckpt_info.get("early_stop_monitor") == compare_metric:
        return safe_float(ckpt_info.get("best_monitor"))
    return None


def _edge_update_choices(args: argparse.Namespace) -> List[str]:
    """Edge-state options the graphs of this study allow."""
    if not getattr(args, "use_segment_edges", True):
        return ["none"]
    return [c for c in EDGE_UPDATE_CHOICES if c != "seg_tracklet" or getattr(args, "use_tracklets", True)]


def _arch_hparams(args: argparse.Namespace, h: Dict[str, Any], pick) -> None:
    """Trainer architecture options into h. pick(name, choices, fixed, default) gives the fixed value when it is not
    None, else the trial's choice (suggested for a new trial, read from the params for a refit). Only GAT trials get
    the non-legacy options; edge_update / calo_context only with relation_mp."""
    is_gat = h["layer_type"] == "gat_residual"
    h["block"] = pick("block", list(BLOCK_CHOICES), args.fixed_block, "legacy") if is_gat else "legacy"
    h["relation_mp"] = bool(pick("relation_mp", [False, True], args.fixed_relation_mp, False)) if is_gat else False
    if h["relation_mp"]:
        h["edge_update"] = pick("edge_update", _edge_update_choices(args), args.fixed_edge_update, "none")
        h["calo_context"] = args.calo_context
    else:
        h["edge_update"] = "none"
        h["calo_context"] = "two_way"
    h["readout"] = pick("readout", list(READOUT_CHOICES), args.fixed_readout, "global")
    h["readout_heads"] = int(args.readout_heads) if h["hidden_dim"] % int(args.readout_heads) == 0 else 4
    h["ffn_mult"] = int(args.ffn_mult)
    h["grad_clip"] = float(args.grad_clip)
    h["relation_aggr"] = args.relation_aggr


def _suggest_loss_hparams(trial: optuna.Trial, args: argparse.Namespace, h: Dict[str, Any]) -> None:
    h["loss_type"] = args.fixed_loss_type or trial.suggest_categorical("loss_type", list(args.loss_types))

    if h["loss_type"] == "bce":
        h["label_smoothing"] = 0.0
        h["focal_gamma"] = 2.0
        h["focal_alpha"] = "none"
        h["asym_gamma_pos"] = 0.0
        h["asym_gamma_neg"] = 4.0
    elif h["loss_type"] == "bce_smooth":
        h["label_smoothing"] = trial.suggest_float("label_smoothing", args.label_smoothing_min, args.label_smoothing_max)
        h["focal_gamma"] = 2.0
        h["focal_alpha"] = "none"
        h["asym_gamma_pos"] = 0.0
        h["asym_gamma_neg"] = 4.0
    elif h["loss_type"] == "focal":
        h["label_smoothing"] = trial.suggest_float("label_smoothing", args.label_smoothing_min, args.label_smoothing_max)
        h["focal_gamma"] = trial.suggest_float("focal_gamma", args.focal_gamma_min, args.focal_gamma_max)
        h["focal_alpha"] = trial.suggest_float("focal_alpha", args.focal_alpha_min, args.focal_alpha_max)
        h["asym_gamma_pos"] = 0.0
        h["asym_gamma_neg"] = 4.0
    elif h["loss_type"] == "asymmetric_focal":
        h["label_smoothing"] = trial.suggest_float("label_smoothing", args.label_smoothing_min, args.label_smoothing_max)
        h["focal_gamma"] = 2.0
        h["focal_alpha"] = trial.suggest_float("focal_alpha", args.focal_alpha_min, args.focal_alpha_max)
        h["asym_gamma_pos"] = trial.suggest_float("asym_gamma_pos", args.asym_gamma_pos_min, args.asym_gamma_pos_max)
        h["asym_gamma_neg"] = trial.suggest_float("asym_gamma_neg", args.asym_gamma_neg_min, args.asym_gamma_neg_max)
    else:
        raise ValueError(f"Unknown loss_type={h['loss_type']!r}")


def build_hparams(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    h: Dict[str, Any] = {}
    h["lr"] = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
    h["hidden_dim"] = trial.suggest_categorical("hidden_dim", args.hidden_dims)
    h["layers"] = trial.suggest_int("layers", args.layers_min, args.layers_max)
    h["dropout"] = trial.suggest_float("dropout", args.dropout_min, args.dropout_max)
    h["weight_decay"] = trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True)
    h["edge_dropout"] = trial.suggest_float("edge_dropout", args.edge_dropout_min, args.edge_dropout_max)
    h["feat_noise_std"] = trial.suggest_float("feat_noise_std", args.feat_noise_min, args.feat_noise_max)
    h["layer_type"] = args.fixed_layer_type or trial.suggest_categorical("layer_type", ["mpnn", "sage_residual", "gat_residual", "edge_residual"])
    h["pool"] = args.fixed_pool or trial.suggest_categorical("pool", ["mean", "meanmax"])
    h["fourier"] = args.fixed_fourier if args.fixed_fourier is not None else trial.suggest_categorical("fourier", [True, False])
    h["pos_weight"] = args.fixed_pos_weight or trial.suggest_categorical("pos_weight", ["auto", "none"])
    if h["layer_type"] == "gat_residual":
        h["gat_heads"] = trial.suggest_categorical("gat_heads", list(args.gat_heads_choices))
        if h["hidden_dim"] % h["gat_heads"] != 0:
            h["gat_heads"] = 4 if h["hidden_dim"] % 4 == 0 else 2
    else:
        h["gat_heads"] = 4
    is_gat = h["layer_type"] == "gat_residual"
    h["gat_edge_attn"] = bool(args.gat_edge_attn and is_gat)
    h["gatv2_edge_attn"] = bool(args.gatv2_edge_attn and is_gat)
    _arch_hparams(args, h, lambda name, choices, fixed, default:
                  fixed if fixed is not None else trial.suggest_categorical(name, choices))
    _suggest_loss_hparams(trial, args, h)
    return h


def refit_hparams_from_trial(trial: optuna.trial.FrozenTrial, args: argparse.Namespace) -> Dict[str, Any]:
    h = _refit_hparams_from_trial(trial, args)
    if auto_batch(args) and "_global_batch" not in h:
        # --batch-size auto: confirm runs and refit train with the global batch the trial was ranked with
        g = trial.user_attrs.get("global_batch_size")
        if g:
            h["_global_batch"] = int(g)
        else:
            _log(f"[warn] trial {trial.number}: no recorded batch size; its re-runs probe the batch again")
    return h


def _refit_hparams_from_trial(trial: optuna.trial.FrozenTrial, args: argparse.Namespace) -> Dict[str, Any]:
    if is_baseline_trial(trial):
        return dict(trial.user_attrs["hparams"])
    p = dict(trial.params)
    layer_type = args.fixed_layer_type or p.get("layer_type", "mpnn")
    is_gat = layer_type == "gat_residual"
    h: Dict[str, Any] = {
        "lr": p.get("lr", args.lr_min),
        "hidden_dim": p.get("hidden_dim", args.hidden_dims[0]),
        "layers": p.get("layers", args.layers_min),
        "dropout": p.get("dropout", args.dropout_min),
        "weight_decay": p.get("weight_decay", args.weight_decay_min),
        "edge_dropout": p.get("edge_dropout", args.edge_dropout_min),
        "feat_noise_std": p.get("feat_noise_std", args.feat_noise_min),
        "layer_type": layer_type,
        "pool": args.fixed_pool or p.get("pool", "meanmax"),
        "fourier": args.fixed_fourier if args.fixed_fourier is not None else p.get("fourier", True),
        "pos_weight": args.fixed_pos_weight or p.get("pos_weight", "auto"),
        "gat_heads": p.get("gat_heads", 4),
        "gat_edge_attn": bool(args.gat_edge_attn and is_gat),
        "gatv2_edge_attn": bool(args.gatv2_edge_attn and is_gat),
        "loss_type": args.fixed_loss_type or p.get("loss_type", args.loss_types[0]),
        "label_smoothing": p.get("label_smoothing", 0.0),
        "focal_gamma": p.get("focal_gamma", 2.0),
        "focal_alpha": p.get("focal_alpha", "none"),
        "asym_gamma_pos": p.get("asym_gamma_pos", 0.0),
        "asym_gamma_neg": p.get("asym_gamma_neg", 4.0),
    }
    _arch_hparams(args, h, lambda name, choices, fixed, default: fixed if fixed is not None else p.get(name, default))
    return h


# ----------------------------------------------------------------------------
# Baseline trial 0: the train script's own defaults (the current best model),
# read from its argparse calls without importing it.
# ----------------------------------------------------------------------------
def read_trainer_defaults(train_script: Path) -> Dict[str, Any]:
    """dest -> default from add_argument(..., default=<literal>) in the train script and
    the modules it imports from its own directory, then its ap.set_defaults(...)."""
    train_script = Path(train_script)
    tree = ast.parse(train_script.read_text(encoding="utf-8"))
    trees = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mod = train_script.parent / (node.module.replace(".", "/") + ".py")
            if mod.is_file():
                trees.append(ast.parse(mod.read_text(encoding="utf-8")))
    trees.append(tree)  # the script last: its set_defaults win
    defaults: Dict[str, Any] = {}
    overrides: Dict[str, Any] = {}
    for t in trees:
        for node in ast.walk(t):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if node.func.attr == "set_defaults":
                for k, v in kw.items():
                    overrides[k] = ast.literal_eval(v)
                continue
            if node.func.attr != "add_argument" or "default" not in kw:
                continue
            if "dest" in kw:
                dest = ast.literal_eval(kw["dest"])
            else:
                flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                longs = [f for f in flags if f.startswith("--")]
                if not longs:
                    continue
                dest = longs[0][2:].replace("-", "_")
            defaults.setdefault(dest, ast.literal_eval(kw["default"]))
    defaults.update(overrides)
    return defaults


def trainer_baseline_hparams(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Trainer defaults in the tuner's hparams format."""
    need = ["lr", "hidden_dim", "layers", "dropout", "weight_decay", "edge_dropout", "feat_noise_std",
            "layer_type", "pool", "fourier", "pos_weight", "gat_heads", "gat_edge_attn", "gatv2_edge_attn",
            "loss_type", "label_smoothing", "focal_gamma", "focal_alpha", "asym_gamma_pos", "asym_gamma_neg"]
    missing = [k for k in need if k not in defaults]
    if missing:
        raise SystemExit(f"[baseline] train script has no default for {missing}")
    h = {k: defaults[k] for k in need}
    for k, v in ARCH_LEGACY_DEFAULTS.items():
        h[k] = defaults.get(k, v)
    fa = str(h["focal_alpha"])
    h["focal_alpha"] = "none" if fa.lower() == "none" else float(fa)
    h["pos_weight"] = str(h["pos_weight"])
    return h


def baseline_enqueue_params(h: Dict[str, Any], args: argparse.Namespace):
    """Optuna params that make build_hparams() reproduce h in this study's search space,
    or (None, reason) if the baseline lies outside it (then it runs without params)."""
    p: Dict[str, Any] = {}

    def fixed(name, value, fixed_value):
        if fixed_value is None:
            return True
        return value == fixed_value

    def flt(name, lo, hi):
        if not (lo <= float(h[name]) <= hi):
            raise ValueError(f"{name}={h[name]} outside [{lo}, {hi}]")
        p[name] = float(h[name])

    try:
        flt("lr", args.lr_min, args.lr_max)
        if h["hidden_dim"] not in args.hidden_dims:
            raise ValueError(f"hidden_dim={h['hidden_dim']} not in {args.hidden_dims}")
        p["hidden_dim"] = h["hidden_dim"]
        if not (args.layers_min <= h["layers"] <= args.layers_max):
            raise ValueError(f"layers={h['layers']} outside [{args.layers_min}, {args.layers_max}]")
        p["layers"] = h["layers"]
        flt("dropout", args.dropout_min, args.dropout_max)
        flt("weight_decay", args.weight_decay_min, args.weight_decay_max)
        flt("edge_dropout", args.edge_dropout_min, args.edge_dropout_max)
        flt("feat_noise_std", args.feat_noise_min, args.feat_noise_max)
        for name, fv, choices in (("layer_type", args.fixed_layer_type, ["mpnn", "sage_residual", "gat_residual", "edge_residual"]),
                                  ("pool", args.fixed_pool, ["mean", "meanmax"]),
                                  ("fourier", args.fixed_fourier, [True, False]),
                                  ("pos_weight", args.fixed_pos_weight, ["auto", "none"])):
            if fv is not None:
                if h[name] != fv:
                    raise ValueError(f"{name}={h[name]!r} but the study fixes {fv!r}")
            elif h[name] in choices:
                p[name] = h[name]
            else:
                raise ValueError(f"{name}={h[name]!r} not in {choices}")
        is_gat = h["layer_type"] == "gat_residual"
        if bool(h["gat_edge_attn"]) != bool(args.gat_edge_attn and is_gat) or \
           bool(h["gatv2_edge_attn"]) != bool(args.gatv2_edge_attn and is_gat):
            raise ValueError("GAT edge-attention flags differ from the study's")
        if is_gat:
            if h["gat_heads"] not in args.gat_heads_choices or h["hidden_dim"] % h["gat_heads"]:
                raise ValueError(f"gat_heads={h['gat_heads']} not in {args.gat_heads_choices}")
            p["gat_heads"] = h["gat_heads"]
        # architecture options: categorical ones as Optuna params, the others must equal the study's fixed values
        arch_cats = [("readout", args.fixed_readout, list(READOUT_CHOICES))]
        if is_gat:
            arch_cats += [("block", args.fixed_block, list(BLOCK_CHOICES)),
                          ("relation_mp", args.fixed_relation_mp, [False, True])]
            if h["relation_mp"]:
                arch_cats.append(("edge_update", args.fixed_edge_update, _edge_update_choices(args)))
        elif h["block"] != "legacy" or h["relation_mp"]:
            raise ValueError(f"block={h['block']} / relation_mp={h['relation_mp']} need gat_residual")
        if not h["relation_mp"] and (h["edge_update"] != "none" or h["calo_context"] != "two_way"):
            raise ValueError("edge_update / calo_context need relation_mp")
        for name, fv, choices in arch_cats:
            if fv is not None:
                if h[name] != fv:
                    raise ValueError(f"{name}={h[name]!r} but the study fixes {fv!r}")
            elif h[name] in choices:
                p[name] = h[name]
            else:
                raise ValueError(f"{name}={h[name]!r} not in {choices}")
        knobs = [("grad_clip", float(args.grad_clip))]
        if h["block"] == "prenorm":
            knobs.append(("ffn_mult", int(args.ffn_mult)))
        if h["relation_mp"]:
            knobs += [("relation_aggr", args.relation_aggr), ("calo_context", args.calo_context)]
        if h["readout"] == "typed_attn":
            knobs.append(("readout_heads", int(args.readout_heads)))
        for name, study_value in knobs:
            if h[name] != study_value:
                raise ValueError(f"{name}={h[name]!r} but the study uses {study_value!r}")
        lt = h["loss_type"]
        if args.fixed_loss_type is not None:
            if lt != args.fixed_loss_type:
                raise ValueError(f"loss_type={lt} but the study fixes {args.fixed_loss_type}")
        elif lt in args.loss_types:
            p["loss_type"] = lt
        else:
            raise ValueError(f"loss_type={lt} not in {args.loss_types}")
        if lt in ("bce_smooth", "focal", "asymmetric_focal"):
            flt("label_smoothing", args.label_smoothing_min, args.label_smoothing_max)
        if lt == "focal":
            flt("focal_gamma", args.focal_gamma_min, args.focal_gamma_max)
        if lt in ("focal", "asymmetric_focal"):
            flt("focal_alpha", args.focal_alpha_min, args.focal_alpha_max)
        if lt == "asymmetric_focal":
            flt("asym_gamma_pos", args.asym_gamma_pos_min, args.asym_gamma_pos_max)
            flt("asym_gamma_neg", args.asym_gamma_neg_min, args.asym_gamma_neg_max)
    except ValueError as e:
        return None, str(e)
    return p, ""


def is_baseline_trial(t) -> bool:
    return bool(t.user_attrs.get("baseline"))


def build_command(
    *,
    python_exe: str,
    train_script: Path,
    data_glob: str,
    split_file: Path,
    save_path: Path,
    save_base: str,
    run_id: str,
    epochs: int,
    max_train_events: int,
    num_workers: int,
    hparams: Dict[str, Any],
    args: argparse.Namespace,
    nproc: int,
    master_port: int,
    rdzv_id: str,
    resume: bool = False,
) -> List[str]:
    if nproc > 0:
        cmd = [
            python_exe, "-m", "torch.distributed.run",
            "--standalone",
            f"--master_port={int(master_port)}",
            f"--nproc_per_node={int(nproc)}",
            str(train_script),
        ]
    else:
        cmd = [python_exe, str(train_script)]

    cmd += [
        "--data-glob", data_glob,
        "--split-file", str(split_file),
        "--epochs", str(int(epochs)),
        "--lr", str(hparams["lr"]),
        "--hidden-dim", str(hparams["hidden_dim"]),
        "--layers", str(hparams["layers"]),
        "--dropout", str(hparams["dropout"]),
        "--layer-type", str(hparams["layer_type"]),
        "--gat-heads", str(hparams.get("gat_heads", 4)),
        "--pool", str(hparams["pool"]),
        "--weight-decay", str(hparams["weight_decay"]),
        "--edge-dropout", str(hparams["edge_dropout"]),
        "--feat-noise-std", str(hparams["feat_noise_std"]),
        "--pos-weight", str(hparams["pos_weight"]),
        "--loss-type", str(hparams["loss_type"]),
        "--label-smoothing", str(hparams["label_smoothing"]),
        "--focal-gamma", str(hparams["focal_gamma"]),
        "--focal-alpha", str(hparams["focal_alpha"]),
        "--asym-gamma-pos", str(hparams["asym_gamma_pos"]),
        "--asym-gamma-neg", str(hparams["asym_gamma_neg"]),
        "--target-fpr", str(args.target_fpr),
        "--num-workers", str(int(num_workers)),
        "--batch-size", str(per_rank_batch_size(args, max(1, int(nproc)), hparams.get("_global_batch"))),
        "--max-open-h5-files", str(int(args.max_open_h5_files)),
        "--save", str(save_base),
        "--save-dir", str(save_path.parent),
        "--run-id", str(run_id),
        "--resume" if resume else "--no-resume",
        "--early-stop-monitor", str(args.compare_metric),
        "--early-stop-patience", str(args.early_stop_patience),
        "--wandb-mode", str(args.wandb_mode),
    ]

    if bool(hparams.get("gat_edge_attn", False)):
        cmd += ["--gat-edge-attn"]
    else:
        cmd += ["--no-gat-edge-attn"]
    if bool(hparams.get("gatv2_edge_attn", False)):
        cmd += ["--gatv2-edge-attn"]
    else:
        cmd += ["--no-gatv2-edge-attn"]
    if "_seed" in hparams:  # confirmation runs only: the same config with another seed
        cmd += ["--seed", str(int(hparams["_seed"]))]
    if cmd[cmd.index("--batch-size") + 1] == "auto":
        cmd += ["--auto-batch-candidates", *[str(c) for c in args.auto_batch_candidates],
                "--auto-batch-mem-frac", str(args.auto_batch_mem_frac),
                "--auto-batch-tolerance", str(args.auto_batch_tolerance)]
    if getattr(args, "lr_batch_scaling", "none") != "none":  # --lr is defined at the global batch --lr-ref-batch
        cmd += ["--lr-batch-scaling", str(args.lr_batch_scaling), "--lr-ref-batch", str(int(args.lr_ref_batch))]
        if args.lr_scaled_max is not None:
            cmd += ["--lr-max", str(args.lr_scaled_max)]
    arch = {k: hparams.get(k, v) for k, v in ARCH_LEGACY_DEFAULTS.items()}
    cmd += [
        "--block", str(arch["block"]),
        "--ffn-mult", str(int(arch["ffn_mult"])),
        "--grad-clip", str(float(arch["grad_clip"])),
        "--relation-mp" if bool(arch["relation_mp"]) else "--no-relation-mp",
        "--relation-aggr", str(arch["relation_aggr"]),
        "--calo-context", str(arch["calo_context"]),
        "--edge-update", str(arch["edge_update"]),
        "--readout", str(arch["readout"]),
        "--readout-heads", str(int(arch["readout_heads"])),
    ]
    if max_train_events > 0:
        cmd += ["--max-train-events", str(int(max_train_events))]
    if args.feature_stats_json:
        cmd += ["--feature-stats-json", str(args.feature_stats_json), "--feature-norm-kind", args.feature_norm_kind]
    if args.normalize_node_features:
        cmd += ["--normalize-node-features"]
    else:
        cmd += ["--no-normalize-node-features"]
    if args.normalize_edge_features:
        cmd += ["--normalize-edge-features"]
    else:
        cmd += ["--no-normalize-edge-features"]
    if args.pin_memory:
        cmd += ["--pin-memory"]
    else:
        cmd += ["--no-pin-memory"]
    if hparams["fourier"]:
        cmd += ["--fourier"]
    else:
        cmd += ["--no-fourier"]
    if args.no_amp:
        cmd += ["--no-amp"]
    if args.no_ema:
        cmd += ["--no-ema"]
    if getattr(args, "use_tracks", False):
        cmd += ["--use-tracks"]
    if not getattr(args, "use_segment_edges", True):
        cmd += ["--no-use-segment-edges"]
    if not getattr(args, "use_tracklets", True):
        cmd += ["--no-use-tracklets"]
    if args.extra_train_args:
        cmd += shlex.split(args.extra_train_args)
    return cmd


def run_one_training(
    *,
    cmd: List[str],
    log_path: Path,
    gpu_ids: List[str],
    cwd: Path,
    env_extra: Optional[Dict[str, str]] = None,
    save_dir: Optional[Path] = None,
    trial_label: str = "",
    on_epoch: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> int:
    """Run one training, tee its output to log_path and stdout. on_epoch(row) is called with the metrics of every
    epoch line; when it returns True the run is stopped (stop_process_tree) and its return code is returned."""
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    if env_extra:
        env.update(env_extra)

    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    # Ensure subprocess output is unbuffered
    env["PYTHONUNBUFFERED"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = " ".join(shlex.quote(x) for x in cmd)
    if save_dir is not None:
        shell_cmd = f"mkdir -p {shlex.quote(str(Path(save_dir).expanduser().resolve()))} && exec {shell_cmd}"

    _log(f"[run]  log → {log_path}")
    _log(f"[run]  GPUs: {gpu_ids if gpu_ids else 'CPU'}")

    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + shell_cmd + "\n")
        log.flush()
        proc = subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        stopping = False
        for line in proc.stdout:
            # Always write to log file
            log.write(line)
            log.flush()
            stripped = line.rstrip()
            # Write directly to stdout with explicit flush (not print)
            _sys.stdout.write(stripped + "\n")
            _sys.stdout.flush()
            # Also emit a highlighted marker for epoch lines
            if stripped.startswith("[epoch"):
                _log(f"  >> {trial_label} | {stripped}")
                row = epoch_row_from_line(stripped)
                if on_epoch is not None and row is not None and not stopping and on_epoch(row):
                    stopping = True
                    msg = f"[prune] {trial_label}: stopped after epoch {row['epoch']}\n"
                    log.write(msg)
                    log.flush()
                    _log(msg.rstrip())
                    stop_process_tree(proc)
        rc = proc.wait()
    return rc


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def arch_tag(h: Dict[str, Any]) -> str:
    """Short architecture name of a trial: [layer type+] block [+relmp[-1way] [+edge-X]] +readout.
    'legacy+global' = the current GATv2 model."""
    parts = []
    lt = h.get("layer_type", "gat_residual")
    if lt != "gat_residual":
        parts.append(str(lt))
    parts.append(str(h.get("block", "legacy")))
    if h.get("relation_mp"):
        parts.append("relmp" + ("-1way" if h.get("calo_context") == "one_way" else ""))
        if h.get("edge_update", "none") != "none":
            parts.append(f"edge-{h['edge_update']}")
    parts.append(str(h.get("readout", "global")))
    return "+".join(parts)


LEADERBOARD_LOCK = threading.Lock()
LEADERBOARD_TOP = 30
_IMPORTANCE_CACHE: Dict[str, Any] = {"n": -1, "value": None, "method": None}
HPARAM_COLUMNS = ("lr", "hidden_dim", "layers", "gat_heads", "dropout", "weight_decay", "edge_dropout",
                  "feat_noise_std", "loss_type", "label_smoothing", "focal_gamma", "focal_alpha", "asym_gamma_pos",
                  "asym_gamma_neg", "block", "relation_mp", "calo_context", "edge_update", "readout", "pool", "fourier")


def _param_importances(study) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Optuna parameter importances over the complete trials (fANOVA, needs scikit-learn; else PED-ANOVA); cached
    per number of complete trials."""
    n = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if n < 10:
        return None, None
    if _IMPORTANCE_CACHE["n"] == n:
        return _IMPORTANCE_CACHE["value"], _IMPORTANCE_CACHE["method"]
    value, method = None, None
    for method_name, make in (("fANOVA", lambda: None),
                              ("PED-ANOVA", lambda: optuna.importance.PedAnovaImportanceEvaluator())):
        try:
            value = dict(optuna.importance.get_param_importances(study, evaluator=make()))
            method = method_name
            break
        except Exception:
            continue
    _IMPORTANCE_CACHE.update(n=n, value=value, method=method)
    return value, method


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "-"
        return f"{v:.{nd}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
    return str(v)


def _median(vals: List[float]) -> Optional[float]:
    vals = sorted(vals)
    if not vals:
        return None
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else 0.5 * (vals[m - 1] + vals[m])


def _most_common(vals: List[Any]) -> Any:
    vals = [v for v in vals if v is not None]
    return max(set(vals), key=vals.count) if vals else None


def collect_leaderboard_rows(out_dir: Path, args: argparse.Namespace, study=None) -> Dict[str, Any]:
    """Fast trials (trials.jsonl, the last record of every trial number) joined with the confirmed ranking
    (confirm_summary.json) and the refits (refit.jsonl), sorted: complete by score, then pruned, then failed."""
    maximize = metric_direction(args.compare_metric) == "maximize"
    fast: Dict[int, Dict[str, Any]] = {}
    for r in load_jsonl(out_dir / "trials.jsonl"):
        if r.get("phase", "fast") == "fast" and r.get("trial_number") is not None:
            fast[int(r["trial_number"])] = r
    baseline_numbers, running = set(), 0
    if study is not None:
        for t in study.trials:
            if is_baseline_trial(t):
                baseline_numbers.add(t.number)
            if t.state == optuna.trial.TrialState.RUNNING:
                running += 1

    confirm: Dict[int, Dict[str, Any]] = {}
    confirm_meta: Dict[str, Any] = {}
    cpath = out_dir / "confirm_summary.json"
    if cpath.exists():
        try:
            cs = json.loads(cpath.read_text(encoding="utf-8"))
            ranking = cs["ranking"] if isinstance(cs, dict) else cs
            confirm_meta = cs if isinstance(cs, dict) else {}
            for row in ranking:
                confirm[int(row["trial_number"])] = row
        except (OSError, ValueError, KeyError, TypeError):
            pass

    refits: Dict[int, Dict[str, Any]] = {}
    for r in load_jsonl(out_dir / "refit.jsonl"):
        n = r.get("source_trial_number")
        if n is None:
            continue
        ok = r.get("returncode") == 0 and r.get("objective_value") is not None
        if ok or int(n) not in refits or refits[int(n)].get("returncode") != 0:
            refits[int(n)] = r

    rows = []
    for n, r in fast.items():
        h = r.get("hparams") or {}
        pruned = bool(r.get("pruned"))
        ok = r.get("returncode") == 0 and r.get("objective_value") is not None and not pruned
        best = r.get("best_row") or {}
        win = r.get("window_row") or best
        info = r.get("run_info") or {}
        c = confirm.get(n, {})
        rf = refits.get(n, {})
        rows.append({
            "trial": n, "state": "complete" if ok else (f"pruned@{r.get('pruned_at_epoch')}" if pruned else
                                                        f"failed(rc={r.get('returncode')})"),
            "ok": ok, "pruned": pruned, "baseline": bool(r.get("baseline")) or n in baseline_numbers,
            "arch": arch_tag(h), "score": safe_float(r.get("objective_value")),
            "best_epoch_value": safe_float(r.get("best_epoch_value", best.get(args.compare_metric))),
            "best_epoch": best.get("epoch"), "window_epochs": r.get("window_epochs"),
            "auc": safe_float(win.get("val_auc")), "val_loss": safe_float(win.get("val_loss")),
            "n_epochs": r.get("n_epochs_seen"), "minutes": (r.get("seconds") or 0.0) / 60.0,
            "trainable_params": info.get("trainable_params"), "data_format": info.get("data_format"),
            "sampler": info.get("sampler"), "indexing_seconds": info.get("indexing_seconds"),
            "global_batch": info.get("global_batch_size"),
            "confirm_mean": c.get("mean"), "confirm_sem": c.get("sem"), "confirm_n": c.get("n"),
            "confirm_rank": c.get("rank"),
            "refit_score": safe_float(rf.get("objective_value")) if rf.get("returncode") == 0 else None,
            "refit_state": None if not rf else ("complete" if rf.get("returncode") == 0 else
                                                f"failed(rc={rf.get('returncode')})"),
            "refit_best_epoch": (rf.get("best_row") or {}).get("epoch"),
            "hparams": h,
        })

    def key(row):
        group = 0 if row["ok"] else (1 if row["pruned"] else 2)
        s = row["score"]
        return (group, s is None, -(s or 0.0) if maximize else (s or 0.0), row["trial"])

    rows.sort(key=key)
    rank = 0
    for row in rows:
        if row["ok"]:
            rank += 1
            row["rank"] = rank
        else:
            row["rank"] = None
    return {"rows": rows, "running": running, "confirm_meta": confirm_meta, "refits": refits, "maximize": maximize}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _leaderboard_markdown(res: Dict[str, Any], args: argparse.Namespace, study=None) -> str:
    rows, maximize = res["rows"], res["maximize"]
    ok = [r for r in rows if r["ok"]]
    by_trial = {r["trial"]: r for r in rows}
    n_pruned = sum(r["pruned"] for r in rows)
    n_failed = len(rows) - len(ok) - n_pruned
    metric = args.compare_metric
    proto = (study.user_attrs.get("protocol") if study is not None else None) or {}  # the study's, else this run's
    fast_gb = proto.get("fast_global_batch")
    if fast_gb is None:
        fast_gb = ("auto" if auto_batch(args) else None) or _most_common([r["global_batch"] for r in rows])
    window = int(proto.get("objective_window", args.objective_window))
    target_fpr = float(proto.get("target_fpr", args.target_fpr))
    fast_epochs = proto.get("fast_epochs", args.fast_epochs)
    fast_events = int(proto.get("fast_max_train_events", args.fast_max_train_events))
    L = [f"# Leaderboard: study `{args.study_name}`", ""]
    L.append(f"Updated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}. Trials: {len(ok)} complete, "
             f"{n_pruned} pruned, {n_failed} failed" + (f", {res['running']} running" if res["running"] else "") + ".")
    L.append("")
    score_txt = (f"`{metric}`" + (f" at FPR {target_fpr:g}" if metric == "val_tpr_at_target_fpr" else "")
                 + " on the validation split, " + (f"mean of the best {window} consecutive epochs" if window > 1
                                                   else "best single epoch")
                 + f" ({'higher' if maximize else 'lower'} is better).")
    L.append(f"- Score: {score_txt}")
    pr = (study.user_attrs.get("pruning") if study is not None else None) or (
        {"percentile": args.prune_percentile, "warmup_epochs": args.prune_warmup_epochs,
         "startup_trials": args.prune_startup_trials} if args.prune else None)
    batch_txt = (f"batch probed per architecture (column `batch`; confirm runs and refits keep it; column `lr` = "
                 f"the lr at global batch {args.lr_ref_batch}, scaled {args.lr_batch_scaling} in the runs)"
                 if isinstance(fast_gb, str) and fast_gb.startswith("auto") else
                 f"global batch {fast_gb if fast_gb is not None else '?'}")
    L.append(f"- Fast trials: {fast_epochs} epochs, "
             f"{'all' if fast_events <= 0 else fast_events} training events, "
             f"{batch_txt}; pruning "
             + (f"on (stop below the best {pr['percentile']:g}% at the same epoch, from epoch "
                f"{pr['warmup_epochs']}, after {pr['startup_trials']} complete trials)" if pr else "off")
             + (f", {n_pruned} trials pruned" if n_pruned else "") + ".")
    fmt_data = _most_common([r["data_format"] for r in rows])
    if fmt_data is not None:
        idx = _median([r["indexing_seconds"] for r in rows if r["indexing_seconds"] is not None])
        L.append(f"- Data: {fmt_data}, sampler: {_most_common([r['sampler'] for r in rows])}, dataset indexing "
                 f"median {_fmt(idx, 1)} s per trial.")
    L.append("")

    # best configuration
    L.append("## Best configuration")
    L.append("")
    refit_ok = sorted([r for r in rows if r["refit_score"] is not None],
                      key=lambda r: -r["refit_score"] if maximize else r["refit_score"])
    cm = res["confirm_meta"]
    confirmed = sorted([r for r in rows if r["confirm_mean"] is not None],
                       key=lambda r: r["confirm_rank"] if r["confirm_rank"] is not None else 1e9)

    def describe(r):
        txt = f"trial {r['trial']}" + (" (trial 0: train-script defaults)" if r["baseline"] else "")
        txt += f", `{r['arch']}`, fast {_fmt(r['score'])}"
        if r["confirm_mean"] is not None:
            txt += f", confirmed {_fmt(r['confirm_mean'])} ± {_fmt(r['confirm_sem'])} (n={r['confirm_n']})"
        if r["refit_score"] is not None:
            txt += f", refit {_fmt(r['refit_score'])}"
        return txt

    if refit_ok:
        b = refit_ok[0]
        L.append(f"**By refit score: {describe(b)}.**")
        if len(refit_ok) > 1:
            n2 = refit_ok[1]
            L.append(f"Next refit: trial {n2['trial']} `{n2['arch']}` {_fmt(n2['refit_score'])} "
                     f"(difference {_fmt(b['refit_score'] - n2['refit_score'])}; refits are single runs).")
    elif confirmed:
        b = confirmed[0]
        L.append(f"**By confirmed mean: {describe(b)}.**")
        fvs = cm.get("first_vs_second") if cm else None
        if fvs and fvs.get("sigma") is not None:
            L.append(f"#1 vs #2: {fvs['diff']:+.4f} ± {fvs['err']:.4f} ({fvs['sigma']:.1f} sigma).")
        if cm and not cm.get("complete", True):
            L.append(f"Confirm phase running: {cm.get('runs_finished')}/{cm.get('runs_total')} runs finished.")
    elif ok:
        b = ok[0]
        L.append(f"**By fast score: {describe(b)}.** Single fast runs: confirm the top trials "
                 "(--confirm-top-k) before choosing.")
        if len(ok) > 1:
            L.append(f"Top 5 span {_fmt(abs(ok[0]['score'] - ok[min(4, len(ok) - 1)]['score']))} in score.")
    else:
        L.append("No complete trial yet.")
    base = [r for r in rows if r["baseline"]]
    if base:
        r0 = base[0]
        L.append("")
        L.append(f"Trial 0 (current best model, train-script defaults): {r0['state']}, score {_fmt(r0['score'])}, "
                 f"rank {r0['rank'] if r0['rank'] is not None else '-'} of {len(ok)}.")
    L.append("")

    if res["refits"]:
        L.append("## Refits (full training split)")
        L.append("")
        L.append("| refit score | trial | arch | state | best epoch | fast score | confirmed | minutes |")
        L.append("|---:|---:|---|---|---:|---:|---|---:|")
        for n, rf in sorted(res["refits"].items(), key=lambda kv: (
                safe_float(kv[1].get("objective_value")) is None,
                -(safe_float(kv[1].get("objective_value")) or 0.0) if maximize else
                (safe_float(kv[1].get("objective_value")) or 0.0))):
            r = by_trial.get(n, {})
            conf = (f"{_fmt(r.get('confirm_mean'))} ± {_fmt(r.get('confirm_sem'))}"
                    if r.get("confirm_mean") is not None else "-")
            L.append(f"| {_fmt(safe_float(rf.get('objective_value')))} | {n} | {r.get('arch', arch_tag(rf.get('hparams') or {}))} | "
                     f"{'complete' if rf.get('returncode') == 0 else 'failed'} | "
                     f"{(rf.get('best_row') or {}).get('epoch', '-')} | {_fmt(r.get('score'))} | {conf} | "
                     f"{_fmt((rf.get('seconds') or 0) / 60, 0)} |")
        L.append("")

    if confirmed:
        p = cm.get("protocol") or {}
        L.append("## Confirmed ranking")
        L.append("")
        if p:
            L.append(f"Re-runs with new seeds: {p.get('epochs')} epochs, "
                     f"{'all' if p.get('max_train_events', -1) <= 0 else p.get('max_train_events')} training events, "
                     f"global batch {p.get('global_batch')}"
                     + ("; the fast trial counts as one seed." if cm.get("includes_fast_value") else
                        "; higher fidelity than the fast trials, so only the confirm runs count."))
            L.append("")
        L.append("| rank | trial | arch | mean | ± sem | n | fast score | refit |")
        L.append("|---:|---:|---|---:|---:|---:|---:|---:|")
        for r in confirmed:
            L.append(f"| {r['confirm_rank']} | {r['trial']} | {r['arch']} | {_fmt(r['confirm_mean'])} | "
                     f"{_fmt(r['confirm_sem'])} | {r['confirm_n']} | {_fmt(r['score'])} | {_fmt(r['refit_score'])} |")
        L.append("")

    if ok:
        L.append("## Per architecture (complete fast trials)")
        L.append("")
        L.append("| arch | complete | pruned | best score | best trial | median | trainable params | batch |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---|")
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r["arch"], []).append(r)
        summary = []
        for arch, rs in groups.items():
            done = [r for r in rs if r["ok"]]
            batches = sorted({r["global_batch"] for r in rs if r["global_batch"] is not None})
            btxt = ",".join(str(b) for b in batches) if batches else "-"
            if not done:
                summary.append((arch, 0, sum(r["pruned"] for r in rs), None, None, None, None, btxt))
                continue
            best = done[0]  # rows are sorted by score
            summary.append((arch, len(done), sum(r["pruned"] for r in rs), best["score"], best["trial"],
                            _median([r["score"] for r in done]), best["trainable_params"], btxt))
        summary.sort(key=lambda s: (s[3] is None, -(s[3] or 0.0) if maximize else (s[3] or 0.0)))
        for arch, n_ok, n_pr, best, bt, med, npar, btxt in summary:
            L.append(f"| {arch} | {n_ok} | {n_pr} | {_fmt(best)} | {bt if bt is not None else '-'} | {_fmt(med)} | "
                     f"{npar if npar is not None else '-'} | {btxt} |")
        L.append("")

        L.append(f"## Fast trials (top {min(LEADERBOARD_TOP, len(ok))} of {len(ok)}; all trials in leaderboard.csv)")
        L.append("")
        L.append("| rank | trial | arch | score | best epoch | window | AUC | epochs | params | batch | lr | "
                 "hidden x layers | loss | minutes | confirmed | refit |")
        L.append("|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|")
        for r in ok[:LEADERBOARD_TOP]:
            h = r["hparams"]
            win = r["window_epochs"]
            L.append(f"| {r['rank']} | {r['trial']}{' (defaults)' if r['baseline'] else ''} | {r['arch']} | "
                     f"{_fmt(r['score'])} | {_fmt(r['best_epoch_value'])} @ {r['best_epoch']} | "
                     f"{f'{win[0]}-{win[1]}' if win else '-'} | {_fmt(r['auc'])} | {r['n_epochs']} | "
                     f"{r['trainable_params'] if r['trainable_params'] is not None else '-'} | "
                     f"{r['global_batch'] if r['global_batch'] is not None else '-'} | {_fmt(h.get('lr'))} | "
                     f"{h.get('hidden_dim')} x {h.get('layers')} | {h.get('loss_type')} | {_fmt(r['minutes'], 0)} | "
                     f"{_fmt(r['confirm_mean'])} | {_fmt(r['refit_score'])} |")
        L.append("")

    if study is not None:
        imp, method = _param_importances(study)
        if imp:
            L.append(f"## Parameter importances ({method}, complete trials)")
            L.append("")
            L.append("| parameter | importance |")
            L.append("|---|---:|")
            for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
                L.append(f"| {k} | {v:.3f} |")
            L.append("")
    return "\n".join(L) + "\n"


def _leaderboard_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    cols = ["rank", "trial", "state", "baseline", "arch", "score", "best_epoch_value", "best_epoch", "window_epochs",
            "auc", "val_loss", "n_epochs", "minutes", "trainable_params", "confirm_rank", "confirm_mean",
            "confirm_sem", "confirm_n", "refit_score", "refit_state", "refit_best_epoch", "global_batch",
            "data_format", "indexing_seconds"]
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + list(HPARAM_COLUMNS))
        for r in rows:
            win = r["window_epochs"]
            vals = [r.get(c) for c in cols]
            vals[cols.index("window_epochs")] = f"{win[0]}-{win[1]}" if win else ""
            w.writerow(["" if v is None else v for v in vals] + ["" if r["hparams"].get(k) is None else r["hparams"][k]
                                                               for k in HPARAM_COLUMNS])
    os.replace(tmp, path)


def write_leaderboard(out_dir: Path, args: argparse.Namespace, study=None, print_top: int = 0) -> Optional[Path]:
    """Rewrite <out-dir>/leaderboard.md and leaderboard.csv; optionally print the top trials. Never raises (a
    failure here must not stop the tuning)."""
    try:
        with LEADERBOARD_LOCK:
            res = collect_leaderboard_rows(out_dir, args, study)
            md = out_dir / "leaderboard.md"
            _write_atomic(md, _leaderboard_markdown(res, args, study))
            _leaderboard_csv(res["rows"], out_dir / "leaderboard.csv")
        if print_top > 0:
            ok = [r for r in res["rows"] if r["ok"]]
            _log(f"  Leaderboard (top {min(print_top, len(ok))} of {len(ok)} complete; {md}):")
            for r in ok[:print_top]:
                _log(f"    #{r['rank']:<3} trial {r['trial']:<4} {args.compare_metric}={_fmt(r['score'], 5)}  "
                     f"{r['arch']}" + ("  (train-script defaults)" if r["baseline"] else ""))
        return md
    except Exception as exc:  # pragma: no cover - reporting only
        _log(f"[leaderboard] not written: {type(exc).__name__}: {exc}")
        return None


def run_protocol(args: argparse.Namespace, phase: str) -> Dict[str, Any]:
    """Training protocol of a phase ('fast', 'confirm', 'refit'): epochs, training events, GPUs, global batch."""
    n_dev = int(getattr(args, "n_devices", 0))
    if phase == "fast":
        epochs, events, gpus = args.fast_epochs, args.fast_max_train_events, args.fast_gpus_per_trial
    elif phase == "confirm":
        epochs = args.fast_epochs if args.confirm_epochs is None else args.confirm_epochs
        events = args.fast_max_train_events if args.confirm_max_train_events is None else args.confirm_max_train_events
        gpus = args.fast_gpus_per_trial if args.confirm_gpus_per_trial is None else args.confirm_gpus_per_trial
    elif phase == "refit":
        epochs, events, gpus = args.refit_epochs, args.refit_max_train_events, args.refit_gpus_per_trial
    else:
        raise ValueError(phase)
    world = phase_world(gpus, n_dev)
    return {"epochs": int(epochs), "max_train_events": int(events), "gpus_per_trial": int(gpus), "world": world,
            "batch_size_per_rank": per_rank_batch_size(args, world), "global_batch": effective_global_batch(args, world)}


def _same_training(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return all(a[k] == b[k] for k in ("epochs", "max_train_events", "global_batch"))


def _protocol_tag(p: Dict[str, Any]) -> str:
    n = p["max_train_events"]
    return f"e{p['epochs']}_n{n if n > 0 else 'all'}_g{p['global_batch']}"


def mean_std_sem(vals: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    n = len(vals)
    if n == 0:
        return None, None, None
    mean = sum(vals) / n
    if n < 2:
        return mean, None, None
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
    return mean, std, std / n ** 0.5


def confirm_top_trials(
    *,
    args: argparse.Namespace,
    trials: list,
    allocator: "GPUAllocator",
    train_script: Path,
    split_file: Path,
    cwd: Path,
    ckpt_dir: Path,
    log_dir: Path,
    out_dir: Path,
    maximize: bool,
    on_update: Optional[Callable[[], None]] = None,
) -> list:
    """Re-run the top --confirm-top-k fast trials with --confirm-seeds seeds each (in parallel like the fast phase)
    and rank the configurations by the mean of their score over the seeds. One fast run is a single draw of the
    training noise, and the best trials of a study often differ by less than that noise.
    Fast protocol (default): the original fast trial counts as one seed, the new seeds are trainer seed + 100*s,
    s = 1..N. Higher fidelity (--confirm-max-train-events / --confirm-epochs / --confirm-gpus-per-trial with another
    global batch): only the N confirm runs count, seeds trainer seed + 100*s, s = 0..N-1 (s = 0: the fast trials'
    seed). Finished runs are read back from <out-dir>/confirm.jsonl (a restarted tuner only runs what is missing);
    <out-dir>/confirm_summary.json is rewritten after every run. Returns the ranked trials."""
    from concurrent.futures import ThreadPoolExecutor

    k, n_seeds = int(args.confirm_top_k), int(args.confirm_seeds)
    candidates = trials[:k]
    if not candidates or n_seeds < 1:
        return candidates
    proto = run_protocol(args, "confirm")
    fast_proto = run_protocol(args, "fast")
    same = _same_training(proto, fast_proto)
    seeds = list(range(1, n_seeds + 1)) if same else list(range(n_seeds))
    tag = "" if same else _protocol_tag(proto) + "_"
    results_path = out_dir / "confirm.jsonl"
    done = {str(r.get("run_id")): r for r in load_jsonl(results_path)
            if r.get("returncode") == 0 and r.get("objective_value") is not None}
    lock = threading.Lock()
    _banner(f"CONFIRM PHASE  —  top {len(candidates)} trials x {n_seeds} seed(s)  "
            + (f"(fast protocol: {proto['epochs']} epochs, max_train_events={proto['max_train_events']}; "
               "the fast trial counts as one seed)" if same else
               f"(higher fidelity: {proto['epochs']} epochs, max_train_events={proto['max_train_events']}, "
               f"global batch {proto['global_batch']}; fast trials: {fast_proto['epochs']} epochs, "
               f"max_train_events={fast_proto['max_train_events']}, global batch {fast_proto['global_batch']})")
            + ("; auto batch: every run keeps its trial's global batch" if auto_batch(args) else ""))

    jobs = [(t, s) for t in candidates for s in seeds]
    per_trial: Dict[int, List[float]] = {t.number: ([float(t.value)] if same else []) for t in candidates}
    n_finished = [0]
    summary_path = out_dir / "confirm_summary.json"

    def rank_and_write(final: bool) -> list:
        def key(t):
            m = mean_std_sem(per_trial[t.number])[0]
            return m if m is not None else (-math.inf if maximize else math.inf)

        ranked = sorted(candidates, key=key, reverse=maximize)
        rows = []
        for rank, t in enumerate(ranked, start=1):
            vals = per_trial[t.number]
            mean, std, sem = mean_std_sem(vals)
            h = _refit_hparams_from_trial(t, args)
            rows.append({"rank": rank, "trial_number": t.number, "baseline": is_baseline_trial(t),
                         "arch": arch_tag(h), "n": len(vals), "mean": mean, "std": std, "sem": sem,
                         "fast_value": t.value, "values": vals, "global_batch": t.user_attrs.get("global_batch_size"),
                         "hparams": {k: v for k, v in h.items() if not k.startswith("_")}})
        first_vs_second = None
        if len(rows) > 1 and rows[0]["sem"] is not None and rows[1]["sem"] is not None:
            diff = rows[0]["mean"] - rows[1]["mean"]
            err = (rows[0]["sem"] ** 2 + rows[1]["sem"] ** 2) ** 0.5
            first_vs_second = {"diff": diff, "err": err, "sigma": abs(diff) / err if err > 0 else None}
        summary = {"objective": args.compare_metric, "target_fpr": args.target_fpr,
                   "objective_window": args.objective_window, "protocol": proto,
                   "fidelity": "fast" if same else "higher", "includes_fast_value": same,
                   "complete": final or n_finished[0] == len(jobs), "runs_finished": n_finished[0],
                   "runs_total": len(jobs), "first_vs_second": first_vs_second, "ranking": rows}
        tmp = summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, summary_path)
        return ranked

    def run_one(job_idx: int) -> None:
        t, s = jobs[job_idx]
        run_id = f"confirm_{tag}t{t.number:05d}_s{s}"
        hparams = refit_hparams_from_trial(t, args)
        hparams["_seed"] = int(args.trainer_seed) + 100 * s
        if run_id in done:
            value, rc = float(done[run_id]["objective_value"]), 0
            _log(f"[confirm] {run_id}: done earlier → {args.compare_metric}={value}")
        else:
            save_path = build_trial_ckpt_path(ckpt_dir, args.save, run_id)
            log_path = log_dir / f"{run_id}.log"
            with allocator.acquire(proto["gpus_per_trial"]) as gpu_ids:
                nproc = torchrun_nproc(proto["gpus_per_trial"], gpu_ids)
                cmd = build_command(
                    python_exe=os.environ.get("PYTHON", "python"), train_script=train_script,
                    data_glob=args.data_glob, split_file=split_file, save_path=save_path, save_base=args.save,
                    run_id=run_id, epochs=proto["epochs"], max_train_events=proto["max_train_events"],
                    num_workers=args.num_workers, hparams=hparams, args=args, nproc=nproc,
                    master_port=pick_master_port(args.master_port_base, 1500 + job_idx),
                    rdzv_id=f"{re.sub(r'[^A-Za-z0-9_.-]', '_', args.study_name)}_{run_id}", resume=False,
                )
                t0 = time.time()
                rc = run_one_training(cmd=cmd, log_path=log_path, gpu_ids=gpu_ids, cwd=cwd, save_dir=ckpt_dir,
                                      trial_label=f"Confirm trial {t.number} seed {s}")
                seconds = time.time() - t0
            parsed = parse_best_metrics_from_log(log_path, args.compare_metric, args.objective_window)
            value = parsed["objective_value"]
            if value is None:
                value = extract_objective_from_ckpt(read_ckpt_metrics(save_path), args.compare_metric)
            with lock:
                append_jsonl(results_path, {
                    "phase": "confirm", "source_trial_number": t.number, "source_objective_value": t.value,
                    "seed_idx": s, "seed": hparams["_seed"], "run_id": run_id, "returncode": rc, "seconds": seconds,
                    "protocol": proto, "objective_name": args.compare_metric, "objective_value": value,
                    "objective_window": args.objective_window, "best_epoch_value": parsed["best_epoch_value"],
                    "window_epochs": parsed["window_epochs"], "best_row": parsed["best_row"],
                    "window_row": parsed["window_row"], "n_epochs_seen": parsed["n_epochs_seen"],
                    "run_info": parsed["run_info"],
                    "hparams": {kk: v for kk, v in hparams.items() if not kk.startswith("_")},
                    "ckpt_path": str(save_path), "log_path": str(log_path),
                })
            _log(f"[confirm] trial {t.number} seed {s}: rc={rc} {args.compare_metric}={value}")
        with lock:
            n_finished[0] += 1
            if rc == 0 and value is not None:
                per_trial[t.number].append(float(value))
            rank_and_write(final=False)
        if on_update is not None:
            on_update()

    with ThreadPoolExecutor(max_workers=max(1, int(args.n_jobs))) as pool:
        list(pool.map(run_one, range(len(jobs))))

    ranked = rank_and_write(final=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _banner(f"CONFIRMED RANKING  ({args.compare_metric}: mean over seeds, sem = std/sqrt(n))")
    _log(f"{'rank':>4} {'trial':>6} {'n':>2} {'mean':>9} {'sem':>8} {'std':>8} {'fast':>9}  {'arch':<40} values")
    for row in summary["ranking"]:
        f = lambda v: f"{v:.5f}" if v is not None else "-"
        tag0 = "  (trial 0: train-script defaults)" if row["baseline"] else ""
        _log(f"{row['rank']:>4} {row['trial_number']:>6} {row['n']:>2} {f(row['mean']):>9} {f(row['sem']):>8} "
             f"{f(row['std']):>8} {f(row['fast_value']):>9}  {row['arch']:<40} "
             + " ".join(f"{v:.5f}" for v in row["values"]) + tag0)
    fvs = summary["first_vs_second"]
    if fvs is not None:
        _log(f"[confirm] #1 vs #2: diff={fvs['diff']:+.5f} +- {fvs['err']:.5f} "
             f"({fvs['sigma']:.1f} sigma)" if fvs["sigma"] is not None else f"[confirm] #1 vs #2: diff={fvs['diff']:+.5f}")
    _log(f"[confirm] summary: {summary_path}  runs: {results_path}")
    if on_update is not None:
        on_update()
    return ranked


def study_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    """The settings that define the score of a fast trial: they must stay the same within a study."""
    return {"compare_metric": args.compare_metric, "target_fpr": float(args.target_fpr),
            "objective_window": int(args.objective_window), "fast_epochs": int(args.fast_epochs),
            "fast_max_train_events": int(args.fast_max_train_events),
            "fast_global_batch": batch_protocol(args)}


def check_study_protocol(study, args: argparse.Namespace, results_path: Path) -> None:
    """Stored as the study user attr 'protocol' at the first run of a study; a later run (more trials, --refit-only,
    confirm) with other values stops here, since its scores or confirm runs would not be comparable with the stored
    trials. A study from before this record has none: its trials used the single best epoch (window 1)."""
    proto = study_protocol(args)
    stored = study.user_attrs.get("protocol")
    finished = [t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    if stored is None:
        if not finished:
            study.set_user_attr("protocol", proto)
            _log("[protocol] stored in the study: " + " ".join(f"{k}={v}" for k, v in proto.items()))
            return
        stored = {"objective_window": 1}
        recs = [r for r in load_jsonl(results_path) if r.get("phase", "fast") == "fast"]
        if recs:
            stored["compare_metric"] = recs[0].get("objective_name")
            stored["target_fpr"] = safe_float(recs[0].get("target_fpr"))
        _log("[protocol] study from before the protocol record: checking only "
             + " ".join(f"{k}={v}" for k, v in stored.items() if v is not None))
    diff = {k: (stored.get(k), proto[k]) for k in proto if stored.get(k) is not None and stored.get(k) != proto[k]}
    if diff:
        flags = {"compare_metric": "--compare-metric", "target_fpr": "--target-fpr",
                 "objective_window": "--objective-window", "fast_epochs": "--fast-epochs",
                 "fast_max_train_events": "--fast-max-train-events",
                 "fast_global_batch": "--global-batch-size (or --batch-size x --fast-gpus-per-trial; --batch-size "
                                      "auto: --auto-batch-* / --lr-batch-scaling / --lr-ref-batch / --lr-scaled-max)"}
        raise SystemExit("[protocol] study '" + str(args.study_name) + "' was run with other settings: "
                         + "; ".join(f"{flags[k]}: study {a}, this run {b}" for k, (a, b) in diff.items())
                         + ". Pass the study's values (or use a new --study-name / --out-dir).")


def validate_inputs(args: argparse.Namespace) -> None:
    if not Path(args.train_script).exists():
        raise SystemExit(f"train script does not exist: {args.train_script}")
    if not Path(args.split_file).exists():
        raise SystemExit(f"split file does not exist: {args.split_file}")
    if not glob.glob(args.data_glob):
        raise SystemExit(f"data_glob matched 0 files: {args.data_glob}")
    if (args.normalize_node_features or args.normalize_edge_features) and not args.feature_stats_json:
        # Same fallback as the trainer: normalization_stats_raw.json beside the split file.
        default_stats = Path(args.split_file).resolve().parent / "normalization_stats_raw.json"
        if default_stats.is_file():
            args.feature_stats_json = str(default_stats)
            print(f"[i] feature stats: {default_stats} (beside the split file)")
    if (args.normalize_node_features or args.normalize_edge_features) and not args.feature_stats_json:
        raise SystemExit(
            "--feature-stats-json is required when --normalize-node-features or "
            "--normalize-edge-features is enabled; the trainer embeds these stats "
            "as model buffers for inference/ONNX export."
        )
    if args.feature_stats_json and not Path(args.feature_stats_json).exists():
        raise SystemExit(f"feature stats JSON does not exist: {args.feature_stats_json}")
    if args.gat_edge_attn and args.gatv2_edge_attn:
        raise SystemExit(
            "--gat-edge-attn and --gatv2-edge-attn select different GAT "
            "architectures and cannot be enabled together."
        )
    # The GAT edge-attention flags only reach GAT trials (build_hparams sets them per
    # trial), so they are harmless with another --fixed-layer-type.
    if args.fixed_relation_mp is False and args.fixed_edge_update not in ("none", None):
        raise SystemExit("--fixed-edge-update needs relation-mp trials: add --fixed-relation-mp true (or any).")
    if args.fixed_relation_mp is False and args.calo_context != "two_way":
        raise SystemExit("--calo-context one_way needs relation-mp trials: add --fixed-relation-mp true (or any).")
    if args.fixed_edge_update not in (None, "none") and args.fixed_edge_update not in _edge_update_choices(args):
        raise SystemExit(f"--fixed-edge-update {args.fixed_edge_update} needs the segment-segment edges "
                         "(and the tracklets for seg_tracklet).")
    if args.gatv2_edge_attn is False and args.fixed_relation_mp is not False:
        print("[i] note: --relation-mp trials use their own GATv2 attention (--no-gatv2-edge-attn does not apply)")

def build_optuna_storage(args: argparse.Namespace, out_dir: Path):
    """
    Build an Optuna storage backend that is safe for the requested parallelism.

    Important:
      - SQLite RDBStorage is fragile with n_jobs > 1.
      - JournalStorage is a better file-based backend for single-node parallel jobs.
      - For multi-node / shared filesystem production runs, use PostgreSQL/MySQL via
        --storage-url.
    """
    backend = args.storage_backend

    if args.storage_url:
        if backend not in ("auto", "rdb"):
            raise SystemExit("--storage-url can only be used with --storage-backend auto/rdb")
        print(f"[i] optuna storage: RDBStorage url={args.storage_url!r}")
        return optuna.storages.RDBStorage(url=args.storage_url)

    if backend == "auto":
        backend = "journal" if int(args.n_jobs) > 1 else "sqlite"

    if backend == "journal":
        storage_path = (
            Path(args.storage_path).resolve()
            if args.storage_path
            else out_dir / "optuna.journal"
        )
        try:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
        except Exception as e:
            raise SystemExit(
                "This Optuna installation does not expose JournalStorage / "
                "JournalFileBackend. Either upgrade Optuna or run with "
                "--storage-backend sqlite --n-jobs 1."
            ) from e

        print(f"[i] optuna storage: JournalStorage path={storage_path}")
        return JournalStorage(JournalFileBackend(str(storage_path)))

    if backend == "sqlite":
        storage_path = (
            Path(args.storage_path).resolve()
            if args.storage_path
            else out_dir / "optuna.db"
        )
        if int(args.n_jobs) > 1:
            print("[w] SQLite + n_jobs > 1 may fail with 'database is locked'; "
                  "prefer --storage-backend journal or --storage-url postgresql/mysql.")
        print(f"[i] optuna storage: SQLite RDBStorage path={storage_path}")
        return optuna.storages.RDBStorage(
            url=f"sqlite:///{storage_path}",
            engine_kwargs={"connect_args": {"timeout": float(args.sqlite_timeout)}},
        )

    raise SystemExit("--storage-backend rdb requires --storage-url, e.g. postgresql://...")


def main() -> None:
    # Emit the exact invocation so shell logs captured via tee include reproducible command details.
    invoked_cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
    print(f"[cmd] {invoked_cmd}")
    print(f"[cmd] started_at={datetime.now().isoformat(timespec='seconds')} cwd={os.getcwd()}")

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train-script", required=True)
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--storage-path", default=None,
        help="File storage path. For sqlite: <out-dir>/optuna.db. For journal: <out-dir>/optuna.journal.")
    ap.add_argument("--storage-backend", default="auto", choices=["auto", "sqlite", "journal", "rdb"],
        help="Optuna storage backend. auto uses journal when n_jobs > 1, sqlite otherwise.")
    ap.add_argument("--storage-url", default=None, help="Full RDB URL, e.g. postgresql://user:pass@host/db")
    ap.add_argument("--sqlite-timeout", type=float, default=120.0, help="SQLite lock wait timeout in seconds.")
    ap.add_argument("--study-name", default="dv_classifier_tpr_fpr1")
    ap.add_argument("--compare-metric", default="val_tpr_at_target_fpr", choices=["val_loss", "val_auc", "val_acc", "val_f1", "val_balanced_acc", "val_tpr_at_target_fpr"])
    ap.add_argument("--target-fpr", type=float, default=0.001,
                    help="FPR working point of val_tpr_at_target_fpr: 0.001 = 0.1%% background (default since "
                         "2026-11-24; before: 0.01).")
    ap.add_argument("--objective-window", type=int, default=1,
                    help="Trial score = best mean of the metric over this many consecutive epochs (1: the single "
                         "best epoch, the former score; 3 is less biased by the epoch-to-epoch noise). The same in "
                         "the fast, confirm and refit phases; stored in the study.")
    ap.add_argument("--n-trials", type=int, default=50,
                    help="Total number of completed fast trials desired in the study. On resume, only the missing trials are launched.")
    ap.add_argument("--n-trials-extra", action="store_true", default=False,
                    help="Interpret --n-trials as Optuna's native count of additional trials for this invocation.")
    ap.add_argument("--n-startup-trials", type=int, default=10)    
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--fast-epochs", type=int, default=40)
    ap.add_argument("--fast-max-train-events", type=int, default=-1)
    ap.add_argument("--fast-gpus-per-trial", type=int, default=1)
    ap.add_argument("--trial-launch-stagger-seconds", type=float, default=15.0,
                    help="Delay between concurrent trial launches to avoid HDF5/EOS/DDP startup stampedes.")
    ap.add_argument("--prune", action="store_true", default=False,
                    help="Stop fast trials whose running score is outside the best --prune-percentile %% of the "
                         "complete trials at the same epoch (Optuna PercentilePruner); trial 0 is never pruned.")
    ap.add_argument("--prune-warmup-epochs", type=int, default=10, help="No pruning before this epoch.")
    ap.add_argument("--prune-startup-trials", type=int, default=8,
                    help="No pruning before this many complete trials; also the minimum number of complete trials "
                         "with a value at the epoch.")
    ap.add_argument("--prune-percentile", type=float, default=50.0,
                    help="Trials outside the best this %% at the same epoch are stopped (50 = median pruner).")
    ap.add_argument("--refit-top-k", type=int, default=0)
    ap.add_argument("--confirm-top-k", "--confirm-topk", dest="confirm_top_k", type=int, default=0,
                    help="Before the refit: re-run the top K fast trials with --confirm-seeds new seeds each (fast "
                         "protocol) and rank them by the mean over the seeds (the fast trial counts as one); the "
                         "refit then takes its --refit-top-k from this ranking. 0: off.")
    ap.add_argument("--confirm-epochs", type=int, default=None, help="Epochs of a confirm run (default: --fast-epochs).")
    ap.add_argument("--confirm-max-train-events", type=int, default=None,
                    help="Training events of a confirm run (default: --fast-max-train-events). More events than the "
                         "fast trials = a higher-fidelity check (only the confirm runs are then averaged).")
    ap.add_argument("--confirm-gpus-per-trial", type=int, default=None,
                    help="GPUs per confirm run (default: --fast-gpus-per-trial).")
    ap.add_argument("--confirm-seeds", type=int, default=2,
                    help="New seeds per confirmed config (trainer --seed = train-script default + 100 * i).")
    ap.add_argument("--refit-epochs", type=int, default=100)
    ap.add_argument("--refit-max-train-events", type=int, default=-1)
    ap.add_argument("--refit-gpus-per-trial", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=tuner_batch_arg, default=None,
                    help="Graphs per GPU step passed to the trainer, the same in every phase (so the global batch "
                         "grows with the GPUs of a run). Default: the train script's default (64, as the best "
                         "model); the old studies used 16. Not with --global-batch-size. "
                         "'auto': every searched fast trial probes --auto-batch-candidates per GPU for its own "
                         "architecture (the trainer takes the fastest that fits --auto-batch-mem-frac), --lr is "
                         "scaled from the global batch --lr-ref-batch (--lr-batch-scaling, default sqrt), and the "
                         "trial's confirm runs and refit keep its global batch (split over their GPUs). Trial 0 "
                         "runs at --lr-ref-batch.")
    ap.add_argument("--auto-batch-candidates", type=int, nargs="+", default=[16, 32, 64, 128],
                    help="--batch-size auto: per-GPU batch sizes probed by the trainer.")
    ap.add_argument("--auto-batch-mem-frac", type=float, default=0.7,
                    help="--batch-size auto: the worst-case batch must stay under this fraction of the GPU memory.")
    ap.add_argument("--auto-batch-tolerance", type=float, default=0.05,
                    help="--batch-size auto: sizes within this fraction of the best throughput count as equal; the "
                         "smallest wins.")
    ap.add_argument("--lr-batch-scaling", default=None, choices=["none", "sqrt", "linear"],
                    help="Forwarded to the trainer (all phases): --lr is defined at the global batch --lr-ref-batch "
                         "and scaled to the global batch of the run (the EMA decay too). Default: sqrt with "
                         "--batch-size auto, none otherwise.")
    ap.add_argument("--lr-ref-batch", type=int, default=None,
                    help="Global batch at which the searched --lr is defined (default: the trainer's, 64).")
    ap.add_argument("--lr-scaled-max", type=float, default=None,
                    help="Forwarded as the trainer's --lr-max: cap on the batch-scaled lr (default: none). Not the "
                         "search range (--lr-min / --lr-max, defined at --lr-ref-batch).")
    ap.add_argument("--global-batch-size", type=int, default=None,
                    help="Graphs per optimizer step summed over the GPUs of a run, the same in every phase: a run "
                         "on k GPUs gets N/k per GPU, so the refit sees the batch the fast trials were tuned at "
                         "(the old study refitted at 4x the fast-trial batch). Stored in the study.")
    ap.add_argument("--max-open-h5-files", type=int, default=16,
                    help="Maximum HDF5 files open per trainer process.")
    ap.add_argument("--pin-memory", action="store_true", default=False)
    ap.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    ap.add_argument("--feature-stats-json", default=None,
                    help="Stats JSON loaded by the trainer and embedded as model normalization buffers.")
    ap.add_argument("--feature-norm-kind", default="standard", choices=["standard", "robust"])
    ap.add_argument("--normalize-node-features", action="store_true", default=True,
                    help="Ask the trainer to normalize node features inside the model (default, as the best model).")
    ap.add_argument("--no-normalize-node-features", dest="normalize_node_features", action="store_false")
    ap.add_argument("--normalize-edge-features", action="store_true", default=True,
                    help="Ask the trainer to normalize edge features inside the model (default, as the best model).")
    ap.add_argument("--no-normalize-edge-features", dest="normalize_edge_features", action="store_false")
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--master-port-base", type=int, default=29500)
    ap.add_argument("--save", default="dv_classifier.pt",
                    help="Base checkpoint filename passed to the trainer. The trainer appends _<run_id>.pt.")
    ap.add_argument("--resume-skip-existing", dest="resume_skip_existing", action="store_true", default=True,
                    help="If the expected checkpoint for a trial already exists, reuse its saved metric instead of rerunning it.")
    ap.add_argument("--no-resume-skip-existing", dest="resume_skip_existing", action="store_false")
    ap.add_argument("--report-only", action="store_true", default=False,
                    help="Only rewrite <out-dir>/leaderboard.md / .csv from the study and the jsonl files, then exit "
                         "(no training; also while the study runs).")
    ap.add_argument("--refit-only", action="store_true", default=False,
                    help="Skip fast Optuna optimization and only refit the current best completed trials.")
    ap.add_argument("--extra-train-args", default="", help="Extra quoted arguments appended to every train command")
    ap.add_argument("--use-tracks", action="store_true", default=False,
                    help="Train every trial with the track block of the graphs (train_DisplacedVertex.py --use-tracks)")
    ap.add_argument("--no-use-segment-edges", dest="use_segment_edges", action="store_false", default=True,
                    help="Train without the segment-segment edges (on by default)")
    ap.add_argument("--no-use-tracklets", dest="use_tracklets", action="store_false", default=True,
                    help="Train without the MS tracklet nodes (on by default)")
    ap.add_argument("--no-amp", action="store_true", default=False)
    ap.add_argument("--no-ema", action="store_true", default=False)
    ap.add_argument("--lr-min", type=float, default=5e-5, help="Old: 1e-5.")
    ap.add_argument("--lr-max", type=float, default=6e-4, help="Old: 1e-3.")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 256], help="Old: 64 128 256.")
    ap.add_argument("--layers-min", type=int, default=3, help="Old: 2.")
    ap.add_argument("--layers-max", type=int, default=6)
    ap.add_argument("--dropout-min", type=float, default=0.0)
    ap.add_argument("--dropout-max", type=float, default=0.12, help="Old: 0.3.")
    ap.add_argument("--weight-decay-min", type=float, default=1e-6, help="Old: 1e-5.")
    ap.add_argument("--weight-decay-max", type=float, default=2e-3, help="Old: 5e-2.")
    ap.add_argument("--edge-dropout-min", type=float, default=0.0)
    ap.add_argument("--edge-dropout-max", type=float, default=0.2)
    ap.add_argument("--feat-noise-min", type=float, default=5e-4, help="Old: 0.")
    ap.add_argument("--feat-noise-max", type=float, default=0.01, help="Old: 0.03.")
    ap.add_argument("--loss-types", nargs="+", default=["bce", "focal", "asymmetric_focal"], choices=list(LOSS_TYPES),
                    help="Loss functions Optuna may sample. Old: all four (bce_smooth: 0/16 in the top 20%%).")
    ap.add_argument("--fixed-loss-type", default=None, choices=list(LOSS_TYPES))
    ap.add_argument("--label-smoothing-min", type=float, default=0.01, help="Old: 0.")
    ap.add_argument("--label-smoothing-max", type=float, default=0.05)
    ap.add_argument("--focal-gamma-min", type=float, default=0.5)
    ap.add_argument("--focal-gamma-max", type=float, default=2.0, help="Old: 5.")
    ap.add_argument("--focal-alpha-min", type=float, default=0.1)
    ap.add_argument("--focal-alpha-max", type=float, default=0.9)
    ap.add_argument("--asym-gamma-pos-min", type=float, default=0.0)
    ap.add_argument("--asym-gamma-pos-max", type=float, default=1.0, help="Old: 2.")
    ap.add_argument("--asym-gamma-neg-min", type=float, default=1.0)
    ap.add_argument("--asym-gamma-neg-max", type=float, default=3.5, help="Old: 6.")
    ap.add_argument("--fixed-layer-type", choices=["mpnn", "edge_residual", "sage_residual", "gat_residual", "any"],
                    default="gat_residual", help="'any' searches the layer type (old default).")
    ap.add_argument("--gat-edge-attn", action="store_true", default=False,
                    help="Enable edge_attr encoder in GAT attention logits for all GAT trials.")
    ap.add_argument(
        "--gatv2-edge-attn",
        action="store_true",
        default=True,
        help=(
            "Enable nonlinear source-destination-edge GATv2 attention for all "
            "GAT trials (default, as the best model)."
        ),
    )
    ap.add_argument("--no-gatv2-edge-attn", dest="gatv2_edge_attn", action="store_false")
    ap.add_argument("--gat-heads-choices", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--fixed-pool", choices=["mean", "max", "sum", "meanmax", "any"], default="meanmax",
                    help="'any' searches mean/meanmax (old default).")
    ap.add_argument("--fixed-fourier", default="false",
                    help="true/false, or 'any' to search it (old default).")
    ap.add_argument("--fixed-pos-weight", default="auto", help="auto/none, or 'any' to search it (old default).")
    ap.add_argument("--baseline-trial", dest="baseline_trial", action="store_true", default=True,
                    help="Trial 0 = train script defaults (the best model); added on top of --n-trials.")
    ap.add_argument("--no-baseline-trial", dest="baseline_trial", action="store_false")
    # Architecture options of the trainer (GAT trials; the defaults give the legacy model). 'any' searches an option.
    ap.add_argument("--fixed-block", choices=list(BLOCK_CHOICES) + ["any"], default="legacy",
                    help="Trainer --block for all GAT trials: legacy or prenorm (pre-LayerNorm blocks + FFN).")
    ap.add_argument("--ffn-mult", type=int, default=2, help="Trainer --ffn-mult (prenorm feed-forward width).")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="Trainer --grad-clip for all trials (<= 0: off).")
    ap.add_argument("--fixed-relation-mp", default="false",
                    help="true/false: trainer --relation-mp (relation-typed message passing) for all GAT trials, "
                         "or 'any' to search it.")
    ap.add_argument("--relation-aggr", choices=["attn", "attn_max"], default="attn_max",
                    help="Trainer --relation-aggr (relation-mp trials).")
    ap.add_argument("--calo-context", choices=["two_way", "one_way"], default="two_way",
                    help="Trainer --calo-context (relation-mp trials).")
    ap.add_argument("--fixed-edge-update", choices=list(EDGE_UPDATE_CHOICES) + ["any"], default="none",
                    help="Trainer --edge-update for the relation-mp trials (hidden segment / tracklet edge states).")
    ap.add_argument("--fixed-readout", choices=list(READOUT_CHOICES) + ["any"], default="global",
                    help="Trainer --readout for all trials: global, typed (per node type) or typed_attn.")
    ap.add_argument("--readout-heads", type=int, default=4, help="Trainer --readout-heads (typed_attn trials).")
    args = ap.parse_args()
    for name in ("fixed_layer_type", "fixed_pool", "fixed_pos_weight", "fixed_block", "fixed_edge_update",
                 "fixed_readout"):
        if getattr(args, name) == "any":
            setattr(args, name, None)
    args.fixed_fourier = None if str(args.fixed_fourier).lower() == "any" else \
        str(args.fixed_fourier).lower() in ("1", "true", "yes", "y")
    args.fixed_relation_mp = None if str(args.fixed_relation_mp).lower() == "any" else \
        str(args.fixed_relation_mp).lower() in ("1", "true", "yes", "y")
    trainer_defaults = read_trainer_defaults(Path(args.train_script))
    if args.global_batch_size is not None:
        if args.batch_size is not None:
            raise SystemExit("--batch-size (per GPU) and --global-batch-size (summed over the GPUs) exclude each other")
        if args.global_batch_size < 1:
            raise SystemExit("--global-batch-size must be >= 1")
    if auto_batch(args):
        if args.global_batch_size is not None:
            raise SystemExit("--batch-size auto and --global-batch-size exclude each other")
        if min(args.auto_batch_candidates) < 1:
            raise SystemExit("--auto-batch-candidates must be >= 1")
    if args.batch_size is None:
        args.batch_size = int(trainer_defaults.get("batch_size", 16))
    if args.lr_batch_scaling is None:
        args.lr_batch_scaling = "sqrt" if auto_batch(args) else "none"
    if args.lr_ref_batch is None:
        args.lr_ref_batch = int(trainer_defaults.get("lr_ref_batch", trainer_defaults.get("batch_size", 64)))
    if args.lr_ref_batch < 1 or (args.lr_scaled_max is not None and args.lr_scaled_max <= 0):
        raise SystemExit("--lr-ref-batch must be >= 1 and --lr-scaled-max > 0")
    if args.objective_window < 1:
        raise SystemExit("--objective-window must be >= 1")
    if not 0.0 < args.prune_percentile <= 100.0:
        raise SystemExit("--prune-percentile must be in (0, 100]")
    args.baseline_hparams = trainer_baseline_hparams(trainer_defaults)
    args.trainer_seed = int(trainer_defaults.get("seed", 12345))  # seed of the fast trials and refits

    out_dir = Path(args.out_dir).resolve()
    mkdir(out_dir)
    if args.storage_backend == "auto" and not args.storage_url and not args.storage_path:
        # an existing study keeps its storage file, whatever --n-jobs this run uses
        existing = [b for b, f in (("journal", "optuna.journal"), ("sqlite", "optuna.db")) if (out_dir / f).exists()]
        if len(existing) == 1:
            args.storage_backend = existing[0]
    if args.report_only:
        args.n_devices = 0
        try:
            study = optuna.load_study(study_name=args.study_name, storage=build_optuna_storage(args, out_dir))
        except Exception as exc:
            print(f"[report] no Optuna study {args.study_name!r} in {out_dir} ({type(exc).__name__}: {exc}); "
                  "leaderboard from the jsonl files only")
            study = None
        md = write_leaderboard(out_dir, args, study, print_top=10)
        print(f"[report] {md}  and  {out_dir / 'leaderboard.csv'}")
        return
    validate_inputs(args)

    devices = detect_gpus()
    args.n_devices = len(devices)
    actual_cuda_devices = int(torch.cuda.device_count())
    if devices and actual_cuda_devices < len(devices):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES lists "
            f"{len(devices)} device(s), but PyTorch can use only {actual_cuda_devices}. "
            "Check for malformed, wrapped, unavailable, or duplicate GPU/MIG identifiers. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
        )
    requested_gpus = max(int(args.fast_gpus_per_trial), int(args.refit_gpus_per_trial),
                         int(args.confirm_gpus_per_trial or 0))
    if devices and requested_gpus > actual_cuda_devices:
        raise RuntimeError(
            f"Requested up to {requested_gpus} GPU(s) per trial, but PyTorch can use only "
            f"{actual_cuda_devices}."
        )
    allocator = GPUAllocator(devices)
    print(f"[i] visible GPUs: {devices if devices else 'none; running train script without torchrun'}")
    print(f"[i] objective: {args.compare_metric}; target_fpr={args.target_fpr}; trial score = "
          + (f"best mean over {args.objective_window} consecutive epochs" if args.objective_window > 1
             else "best single epoch"))
    phases = ["fast"] + (["confirm"] if args.confirm_top_k > 0 else []) + (["refit"] if args.refit_top_k > 0 else [])
    if auto_batch(args):
        # every global batch a trial can get (candidates x fast GPUs; trial 0: --lr-ref-batch) must split evenly
        # over the GPUs of its confirm runs and refit
        fast_world = run_protocol(args, "fast")["world"]
        globals_ = sorted({int(c) * fast_world for c in args.auto_batch_candidates} | {int(args.lr_ref_batch)})
        for ph in phases:
            w = run_protocol(args, ph)["world"]
            bad = [g for g in globals_ if g % w]
            if bad:
                raise SystemExit(f"--batch-size auto: the global batches {bad} (--auto-batch-candidates x "
                                 f"{fast_world} fast GPU(s), or --lr-ref-batch) do not split over the {w} GPUs of the "
                                 f"{ph} runs: change --auto-batch-candidates / --lr-ref-batch or the GPUs per run")
        print(f"[i] batch (auto, per architecture): fast trials probe {sorted(set(args.auto_batch_candidates))} per "
              f"GPU on {fast_world} GPU(s) (mem frac {args.auto_batch_mem_frac:g}, tolerance "
              f"{args.auto_batch_tolerance:g}); trial 0: global {args.lr_ref_batch}; confirm runs and refits keep "
              f"the trial's global batch, split over their GPUs")
    else:
        batch_txt = []
        for ph in phases:
            pr = run_protocol(args, ph)  # raises if --global-batch-size does not divide
            batch_txt.append(f"{ph} {pr['world']} GPU(s) x {pr['batch_size_per_rank']} = {pr['global_batch']}")
        print(f"[i] batch ({'--global-batch-size' if args.global_batch_size is not None else '--batch-size per GPU'}): "
              + " | ".join(batch_txt))
    if args.lr_batch_scaling != "none":
        print(f"[i] lr: the searched lr is defined at global batch {args.lr_ref_batch}, scaled "
              f"{args.lr_batch_scaling} to the global batch of each run"
              + (f" (capped at {args.lr_scaled_max:g})" if args.lr_scaled_max is not None else "")
              + "; the EMA decay too")
    if args.global_batch_size is None and len({run_protocol(args, ph)["global_batch"] for ph in phases}) > 1:
        print("[w] the phases train with different global batches (--batch-size is per GPU): the learning rate of "
              "the fast trials is not the one the refit needs; --global-batch-size keeps it the same.")
    if args.prune:
        print(f"[i] pruning: stop trials outside the best {args.prune_percentile:g}% of the complete trials at the "
              f"same epoch, from epoch {args.prune_warmup_epochs}, after {args.prune_startup_trials} complete trials")
    data_paths = glob.glob(args.data_glob)
    if data_paths and not any(Path(x).name.endswith(".packed") for x in data_paths):
        print("[w] the data are H5 parts: every trial re-indexes them (old study: median 41 min per trial) and its "
              "batches mix only --max-open-h5-files files. Packed parts (DisplacedVertex_converter.py default, or "
              "pack_DisplacedVertex_dataset.py) remove both.")
    print(f"[i] loss search space: {args.loss_types if not args.fixed_loss_type else [args.fixed_loss_type]}")
    print(f"[i] search space: lr [{args.lr_min:g}, {args.lr_max:g}] log | hidden {args.hidden_dims} | "
          f"layers [{args.layers_min}, {args.layers_max}] | dropout [{args.dropout_min:g}, {args.dropout_max:g}] | "
          f"wd [{args.weight_decay_min:g}, {args.weight_decay_max:g}] log | edge_dropout "
          f"[{args.edge_dropout_min:g}, {args.edge_dropout_max:g}] | feat_noise [{args.feat_noise_min:g}, {args.feat_noise_max:g}] | "
          f"label_smoothing [{args.label_smoothing_min:g}, {args.label_smoothing_max:g}] | focal_gamma "
          f"[{args.focal_gamma_min:g}, {args.focal_gamma_max:g}] | focal_alpha [{args.focal_alpha_min:g}, {args.focal_alpha_max:g}] | "
          f"asym_gamma_pos [{args.asym_gamma_pos_min:g}, {args.asym_gamma_pos_max:g}] | asym_gamma_neg "
          f"[{args.asym_gamma_neg_min:g}, {args.asym_gamma_neg_max:g}] | gat_heads {args.gat_heads_choices}")
    print(f"[i] fixed: layer_type={args.fixed_layer_type or 'searched'} pool={args.fixed_pool or 'searched'} "
          f"fourier={'searched' if args.fixed_fourier is None else args.fixed_fourier} "
          f"pos_weight={args.fixed_pos_weight or 'searched'} gatv2_edge_attn={args.gatv2_edge_attn} "
          f"gat_edge_attn={args.gat_edge_attn} batch_size={args.batch_size} "
          f"normalize node/edge={args.normalize_node_features}/{args.normalize_edge_features}")
    print(f"[i] architecture: block={args.fixed_block or 'searched'} "
          f"relation_mp={'searched' if args.fixed_relation_mp is None else args.fixed_relation_mp} "
          f"edge_update={args.fixed_edge_update or 'searched'} readout={args.fixed_readout or 'searched'} | "
          f"ffn_mult={args.ffn_mult} grad_clip={args.grad_clip} relation_aggr={args.relation_aggr} "
          f"calo_context={args.calo_context} readout_heads={args.readout_heads}")
    if args.baseline_trial:
        print("[i] baseline trial 0 (train script defaults): " +
              " ".join(f"{k}={v}" for k, v in sorted(args.baseline_hparams.items())))

    storage = build_optuna_storage(args, out_dir)
    direction = metric_direction(args.compare_metric)
    maximize = direction == "maximize"
    pruner = optuna.pruners.NopPruner()
    if args.prune:
        # PercentilePruner(p): stop a trial whose best reported value so far is outside the best p% of the complete
        # trials' values at the same step (an epoch; the reported value is the running score, see on_epoch).
        pruner = optuna.pruners.PercentilePruner(
            float(args.prune_percentile), n_startup_trials=max(1, int(args.prune_startup_trials)),
            n_warmup_steps=max(0, int(args.prune_warmup_epochs)), interval_steps=1,
            n_min_trials=max(1, int(args.prune_startup_trials)))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        # the baseline trial 0 is not counted in --n-startup-trials
        sampler=optuna.samplers.TPESampler(
            seed=12345,
            n_startup_trials=max(1, int(args.n_startup_trials)) + (1 if args.baseline_trial else 0),
        ),
        pruner=pruner,
    )

    train_script = Path(args.train_script).resolve()
    split_file = Path(args.split_file).resolve()
    cwd = train_script.parent
    results_path = out_dir / "trials.jsonl"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    mkdir(ckpt_dir)
    mkdir(log_dir)
    check_study_protocol(study, args, results_path)
    write_leaderboard(out_dir, args, study)

    # Track best value seen so far for the running leaderboard
    _best_so_far: Dict[str, Any] = {"value": None, "trial": None}

    def objective(trial: optuna.Trial) -> float:
        if is_baseline_trial(trial):
            if trial.user_attrs.get("baseline_has_params"):
                build_hparams(trial, args)  # registers the enqueued params (= the defaults) for TPE
            hparams = dict(args.baseline_hparams)
            if auto_batch(args):  # trial 0 = the defaults, at the global batch where --lr is defined
                hparams["_global_batch"] = int(args.lr_ref_batch)
            trial.set_user_attr("hparams", hparams)
        else:
            hparams = build_hparams(trial, args)
        run_id = f"trial{trial.number:05d}_{now_utc_compact()}"
        save_path = build_trial_ckpt_path(ckpt_dir, args.save, run_id)
        log_path = log_dir / f"{run_id}.log"
        trial_label = f"Trial {trial.number}"

        if args.trial_launch_stagger_seconds > 0 and args.n_jobs > 1:
            time.sleep(float(args.trial_launch_stagger_seconds) * (trial.number % int(args.n_jobs)))

        master_port = pick_master_port(args.master_port_base, trial.number)
        rdzv_id = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', args.study_name)}_{run_id}"

        # ── Trial start banner ────────────────────────────────────────────
        _banner(f"TRIAL {trial.number} STARTED  [{datetime.now().strftime('%H:%M:%S')}]")
        _log(f"  run_id : {run_id}")
        _log(f"  epochs : {args.fast_epochs}")
        _log(f"  objective: {args.compare_metric}")
        _log(f"  arch   : {arch_tag(hparams)}" + ("  (trial 0: train-script defaults)" if is_baseline_trial(trial) else ""))
        _log(f"  Hyperparameters:")
        for k, v in sorted(hparams.items()):
            _log(f"    {k:<22} = {v}")
        _log(SEP2)

        with allocator.acquire(args.fast_gpus_per_trial) as gpu_ids:
            nproc = torchrun_nproc(args.fast_gpus_per_trial, gpu_ids)

            # ── Skip if checkpoint already exists ─────────────────────────
            if args.resume_skip_existing:
                for candidate_ckpt in (save_path, legacy_duplicated_ckpt_path(save_path, run_id)):
                    if not candidate_ckpt.exists():
                        continue
                    ckpt_info = read_ckpt_metrics(candidate_ckpt)
                    objective_value = extract_objective_from_ckpt(ckpt_info, args.compare_metric)
                    if objective_value is not None:
                        _log(f"[skip] Trial {trial.number}: checkpoint exists → {args.compare_metric}={objective_value:.5f}")
                        return float(objective_value)

            cmd = build_command(
                python_exe=os.environ.get("PYTHON", "python"),
                train_script=train_script,
                data_glob=args.data_glob,
                split_file=split_file,
                save_path=save_path,
                save_base=args.save,
                run_id=run_id,
                epochs=args.fast_epochs,
                max_train_events=args.fast_max_train_events,
                num_workers=args.num_workers,
                hparams=hparams,
                args=args,
                nproc=nproc,
                master_port=master_port,
                rdzv_id=rdzv_id,
                resume=False,
            )
            # every epoch: report the running score (= the final score if the trial ended here) to Optuna; with
            # --prune, stop the trial when the pruner says so (never trial 0)
            epoch_values: List[Optional[float]] = []
            prune_state: Dict[str, Any] = {"epoch": None, "value": None}
            baseline = is_baseline_trial(trial)

            def on_epoch(row: Dict[str, Any]) -> bool:
                epoch_values.append(row.get(args.compare_metric))
                value, _ = windowed_best(epoch_values, args.objective_window, maximize)
                if value is None:
                    return False
                try:
                    trial.report(float(value), step=int(row["epoch"]))
                    if args.prune and not baseline and trial.should_prune():
                        prune_state.update(epoch=int(row["epoch"]), value=float(value))
                        return True
                except Exception as exc:  # storage hiccup: keep training, never prune on an error
                    _log(f"[warn] {trial_label}: Optuna report failed at epoch {row.get('epoch')}: {exc}")
                return False

            t0 = time.time()
            _log(f"[run]  Training subprocess launched...")
            rc = run_one_training(
                cmd=cmd, log_path=log_path, gpu_ids=gpu_ids,
                cwd=cwd, save_dir=ckpt_dir, trial_label=trial_label, on_epoch=on_epoch,
            )
            seconds = time.time() - t0

        # ── Parse results ─────────────────────────────────────────────────
        pruned = prune_state["epoch"] is not None
        parsed = parse_best_metrics_from_log(log_path, args.compare_metric, args.objective_window)
        objective_value = parsed["objective_value"]
        if parsed["run_info"].get("global_batch_size"):
            # the batch the trial was ranked with (--batch-size auto: probed for its architecture)
            trial.set_user_attr("global_batch_size", int(parsed["run_info"]["global_batch_size"]))
        if objective_value is None:
            for candidate_ckpt in (save_path, legacy_duplicated_ckpt_path(save_path, run_id)):
                objective_value = extract_objective_from_ckpt(
                    read_ckpt_metrics(candidate_ckpt), args.compare_metric
                )
                if objective_value is not None:
                    break

        record = {
            "phase": "fast",
            "trial_number": trial.number,
            "run_id": run_id,
            "returncode": rc,
            "seconds": seconds,
            "gpu_ids": gpu_ids,
            "objective_name": args.compare_metric,
            "objective_value": objective_value,
            "target_fpr": args.target_fpr,
            "objective_window": args.objective_window,
            "best_epoch_value": parsed["best_epoch_value"],
            "window_epochs": parsed["window_epochs"],
            "window_row": parsed["window_row"],
            "best_row": parsed["best_row"],
            "last_row": parsed["last_row"],
            "n_epochs_seen": parsed["n_epochs_seen"],
            "run_info": parsed["run_info"],
            "protocol": run_protocol(args, "fast"),
            "baseline": baseline,
            "pruned": pruned,
            "pruned_at_epoch": prune_state["epoch"],
            "hparams": hparams,
            "ckpt_path": str(save_path),
            "log_path": str(log_path),
        }
        append_jsonl(results_path, record)

        # ── Trial complete banner ─────────────────────────────────────────
        _banner(f"TRIAL {trial.number} FINISHED  [{datetime.now().strftime('%H:%M:%S')}]", char="─")
        status = (f"✂ PRUNED after epoch {prune_state['epoch']}" if pruned else
                  "✓ OK" if rc == 0 else f"✗ FAILED (rc={rc})")
        _log(f"  status         : {status}")
        _log(f"  elapsed        : {seconds/60:.1f} min ({seconds:.0f}s)")
        _log(f"  epochs seen    : {parsed['n_epochs_seen']}")
        _log(f"  {args.compare_metric:<20} = {objective_value if objective_value is not None else 'N/A'}")
        if parsed.get("window_epochs") and args.objective_window > 1:
            _log(f"  score window   : epochs {parsed['window_epochs'][0]}-{parsed['window_epochs'][1]} "
                 f"(single best epoch: {parsed['best_epoch_value']:.5f})")
        if parsed.get("best_row"):
            br = parsed["best_row"]
            _log(f"  best epoch     : {br.get('epoch', '?')}")
            for k in ("val_loss", "val_auc", "val_acc", "val_f1", "val_tpr_at_target_fpr", "val_fpr_at_target_fpr"):
                v = br.get(k)
                if v is not None:
                    _log(f"    {k:<28} = {v:.5f}")

        if pruned:
            trial.set_user_attr("pruned_at_epoch", prune_state["epoch"])
            write_leaderboard(out_dir, args, study)
            _log(SEP2)
            raise optuna.exceptions.TrialPruned(
                f"pruned after epoch {prune_state['epoch']}: running score {prune_state['value']:.5f}")

        # ── Update and print running best-so-far leaderboard ─────────────
        if objective_value is not None:
            is_new_best = (
                _best_so_far["value"] is None
                or (
                    direction == "maximize" and objective_value > _best_so_far["value"]
                )
                or (
                    direction == "minimize" and objective_value < _best_so_far["value"]
                )
            )
            if is_new_best:
                _best_so_far["value"] = objective_value
                _best_so_far["trial"] = trial.number
                _log(f"  ★ NEW BEST! Trial {trial.number} → {args.compare_metric} = {objective_value:.5f}")
            else:
                _log(f"  ★ Best so far: Trial {_best_so_far['trial']} → {args.compare_metric} = {_best_so_far['value']:.5f}")
        write_leaderboard(out_dir, args, study, print_top=5)
        _log(SEP2)

        if rc != 0 or objective_value is None:
            raise optuna.exceptions.TrialPruned(
                f"training failed or metric missing; rc={rc}, metric={objective_value}"
            )
        return float(objective_value)

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and not is_baseline_trial(t)
    ]
    # trials stopped by the pruner count as done for --n-trials (failed trials do not)
    n_pruned_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
                        and t.user_attrs.get("pruned_at_epoch") is not None and not is_baseline_trial(t))
    if args.refit_only:
        _log("[mode] --refit-only set: skipping FAST Optuna optimization.")
    else:
        if args.n_trials_extra:
            n_to_run = int(args.n_trials)
        else:
            n_to_run = max(0, int(args.n_trials) - len(complete_trials) - n_pruned_done)
        _banner(
            f"OPTUNA SEARCH  |  completed={len(complete_trials)}  pruned={n_pruned_done}  "
            f"requested={args.n_trials}  launching={n_to_run}  "
            f"objective={args.compare_metric}"
        )
        if args.baseline_trial and not any(
            is_baseline_trial(t) and t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.WAITING)
            for t in study.trials
        ):
            params, why = baseline_enqueue_params(args.baseline_hparams, args)
            study.enqueue_trial(params or {}, user_attrs={"baseline": True, "baseline_has_params": params is not None})
            n_to_run += 1
            _log("[baseline] enqueued the train-script-default trial as the next trial" +
                 ("" if params is not None else f" (no Optuna params: {why}; TPE skips it)"))
        if n_to_run > 0:
            if args.prune:  # shown in the leaderboard (also by later --refit-only / --report-only runs)
                study.set_user_attr("pruning", {"percentile": float(args.prune_percentile),
                                                "warmup_epochs": int(args.prune_warmup_epochs),
                                                "startup_trials": int(args.prune_startup_trials)})
            study.optimize(objective, n_trials=n_to_run, n_jobs=args.n_jobs, gc_after_trial=True)

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    _banner("OPTIMIZATION COMPLETE")
    if complete_trials:
        _log(f"  Best trial    : #{study.best_trial.number}")
        _log(f"  {args.compare_metric:<20} = {study.best_value}")
        _log(f"  Best params:" + ("  (baseline trial: train script defaults)" if is_baseline_trial(study.best_trial) else ""))
        best_items = study.best_trial.user_attrs["hparams"] if is_baseline_trial(study.best_trial) else study.best_trial.params
        for k, v in sorted(best_items.items()):
            _log(f"    {k:<28} = {v}")
    else:
        _log("  No completed trials — check the latest log files for training failures.")

    trials = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
    trials.sort(key=lambda t: t.value, reverse=direction == "maximize")
    if args.confirm_top_k > 0 and complete_trials:
        if args.refit_top_k > args.confirm_top_k:
            _log(f"[warn] --refit-top-k {args.refit_top_k} > --confirm-top-k {args.confirm_top_k}: "
                 f"refitting only the {args.confirm_top_k} confirmed configs.")
        trials = confirm_top_trials(
            args=args, trials=trials, allocator=allocator, train_script=train_script, split_file=split_file,
            cwd=cwd, ckpt_dir=ckpt_dir, log_dir=log_dir, out_dir=out_dir, maximize=direction == "maximize",
            on_update=lambda: write_leaderboard(out_dir, args, study),
        )
        _log(f"[refit] using the confirmed ranking: trials {[t.number for t in trials[:args.refit_top_k]]}")

    if args.refit_top_k > 0 and complete_trials:
        finalists = trials[: args.refit_top_k]
        refit_results = out_dir / "refit.jsonl"
        _banner(f"REFIT PHASE  —  top {len(finalists)} trials  ({args.refit_epochs} epochs each)")
        for rank, t in enumerate(finalists, start=1):
            hparams = refit_hparams_from_trial(t, args)
            run_id = f"refit_rank{rank:02d}_trial{t.number:05d}_{now_utc_compact()}"
            save_path = build_trial_ckpt_path(ckpt_dir, args.save, run_id)
            log_path = log_dir / f"{run_id}.log"
            master_port = pick_master_port(args.master_port_base, 100000 + rank)
            rdzv_id = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', args.study_name)}_{run_id}"
            trial_label = f"Refit rank {rank} (trial {t.number})"
            _banner(
                f"REFIT {rank}/{len(finalists)}  trial={t.number}  "
                f"fast_score={t.value:.5f}  [{datetime.now().strftime('%H:%M:%S')}]",
                char="─",
            )
            _log(f"  arch   : {arch_tag(hparams)}")
            _log(f"  Hyperparameters:")
            for k, v in sorted(hparams.items()):
                _log(f"    {k:<22} = {v}")
            with allocator.acquire(args.refit_gpus_per_trial) as gpu_ids:
                nproc = torchrun_nproc(args.refit_gpus_per_trial, gpu_ids)
                cmd = build_command(
                    python_exe=os.environ.get("PYTHON", "python"),
                    train_script=train_script,
                    data_glob=args.data_glob,
                    split_file=split_file,
                    save_path=save_path,
                    save_base=args.save,
                    run_id=run_id,
                    epochs=args.refit_epochs,
                    max_train_events=args.refit_max_train_events,
                    num_workers=args.num_workers,
                    hparams=hparams,
                    args=args,
                    nproc=nproc,
                    master_port=master_port,
                    rdzv_id=rdzv_id,
                    resume=False,
                )
                t0 = time.time()
                rc = run_one_training(
                    cmd=cmd, log_path=log_path, gpu_ids=gpu_ids,
                    cwd=cwd, save_dir=ckpt_dir, trial_label=trial_label,
                )
                seconds = time.time() - t0
            parsed = parse_best_metrics_from_log(log_path, args.compare_metric, args.objective_window)
            _log(f"  [refit {rank}] elapsed={seconds/60:.1f} min | "
                 f"{args.compare_metric}={parsed['objective_value']}")
            append_jsonl(refit_results, {
                "phase": "refit",
                "rank": rank,
                "source_trial_number": t.number,
                "run_id": run_id,
                "returncode": rc,
                "seconds": seconds,
                "objective_name": args.compare_metric,
                "objective_value": parsed["objective_value"],
                "target_fpr": args.target_fpr,
                "objective_window": args.objective_window,
                "best_epoch_value": parsed["best_epoch_value"],
                "window_epochs": parsed["window_epochs"],
                "window_row": parsed["window_row"],
                "best_row": parsed["best_row"],
                "last_row": parsed["last_row"],
                "n_epochs_seen": parsed["n_epochs_seen"],
                "run_info": parsed["run_info"],
                "protocol": run_protocol(args, "refit"),
                "hparams": hparams,
                "ckpt_path": str(save_path),
                "log_path": str(log_path),
            })
            write_leaderboard(out_dir, args, study)

    md = write_leaderboard(out_dir, args, study, print_top=10)
    if md is not None:
        _log(f"[leaderboard] {md}  and  {out_dir / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
