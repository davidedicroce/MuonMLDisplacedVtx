#!/usr/bin/env python3
"""
Optuna tuner for DisplacedVertex graph-level binary classifier training.

Default objective:
  maximize val_tpr_at_target_fpr, with --target-fpr 0.01 by default.
That means trials are ranked by the highest validation TPR achievable while the
validation FPR is constrained to be <= 1%.

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

Architecture options of the trainer (GAT trials; the defaults keep the legacy model, the search space above was
narrowed on it): --fixed-block legacy|prenorm, --fixed-relation-mp false|true, --fixed-edge-update
none|seg|seg_tracklet (relation-mp trials), --fixed-readout global|typed|typed_attn ('any' searches one of them),
plus the fixed knobs --ffn-mult, --grad-clip, --relation-aggr, --calo-context, --readout-heads. For a comparison of
the options fix one at a time; deeper prenorm stacks: --layers-max 8. The baseline trial keeps the trainer defaults
(it lies outside a study with other fixed options and then runs without Optuna params).

Example:
TUNE_FLAGS=(
  --train-script ./train_DisplacedVertex.py
  --data-glob "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/*.h5"
  --split-file "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/split_displaced_vertex_seed12345.npz"
  --feature-stats-json "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json"
  --out-dir "/shared/wp2p5/models/tuning_dv_classifier_gatv2"
  --study-name dv_classifier_gatv2
  --compare-metric val_tpr_at_target_fpr --target-fpr 0.01
  --num-workers 4 --pin-memory --wandb-mode disabled
)
python -u tune_DisplacedVertex_optuna.py "${TUNE_FLAGS[@]}" \
  --n-trials 120 --n-startup-trials 60 \
  --fast-epochs 60 --fast-max-train-events 100000 \
  --fast-gpus-per-trial 2 --n-jobs 4 \
  --refit-top-k 0 2>&1 | tee -a log_tune_dv_classifier_gatv2.txt

# later: refit the best 5 on the full train split (same study, same flags)
python -u tune_DisplacedVertex_optuna.py "${TUNE_FLAGS[@]}" \
  --refit-only --refit-top-k 5 --refit-epochs 200 \
  --refit-gpus-per-trial 8 2>&1 | tee -a log_tune_dv_classifier_gatv2_refit.txt
"""

import argparse
import ast
from datetime import datetime
import glob
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def parse_best_metrics_from_log(log_path: Path, compare_metric: str) -> Dict[str, Any]:
    if not log_path.exists():
        return {"objective_value": None, "best_row": {}, "last_row": {}, "n_epochs_seen": 0}

    last_error: Optional[OSError] = None
    for attempt in range(6):
        best_value = None
        best_row: Dict[str, Any] = {}
        last_row: Dict[str, Any] = {}
        n_epochs = 0
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = EPOCH_LINE_RE.search(line.strip())
                    if not m:
                        continue
                    n_epochs += 1
                    row: Dict[str, Any] = {"epoch": int(m.group("epoch"))}
                    for key in METRICS:
                        row[key] = safe_float(m.groupdict().get(key))
                    last_row = row
                    value = row.get(compare_metric)
                    if value is not None and is_better(compare_metric, value, best_value):
                        best_value = value
                        best_row = row
            return {
                "objective_value": best_value,
                "best_row": best_row,
                "last_row": last_row,
                "n_epochs_seen": n_epochs,
            }
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
        "--batch-size", str(int(args.batch_size)),
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
) -> int:
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
        last_epoch_summary = ""
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
                last_epoch_summary = stripped
                _log(f"  >> {trial_label} | {stripped}")
        rc = proc.wait()
    return rc


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
    ap.add_argument("--target-fpr", type=float, default=0.01, help="FPR working point. 0.01 means 1%% FPR.")
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
    ap.add_argument("--refit-top-k", type=int, default=0)
    ap.add_argument("--refit-epochs", type=int, default=100)
    ap.add_argument("--refit-max-train-events", type=int, default=-1)
    ap.add_argument("--refit-gpus-per-trial", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Graphs per GPU step passed to the trainer. Default: the train script's default "
                         "(64, as the best model); the old studies used 16.")
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
    if args.batch_size is None:
        args.batch_size = int(trainer_defaults.get("batch_size", 16))
    args.baseline_hparams = trainer_baseline_hparams(trainer_defaults)

    out_dir = Path(args.out_dir).resolve()
    mkdir(out_dir)
    validate_inputs(args)

    devices = detect_gpus()
    actual_cuda_devices = int(torch.cuda.device_count())
    if devices and actual_cuda_devices < len(devices):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES lists "
            f"{len(devices)} device(s), but PyTorch can use only {actual_cuda_devices}. "
            "Check for malformed, wrapped, unavailable, or duplicate GPU/MIG identifiers. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
        )
    requested_gpus = max(int(args.fast_gpus_per_trial), int(args.refit_gpus_per_trial))
    if devices and requested_gpus > actual_cuda_devices:
        raise RuntimeError(
            f"Requested up to {requested_gpus} GPU(s) per trial, but PyTorch can use only "
            f"{actual_cuda_devices}."
        )
    allocator = GPUAllocator(devices)
    print(f"[i] visible GPUs: {devices if devices else 'none; running train script without torchrun'}")
    print(f"[i] objective: {args.compare_metric}; target_fpr={args.target_fpr}")
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
    )

    train_script = Path(args.train_script).resolve()
    split_file = Path(args.split_file).resolve()
    cwd = train_script.parent
    results_path = out_dir / "trials.jsonl"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    mkdir(ckpt_dir)
    mkdir(log_dir)

    # Track best value seen so far for the running leaderboard
    _best_so_far: Dict[str, Any] = {"value": None, "trial": None}

    def objective(trial: optuna.Trial) -> float:
        if is_baseline_trial(trial):
            if trial.user_attrs.get("baseline_has_params"):
                build_hparams(trial, args)  # registers the enqueued params (= the defaults) for TPE
            hparams = dict(args.baseline_hparams)
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
            t0 = time.time()
            _log(f"[run]  Training subprocess launched...")
            rc = run_one_training(
                cmd=cmd, log_path=log_path, gpu_ids=gpu_ids,
                cwd=cwd, save_dir=ckpt_dir, trial_label=trial_label,
            )
            seconds = time.time() - t0

        # ── Parse results ─────────────────────────────────────────────────
        parsed = parse_best_metrics_from_log(log_path, args.compare_metric)
        objective_value = parsed["objective_value"]
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
            "best_row": parsed["best_row"],
            "last_row": parsed["last_row"],
            "n_epochs_seen": parsed["n_epochs_seen"],
            "hparams": hparams,
            "ckpt_path": str(save_path),
            "log_path": str(log_path),
        }
        append_jsonl(results_path, record)

        # ── Trial complete banner ─────────────────────────────────────────
        _banner(f"TRIAL {trial.number} FINISHED  [{datetime.now().strftime('%H:%M:%S')}]", char="─")
        status = "✓ OK" if rc == 0 else f"✗ FAILED (rc={rc})"
        _log(f"  status         : {status}")
        _log(f"  elapsed        : {seconds/60:.1f} min ({seconds:.0f}s)")
        _log(f"  epochs seen    : {parsed['n_epochs_seen']}")
        _log(f"  {args.compare_metric:<20} = {objective_value if objective_value is not None else 'N/A'}")
        if parsed.get("best_row"):
            br = parsed["best_row"]
            _log(f"  best epoch     : {br.get('epoch', '?')}")
            for k in ("val_loss", "val_auc", "val_acc", "val_f1", "val_tpr_at_target_fpr", "val_fpr_at_target_fpr"):
                v = br.get(k)
                if v is not None:
                    _log(f"    {k:<28} = {v:.5f}")

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
    if args.refit_only:
        _log("[mode] --refit-only set: skipping FAST Optuna optimization.")
    else:
        if args.n_trials_extra:
            n_to_run = int(args.n_trials)
        else:
            n_to_run = max(0, int(args.n_trials) - len(complete_trials))
        _banner(
            f"OPTUNA SEARCH  |  completed={len(complete_trials)}  "
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

    if args.refit_top_k > 0 and complete_trials:
        trials = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
        reverse = direction == "maximize"
        trials.sort(key=lambda t: t.value, reverse=reverse)
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
            parsed = parse_best_metrics_from_log(log_path, args.compare_metric)
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
                "best_row": parsed["best_row"],
                "last_row": parsed["last_row"],
                "hparams": hparams,
                "ckpt_path": str(save_path),
                "log_path": str(log_path),
            })


if __name__ == "__main__":
    main()
