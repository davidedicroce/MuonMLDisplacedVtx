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

Example:
python tune_DisplacedVertex_optuna.py \
  --train-script ./train_DisplacedVertex.py \
  --data-glob "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/*.h5" \
  --split-file "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/split_displaced_vertex_seed12345.npz" \
  --feature-stats-json "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json" \
  --normalize-node-features \
  --normalize-edge-features \
  --out-dir "/shared/wp2p5/models/tuning_dv_classifier_edge" \
  --study-name "/shared/wp2p5/sqlite/dv_classifier_edge" \
  --fixed-layer-type edge_residual \
  --compare-metric val_tpr_at_target_fpr \
  --target-fpr 0.01 \
  --loss-types bce bce_smooth focal asymmetric_focal \
  --n-trials 200 \
  --fast-epochs 60 \
  --fast-gpus-per-trial 2 \
  --n-jobs 4 \
  --fast-max-train-events 50000 \
  --refit-max-train-events -1 \
  --refit-top-k 5 \
  --refit-epochs 200 \
  --refit-gpus-per-trial 8 \
  --num-workers 4 \
  --pin-memory \
  --wandb-mode disabled \
2>&1 | tee log_tune_dv_classifier_edge.txt              
"""

import argparse
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
import torch


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
    best_value = None
    best_row: Dict[str, Any] = {}
    last_row: Dict[str, Any] = {}
    if not log_path.exists():
        return {"objective_value": None, "best_row": {}, "last_row": {}, "n_epochs_seen": 0}

    n_epochs = 0
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
        h["gat_heads"] = trial.suggest_categorical("gat_heads", [2, 4, 8])
        if h["hidden_dim"] % h["gat_heads"] != 0:
            h["gat_heads"] = 4 if h["hidden_dim"] % 4 == 0 else 2
    else:
        h["gat_heads"] = 4
    _suggest_loss_hparams(trial, args, h)
    return h


def refit_hparams_from_trial(trial: optuna.trial.FrozenTrial, args: argparse.Namespace) -> Dict[str, Any]:
    p = dict(trial.params)
    h: Dict[str, Any] = {
        "lr": p.get("lr", args.lr_min),
        "hidden_dim": p.get("hidden_dim", args.hidden_dims[0]),
        "layers": p.get("layers", args.layers_min),
        "dropout": p.get("dropout", args.dropout_min),
        "weight_decay": p.get("weight_decay", args.weight_decay_min),
        "edge_dropout": p.get("edge_dropout", args.edge_dropout_min),
        "feat_noise_std": p.get("feat_noise_std", args.feat_noise_min),
        "layer_type": args.fixed_layer_type or p.get("layer_type", "mpnn"),
        "pool": args.fixed_pool or p.get("pool", "meanmax"),
        "fourier": args.fixed_fourier if args.fixed_fourier is not None else p.get("fourier", True),
        "pos_weight": args.fixed_pos_weight or p.get("pos_weight", "auto"),
        "gat_heads": p.get("gat_heads", 4),
        "loss_type": args.fixed_loss_type or p.get("loss_type", args.loss_types[0]),
        "label_smoothing": p.get("label_smoothing", 0.0),
        "focal_gamma": p.get("focal_gamma", 2.0),
        "focal_alpha": p.get("focal_alpha", "none"),
        "asym_gamma_pos": p.get("asym_gamma_pos", 0.0),
        "asym_gamma_neg": p.get("asym_gamma_neg", 4.0),
    }
    return h


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
            "torchrun",
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
        "--save", str(save_base),
        "--save-dir", str(save_path.parent),
        "--run-id", str(run_id),
        "--resume" if resume else "--no-resume",
        "--early-stop-monitor", str(args.compare_metric),
        "--early-stop-patience", str(args.early_stop_patience),
        "--wandb-mode", str(args.wandb_mode),
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

    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = " ".join(shlex.quote(x) for x in cmd)
    if save_dir is not None:
        shell_cmd = f"mkdir -p {shlex.quote(str(Path(save_dir).expanduser().resolve()))} && exec {shell_cmd}"

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
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def validate_inputs(args: argparse.Namespace) -> None:
    if not Path(args.train_script).exists():
        raise SystemExit(f"train script does not exist: {args.train_script}")
    if not Path(args.split_file).exists():
        raise SystemExit(f"split file does not exist: {args.split_file}")
    if not glob.glob(args.data_glob):
        raise SystemExit(f"data_glob matched 0 files: {args.data_glob}")
    if (args.normalize_node_features or args.normalize_edge_features) and not args.feature_stats_json:
        raise SystemExit(
            "--feature-stats-json is required when --normalize-node-features or "
            "--normalize-edge-features is enabled; the trainer embeds these stats "
            "as model buffers for inference/ONNX export."
        )
    if args.feature_stats_json and not Path(args.feature_stats_json).exists():
        raise SystemExit(f"feature stats JSON does not exist: {args.feature_stats_json}")

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
    ap.add_argument("--target-fpr", type=float, default=0.01, help="FPR working point. 0.01 means 1% FPR.")
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
    ap.add_argument("--pin-memory", action="store_true", default=False)
    ap.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    ap.add_argument("--feature-stats-json", default=None,
                    help="Stats JSON loaded by the trainer and embedded as model normalization buffers.")
    ap.add_argument("--feature-norm-kind", default="standard", choices=["standard", "robust"])
    ap.add_argument("--normalize-node-features", action="store_true", default=False,
                    help="Ask the trainer to normalize node features inside the model.")
    ap.add_argument("--normalize-edge-features", action="store_true", default=False,
                    help="Ask the trainer to normalize edge features inside the model.")
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
    ap.add_argument("--no-amp", action="store_true", default=False)
    ap.add_argument("--no-ema", action="store_true", default=False)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--lr-max", type=float, default=1e-3)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--layers-min", type=int, default=2)
    ap.add_argument("--layers-max", type=int, default=6)
    ap.add_argument("--dropout-min", type=float, default=0.0)
    ap.add_argument("--dropout-max", type=float, default=0.3)
    ap.add_argument("--weight-decay-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay-max", type=float, default=5e-2)
    ap.add_argument("--edge-dropout-min", type=float, default=0.0)
    ap.add_argument("--edge-dropout-max", type=float, default=0.2)
    ap.add_argument("--feat-noise-min", type=float, default=0.0)
    ap.add_argument("--feat-noise-max", type=float, default=0.03)
    ap.add_argument("--loss-types", nargs="+", default=list(LOSS_TYPES), choices=list(LOSS_TYPES),
                    help="Loss functions Optuna may sample.")
    ap.add_argument("--fixed-loss-type", default=None, choices=list(LOSS_TYPES))
    ap.add_argument("--label-smoothing-min", type=float, default=0.0)
    ap.add_argument("--label-smoothing-max", type=float, default=0.05)
    ap.add_argument("--focal-gamma-min", type=float, default=0.5)
    ap.add_argument("--focal-gamma-max", type=float, default=5.0)
    ap.add_argument("--focal-alpha-min", type=float, default=0.1)
    ap.add_argument("--focal-alpha-max", type=float, default=0.9)
    ap.add_argument("--asym-gamma-pos-min", type=float, default=0.0)
    ap.add_argument("--asym-gamma-pos-max", type=float, default=2.0)
    ap.add_argument("--asym-gamma-neg-min", type=float, default=1.0)
    ap.add_argument("--asym-gamma-neg-max", type=float, default=6.0)
    ap.add_argument("--fixed-layer-type", choices=["mpnn", "edge_residual", "sage_residual", "gat_residual"], default=None)
    ap.add_argument("--fixed-pool", choices=["mean", "max", "sum", "meanmax"], default=None)
    ap.add_argument("--fixed-fourier", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=None)
    ap.add_argument("--fixed-pos-weight", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    mkdir(out_dir)
    validate_inputs(args)

    devices = detect_gpus()
    allocator = GPUAllocator(devices)
    print(f"[i] visible GPUs: {devices if devices else 'none; running train script without torchrun'}")
    print(f"[i] objective: {args.compare_metric}; target_fpr={args.target_fpr}")
    print(f"[i] loss search space: {args.loss_types if not args.fixed_loss_type else [args.fixed_loss_type]}")

    storage = build_optuna_storage(args, out_dir)
    direction = metric_direction(args.compare_metric)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=12345, n_startup_trials=max(1, int(args.n_startup_trials))),
    )

    train_script = Path(args.train_script).resolve()
    split_file = Path(args.split_file).resolve()
    cwd = train_script.parent
    results_path = out_dir / "trials.jsonl"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    mkdir(ckpt_dir)
    mkdir(log_dir)

    def objective(trial: optuna.Trial) -> float:
        hparams = build_hparams(trial, args)
        run_id = f"trial{trial.number:05d}_{now_utc_compact()}"
        save_path = build_trial_ckpt_path(ckpt_dir, args.save, run_id)
        log_path = log_dir / f"{run_id}.log"
        if args.trial_launch_stagger_seconds > 0 and args.n_jobs > 1:
            time.sleep(float(args.trial_launch_stagger_seconds) * (trial.number % int(args.n_jobs)))

        master_port = pick_master_port(args.master_port_base, trial.number)
        rdzv_id = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', args.study_name)}_{run_id}"


        with allocator.acquire(args.fast_gpus_per_trial) as gpu_ids:
            nproc = torchrun_nproc(args.fast_gpus_per_trial, gpu_ids)
            if args.resume_skip_existing:
                for candidate_ckpt in (save_path, legacy_duplicated_ckpt_path(save_path, run_id)):
                    if not candidate_ckpt.exists():
                        continue
                    ckpt_info = read_ckpt_metrics(candidate_ckpt)
                    objective_value = extract_objective_from_ckpt(ckpt_info, args.compare_metric)
                    if objective_value is not None:
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
            rc = run_one_training(cmd=cmd, log_path=log_path, gpu_ids=gpu_ids, cwd=cwd, save_dir=ckpt_dir)
            seconds = time.time() - t0

        parsed = parse_best_metrics_from_log(log_path, args.compare_metric)
        objective_value = parsed["objective_value"]
        if objective_value is None:
            for candidate_ckpt in (save_path, legacy_duplicated_ckpt_path(save_path, run_id)):
                objective_value = extract_objective_from_ckpt(read_ckpt_metrics(candidate_ckpt), args.compare_metric)
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

        if rc != 0 or objective_value is None:
            raise optuna.exceptions.TrialPruned(f"training failed or metric missing; rc={rc}, metric={objective_value}")
        return float(objective_value)

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if args.refit_only:
        print("[mode] --refit-only set: skipping FAST Optuna optimization.", flush=True)
    else:
        if args.n_trials_extra:
            n_to_run = int(args.n_trials)
        else:
            n_to_run = max(0, int(args.n_trials) - len(complete_trials))
        print(
            f"[resume] completed_fast_trials={len(complete_trials)} requested_total={args.n_trials} launching={n_to_run}",
            flush=True,
        )
        if n_to_run > 0:
            study.optimize(objective, n_trials=n_to_run, n_jobs=args.n_jobs, gc_after_trial=True)

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if complete_trials:
        print(f"[done] best trial: {study.best_trial.number} {args.compare_metric}={study.best_value}")
        print(f"[done] best params: {study.best_trial.params}")
    else:
        print("[done] no completed trials in study; check the latest log files for training failures.")

    if args.refit_top_k > 0 and complete_trials:
        trials = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
        reverse = direction == "maximize"
        trials.sort(key=lambda t: t.value, reverse=reverse)
        finalists = trials[: args.refit_top_k]
        refit_results = out_dir / "refit.jsonl"
        print(f"[refit] refitting top {len(finalists)} trials")
        for rank, t in enumerate(finalists, start=1):
            hparams = refit_hparams_from_trial(t, args)
            run_id = f"refit_rank{rank:02d}_trial{t.number:05d}_{now_utc_compact()}"
            save_path = build_trial_ckpt_path(ckpt_dir, args.save, run_id)
            log_path = log_dir / f"{run_id}.log"
            master_port = pick_master_port(args.master_port_base, 100000 + rank)
            rdzv_id = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', args.study_name)}_{run_id}"
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
                rc = run_one_training(cmd=cmd, log_path=log_path, gpu_ids=gpu_ids, cwd=cwd, save_dir=ckpt_dir)
                seconds = time.time() - t0
            parsed = parse_best_metrics_from_log(log_path, args.compare_metric)
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
