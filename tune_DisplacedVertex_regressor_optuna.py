#!/usr/bin/env python3
"""
Two-phase Optuna tuner for DisplacedVertex training scripts (DDP on multiple GPUs).

Supports tuning any of:
  - train_DisplacedVertex_position.py
  - train_DisplacedVertex_position_cylindrical.py
  - train_DisplacedVertex_position_polar.py

Phase A (fast/proxy):
- runs n_trials with fewer epochs and optional capped max_train_events
- objective: minimize one chosen metric among:
    val_loss, val_mae, val_rmse, val_mape
- first n_startup_trials random, rest TPE

Phase B (refit finalists):
- retrains top K configs with longer schedule / more events
- writes separate JSONL for refit

Important note on val_mape:
- The trainer supports early stopping only on:
    val_loss, val_mae, val_rmse
- If --compare-metric val_mape is used, this tuner parses the training log and
  uses the best val_mape observed across epochs as the Optuna objective.
- In that case, early stopping still uses --trainer-monitor.

Example:
python tune_DisplacedVertex_optuna.py \
  --train-script ./train_DisplacedVertex_position_cylindrical.py \
  --data-glob "./data_cylindrical/displaced_vertex_dataset_part*.h5" \
  --split-file "./data_cylindrical/split_displaced_vertex_cylindrical_seed12345.npz" \
  --out-dir "./tuning_runs_dv_cylindrical" \
  --storage-path "/shared/wp2p5/sqlite/optuna_dv_cyl.db" \
  --study-name "dv_cyl_val_mape" \
  --compare-metric val_mape \
  --trainer-monitor val_mae \
  --fixed-layer-type mpnn \
  --n-trials 200 \
  --fast-gpus-per-trial 2 \
  --n-jobs 4 \
  --num-workers 4 \
  --pin-memory \
  --wandb-mode disabled 2>&1 | tee log_tune_dv.txt
"""

import argparse
import glob
import json
import os
import random
import re
import shlex
import sqlite3
import subprocess
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import optuna.exceptions
import optuna.storages
import torch
from sqlalchemy import event
from sqlalchemy.pool import NullPool


# ============================================================
# Progress DB (separate from Optuna storage)
# ============================================================

PROGRESS_DB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trial_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL,

  phase TEXT NOT NULL,
  study_name TEXT,

  trial_number INTEGER,
  source_trial_number INTEGER,
  rank INTEGER,

  run_id TEXT NOT NULL,
  status TEXT NOT NULL,
  returncode INTEGER,

  seconds REAL,
  gpu_ids TEXT,
  master_port INTEGER,

  objective_name TEXT,
  objective_value REAL,

  ckpt_path TEXT,
  log_path TEXT,

  hparams_json TEXT,
  error TEXT
);

CREATE INDEX IF NOT EXISTS idx_trial_runs_phase_status ON trial_runs(phase, status);
CREATE INDEX IF NOT EXISTS idx_trial_runs_trial_number ON trial_runs(trial_number);
CREATE INDEX IF NOT EXISTS idx_trial_runs_run_id ON trial_runs(run_id);
"""

_progress_db_lock = threading.Lock()


def now_utc_compact() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _progress_db_connect(path: Path, *, timeout_s: int = 30) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=float(timeout_s), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)};")
    return conn


def progress_db_init(db_path: Path, *, timeout_s: int = 30) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _progress_db_lock:
        conn = _progress_db_connect(db_path, timeout_s=timeout_s)
        try:
            conn.executescript(PROGRESS_DB_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()


def progress_db_insert_running(
    db_path: Path,
    *,
    phase: str,
    study_name: Optional[str],
    trial_number: Optional[int],
    source_trial_number: Optional[int],
    rank: Optional[int],
    run_id: str,
    gpu_ids: List[str],
    master_port: int,
    objective_name: str,
    hparams: Dict[str, Any],
    ckpt_path: Path,
    log_path: Path,
    timeout_s: int = 30,
) -> int:
    now = now_utc_compact()
    with _progress_db_lock:
        conn = _progress_db_connect(db_path, timeout_s=timeout_s)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO trial_runs (
                  created_at_utc, updated_at_utc, phase, study_name,
                  trial_number, source_trial_number, rank,
                  run_id, status, gpu_ids, master_port,
                  objective_name, ckpt_path, log_path, hparams_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now, now, str(phase), study_name,
                    trial_number, source_trial_number, rank,
                    str(run_id), "running", ",".join(gpu_ids), int(master_port),
                    str(objective_name), str(ckpt_path), str(log_path),
                    json.dumps(hparams, sort_keys=True),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def progress_db_update_done(
    db_path: Path,
    row_id: int,
    *,
    status: str,
    returncode: Optional[int],
    seconds: Optional[float],
    objective_value: Optional[float],
    error: Optional[str],
    timeout_s: int = 30,
) -> None:
    now = now_utc_compact()
    with _progress_db_lock:
        conn = _progress_db_connect(db_path, timeout_s=timeout_s)
        try:
            conn.execute(
                """
                UPDATE trial_runs
                SET updated_at_utc=?,
                    status=?,
                    returncode=?,
                    seconds=?,
                    objective_value=?,
                    error=?
                WHERE id=?
                """,
                (now, str(status), returncode, seconds, objective_value, error, int(row_id)),
            )
            conn.commit()
        finally:
            conn.close()


# ============================================================
# GPU helpers
# ============================================================

def _detect_gpus_via_nvidia_smi() -> Optional[List[str]]:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        ids = []
        for line in out.splitlines():
            m = re.match(r"^\s*GPU\s+(\d+)\s*:", line)
            if m:
                ids.append(m.group(1))
        return ids if ids else None
    except Exception:
        return None


def parse_cuda_visible_devices(cvd: Optional[str]) -> List[str]:
    if cvd is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None and str(cvd).strip() != "":
        parts = [p.strip() for p in str(cvd).split(",") if p.strip() != ""]
        if parts:
            return parts

    smi_ids = _detect_gpus_via_nvidia_smi()
    if smi_ids is not None:
        return smi_ids

    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 0
    return [str(i) for i in range(max(1, n))]


class GPUAllocator:
    def __init__(self, devices: List[str]):
        self._all = list(devices)
        self._free = list(devices)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    @contextmanager
    def acquire(self, n: int, *, timeout_s: Optional[float] = None):
        if n <= 0:
            raise ValueError("n must be >= 1")
        t0 = time.time()
        with self._cv:
            while len(self._free) < n:
                if timeout_s is not None:
                    remain = timeout_s - (time.time() - t0)
                    if remain <= 0:
                        raise TimeoutError(
                            f"Timed out waiting for {n} GPUs (free={len(self._free)}/{len(self._all)})."
                        )
                    self._cv.wait(timeout=remain)
                else:
                    self._cv.wait()
            got = self._free[:n]
            self._free = self._free[n:]
        try:
            yield got
        finally:
            with self._cv:
                self._free = list(got) + self._free
                self._cv.notify_all()


def pick_master_port(base: int, trial_number: int) -> int:
    jitter = random.randint(0, 127)
    return int(base + (trial_number % 896) + jitter)


def infer_nproc_per_node(requested: int, env: Optional[dict] = None) -> int:
    if requested and requested > 0:
        return requested
    env = env or os.environ
    cvd = env.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None and str(cvd).strip() != "":
        n = len([x for x in str(cvd).split(",") if x.strip() != ""])
        if n > 0:
            return n
    try:
        n = torch.cuda.device_count()
        return max(1, int(n))
    except Exception:
        return 1


# ============================================================
# IO / utility helpers
# ============================================================

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ensure_dir_writable(p: Path, what: str) -> None:
    p = Path(p)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"[io] Cannot create {what} directory: {p} ({e!r})") from e

    probe = p / f".write_probe_{os.getpid()}"
    try:
        with probe.open("w", encoding="utf-8") as f:
            f.write("ok\n")
        probe.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(f"[io] {what} directory not writable: {p} ({e!r})") from e


def _abspath_glob(pattern: str, *, base_dir: Path) -> str:
    pattern = str(pattern)
    if os.path.isabs(pattern):
        return pattern
    return str((base_dir / pattern).resolve())


def validate_training_inputs(*, data_glob: str, split_file: Path, train_script: Path) -> None:
    if not train_script.exists():
        raise RuntimeError(f"[input] train script does not exist: {train_script}")
    if not split_file.exists():
        raise RuntimeError(f"[input] split file does not exist: {split_file}")
    matches = glob.glob(data_glob)
    if not matches:
        raise RuntimeError(
            f"[input] data_glob matched 0 files: {data_glob}\n"
            f"        (tip) pass an absolute path, or run from the repo root."
        )


def append_jsonl(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(json.loads(s))
                except Exception:
                    continue
    except Exception:
        return out
    return out


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def find_completed_refit_record(refit_results_path: Path, *, source_trial_number: int) -> Optional[Dict[str, Any]]:
    records = load_jsonl_records(refit_results_path)
    matches: List[Dict[str, Any]] = []
    for r in records:
        try:
            if r.get("phase") != "refit":
                continue
            if int(r.get("source_trial_number")) != int(source_trial_number):
                continue
            if int(r.get("returncode", 1)) != 0:
                continue
            if safe_float(r.get("objective_value")) is None:
                continue
            matches.append(r)
        except Exception:
            continue
    if not matches:
        return None
    matches.sort(key=lambda r: str(r.get("timestamp_utc", "")))
    return matches[-1]


def build_trial_ckpt_path(ckpt_dir: Path, save_base: str, run_id: str) -> Path:
    base = os.path.basename(save_base)
    stem, ext = os.path.splitext(base)
    ext = ext if ext else ".pt"
    return ckpt_dir / f"{stem}_{run_id}{ext}"


# ============================================================
# Metric parsing
# ============================================================

# Matches lines like:
# [epoch 001] train loss=... mae=... rmse=... mape=... | val loss=0.123 mae=0.456 rmse=0.789 mape=12.34% (...)
EPOCH_LINE_RE = re.compile(
    r"""
    ^\[epoch\s+(?P<epoch>\d+)\]
    .*?
    \|\s+val\s+loss=(?P<val_loss>[-+0-9.eE]+)
    \s+mae=(?P<val_mae>[-+0-9.eE]+)
    \s+rmse=(?P<val_rmse>[-+0-9.eE]+)
    \s+mape=(?P<val_mape>[-+0-9.eE]+)%
    """,
    re.VERBOSE,
)


def parse_best_metrics_from_log(log_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "epochs_seen": 0,
        "best_val_loss": None,
        "best_val_mae": None,
        "best_val_rmse": None,
        "best_val_mape": None,
        "best_epoch_val_loss": None,
        "best_epoch_val_mae": None,
        "best_epoch_val_rmse": None,
        "best_epoch_val_mape": None,
        "last_val_loss": None,
        "last_val_mae": None,
        "last_val_rmse": None,
        "last_val_mape": None,
    }

    if not log_path.exists():
        return out

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = EPOCH_LINE_RE.search(line)
                if not m:
                    continue

                epoch = int(m.group("epoch"))
                vals = {
                    "val_loss": float(m.group("val_loss")),
                    "val_mae": float(m.group("val_mae")),
                    "val_rmse": float(m.group("val_rmse")),
                    "val_mape": float(m.group("val_mape")),
                }

                out["epochs_seen"] = max(out["epochs_seen"], epoch)
                for k, v in vals.items():
                    out[f"last_{k}"] = v
                    best_k = f"best_{k}"
                    best_epoch_k = f"best_epoch_{k}"
                    if out[best_k] is None or v < out[best_k]:
                        out[best_k] = v
                        out[best_epoch_k] = epoch
    except Exception:
        pass

    return out


def read_ckpt_info(ckpt_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ckpt_path": str(ckpt_path), "ckpt_exists": False}
    if not ckpt_path.exists():
        return out

    out["ckpt_exists"] = True
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        out["ckpt_read_error"] = f"{type(e).__name__}: {e}"
        return out

    out["best_monitor"] = safe_float(ckpt.get("best_monitor"))
    out["early_stop_monitor"] = ckpt.get("early_stop_monitor")
    out["best_ckpt_epoch"] = ckpt.get("best_ckpt_epoch")
    out["run_id"] = ckpt.get("run_id")

    for k in [
        "coordinate_system",
        "target_order",
        "target_scale_mode",
        "normalize_target",
        "periodic_phi_loss",
        "phi_period",
        "phi_index",
        "normalize_node_features",
        "normalize_edge_features",
        "feature_stats_json",
        "feature_norm_kind",
        "layer_type",
        "gat_heads",
        "sage_aggr",
        "edgeconv_aggr",
        "pool",
        "hidden_dim",
        "layers",
        "dropout",
        "weight_decay",
        "edge_dropout",
        "feat_noise_std",
        "ema",
        "ema_decay",
        "lr_schedule",
        "warmup_epochs",
        "min_lr_ratio",
        "fourier",
        "fourier_base",
        "fourier_min_exp",
        "fourier_max_exp",
        "code_version",
    ]:
        if k in ckpt:
            out[k] = ckpt[k]

    return out


def extract_objective_value(
    *,
    compare_metric: str,
    log_metrics: Dict[str, Any],
    ckpt_info: Dict[str, Any],
) -> Optional[float]:
    compare_metric = str(compare_metric)

    # Prefer parsed log metrics because they support val_mape explicitly and work
    # even if the trainer checkpoint does not store the desired metric.
    key = f"best_{compare_metric}"
    if key in log_metrics and safe_float(log_metrics.get(key)) is not None:
        return float(log_metrics[key])

    # Fallback for trainer-supported monitors stored in ckpt.
    if compare_metric in {"val_loss", "val_mae", "val_rmse"}:
        if ckpt_info.get("early_stop_monitor") == compare_metric and safe_float(ckpt_info.get("best_monitor")) is not None:
            return float(ckpt_info["best_monitor"])

    return None


# ============================================================
# SQLite Optuna storage
# ============================================================

def make_sqlite_storage(sqlite_path: Path, *, timeout_s: int, enable_wal: bool, pool: str = "null"):
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{sqlite_path}"

    connect_args = {
        "timeout": int(timeout_s),
        "check_same_thread": False,
    }
    engine_kwargs = {"connect_args": connect_args}
    if pool == "null":
        engine_kwargs["poolclass"] = NullPool

    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs=engine_kwargs,
    )

    @event.listens_for(storage.engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute(f"PRAGMA busy_timeout={int(timeout_s) * 1000};")
            if enable_wal:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA temp_store=MEMORY;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()
        except Exception:
            pass

    return storage


# ============================================================
# Hyperparameter normalization
# ============================================================

def normalize_hparams(hp: Dict[str, Any]) -> Dict[str, Any]:
    layer_type = hp.get("layer_type", "mpnn")
    hp.setdefault("layer_type", layer_type)

    hp.setdefault("gat_heads", 4)
    hp.setdefault("sage_aggr", "mean")
    hp.setdefault("edgeconv_aggr", "mean")
    hp.setdefault("fourier", False)
    hp.setdefault("pool", "meanmax")
    hp.setdefault("loss", "smoothl1")
    hp.setdefault("target_scale", "none")
    hp.setdefault("feature_norm_kind", "standard")

    if layer_type == "gat_residual":
        hidden_dim = int(hp.get("hidden_dim", 128))
        candidate_heads = [1, 2, 4, 8]
        valid_heads = [h for h in candidate_heads if hidden_dim % h == 0]
        if not valid_heads:
            valid_heads = [1]
        if "gat_heads" not in hp or hp["gat_heads"] not in valid_heads:
            hp["gat_heads"] = 4 if 4 in valid_heads else valid_heads[0]
    else:
        hp["gat_heads"] = int(hp.get("gat_heads", 4))

    return hp


# ============================================================
# Command construction / execution
# ============================================================

def build_command(
    args,
    run_id: str,
    hp: Dict[str, Any],
    ckpt_dir: Path,
    env: dict,
    *,
    phase: str,
    epochs: int,
    max_train_events: int,
    early_stop_patience: int,
    master_port: int,
) -> Tuple[List[str], Path, Path]:
    ckpt_path = build_trial_ckpt_path(ckpt_dir, args.save, run_id)
    log_path = args.out_dir / "logs" / f"{phase}_{run_id}.log"

    nproc = infer_nproc_per_node(args.nproc_per_node, env=env)
    save_dir_abs = Path(ckpt_dir).expanduser().resolve()

    cmd = [
        "torchrun",
        "--standalone",
        f"--master_port={int(master_port)}",
        f"--nproc_per_node={nproc}",
        str(args.train_script),
        "--data-glob", args.data_glob,
        "--split-file", args.split_file,
        "--epochs", str(epochs),
        "--num-workers", str(args.num_workers),
        "--save", args.save,
        "--save-dir", str(save_dir_abs),
        "--run-id", run_id,
        "--seed", str(args.seed + hp.get("_trial_idx", 0) * 100),
        "--early-stop-monitor", args.trainer_monitor,
        "--early-stop-patience", str(early_stop_patience),
        "--prefetch-factor", str(args.prefetch_factor),
        "--loss", hp["loss"],
        "--pool", hp["pool"],
        "--target-scale", hp["target_scale"],
    ]

    if getattr(args, "code_version", None):
        cmd += ["--code-version", str(args.code_version)]

    if args.resume_in_refit_only and phase == "refit":
        cmd += ["--resume"]
    else:
        cmd += ["--no-resume"]

    if args.pin_memory:
        cmd.append("--pin-memory")
    else:
        cmd.append("--no-pin-memory")

    if args.no_persistent_workers:
        cmd.append("--no-persistent-workers")

    if args.trainer_amp:
        cmd.append("--amp")
        cmd += ["--amp-dtype", args.amp_dtype]
    else:
        cmd.append("--no-amp")

    if args.strict_split_check:
        cmd.append("--strict-split-check")

    if args.periodic_phi_loss:
        cmd.append("--periodic-phi-loss")
        cmd += ["--phi-period", str(args.phi_period), "--phi-index", str(args.phi_index)]
    else:
        cmd.append("--no-periodic-phi-loss")

    if args.normalize_node_features:
        cmd.append("--normalize-node-features")
    else:
        cmd.append("--no-normalize-node-features")

    if args.normalize_edge_features:
        cmd.append("--normalize-edge-features")
    else:
        cmd.append("--no-normalize-edge-features")

    if args.feature_stats_json:
        cmd += ["--feature-stats-json", str(args.feature_stats_json)]
        cmd += ["--feature-norm-kind", hp["feature_norm_kind"]]
        cmd += ["--feature-norm-clip", str(hp["feature_norm_clip"])]

    if args.normalize_target_legacy:
        cmd.append("--normalize-target")
    else:
        cmd.append("--no-normalize-target")

    if args.reload_best_half_patience:
        cmd.append("--reload-best-half-patience")

    if args.disable_ema:
        cmd.append("--no-ema")
    else:
        cmd.append("--ema")

    if max_train_events > 0:
        cmd += ["--max-train-events", str(max_train_events)]

    if args.target_stats_max_events > 0:
        cmd += ["--target-stats-max-events", str(args.target_stats_max_events)]

    if args.wandb_mode == "disabled":
        cmd += ["--wandb-mode", "disabled"]
    else:
        cmd += ["--wandb", "--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project]
        cmd += ["--wandb-name", f"{phase}_{run_id}"]

    # Tuned hyperparameters
    cmd += ["--lr", f"{hp['lr']:.8g}"]
    cmd += ["--weight-decay", f"{hp['weight_decay']:.8g}"]
    cmd += ["--dropout", f"{hp['dropout']:.8g}"]
    cmd += ["--hidden-dim", str(hp["hidden_dim"])]
    cmd += ["--layers", str(hp["layers"])]
    cmd += ["--edge-dropout", f"{hp['edge_dropout']:.8g}"]
    cmd += ["--feat-noise-std", f"{hp['feat_noise_std']:.8g}"]
    cmd += ["--ema-decay", f"{hp['ema_decay']:.8g}"]
    cmd += ["--lr-schedule", hp["lr_schedule"]]
    cmd += ["--warmup-epochs", f"{hp['warmup_epochs']:.8g}"]
    cmd += ["--min-lr-ratio", f"{hp['min_lr_ratio']:.8g}"]
    cmd += ["--layer-type", hp["layer_type"]]
    cmd += ["--gat-heads", str(hp.get("gat_heads", 4))]
    cmd += ["--sage-aggr", hp.get("sage_aggr", "mean")]
    cmd += ["--edgeconv-aggr", hp.get("edgeconv_aggr", "mean")]

    if hp["fourier"]:
        cmd += ["--fourier"]
    else:
        cmd += ["--no-fourier"]

    return cmd, ckpt_path, log_path


def run_trial(cmd: List[str], log_path: Path, env: dict, *, save_dir: Optional[Path] = None) -> int:
    mkdir(log_path.parent)
    shell_cmd = " ".join(shlex.quote(x) for x in cmd)
    if save_dir is not None:
        save_dir = Path(save_dir).expanduser().resolve()
        shell_cmd = f"mkdir -p {shlex.quote(str(save_dir))} && exec {shell_cmd}"

    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("COMMAND:\n" + shell_cmd + "\n\n")
        lf.flush()
        p = subprocess.run(
            ["bash", "-lc", shell_cmd],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return p.returncode


# ============================================================
# Objective / refit
# ============================================================

def objective_factory(args, ckpt_dir: Path, results_path: Path, allocator: GPUAllocator, progress_db_path: Optional[Path]):
    def objective(trial: optuna.Trial) -> float:
        hp: Dict[str, Any] = {}
        hp["_trial_idx"] = trial.number

        hp["lr"] = trial.suggest_float("lr", 5e-5, 6e-4, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 5e-4, 5e-2, log=True)
        hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.25)
        hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [96, 128, 160, 192, 256])
        hp["layers"] = trial.suggest_categorical("layers", [3, 4, 5, 6])

        hp["edge_dropout"] = trial.suggest_float("edge_dropout", 0.0, 0.25)
        hp["feat_noise_std"] = trial.suggest_categorical(
            "feat_noise_std",
            [0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
        )
        hp["ema_decay"] = trial.suggest_categorical("ema_decay", [0.995, 0.998, 0.999, 0.9995])

        hp["lr_schedule"] = trial.suggest_categorical("lr_schedule", ["plateau", "cosine"])
        hp["warmup_epochs"] = trial.suggest_categorical("warmup_epochs", [0.0, 1.0, 2.0, 3.0])
        hp["min_lr_ratio"] = trial.suggest_categorical("min_lr_ratio", [0.02, 0.05, 0.1])

        if args.fixed_layer_type is not None:
            hp["layer_type"] = args.fixed_layer_type
        else:
            hp["layer_type"] = trial.suggest_categorical(
                "layer_type",
                ["mpnn", "edge_residual", "sage_residual", "gat_residual"],
            )

        if hp["layer_type"] == "gat_residual":
            candidate_heads = [1, 2, 4, 8]
            valid_heads = [h for h in candidate_heads if hp["hidden_dim"] % h == 0]
            if not valid_heads:
                raise optuna.TrialPruned()
            hp["gat_heads"] = trial.suggest_categorical("gat_heads", valid_heads)
        else:
            hp["gat_heads"] = 4

        if hp["layer_type"] == "sage_residual":
            hp["sage_aggr"] = trial.suggest_categorical("sage_aggr", ["mean", "sum", "max"])
        else:
            hp["sage_aggr"] = "mean"

        if hp["layer_type"] == "edge_residual":
            hp["edgeconv_aggr"] = trial.suggest_categorical("edgeconv_aggr", ["mean", "sum", "max"])
        else:
            hp["edgeconv_aggr"] = "mean"

        hp["fourier"] = trial.suggest_categorical("fourier", [True, False])
        hp["pool"] = trial.suggest_categorical("pool", ["mean", "max", "sum", "meanmax"])
        hp["loss"] = trial.suggest_categorical("loss", ["smoothl1", "mse", "l1"])
        hp["target_scale"] = trial.suggest_categorical("target_scale", ["none", "standard", "robust", "minmax"])

        if args.feature_stats_json is not None and (args.normalize_node_features or args.normalize_edge_features):
            hp["feature_norm_kind"] = trial.suggest_categorical("feature_norm_kind", ["standard", "robust"])
            hp["feature_norm_clip"] = trial.suggest_categorical("feature_norm_clip", [-1.0, 5.0, 8.0, 10.0])
        else:
            hp["feature_norm_kind"] = "standard"
            hp["feature_norm_clip"] = -1.0

        run_id = f"t{trial.number:04d}_{now_utc_compact()}"

        gpus_per_trial = int(args.fast_gpus_per_trial)
        progress_row_id: Optional[int] = None

        with allocator.acquire(gpus_per_trial, timeout_s=args.gpu_acquire_timeout_s) as gpu_ids:
            mkdir(Path(ckpt_dir).expanduser().resolve())
            mkdir((args.out_dir / "logs").expanduser().resolve())

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            env.setdefault("NCCL_DEBUG", "WARN")
            env.setdefault("OMP_NUM_THREADS", str(args.omp_num_threads))
            env.setdefault("MKL_NUM_THREADS", str(args.mkl_num_threads))
            env.setdefault("PYTHONUNBUFFERED", "1")

            master_port = pick_master_port(args.master_port_base, trial.number)

            try:
                validate_training_inputs(
                    data_glob=args.data_glob,
                    split_file=Path(args.split_file),
                    train_script=args.train_script,
                )
            except Exception as e:
                raise optuna.TrialPruned(str(e))

            cmd, ckpt_path, log_path = build_command(
                args, run_id, hp, ckpt_dir, env=env,
                phase="fast",
                epochs=args.fast_epochs,
                max_train_events=args.max_train_events,
                early_stop_patience=args.fast_early_stop_patience,
                master_port=master_port,
            )

            if progress_db_path is not None:
                try:
                    progress_row_id = progress_db_insert_running(
                        progress_db_path,
                        phase="fast",
                        study_name=getattr(args, "study_name", None),
                        trial_number=int(trial.number),
                        source_trial_number=None,
                        rank=None,
                        run_id=run_id,
                        gpu_ids=gpu_ids,
                        master_port=int(master_port),
                        objective_name=args.compare_metric,
                        hparams={k: v for k, v in hp.items() if not k.startswith("_")},
                        ckpt_path=ckpt_path,
                        log_path=log_path,
                        timeout_s=int(args.progress_db_timeout),
                    )
                except Exception:
                    progress_row_id = None

            if args.resume_skip_existing and ckpt_path.exists() and log_path.exists():
                ckpt_info = read_ckpt_info(ckpt_path)
                log_metrics = parse_best_metrics_from_log(log_path)
                score = extract_objective_value(
                    compare_metric=args.compare_metric,
                    log_metrics=log_metrics,
                    ckpt_info=ckpt_info,
                )
                if score is not None:
                    if progress_db_path is not None and progress_row_id is not None:
                        progress_db_update_done(
                            progress_db_path,
                            progress_row_id,
                            status="ok",
                            returncode=0,
                            seconds=0.0,
                            objective_value=float(score),
                            error=None,
                            timeout_s=int(args.progress_db_timeout),
                        )
                    return float(score)

            t0 = time.time()
            rc = run_trial(cmd, log_path, env, save_dir=ckpt_dir)
            dt = time.time() - t0

            ckpt_info = read_ckpt_info(ckpt_path)
            log_metrics = parse_best_metrics_from_log(log_path)
            objective_value = extract_objective_value(
                compare_metric=args.compare_metric,
                log_metrics=log_metrics,
                ckpt_info=ckpt_info,
            )

            record = {
                "trial_number": trial.number,
                "returncode": rc,
                "run_id": run_id,
                "seconds": dt,
                "phase": "fast",
                "gpus": gpu_ids,
                "master_port": int(master_port),
                "compare_metric": args.compare_metric,
                "trainer_monitor": args.trainer_monitor,
                "hparams": {k: v for k, v in hp.items() if not k.startswith("_")},
                "ckpt": ckpt_info,
                "log_metrics": log_metrics,
                "objective_value": objective_value,
                "log_path": str(log_path),
                "timestamp_utc": now_utc_compact(),
            }
            append_jsonl(results_path, record)

            if rc != 0 or objective_value is None:
                if progress_db_path is not None and progress_row_id is not None:
                    progress_db_update_done(
                        progress_db_path,
                        progress_row_id,
                        status="fail" if rc != 0 else "pruned",
                        returncode=int(rc),
                        seconds=float(dt),
                        objective_value=None,
                        error=None if rc == 0 else f"nonzero returncode {rc}",
                        timeout_s=int(args.progress_db_timeout),
                    )
                raise optuna.TrialPruned()

            if progress_db_path is not None and progress_row_id is not None:
                progress_db_update_done(
                    progress_db_path,
                    progress_row_id,
                    status="ok",
                    returncode=int(rc),
                    seconds=float(dt),
                    objective_value=float(objective_value),
                    error=None,
                    timeout_s=int(args.progress_db_timeout),
                )

            trial.set_user_attr("ckpt_path", str(ckpt_path))
            trial.set_user_attr("log_path", str(log_path))
            trial.set_user_attr("run_id", run_id)
            trial.set_user_attr("gpus", ",".join(gpu_ids))
            trial.set_user_attr("master_port", int(master_port))
            trial.set_user_attr("compare_metric", args.compare_metric)

            return float(objective_value)

    return objective


def refit_topk(args, ckpt_dir: Path, study: optuna.Study, k: int, refit_results_path: Path, allocator: GPUAllocator, progress_db_path: Optional[Path]):
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value
    )
    all_topk = trials_sorted[:k]

    selected_ranks: Optional[List[int]] = getattr(args, "refit_ranks", None)
    if selected_ranks:
        selected_rank_set = set(int(x) for x in selected_ranks)
        finalists_with_rank = [
            (rank, t)
            for rank, t in enumerate(all_topk, start=1)
            if rank in selected_rank_set
        ]
        missing = sorted(selected_rank_set - {rank for rank, _ in finalists_with_rank})
        if missing:
            print(
                f"[refit] Requested ranks not available within top-{k}: {missing}. "
                f"Available ranks are 1..{len(all_topk)}.",
                flush=True,
            )
    else:
        finalists_with_rank = list(enumerate(all_topk, start=1))

    if not finalists_with_rank:
        print("[refit] No successful trials to refit.")
        return

    print(
        f"\n[refit] Re-training {len(finalists_with_rank)} finalist config(s) with "
        f"epochs={args.refit_epochs}, max_train_events={args.refit_max_train_events}, "
        f"compare_metric={args.compare_metric}\n",
        flush=True,
    )

    for rank, t in finalists_with_rank:
        if args.resume_skip_existing_refit:
            prev = find_completed_refit_record(
                refit_results_path,
                source_trial_number=int(t.number),
            )
            if prev is not None:
                prev_run_id = prev.get("run_id", "<unknown>")
                prev_metric = prev.get("objective_value")
                print(
                    f"[refit top{rank:02d}] SKIP: already completed "
                    f"(source_trial={t.number}, run_id={prev_run_id}, objective={prev_metric})",
                    flush=True,
                )
                continue

        hp = dict(t.params)
        hp["_trial_idx"] = t.number
        if args.fixed_layer_type is not None:
            hp["layer_type"] = args.fixed_layer_type
        hp = normalize_hparams(hp)

        run_id = f"refit_top{rank:02d}_from_t{t.number:04d}"
        gpus_per_trial = int(args.refit_gpus_per_trial)
        progress_row_id: Optional[int] = None

        with allocator.acquire(gpus_per_trial, timeout_s=args.gpu_acquire_timeout_s) as gpu_ids:
            mkdir(Path(ckpt_dir).expanduser().resolve())
            mkdir((args.out_dir / "logs").expanduser().resolve())

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            env.setdefault("NCCL_DEBUG", "WARN")
            env.setdefault("OMP_NUM_THREADS", str(args.omp_num_threads))
            env.setdefault("MKL_NUM_THREADS", str(args.mkl_num_threads))
            env.setdefault("PYTHONUNBUFFERED", "1")

            master_port = pick_master_port(args.master_port_base, 10_000 + rank)

            validate_training_inputs(
                data_glob=args.data_glob,
                split_file=Path(args.split_file),
                train_script=args.train_script,
            )

            cmd, ckpt_path, log_path = build_command(
                args, run_id, hp, ckpt_dir, env=env,
                phase="refit",
                epochs=args.refit_epochs,
                max_train_events=args.refit_max_train_events,
                early_stop_patience=args.refit_early_stop_patience,
                master_port=master_port,
            )

            if progress_db_path is not None:
                try:
                    progress_row_id = progress_db_insert_running(
                        progress_db_path,
                        phase="refit",
                        study_name=getattr(args, "study_name", None),
                        trial_number=None,
                        source_trial_number=int(t.number),
                        rank=int(rank),
                        run_id=run_id,
                        gpu_ids=gpu_ids,
                        master_port=int(master_port),
                        objective_name=args.compare_metric,
                        hparams={k: v for k, v in hp.items() if not k.startswith("_")},
                        ckpt_path=ckpt_path,
                        log_path=log_path,
                        timeout_s=int(args.progress_db_timeout),
                    )
                except Exception:
                    progress_row_id = None

            t0 = time.time()
            rc = run_trial(cmd, log_path, env, save_dir=ckpt_dir)
            dt = time.time() - t0

            ckpt_info = read_ckpt_info(ckpt_path)
            log_metrics = parse_best_metrics_from_log(log_path)
            objective_value = extract_objective_value(
                compare_metric=args.compare_metric,
                log_metrics=log_metrics,
                ckpt_info=ckpt_info,
            )

            record = {
                "source_trial_number": t.number,
                "source_objective_value": t.value,
                "rank": rank,
                "returncode": rc,
                "run_id": run_id,
                "seconds": dt,
                "phase": "refit",
                "gpus": gpu_ids,
                "master_port": int(master_port),
                "compare_metric": args.compare_metric,
                "trainer_monitor": args.trainer_monitor,
                "hparams": {k: v for k, v in hp.items() if not k.startswith("_")},
                "ckpt": ckpt_info,
                "log_metrics": log_metrics,
                "objective_value": objective_value,
                "log_path": str(log_path),
                "timestamp_utc": now_utc_compact(),
            }

            try:
                append_jsonl(refit_results_path, record)
            except Exception as e:
                print(
                    f"[refit top{rank:02d}] WARN: failed to append to {refit_results_path}: {e!r}. "
                    f"Continuing because training already finished.",
                    flush=True,
                )

            if progress_db_path is not None and progress_row_id is not None:
                status = "ok" if (rc == 0 and objective_value is not None) else "fail"
                progress_db_update_done(
                    progress_db_path,
                    progress_row_id,
                    status=status,
                    returncode=int(rc),
                    seconds=float(dt),
                    objective_value=safe_float(objective_value),
                    error=None if status == "ok" else f"refit failed rc={rc}",
                    timeout_s=int(args.progress_db_timeout),
                )

            print(
                f"[refit top{rank:02d}] rc={rc} {args.compare_metric}={objective_value} ckpt={ckpt_path.name}",
                flush=True,
            )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-script", type=Path, required=True)
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--split-file", required=True)

    ap.add_argument("--out-dir", type=Path, default=Path("./tuning_runs_dv"))
    ap.add_argument("--save", default="displaced_vertex_gnn.pt")

    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--n-startup-trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--nproc-per-node", type=int, default=0,
                    help="0 = auto-detect from CUDA_VISIBLE_DEVICES / torch.cuda.device_count().")
    ap.add_argument("--cuda-visible-devices", default=None)

    ap.add_argument("--fixed-layer-type", default=None,
                    choices=["mpnn", "edge_residual", "sage_residual", "gat_residual"])

    ap.add_argument("--compare-metric", default="val_loss",
                    choices=["val_loss", "val_mae", "val_rmse", "val_mape"],
                    help="Optuna objective to minimize. val_mape is obtained by parsing the training log.")
    ap.add_argument("--trainer-monitor", default="val_mae",
                    choices=["val_loss", "val_mae", "val_rmse"],
                    help="Metric passed to trainer --early-stop-monitor. Required because trainer does not support val_mape.")
    ap.add_argument("--mape-eps", type=float, default=1e-6,
                    help="Forwarded only indirectly through trainer default unless you add it below manually.")

    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--fast-gpus-per-trial", type=int, default=2)
    ap.add_argument("--refit-gpus-per-trial", type=int, default=8)
    ap.add_argument("--gpu-acquire-timeout-s", type=float, default=None)
    ap.add_argument("--master-port-base", type=int, default=29500)

    ap.add_argument("--omp-num-threads", type=int, default=1)
    ap.add_argument("--mkl-num-threads", type=int, default=1)

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=1)
    ap.add_argument("--no-persistent-workers", action="store_true", default=False)

    ap.add_argument("--trainer-amp", dest="trainer_amp", action="store_true", default=True)
    ap.add_argument("--no-trainer-amp", dest="trainer_amp", action="store_false")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--fast-epochs", type=int, default=40)
    ap.add_argument("--max-train-events", type=int, default=20000)
    ap.add_argument("--fast-early-stop-patience", type=int, default=10)

    ap.add_argument("--refit-topk", type=int, default=5)
    ap.add_argument("--refit-epochs", type=int, default=200)
    ap.add_argument("--refit-max-train-events", type=int, default=-1)
    ap.add_argument("--refit-ranks", type=int, nargs="+", default=None)
    ap.add_argument("--refit-early-stop-patience", type=int, default=30)

    ap.add_argument("--resume-skip-existing", action="store_true", default=True)
    ap.add_argument("--resume-skip-existing-refit", action="store_true", default=True)
    ap.add_argument("--no-resume-skip-existing-refit", dest="resume_skip_existing_refit", action="store_false")
    ap.add_argument("--resume-in-refit-only", action="store_true", default=False,
                    help="Pass --resume to trainer during refit runs.")

    ap.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-project", default="DisplacedVertex_Tuning")

    ap.add_argument("--study-name", default="displaced_vertex_optuna")
    ap.add_argument("--code-version", default=None)

    ap.add_argument("--refit-only", action="store_true", default=False)

    ap.add_argument("--storage-path", type=Path, default=None)
    ap.add_argument("--sqlite-timeout", type=int, default=180)
    ap.add_argument("--sqlite-wal", dest="sqlite_wal", action="store_true", default=True)
    ap.add_argument("--no-sqlite-wal", dest="sqlite_wal", action="store_false")
    ap.add_argument("--sqlite-pool", default="null", choices=["null", "default"])

    ap.add_argument("--progress-db", type=Path, default=None)
    ap.add_argument("--no-progress-db", action="store_true", default=False)
    ap.add_argument("--progress-db-timeout", type=int, default=30)

    # Forwarded trainer flags / search-space context
    ap.add_argument("--strict-split-check", action="store_true", default=False)

    ap.add_argument("--feature-stats-json", default=None)
    ap.add_argument("--normalize-node-features", action="store_true", default=False)
    ap.add_argument("--normalize-edge-features", action="store_true", default=False)

    ap.add_argument("--normalize-target-legacy", action="store_true", default=True,
                    help="Forward --normalize-target to trainer for backward-compatible behavior when target-scale=none.")
    ap.add_argument("--no-normalize-target-legacy", dest="normalize_target_legacy", action="store_false")

    ap.add_argument("--periodic-phi-loss", action="store_true", default=False)
    ap.add_argument("--phi-period", type=float, default=(2.0 * 3.141592653589793))
    ap.add_argument("--phi-index", type=int, default=1)

    ap.add_argument("--disable-ema", action="store_true", default=False)
    ap.add_argument("--reload-best-half-patience", action="store_true", default=False)
    ap.add_argument("--target-stats-max-events", type=int, default=-1)

    args = ap.parse_args()

    args.out_dir = Path(args.out_dir).expanduser().resolve()
    args.train_script = Path(args.train_script).expanduser().resolve()

    if args.refit_ranks is not None:
        bad = [r for r in args.refit_ranks if int(r) < 1]
        if bad:
            raise SystemExit(f"--refit-ranks must be >= 1, got: {bad}")
        seen = set()
        args.refit_ranks = [
            int(r) for r in args.refit_ranks
            if not (int(r) in seen or seen.add(int(r)))
        ]

    base_dir = Path.cwd()
    args.data_glob = _abspath_glob(args.data_glob, base_dir=base_dir)
    args.split_file = str(Path(args.split_file).expanduser().resolve())

    if args.storage_path is not None:
        args.storage_path = Path(args.storage_path).expanduser().resolve()
    if args.progress_db is not None:
        args.progress_db = Path(args.progress_db).expanduser().resolve()

    validate_training_inputs(
        data_glob=args.data_glob,
        split_file=Path(args.split_file),
        train_script=args.train_script,
    )

    warnings.filterwarnings(
        "ignore",
        message=r"Argument ``multivariate`` is an experimental feature\.",
        category=optuna.exceptions.ExperimentalWarning,
    )

    base_devices = parse_cuda_visible_devices(args.cuda_visible_devices)
    if len(base_devices) < 1:
        raise SystemExit("No CUDA devices visible to tuner.")

    try:
        torch_n = int(torch.cuda.device_count())
    except Exception:
        torch_n = -1

    print(
        f"[gpu-detect] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')!r} "
        f"torch.cuda.device_count()={torch_n} pool={base_devices}",
        flush=True,
    )

    max_parallel = max(1, len(base_devices) // max(1, int(args.fast_gpus_per_trial)))
    if args.n_jobs > max_parallel:
        print(
            f"[warn] --n-jobs={args.n_jobs} is too high for ngpus={len(base_devices)} and "
            f"--fast-gpus-per-trial={args.fast_gpus_per_trial}. Clamping to {max_parallel}.",
            flush=True,
        )
        args.n_jobs = max_parallel

    mkdir(args.out_dir)
    ckpt_dir = (args.out_dir / "ckpts").resolve()
    mkdir(ckpt_dir)

    ensure_dir_writable(args.out_dir, "--out-dir")
    ensure_dir_writable(ckpt_dir, "checkpoint (--out-dir/ckpts)")
    ensure_dir_writable((args.out_dir / "logs").resolve(), "logs (--out-dir/logs)")

    results_fast = args.out_dir / "results_fast.jsonl"
    results_refit = args.out_dir / "results_refit.jsonl"

    ensure_dir_writable(results_fast.parent, "results (--out-dir)")
    ensure_dir_writable(results_refit.parent, "results (--out-dir)")

    if args.storage_path is not None:
        storage_path = Path(args.storage_path)
    else:
        storage_path = (args.out_dir / "optuna_study.db").resolve()

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[storage] Using SQLite DB at: {storage_path}", flush=True)

    storage = make_sqlite_storage(
        storage_path,
        timeout_s=args.sqlite_timeout,
        enable_wal=args.sqlite_wal,
        pool=args.sqlite_pool,
    )

    progress_db_path: Optional[Path] = None
    if not args.no_progress_db:
        progress_db_path = (
            args.progress_db if args.progress_db is not None else (args.out_dir / "progress.sqlite")
        ).resolve()
        try:
            progress_db_init(progress_db_path, timeout_s=int(args.progress_db_timeout))
            print(f"[progress-db] Writing trial progress to: {progress_db_path}", flush=True)
        except Exception as e:
            print(f"[progress-db] WARN: could not init progress DB at {progress_db_path}: {e!r}", flush=True)
            progress_db_path = None

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
    )

    allocator = GPUAllocator(base_devices)
    objective = objective_factory(args, ckpt_dir, results_fast, allocator, progress_db_path)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    if args.refit_only:
        print("[mode] --refit-only set: skipping FAST Optuna optimization; loading existing study only.", flush=True)
    else:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=int(args.n_jobs), gc_after_trial=True)

    print("\n=== FAST PHASE BEST ===")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if completed:
        print(f"best {args.compare_metric}: {study.best_value}")
        print("best params:", json.dumps(study.best_params, indent=2, sort_keys=True))
    else:
        print(f"best {args.compare_metric}: <none> (no completed trials in study)")
        print("best params: <none>")

    print(f"compare_metric: {args.compare_metric}")
    print(f"trainer_monitor: {args.trainer_monitor}")
    print(f"Optuna DB: {storage_path} (sqlite_wal={args.sqlite_wal} timeout={args.sqlite_timeout}s pool={args.sqlite_pool})")
    print(f"Fast JSONL: {results_fast}")

    refit_topk(args, ckpt_dir, study, args.refit_topk, results_refit, allocator, progress_db_path)
    print(f"\nRefit JSONL: {results_refit}")


if __name__ == "__main__":
    main()
