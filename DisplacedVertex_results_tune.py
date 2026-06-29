#!/usr/bin/env python3

'''
Example:
python -u DisplacedVertex_results_tune.py \
  --ckpt-glob "./models/*.pt" \
  --data-glob "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/*.h5" \
  --split-file "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/split_displaced_vertex_seed12345.npz" \
  --feature-stats-json "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json" \
  --target-fprs 0.001 0.005 0.01 0.02 \
  --force-recompute \
   2>&1 | tee log_dv_results_tune_$(date +%Y%m%d_%H%M%S).txt
'''

from __future__ import annotations

import argparse
import copy
import gc
import glob
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc


def maybe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


onnx = maybe_import("onnx")
ort = maybe_import("onnxruntime")
if ort is not None:
    ort.set_default_logger_severity(4)

torch.set_grad_enabled(False)

# Import the training utilities used by train_DisplacedVertex.py.
# In the production area this should be named dv_training_utils.py.  The fallback
# makes the script runnable next to the uploaded file name used in this chat.
try:
    from dv_training_utils import (  # type: ignore
        DisplacedVertexGNN,
        H5EventDataset,
        NODE_FEATURE_NAMES,
        EDGE_FEATURE_NAMES,
        load_feature_stats_json,
        _check_split_paths_compatible,
    )
except Exception:
    _fallback = Path(__file__).with_name("dv_training_utils(2).py")
    if _fallback.exists():
        spec = importlib.util.spec_from_file_location("dv_training_utils_fallback", str(_fallback))
        if spec is None or spec.loader is None:
            raise
        _mod = importlib.util.module_from_spec(spec)
        sys.modules["dv_training_utils_fallback"] = _mod
        spec.loader.exec_module(_mod)
        DisplacedVertexGNN = _mod.DisplacedVertexGNN
        H5EventDataset = _mod.H5EventDataset
        NODE_FEATURE_NAMES = _mod.NODE_FEATURE_NAMES
        EDGE_FEATURE_NAMES = _mod.EDGE_FEATURE_NAMES
        load_feature_stats_json = _mod.load_feature_stats_json
        _check_split_paths_compatible = _mod._check_split_paths_compatible
    else:
        raise


# ============================================================
# DisplacedVertex defaults
# ============================================================

DEFAULT_DATA_DIR = Path("/shared/wp2p5/data/data_displacedVtx_mu200_graphs")
DEFAULT_DATA_GLOB = str(DEFAULT_DATA_DIR / "*.h5")

# Explicit DisplacedVertex samples requested for validation/inference plots.
# These defaults expand to the same H5 part files you listed, and they make the
# signal-vs-background sample definitions available in run_summary.json and
# plot_payload.pkl for downstream plotting.
DEFAULT_SIGNAL_DATA_GLOBS = [
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_a_mumu_m100_ctau5000_mu200*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_a_mumu_m10_ctau500_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_a_mumu_m20_ctau500_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_a_mumu_m2_ctau100_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_a_mumu_m4_ctau100_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_a_mumu_m50_ctau1000_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_a_mumu_m6_ctau100_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_a_mumu_m8_ctau500_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_Haa4mu_m10_ctau500_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_Haa4mu_m4_ctau200_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_Haa4mu_m6_ctau200_mu200_*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_Haa4mu_m8_ctau500_mu200_*.h5"),
]
DEFAULT_BACKGROUND_DATA_GLOBS = [
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_Zmumu_mu200_part*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_JPSI_mu200_part*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_jj_mu200_part*.h5"),
    str(DEFAULT_DATA_DIR / "MuonBucketDump_vertex_ttbar_mu200_part*.h5"),
]
DEFAULT_SPLIT_FILE = DEFAULT_DATA_DIR / "split_displaced_vertex_seed12345.npz"
DEFAULT_FEATURE_STATS_JSON = DEFAULT_DATA_DIR / "normalization_stats_raw.json"


# ============================================================
# Small helpers
# ============================================================

def sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(s))


def sample_name_from_file(filename: str) -> str:
    """Return a stable physics-sample name from an H5 file path or glob pattern.

    Examples:
      MuonBucketDump_vertex_a_mumu_m10_ctau500_mu200_part12.h5
        -> vertex_a_mumu_m10_ctau500_mu200
      MuonBucketDump_vertex_a_mumu_m10_ctau500_mu200_*
        -> vertex_a_mumu_m10_ctau500_mu200

    The glob handling is important because CLI sample definitions may use either
    *_part*.h5 or *_mu200_* style patterns.  Both must map to the same sample
    name as the event metadata stored in the H5 files.
    """
    raw = str(filename)
    name = Path(raw).name if ("/" in raw or "\\" in raw) else raw
    name = re.sub(r"\.(h5|hdf5)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[\*\?\[].*$", "", name)  # strip glob suffixes
    name = re.sub(r"_part[^_]*$", "", name)
    name = re.sub(r"_+$", "", name)
    name = re.sub(r"^MuonBucketDump_", "", name)
    name = re.sub(r"^MuonSegmentDump_", "", name)
    return name


def decode_attr_to_str(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    return str(v)


def checkpoint_display_parts(ckpt_path: Path, ckpt: dict | None = None) -> tuple[str, str, str]:
    run_id = None
    if isinstance(ckpt, dict):
        run_id = ckpt.get("run_id")
    if run_id:
        display_name = str(run_id)
    else:
        display_name = ckpt_path.stem
    run_group = ckpt_path.parent.name if ckpt_path.parent.name else "checkpoints"
    tune_group = ckpt_path.parent.parent.name if ckpt_path.parent.parent.name else run_group
    return run_group, tune_group, display_name


def stable_model_id_from_path(ckpt_path: Path) -> str:
    return str(ckpt_path.expanduser().resolve())


def stable_model_id(item: "LoadedCheckpoint") -> str:
    return stable_model_id_from_path(item.ckpt_path)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _ckpt_get(ckpt: dict, key: str, default=None):
    if not isinstance(ckpt, dict):
        return default
    if key in ckpt:
        return ckpt[key]
    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, dict) and key in hp:
        return hp[key]
    return default


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(k).startswith("module.") for k in state_dict.keys()):
        return {str(k)[7:]: v for k, v in state_dict.items()}
    return state_dict


def discover_checkpoints(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p).expanduser().resolve() for p in glob.glob(str(pat), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError(f"No checkpoints matched: {list(patterns)}")
    return paths


def resolve_input_paths(data_glob: str) -> list[Path]:
    return resolve_input_paths_from_patterns([data_glob], label=f"--data-glob {data_glob!r}")


def resolve_input_paths_from_patterns(patterns: Iterable[str], *, label: str = "input globs") -> list[Path]:
    paths: list[Path] = []
    patterns = [str(p) for p in patterns if str(p)]
    for pat in patterns:
        paths.extend(Path(p).expanduser().resolve() for p in glob.glob(str(pat), recursive=True))
    paths = sorted(set(p for p in paths if p.exists() and p.is_file() and p.suffix.lower() in (".h5", ".hdf5")))
    if not paths:
        raise RuntimeError(f"No H5 files matched {label}: {patterns!r}")
    return paths


def sample_names_from_patterns(patterns: Iterable[str]) -> list[str]:
    names = [sample_name_from_file(str(pat)) for pat in patterns if str(pat)]
    # Preserve the user-specified order, while removing duplicates.
    return list(dict.fromkeys(names))


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class LoadedCheckpoint:
    ckpt_path: Path
    run_group: str
    display_name: str
    layer_type: str
    model: nn.Module
    ckpt: dict


# ============================================================
# Model build / ONNX export
# ============================================================

class DisplacedVertexOnnxExportWrapper(nn.Module):
    """
    ONNX wrapper for the DisplacedVertex graph-level binary classifier.

    ONNX contract:
      inputs:
        x            [num_nodes, 7]      raw node features
        edge_index   [2, num_edges]
        edge_attr    [num_edges, 5]      raw edge features
        n_muon_nodes [1]                 number of leading muon nodes in x
      output:
        logits       [1]

    Feature normalization is embedded in DisplacedVertexGNN.  The exported
    ONNX model therefore consumes the same raw x/edge_attr tensors as the H5
    files, plus n_muon_nodes so the graph can apply the muon/calo node
    normalizers internally.
    """

    def __init__(self, model: nn.Module, internal_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.internal_dtype = internal_dtype
        self.model = copy.deepcopy(model).eval()
        if internal_dtype == torch.float16:
            self.model = self.model.half()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        n_muon_nodes: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(dtype=self.internal_dtype)
        edge_index = edge_index.to(dtype=torch.long)
        edge_attr = edge_attr.to(dtype=self.internal_dtype)
        n_muon_nodes = n_muon_nodes.to(dtype=torch.long)
        logits = self.model(
            x, edge_index, edge_attr,
            n_muon_nodes=n_muon_nodes,
            edge_dropout_p=0.0,
        )
        return logits.to(dtype=torch.float32).view(-1)


NORM_NODE_BUFFER_NAMES = ("mu_center", "mu_scale", "ca_center", "ca_scale")
NORM_EDGE_BUFFER_NAMES = ("edge_center", "edge_scale")


def _normalization_buffers_in_state(state_dict: dict[str, Any]) -> tuple[bool, bool]:
    keys = set(str(k) for k in state_dict.keys())
    has_node = all(k in keys for k in NORM_NODE_BUFFER_NAMES)
    has_edge = all(k in keys for k in NORM_EDGE_BUFFER_NAMES)
    return has_node, has_edge


def _resolve_feature_stats_for_model(
    ckpt: dict,
    *,
    norm_node: bool,
    norm_edge: bool,
    stats_json: str | None,
    norm_kind: str,
    state_has_node_buffers: bool,
    state_has_edge_buffers: bool,
):
    """Return stats only when buffers must be initialized before loading the checkpoint."""
    need_external_stats = (norm_node and not state_has_node_buffers) or (norm_edge and not state_has_edge_buffers)
    if not need_external_stats:
        return None

    feature_stats = _ckpt_get(ckpt, "feature_stats", None)
    if isinstance(feature_stats, dict):
        return feature_stats

    if stats_json is None:
        raise RuntimeError(
            "Checkpoint requests model-side feature normalization, but no normalization "
            "buffers or feature_stats payload were found in the checkpoint. "
            "Pass --feature-stats-json so the ONNX export can initialize the buffers."
        )
    return load_feature_stats_json(str(stats_json), norm_kind=norm_kind)


def _load_state_dict_allowing_initialized_norm_buffers(model: nn.Module, state_dict: dict[str, Any]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = set(NORM_NODE_BUFFER_NAMES + NORM_EDGE_BUFFER_NAMES)
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Checkpoint state_dict is incompatible with DisplacedVertexGNN: "
            f"missing={bad_missing}, unexpected={list(unexpected)}"
        )


def build_displaced_vertex_model_from_checkpoint(
    ckpt_path: Path,
    *,
    internal_dtype: torch.dtype = torch.float32,
    args=None,
) -> tuple[nn.Module, dict]:
    ckpt = _safe_torch_load(ckpt_path)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain 'model_state'.")

    xdim = int(_ckpt_get(ckpt, "xdim", 7))
    edim = int(_ckpt_get(ckpt, "edim", 5))
    hidden_dim = int(_ckpt_get(ckpt, "hidden_dim", 128))
    layers = int(_ckpt_get(ckpt, "layers", _ckpt_get(ckpt, "num_layers", 4)))
    dropout = float(_ckpt_get(ckpt, "dropout", 0.1))
    layer_type = str(_ckpt_get(ckpt, "layer_type", "mpnn"))
    gat_heads = int(_ckpt_get(ckpt, "gat_heads", 4))
    sage_aggr = str(_ckpt_get(ckpt, "sage_aggr", "mean"))
    edgeconv_aggr = str(_ckpt_get(ckpt, "edgeconv_aggr", "mean"))
    pool = str(_ckpt_get(ckpt, "pool", "meanmax"))
    use_fourier = bool(_ckpt_get(ckpt, "fourier", False))
    fourier_base = float(_ckpt_get(ckpt, "fourier_base", 3.0))
    fourier_min_exp = int(_ckpt_get(ckpt, "fourier_min_exp", -6))
    fourier_max_exp = int(_ckpt_get(ckpt, "fourier_max_exp", 6))

    state_dict = _strip_module_prefix(ckpt["model_state"])
    state_has_node_buffers, state_has_edge_buffers = _normalization_buffers_in_state(state_dict)
    norm_node, norm_edge, stats_json, norm_kind, norm_clip = resolve_normalization_for_checkpoint(
        ckpt, args, state_dict=state_dict,
    )
    feature_stats = _resolve_feature_stats_for_model(
        ckpt,
        norm_node=norm_node,
        norm_edge=norm_edge,
        stats_json=stats_json,
        norm_kind=norm_kind,
        state_has_node_buffers=state_has_node_buffers,
        state_has_edge_buffers=state_has_edge_buffers,
    )

    model = DisplacedVertexGNN(
        xdim=xdim,
        edim=edim,
        hdim=hidden_dim,
        n_layers=layers,
        dropout=dropout,
        layer_type=layer_type,
        gat_heads=gat_heads,
        sage_aggr=sage_aggr,
        edgeconv_aggr=edgeconv_aggr,
        pool=pool,
        use_fourier=use_fourier,
        fourier_base=fourier_base,
        fourier_min_exp=fourier_min_exp,
        fourier_max_exp=fourier_max_exp,
        normalize_node_features=norm_node,
        normalize_edge_features=norm_edge,
        feature_stats=feature_stats,
        feature_norm_clip=norm_clip,
    )
    _load_state_dict_allowing_initialized_norm_buffers(model, state_dict)
    model.eval()

    wrapper = DisplacedVertexOnnxExportWrapper(model, internal_dtype=internal_dtype).eval()
    return wrapper, ckpt


def infer_input_feature_metadata(xdim: int, edim: int) -> dict:
    node_names = list(NODE_FEATURE_NAMES) if len(NODE_FEATURE_NAMES) == int(xdim) else [f"x_{i}" for i in range(int(xdim))]
    edge_names = list(EDGE_FEATURE_NAMES) if len(EDGE_FEATURE_NAMES) == int(edim) else [f"edge_attr_{i}" for i in range(int(edim))]
    return {
        "input_names": ["x", "edge_index", "edge_attr", "n_muon_nodes"],
        "input_metadata_names": ["n_muon_nodes"],
        "node_feature_names": node_names,
        "node_feature_names_json": json.dumps(node_names),
        "edge_feature_names": edge_names,
        "edge_feature_names_json": json.dumps(edge_names),
        "input_node_feature_dim": int(xdim),
        "input_edge_feature_dim": int(edim),
        "edge_index_feature_names": ["src_node_index", "dst_node_index"],
    }


def add_onnx_metadata(onnx_path: Path, metadata: dict) -> dict:
    if onnx is None:
        return {"metadata_written": False, "metadata_reason": "onnx package not installed"}
    try:
        onnx_model = onnx.load(str(onnx_path))
        existing = {p.key: p for p in onnx_model.metadata_props}
        for key, value in metadata.items():
            value_str = value if isinstance(value, str) else json.dumps(_json_safe(value))
            if key in existing:
                existing[key].value = value_str
            else:
                prop = onnx_model.metadata_props.add()
                prop.key = key
                prop.value = value_str
        onnx.save(onnx_model, str(onnx_path))
        return {"metadata_written": True, "metadata_reason": ""}
    except Exception as exc:
        return {"metadata_written": False, "metadata_reason": repr(exc)}


def export_model_to_onnx(
    model: nn.Module,
    out_path: Path,
    *,
    dummy_x: torch.Tensor,
    dummy_edge_index: torch.Tensor,
    dummy_edge_attr: torch.Tensor,
    dummy_n_muon_nodes: torch.Tensor,
    onnx_opset: int,
    metadata: dict | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if onnx is None:
        return {"exported": False, "reason": "onnx package not installed", "onnx_path": str(out_path)}

    try:
        torch.onnx.export(
            model,
            (dummy_x.float(), dummy_edge_index.long(), dummy_edge_attr.float(), dummy_n_muon_nodes.long()),
            str(out_path),
            input_names=["x", "edge_index", "edge_attr", "n_muon_nodes"],
            output_names=["logits"],
            dynamic_axes={
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "edge_attr": {0: "num_edges"},
                "logits": {0: "num_graphs"},
            },
            opset_version=int(onnx_opset),
            export_params=True,
            do_constant_folding=True,
        )
        out = {"exported": True, "onnx_path": str(out_path), "reason": ""}
        if metadata:
            out.update(add_onnx_metadata(out_path, metadata))
        return out
    except Exception as exc:
        return {"exported": False, "reason": repr(exc), "onnx_path": str(out_path)}


def inspect_onnx_artifact(onnx_path: Path) -> dict:
    info = {"onnx_model_size_mb": np.nan, "onnx_initializer_elements": np.nan}
    try:
        info["onnx_model_size_mb"] = Path(onnx_path).stat().st_size / 1024**2
    except Exception:
        pass
    if onnx is None:
        return info
    try:
        try:
            model = onnx.load(str(onnx_path), load_external_data=False)
        except TypeError:
            model = onnx.load(str(onnx_path))
        n_elements = 0
        for initializer in model.graph.initializer:
            if initializer.dims:
                n_elements += int(np.prod(initializer.dims))
            else:
                n_elements += 1
        info["onnx_initializer_elements"] = int(n_elements)
    except Exception:
        pass
    return info


# ============================================================
# ONNX Runtime helpers
# ============================================================

def make_ort_session(onnx_path: Path, no_cuda: bool, intra_threads: int, inter_threads: int):
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")

    os.environ["OMP_NUM_THREADS"] = str(intra_threads)
    os.environ["OMP_PROC_BIND"] = "false"

    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.log_verbosity_level = 0
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = int(intra_threads)
    so.inter_op_num_threads = int(inter_threads)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")

    available = ort.get_available_providers()
    providers = []
    provider_options = []
    if not no_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        provider_options.append({
            "device_id": 0,
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
            "arena_extend_strategy": "kNextPowerOfTwo",
        })
    providers.append("CPUExecutionProvider")
    provider_options.append({})

    return ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=providers,
        provider_options=provider_options,
    )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax_second_class(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p = p / np.clip(np.sum(p, axis=1, keepdims=True), 1e-12, None)
    if p.shape[1] == 1:
        return p[:, 0]
    return p[:, 1]


def scores_from_onnx_output(logits, single_output_mode: str = "logit") -> np.ndarray:
    arr = np.asarray(logits)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        arr = arr.reshape(1)
        
    mode = str(single_output_mode).lower()

    if arr.ndim == 1:
        raw = arr.astype(np.float64)
        if mode == "prob":
            return raw.astype(np.float32)
        if mode in ("logit", "auto"):
            return sigmoid(raw).astype(np.float32)
        raise ValueError(f"Unsupported --single-output-mode {single_output_mode!r}")

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return scores_from_onnx_output(arr[:, 0], single_output_mode=mode)
        if mode == "prob":
            # Multi-output probability model: use the signal-class column.
            return arr[:, 1].astype(np.float32) if arr.shape[1] > 1 else arr[:, 0].astype(np.float32)
        # Multi-class logits from an external model.
        return softmax_second_class(arr).astype(np.float32)

    raise ValueError(f"Unsupported ONNX output shape: {np.asarray(logits).shape}")


# ============================================================
# Streaming histogram metrics
# ============================================================

def empty_hist_state(n_bins: int):
    n_bins = int(n_bins)
    return {
        "hist_pos": np.zeros(n_bins, dtype=np.int64),
        "hist_neg": np.zeros(n_bins, dtype=np.int64),
        "sample_hists": {},
        "n_bins": n_bins,
        "score_range": (0.0, 1.0),
        "num_predictions": 0,
    }


def _update_hist_arrays(hist_pos: np.ndarray, hist_neg: np.ndarray, probs: np.ndarray, labels: np.ndarray):
    n_bins = int(hist_pos.shape[0])
    probs = np.clip(np.asarray(probs, dtype=np.float32).reshape(-1), 0.0, 1.0)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    bin_idx = np.floor(probs * n_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    pos_bins = bin_idx[labels == 1]
    neg_bins = bin_idx[labels == 0]
    if pos_bins.size:
        np.add.at(hist_pos, pos_bins, 1)
    if neg_bins.size:
        np.add.at(hist_neg, neg_bins, 1)


def update_hist_state(state: dict, probs: np.ndarray, labels: np.ndarray, sample: str | None = None):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"probs/labels mismatch: {probs.shape[0]} vs {labels.shape[0]}")

    _update_hist_arrays(state["hist_pos"], state["hist_neg"], probs, labels)
    state["num_predictions"] += int(labels.shape[0])

    if sample is not None:
        sample = str(sample)
        if sample not in state["sample_hists"]:
            n_bins = int(state["n_bins"])
            state["sample_hists"][sample] = {
                "hist_pos": np.zeros(n_bins, dtype=np.int64),
                "hist_neg": np.zeros(n_bins, dtype=np.int64),
                "num_predictions": 0,
            }
        sample_state = state["sample_hists"][sample]
        _update_hist_arrays(sample_state["hist_pos"], sample_state["hist_neg"], probs, labels)
        sample_state["num_predictions"] += int(labels.shape[0])


def threshold_bin_from_value(threshold: float, n_bins: int) -> int:
    return int(np.clip(np.floor(float(threshold) * int(n_bins)), 0, int(n_bins) - 1))


def threshold_metrics_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray, threshold_bin: int):
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    hist_neg = np.asarray(hist_neg, dtype=np.int64)
    n_bins = int(hist_pos.shape[0])
    threshold_bin = int(np.clip(threshold_bin, 0, n_bins - 1))

    tp = int(hist_pos[threshold_bin:].sum())
    fp = int(hist_neg[threshold_bin:].sum())
    fn = int(hist_pos[:threshold_bin].sum())
    tn = int(hist_neg[:threshold_bin].sum())
    total = max(tp + tn + fp + fn, 1)
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / total
    balanced_accuracy = 0.5 * (tpr + specificity)
    f1 = 2.0 * precision * tpr / max(precision + tpr, 1e-12)
    return {
        "threshold": float(threshold_bin / n_bins),
        "threshold_bin": int(threshold_bin),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "tpr": float(tpr),
        "recall": float(tpr),
        "signal_efficiency": float(tpr),
        "signal_loss": float(1.0 - tpr),
        "fpr": float(fpr),
        "background_rejection": float(1.0 - fpr),
        "precision": float(precision),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def signal_efficiency_from_hist_pos(hist_pos: np.ndarray, threshold_bin: int) -> dict:
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    n_bins = int(hist_pos.shape[0])
    threshold_bin = int(np.clip(threshold_bin, 0, n_bins - 1))
    num_signal = int(hist_pos.sum())
    num_pass_signal = int(hist_pos[threshold_bin:].sum())
    signal_efficiency = num_pass_signal / max(num_signal, 1)
    return {
        "threshold": float(threshold_bin / n_bins),
        "threshold_bin": int(threshold_bin),
        "num_signal_events": num_signal,
        "num_pass_signal_events": num_pass_signal,
        "signal_efficiency": float(signal_efficiency),
        "signal_efficiency_percent": float(100.0 * signal_efficiency),
        "signal_loss": float(1.0 - signal_efficiency),
    }


def operating_point_payload_key(prefix: str, value: float) -> str:
    return f"{prefix}_{str(float(value)).replace('.', 'p')}"


def operating_point_at_target_tpr_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray, target_tpr: float):
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    hist_neg = np.asarray(hist_neg, dtype=np.int64)
    n_bins = int(hist_pos.shape[0])
    tp_from_bin = np.cumsum(hist_pos[::-1])[::-1]
    total_pos = max(int(hist_pos.sum()), 1)
    tpr_from_bin = tp_from_bin / total_pos
    high_to_low = np.arange(n_bins - 1, -1, -1)
    ok = np.where(tpr_from_bin[high_to_low] >= float(target_tpr))[0]
    threshold_bin = int(high_to_low[int(ok[0])]) if len(ok) else 0
    out = threshold_metrics_from_hist(hist_pos, hist_neg, threshold_bin)
    out["target_tpr"] = float(target_tpr)
    return out


def operating_point_at_target_fpr_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray, target_fpr: float):
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    hist_neg = np.asarray(hist_neg, dtype=np.int64)
    n_bins = int(hist_pos.shape[0])
    fp_from_bin = np.cumsum(hist_neg[::-1])[::-1]
    total_neg = max(int(hist_neg.sum()), 1)
    fpr_from_bin = fp_from_bin / total_neg

    # Pick the lowest threshold that still satisfies FPR <= target_fpr, which
    # gives the highest TPR among valid thresholds. If no threshold satisfies it,
    # use the strictest threshold.
    low_to_high = np.arange(0, n_bins, dtype=np.int64)
    ok = np.where(fpr_from_bin[low_to_high] <= float(target_fpr))[0]
    threshold_bin = int(low_to_high[int(ok[0])]) if len(ok) else n_bins - 1
    out = threshold_metrics_from_hist(hist_pos, hist_neg, threshold_bin)
    out["target_fpr"] = float(target_fpr)
    return out


def roc_curve_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray):
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    hist_neg = np.asarray(hist_neg, dtype=np.int64)
    n_bins = int(hist_pos.shape[0])
    total_pos = max(int(hist_pos.sum()), 1)
    total_neg = max(int(hist_neg.sum()), 1)
    tp_from_bin = np.cumsum(hist_pos[::-1])[::-1]
    fp_from_bin = np.cumsum(hist_neg[::-1])[::-1]
    high_to_low = np.arange(n_bins - 1, -1, -1)
    tpr = tp_from_bin[high_to_low] / total_pos
    fpr = fp_from_bin[high_to_low] / total_neg
    thresholds = high_to_low.astype(np.float64) / float(n_bins)
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    thresholds = np.concatenate([[1.0], thresholds])
    return fpr.astype(np.float32), tpr.astype(np.float32), thresholds.astype(np.float32)


def partial_auc_low_fpr_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray, fpr_max: float):
    fpr, tpr, _ = roc_curve_from_hist(hist_pos, hist_neg)
    x = np.asarray(fpr, dtype=float)
    y = np.asarray(tpr, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    x_unique, y_unique = [], []
    for xv in np.unique(x):
        mask = x == xv
        x_unique.append(float(xv))
        y_unique.append(float(np.max(y[mask])))
    x = np.asarray(x_unique, dtype=float)
    y = np.asarray(y_unique, dtype=float)

    fpr_max = float(fpr_max)
    if fpr_max <= 0:
        return {"fpr_max": fpr_max, "raw_area": 0.0, "normalized_area": np.nan}
    if fpr_max < x[-1] and not np.any(np.isclose(x, fpr_max)):
        y_at_max = np.interp(fpr_max, x, y)
        x = np.concatenate([x, [fpr_max]])
        y = np.concatenate([y, [y_at_max]])
        order = np.argsort(x, kind="mergesort")
        x, y = x[order], y[order]
    mask = x <= fpr_max
    if np.count_nonzero(mask) < 2:
        return {"fpr_max": fpr_max, "raw_area": 0.0, "normalized_area": 0.0}
    raw_area = auc(x[mask], y[mask])
    return {"fpr_max": fpr_max, "raw_area": float(raw_area), "normalized_area": float(raw_area / fpr_max)}


def confusion_matrix_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    n_bins = int(np.asarray(hist_pos).shape[0])
    m = threshold_metrics_from_hist(hist_pos, hist_neg, threshold_bin_from_value(threshold, n_bins))
    return np.asarray([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.int64)


def threshold_scan_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray):
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    hist_neg = np.asarray(hist_neg, dtype=np.int64)
    n_bins = int(hist_pos.shape[0])
    tp = np.cumsum(hist_pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(hist_neg[::-1])[::-1].astype(np.float64)
    total_pos = max(float(hist_pos.sum()), 1.0)
    total_neg = max(float(hist_neg.sum()), 1.0)
    fn = total_pos - tp
    tn = total_neg - fp
    thresholds = np.arange(n_bins, dtype=np.float64) / float(n_bins)
    tpr = tp / total_pos
    fpr = fp / total_neg
    precision = tp / np.clip(tp + fp, 1.0, None)
    accuracy = (tp + tn) / np.clip(tp + tn + fp + fn, 1.0, None)
    specificity = tn / np.clip(tn + fp, 1.0, None)
    return {
        "thresholds": thresholds.astype(np.float32),
        "tpr": tpr.astype(np.float32),
        "fpr": fpr.astype(np.float32),
        "signal_efficiency": tpr.astype(np.float32),
        "signal_loss": (1.0 - tpr).astype(np.float32),
        "background_rejection": (1.0 - fpr).astype(np.float32),
        "precision": precision.astype(np.float32),
        "accuracy": accuracy.astype(np.float32),
        "balanced_accuracy": (0.5 * (tpr + specificity)).astype(np.float32),
    }


def significance_scan_from_hist(hist_pos: np.ndarray, hist_neg: np.ndarray):
    """Return S/sqrt(S+B) as a function of the score cut.

    S and B are cumulative pass counts for events with score >= cut.  This is
    intentionally histogram-based, so the scan can be produced without storing
    every event score.
    """
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    hist_neg = np.asarray(hist_neg, dtype=np.int64)
    if hist_pos.shape != hist_neg.shape:
        raise ValueError(f"hist_pos/hist_neg shape mismatch: {hist_pos.shape} vs {hist_neg.shape}")
    n_bins = int(hist_pos.shape[0])
    thresholds = np.arange(n_bins, dtype=np.float64) / float(n_bins)
    signal = np.cumsum(hist_pos[::-1])[::-1].astype(np.float64)
    background = np.cumsum(hist_neg[::-1])[::-1].astype(np.float64)
    total = signal + background
    significance = np.zeros_like(signal, dtype=np.float64)
    valid = total > 0.0
    significance[valid] = signal[valid] / np.sqrt(total[valid])
    if significance.size and np.any(np.isfinite(significance)):
        best_idx = int(np.nanargmax(significance))
    else:
        best_idx = 0
    return {
        "definition": "S/sqrt(S+B), with S and B counted for score >= threshold",
        "thresholds": thresholds.astype(np.float32),
        "signal": signal.astype(np.float64),
        "background": background.astype(np.float64),
        "significance": significance.astype(np.float32),
        "best_threshold": float(thresholds[best_idx]) if n_bins else np.nan,
        "best_threshold_bin": int(best_idx),
        "best_significance": float(significance[best_idx]) if n_bins else np.nan,
        "best_signal": float(signal[best_idx]) if n_bins else np.nan,
        "best_background": float(background[best_idx]) if n_bins else np.nan,
    }


def plot_sample_payload_from_hist(
    hist_pos: np.ndarray,
    hist_neg: np.ndarray,
    *,
    target_tprs: list[float],
    target_fprs: list[float],
    default_threshold: float,
    include_threshold_scan: bool,
):
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    hist_neg = np.asarray(hist_neg, dtype=np.int64)
    fpr, tpr, thresholds = roc_curve_from_hist(hist_pos, hist_neg)
    out = {
        "num_predictions": int(hist_pos.sum() + hist_neg.sum()),
        "num_events": int(hist_pos.sum() + hist_neg.sum()),
        "num_signal": int(hist_pos.sum()),
        "num_background": int(hist_neg.sum()),
        "num_positive": int(hist_pos.sum()),
        "num_negative": int(hist_neg.sum()),
        "histograms": {
            "hist_pos": hist_pos,
            "hist_neg": hist_neg,
            "n_bins": int(hist_pos.shape[0]),
            "score_range": (0.0, 1.0),
        },
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": float(auc(fpr, tpr))},
        "confusion_matrices": {
            f"threshold_{str(default_threshold).replace('.', 'p')}": confusion_matrix_from_hist(
                hist_pos, hist_neg, threshold=default_threshold,
            )
        },
        "operating_points_at_target_tpr": {},
        "operating_points_at_target_fpr": {},
    }
    for target_tpr in target_tprs:
        op = operating_point_at_target_tpr_from_hist(hist_pos, hist_neg, target_tpr)
        key = f"target_tpr_{str(target_tpr).replace('.', 'p')}"
        out["operating_points_at_target_tpr"][key] = op
        out["confusion_matrices"][key] = np.asarray([[op["tn"], op["fp"]], [op["fn"], op["tp"]]], dtype=np.int64)
    for target_fpr in target_fprs:
        op = operating_point_at_target_fpr_from_hist(hist_pos, hist_neg, target_fpr)
        key = f"target_fpr_{str(target_fpr).replace('.', 'p')}"
        out["operating_points_at_target_fpr"][key] = op
        out["confusion_matrices"][key] = np.asarray([[op["tn"], op["fp"]], [op["fn"], op["tp"]]], dtype=np.int64)
    if include_threshold_scan:
        out["threshold_scan"] = threshold_scan_from_hist(hist_pos, hist_neg)
    return out


def attach_global_threshold_signal_efficiencies(samples: dict, hist_state: dict, *, target_fprs: list[float], default_threshold: float):
    """Add per-signal-sample efficiencies using thresholds from the merged validation sample.

    For signal-only samples, computing a target-FPR working point from that sample
    alone is not meaningful because the sample has no background. This helper
    first chooses the threshold on the merged validation histograms (`__all__`),
    then applies that threshold to each individual sample's signal histogram.
    """
    hist_pos_all = np.asarray(hist_state["hist_pos"], dtype=np.int64)
    hist_neg_all = np.asarray(hist_state["hist_neg"], dtype=np.int64)
    n_bins = int(hist_pos_all.shape[0])

    # Include the default threshold too, useful for quick checks.
    default_threshold_bin = threshold_bin_from_value(default_threshold, n_bins)
    default_global = threshold_metrics_from_hist(hist_pos_all, hist_neg_all, default_threshold_bin)

    for sample, sample_state in sorted(hist_state.get("sample_hists", {}).items()):
        if sample not in samples:
            continue
        sample_hist_pos = np.asarray(sample_state["hist_pos"], dtype=np.int64)
        sample_hist_neg = np.asarray(sample_state["hist_neg"], dtype=np.int64)
        if int(sample_hist_pos.sum()) <= 0:
            continue

        samples[sample]["signal_efficiency_at_global_default_threshold"] = {
            **signal_efficiency_from_hist_pos(sample_hist_pos, default_threshold_bin),
            "global_threshold": float(default_global["threshold"]),
            "global_fpr": float(default_global["fpr"]),
            "global_tpr": float(default_global["tpr"]),
            "global_background_rejection": float(default_global["background_rejection"]),
            "num_background_events_in_sample": int(sample_hist_neg.sum()),
        }

        target_block = {}
        for target_fpr in target_fprs:
            global_op = operating_point_at_target_fpr_from_hist(hist_pos_all, hist_neg_all, target_fpr)
            key = operating_point_payload_key("target_fpr", target_fpr)
            row = signal_efficiency_from_hist_pos(sample_hist_pos, int(global_op["threshold_bin"]))
            row.update({
                "target_fpr": float(target_fpr),
                "global_threshold": float(global_op["threshold"]),
                "global_fpr": float(global_op["fpr"]),
                "global_tpr": float(global_op["tpr"]),
                "global_background_rejection": float(global_op["background_rejection"]),
                "num_background_events_in_sample": int(sample_hist_neg.sum()),
            })
            target_block[key] = row
        samples[sample]["signal_efficiency_at_global_target_fpr"] = target_block


def plot_model_payload_from_hist(
    *,
    model_id: str,
    display_name: str,
    run_group: str,
    layer_type: str,
    ckpt_path: str,
    onnx_path: str,
    hist_state: dict,
    lat_ort: np.ndarray,
    providers: list[str],
    target_tprs: list[float],
    target_fprs: list[float],
    default_threshold: float,
    include_threshold_scan: bool,
):
    samples = {
        "__all__": plot_sample_payload_from_hist(
            hist_state["hist_pos"], hist_state["hist_neg"],
            target_tprs=target_tprs,
            target_fprs=target_fprs,
            default_threshold=default_threshold,
            include_threshold_scan=include_threshold_scan,
        )
    }
    # Store the global validation significance scan once per model.  Per-sample
    # signal efficiencies remain separate below, but the discovery significance
    # S/sqrt(S+B) should use the merged validation signal/background counts.
    samples["__all__"]["significance_scan"] = significance_scan_from_hist(
        hist_state["hist_pos"], hist_state["hist_neg"]
    )
    for sample, sample_state in sorted(hist_state.get("sample_hists", {}).items()):
        samples[sample] = plot_sample_payload_from_hist(
            sample_state["hist_pos"], sample_state["hist_neg"],
            target_tprs=target_tprs,
            target_fprs=target_fprs,
            default_threshold=default_threshold,
            include_threshold_scan=include_threshold_scan,
        )

    attach_global_threshold_signal_efficiencies(
        samples,
        hist_state,
        target_fprs=target_fprs,
        default_threshold=default_threshold,
    )

    return {
        "model_id": model_id,
        "display_name": display_name,
        "plot_name": display_name,
        "run_group": run_group,
        "layer_type": layer_type,
        "ckpt_path": ckpt_path,
        "onnx_path": onnx_path,
        "is_reference": False,
        "providers": list(providers),
        "num_validation_events": int(len(lat_ort)),
        "mean_event_latency_ms_onnxruntime": float(np.mean(lat_ort)),
        "sample_names": sorted(samples.keys()),
        "samples": samples,
    }


# ============================================================
# Dataset / evaluation
# ============================================================

def resolve_normalization_for_checkpoint(ckpt: dict, args=None, *, state_dict: dict[str, Any] | None = None) -> tuple[bool, bool, str | None, str, float]:
    if state_dict is None and isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = _strip_module_prefix(ckpt["model_state"])
    state_has_node_buffers, state_has_edge_buffers = (False, False)
    if state_dict is not None:
        state_has_node_buffers, state_has_edge_buffers = _normalization_buffers_in_state(state_dict)

    norm_node = state_has_node_buffers or bool(_ckpt_get(ckpt, "normalize_node_features", False))
    norm_edge = state_has_edge_buffers or bool(_ckpt_get(ckpt, "normalize_edge_features", False))

    if args is not None:
        if args.normalize_node_features is not None and not state_has_node_buffers:
            norm_node = bool(args.normalize_node_features)
        if args.normalize_edge_features is not None and not state_has_edge_buffers:
            norm_edge = bool(args.normalize_edge_features)

    stats_json = args.feature_stats_json if args is not None else None
    if stats_json is None:
        stats_json = _ckpt_get(ckpt, "feature_stats_json", None)
    if stats_json is None and DEFAULT_FEATURE_STATS_JSON.exists():
        stats_json = str(DEFAULT_FEATURE_STATS_JSON)

    raw_norm_kind = _ckpt_get(ckpt, "feature_norm_kind", None)
    if args is not None and args.feature_norm_kind is not None:
        norm_kind = str(args.feature_norm_kind)
    elif raw_norm_kind in ("standard", "robust"):
        norm_kind = str(raw_norm_kind)
    else:
        norm_kind = "standard"
    norm_clip = float(
        args.feature_norm_clip
        if args is not None and args.feature_norm_clip is not None
        else _ckpt_get(ckpt, "feature_norm_clip", -1.0)
    )
    return norm_node, norm_edge, stats_json, norm_kind, norm_clip


def build_eval_indices(args, input_paths: list[Path], dataset: H5EventDataset) -> tuple[np.ndarray, dict]:
    meta = {"split_file": None, "split_key": args.split_key, "strict_split_check": bool(args.strict_split_check)}
    n = len(dataset)

    if args.split_key == "all":
        idx = np.arange(n, dtype=np.int64)
    else:
        if args.split_file is None:
            raise RuntimeError("--split-file is required unless --split-key all is used.")
        split_path = Path(args.split_file).expanduser().resolve()
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        split = np.load(split_path, allow_pickle=True)
        _check_split_paths_compatible(split, [str(p) for p in input_paths], strict=args.strict_split_check)
        key = args.split_key
        if key not in split.files:
            raise KeyError(f"Split key {key!r} not found in {split_path}. Available keys: {list(split.files)}")
        idx = split[key].astype(np.int64)
        if idx.size == 0:
            raise RuntimeError(f"Split {key!r} is empty in {split_path}")
        if idx.min() < 0 or idx.max() >= n:
            raise RuntimeError(f"Split indices in {split_path} are out of range for dataset length {n}.")
        meta["split_file"] = str(split_path)
        meta["split_keys"] = list(split.files)

    if args.max_val_events is not None and args.max_val_events > 0:
        idx = idx[: int(args.max_val_events)]
    return idx, meta


GENERIC_DATASET_SAMPLE_NAMES = {"", "unknown", "validation", "val", "valid", "train", "training", "test", "all", "displacedvtx", "displaced_vertex"}


def _is_generic_dataset_name(name: str) -> bool:
    return str(name).strip().lower() in GENERIC_DATASET_SAMPLE_NAMES


def dataset_sample_for_index(dataset: H5EventDataset, idx: int) -> str:
    """Return the physics sample for one dataset index.

    Some H5 productions store a split/dataset label such as "validation" in
    dataset_names.  That is useful for the global validation sample, but it is
    not enough for per-signal-sample plots.  In that case, fall back to the
    root file or H5 part file name so each requested physics sample is filled
    separately while still using the validation split indices.
    """
    try:
        name = decode_attr_to_str(dataset.dataset_names[int(idx)])
        sample = sample_name_from_file(name)
        if sample and not _is_generic_dataset_name(sample):
            return sample
    except Exception:
        pass
    try:
        root_file = decode_attr_to_str(dataset.root_files[int(idx)])
        sample = sample_name_from_file(root_file)
        if sample and not _is_generic_dataset_name(sample):
            return sample
    except Exception:
        pass
    try:
        fi, _ = dataset.index[int(idx)]
        return sample_name_from_file(str(dataset.h5_paths[int(fi)]))
    except Exception:
        return "unknown"


def make_ort_feed(sess, item: dict) -> dict[str, np.ndarray]:
    arrays = {
        "x": np.ascontiguousarray(item["x"].numpy(), dtype=np.float32),
        "edge_index": np.ascontiguousarray(item["edge_index"].numpy(), dtype=np.int64),
        "edge_attr": np.ascontiguousarray(item["edge_attr"].numpy(), dtype=np.float32),
        "n_muon_nodes": np.ascontiguousarray(item["n_muon_nodes"].numpy().reshape(1), dtype=np.int64),
    }
    feed = {}
    for inp in sess.get_inputs():
        name = inp.name
        lower = name.lower()
        if name in arrays:
            feed[name] = arrays[name]
        elif "n_muon" in lower or "nmuon" in lower:
            feed[name] = arrays["n_muon_nodes"]
        elif "edge_index" in lower or lower.endswith("edgeindex"):
            feed[name] = arrays["edge_index"]
        elif "edge_attr" in lower or "edgeattr" in lower:
            feed[name] = arrays["edge_attr"]
        elif lower in ("features", "feature", "x", "nodes", "node_features") or "feature" in lower:
            feed[name] = arrays["x"]
        else:
            raise KeyError(
                f"Could not map ONNX input {name!r}. "
                "Available expected inputs are x, edge_index, edge_attr, n_muon_nodes."
            )
    return feed


def evaluate_onnx_model_on_files(
    onnx_path: Path,
    input_paths: list[Path],
    *,
    ckpt: dict,
    args,
    no_cuda: bool = False,
    intra_threads: int = 1,
    inter_threads: int = 1,
    single_output_mode: str = "auto",
    metric_bins: int = 200000,
):
    norm_node, norm_edge, stats_json, norm_kind, norm_clip = resolve_normalization_for_checkpoint(ckpt, args)

    # The patched H5EventDataset returns raw features.  Normalization is handled
    # by DisplacedVertexGNN/ONNX using n_muon_nodes, so inference must not
    # normalize x or edge_attr before feeding the model.
    dataset = H5EventDataset([str(p) for p in input_paths])
    eval_indices, split_meta = build_eval_indices(args, input_paths, dataset)

    sess = make_ort_session(
        onnx_path=onnx_path,
        no_cuda=no_cuda,
        intra_threads=intra_threads,
        inter_threads=inter_threads,
    )
    output_names = [out.name for out in sess.get_outputs()]
    hist_state = empty_hist_state(metric_bins)
    event_latencies_ms: list[float] = []

    print(
        f"[eval] {Path(onnx_path).name}: evaluating {len(eval_indices)} events "
        f"from {len(input_paths)} file(s), split_key={args.split_key}, "
        f"model_side_normalize_node={norm_node}, model_side_normalize_edge={norm_edge}",
        flush=True,
    )

    for n_done, ds_idx in enumerate(eval_indices, start=1):
        item = dataset[int(ds_idx)]
        feed = make_ort_feed(sess, item)
        t0 = time.perf_counter()
        logits = sess.run(output_names, feed)[0]
        event_latencies_ms.append((time.perf_counter() - t0) * 1e3)
        probs = scores_from_onnx_output(logits, single_output_mode=single_output_mode)
        labels = item["y"].numpy().astype(np.int64).reshape(-1)

        if probs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Prediction/label length mismatch for eval index {int(ds_idx)}: "
                f"prediction rows={probs.shape[0]}, label rows={labels.shape[0]}, "
                f"raw output shape={np.asarray(logits).shape}"
            )

        sample = dataset_sample_for_index(dataset, int(ds_idx))
        update_hist_state(hist_state, probs, labels, sample=sample)

        if args.print_every and n_done % int(args.print_every) == 0:
            print(f"[eval] {Path(onnx_path).name}: {n_done}/{len(eval_indices)} events", flush=True)

        del item, feed, logits, probs, labels

    if hist_state["num_predictions"] == 0:
        raise RuntimeError("No validation events were evaluated.")

    providers = sess.get_providers()
    del sess, dataset
    release_memory()

    print(
        f"[eval] {Path(onnx_path).name}: evaluated {len(event_latencies_ms)} events "
        f"({hist_state['hist_pos'].sum()} signal, {hist_state['hist_neg'].sum()} background)",
        flush=True,
    )

    sample_keys = sorted(hist_state.get("sample_hists", {}).keys())
    print(
        f"[eval] {Path(onnx_path).name}: filled {len(sample_keys)} per-sample histogram(s): "
        f"{sample_keys[:20]}{' ...' if len(sample_keys) > 20 else ''}",
        flush=True,
    )

    eval_meta = {
        **split_meta,
        "normalize_node_features": bool(norm_node),
        "normalize_edge_features": bool(norm_edge),
        "feature_stats_json": str(stats_json) if stats_json is not None else None,
        "feature_norm_kind": str(norm_kind),
        "feature_norm_clip": float(norm_clip),
        "feature_norm_in_model": bool(norm_node or norm_edge),
    }
    return hist_state, np.asarray(event_latencies_ms, dtype=float), providers, eval_meta


# ============================================================
# Tables / metadata
# ============================================================

def load_existing_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def upsert_rows(existing: pd.DataFrame, new_rows: list[dict], key_cols: list[str]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        return new_df
    if new_df.empty:
        return existing
    for c in key_cols:
        if c not in existing.columns:
            existing[c] = np.nan
        if c not in new_df.columns:
            new_df[c] = np.nan
    combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    return combined


def load_existing_plot_entries(plot_payload_path: Path) -> dict[str, dict]:
    if not plot_payload_path.exists():
        return {}
    try:
        payload = pd.read_pickle(plot_payload_path)
        entries = payload.get("models", [])
        return {str(entry["model_id"]): entry for entry in entries if "model_id" in entry}
    except Exception as exc:
        print(f"[plot-payload] Could not read existing payload {plot_payload_path}: {exc}", flush=True)
        return {}


def plot_entry_has_samples(entry: dict | None, sample_names: Iterable[str]) -> bool:
    """Return True when a cached plot_payload model entry contains all required samples."""
    if not isinstance(entry, dict):
        return False
    available = set(entry.get("samples", {}).keys())
    required = set(str(s) for s in sample_names)
    return required.issubset(available)


def metadata_from_checkpoint(item: LoadedCheckpoint, args, feature_metadata: dict, export_info: dict, eval_meta: dict | None = None) -> dict:
    keys = [
        "task", "model_type", "xdim", "edim", "hidden_dim", "layers", "dropout", "layer_type",
        "gat_heads", "sage_aggr", "edgeconv_aggr", "pool", "fourier", "fourier_base",
        "fourier_min_exp", "fourier_max_exp", "lr", "weight_decay", "edge_dropout",
        "feat_noise_std", "loss_type", "threshold", "target_fpr", "best_monitor",
        "early_stop_monitor", "best_ckpt_epoch", "run_id", "epoch", "ema", "ema_decay",
        "normalize_node_features", "normalize_edge_features", "feature_norm_kind", "feature_norm_clip",
        "feature_norm_in_model",
    ]
    metadata = {
        "model_id": stable_model_id(item),
        "source_checkpoint": str(item.ckpt_path),
        "display_name": item.display_name,
        "run_group": item.run_group,
        "layer_type": item.layer_type,
        "opset": int(args.onnx_opset),
        **feature_metadata,
        **export_info,
    }
    if eval_meta:
        metadata.update(eval_meta)
    for key in keys:
        value = _ckpt_get(item.ckpt, key, None)
        if value is not None:
            metadata[key] = _json_safe(value)
    return metadata


def append_metric_rows_from_hist(
    *,
    summary_rows,
    op_tpr_rows,
    op_fpr_rows,
    pauc_fpr_rows,
    target_tprs,
    target_fprs,
    partial_auc_fpr_max,
    model_id,
    display_name,
    run_group,
    layer_type,
    ckpt_path,
    threshold_ckpt,
    target_fpr_ckpt,
    best_monitor,
    hist_state,
    lat_ort,
    providers,
    export_info,
    eval_meta,
    metrics_source="onnxruntime_hist",
):
    hist_pos = hist_state["hist_pos"]
    hist_neg = hist_state["hist_neg"]
    roc_fpr, roc_tpr, _ = roc_curve_from_hist(hist_pos, hist_neg)
    try:
        threshold_value = float(threshold_ckpt)
        if not np.isfinite(threshold_value):
            threshold_value = 0.5
    except Exception:
        threshold_value = 0.5
    default_op = threshold_metrics_from_hist(hist_pos, hist_neg, threshold_bin_from_value(threshold_value, len(hist_pos)))

    summary = {
        "model_id": model_id,
        "display_name": display_name,
        "run_group": run_group,
        "layer_type": layer_type,
        "ckpt_path": ckpt_path,
        "threshold_ckpt": float(threshold_ckpt) if threshold_ckpt is not None else np.nan,
        "target_fpr_ckpt": float(target_fpr_ckpt) if target_fpr_ckpt is not None else np.nan,
        "best_monitor": best_monitor,
        "num_events": int(hist_state["num_predictions"]),
        "num_validation_events": int(len(lat_ort)),
        "num_signal_events": int(hist_pos.sum()),
        "num_background_events": int(hist_neg.sum()),
        "roc_auc": float(auc(roc_fpr, roc_tpr)),
        "mean_event_latency_ms_onnxruntime": float(np.mean(lat_ort)),
        "median_event_latency_ms_onnxruntime": float(np.median(lat_ort)),
        "providers": ",".join(providers),
        "exported_onnx": True,
        "onnx_export_reason": export_info.get("reason", ""),
        "metrics_source": metrics_source,
        "default_threshold_accuracy": default_op["accuracy"],
        "default_threshold_tpr": default_op["tpr"],
        "default_threshold_fpr": default_op["fpr"],
        "default_threshold_precision": default_op["precision"],
        **{f"eval_{k}": v for k, v in eval_meta.items() if isinstance(v, (str, int, float, bool)) or v is None},
    }
    summary_rows.append(summary)

    for target_tpr in target_tprs:
        op = operating_point_at_target_tpr_from_hist(hist_pos, hist_neg, target_tpr)
        op.update({
            "model_id": model_id,
            "display_name": display_name,
            "run_group": run_group,
            "layer_type": layer_type,
            "metrics_source": metrics_source,
        })
        op_tpr_rows.append(op)

    for target_fpr in target_fprs:
        op = operating_point_at_target_fpr_from_hist(hist_pos, hist_neg, target_fpr)
        op.update({
            "model_id": model_id,
            "display_name": display_name,
            "run_group": run_group,
            "layer_type": layer_type,
            "metrics_source": metrics_source,
        })
        op_fpr_rows.append(op)

    for fpr_max in partial_auc_fpr_max:
        pa = partial_auc_low_fpr_from_hist(hist_pos, hist_neg, fpr_max)
        pauc_fpr_rows.append({
            "model_id": model_id,
            "display_name": display_name,
            "run_group": run_group,
            "layer_type": layer_type,
            "fpr_max": pa["fpr_max"],
            "raw_area": pa["raw_area"],
            "normalized_area": pa["normalized_area"],
            "metrics_source": metrics_source,
        })



def append_signal_efficiency_rows_from_hist(
    *,
    signal_efficiency_rows,
    target_fprs,
    model_id,
    display_name,
    run_group,
    layer_type,
    hist_state,
    metrics_source="onnxruntime_hist",
):
    """Append per-signal-sample efficiencies at global target-FPR thresholds.

    The threshold for each target FPR is computed once on the merged validation
    sample. It is then applied to every sample that contains signal events.
    """
    hist_pos_all = np.asarray(hist_state["hist_pos"], dtype=np.int64)
    hist_neg_all = np.asarray(hist_state["hist_neg"], dtype=np.int64)
    for target_fpr in target_fprs:
        global_op = operating_point_at_target_fpr_from_hist(hist_pos_all, hist_neg_all, target_fpr)
        threshold_bin = int(global_op["threshold_bin"])
        for sample, sample_state in sorted(hist_state.get("sample_hists", {}).items()):
            sample_hist_pos = np.asarray(sample_state["hist_pos"], dtype=np.int64)
            sample_hist_neg = np.asarray(sample_state["hist_neg"], dtype=np.int64)
            if int(sample_hist_pos.sum()) <= 0:
                continue
            row = signal_efficiency_from_hist_pos(sample_hist_pos, threshold_bin)
            row.update({
                "model_id": model_id,
                "display_name": display_name,
                "run_group": run_group,
                "layer_type": layer_type,
                "sample": str(sample),
                "target_fpr": float(target_fpr),
                "global_threshold": float(global_op["threshold"]),
                "global_fpr": float(global_op["fpr"]),
                "global_tpr": float(global_op["tpr"]),
                "global_background_rejection": float(global_op["background_rejection"]),
                "num_background_events_in_sample": int(sample_hist_neg.sum()),
                "metrics_source": metrics_source,
            })
            signal_efficiency_rows.append(row)


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Export tuned DisplacedVertex graph classifiers to ONNX and evaluate them with ONNX Runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt-glob", nargs="+", default=["tuning_*/checkpoints/*.pt"],
                    help="One or more glob patterns for tuned/refit .pt checkpoints.")
    ap.add_argument("--data-glob", default=DEFAULT_DATA_GLOB,
                    help="Fallback glob used when --no-explicit-dv-samples is set.")
    ap.add_argument("--signal-data-globs", nargs="*", default=DEFAULT_SIGNAL_DATA_GLOBS,
                    help="Signal H5 glob patterns. Defaults to the explicit DV signal samples.")
    ap.add_argument("--background-data-globs", nargs="*", default=DEFAULT_BACKGROUND_DATA_GLOBS,
                    help="Background H5 glob patterns. Defaults to the explicit DV background samples.")
    ap.add_argument("--no-explicit-dv-samples", action="store_true", default=False,
                    help="Use --data-glob instead of the explicit signal/background glob lists.")
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--split-key", default="val_idx", help="Split key to evaluate, e.g. val_idx, train_idx, or all.")
    ap.add_argument("--strict-split-check", action="store_true", default=False)
    ap.add_argument("--feature-stats-json", default=None,
                    help=(
                        "Override feature stats JSON. This is only needed for legacy checkpoints "
                        "that requested normalization but did not save normalization buffers. "
                        "New checkpoints carry normalization inside the model state."
                    ))
    ap.add_argument("--feature-norm-kind", default=None, choices=["standard", "robust"])
    ap.add_argument("--normalize-node-features", dest="normalize_node_features", action="store_true", default=None,
                    help="Override checkpoint metadata for model-side node normalization when exporting old checkpoints.")
    ap.add_argument("--no-normalize-node-features", dest="normalize_node_features", action="store_false",
                    help="Disable model-side node normalization only for checkpoints without normalization buffers.")
    ap.add_argument("--normalize-edge-features", dest="normalize_edge_features", action="store_true", default=None,
                    help="Override checkpoint metadata for model-side edge normalization when exporting old checkpoints.")
    ap.add_argument("--no-normalize-edge-features", dest="normalize_edge_features", action="store_false",
                    help="Disable model-side edge normalization only for checkpoints without normalization buffers.")
    ap.add_argument("--feature-norm-clip", type=float, default=None)

    ap.add_argument("--output-dir", default="onnx_eval_compare_displaced_vertex")
    ap.add_argument("--max-val-events", type=int, default=None)

    ap.add_argument("--onnx-opset", type=int, default=19)
    ap.add_argument("--onnx-dummy-nodes", type=int, default=64)
    ap.add_argument("--onnx-dummy-edges", type=int, default=512)
    ap.add_argument("--export-onnx", action="store_true", default=True)
    ap.add_argument("--run-ort-validation", action="store_true", default=True)
    ap.add_argument("--force-recompute", action="store_true", default=False)
    ap.add_argument("--save-pytorch-parity", action="store_true", default=False,
                    help="Reserved for symmetry with Bucket_results_tune.py. Full parity arrays are not stored.")

    ap.add_argument("--no-cuda", action="store_true", default=False)
    ap.add_argument("--ort-intra-threads", type=int, default=1)
    ap.add_argument("--ort-inter-threads", type=int, default=1)
    ap.add_argument("--single-output-mode", choices=["auto", "logit", "prob"], default="logit")

    ap.add_argument("--target-fprs", nargs="*", type=float, default=[0.001, 0.005, 0.01, 0.02])
    ap.add_argument("--target-tprs", nargs="*", type=float, default=[0.90, 0.95, 0.99])
    ap.add_argument("--partial-auc-fpr-max", nargs="*", type=float, default=[0.001, 0.005, 0.01, 0.02])
    ap.add_argument("--metric-bins", type=int, default=200000)

    ap.add_argument("--plot-pkl-name", default="plot_payload.pkl")
    ap.add_argument("--plot-threshold", type=float, default=0.5)
    ap.add_argument("--no-threshold-scan-in-pkl", action="store_true", default=False)
    ap.add_argument("--print-every", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.export_onnx:
        raise RuntimeError("This script is intended to evaluate ONNX models. Use --export-onnx.")
    if not args.run_ort_validation:
        raise RuntimeError("This script is intended to compute metrics from ONNX Runtime. Use --run-ort-validation.")
    if ort is None:
        raise RuntimeError("onnxruntime is not installed in this environment.")
    if onnx is None:
        raise RuntimeError("onnx is not installed in this environment.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "model_summary.csv"
    op_tpr_path = output_dir / "accuracy_at_target_tpr.csv"  # kept for Bucket-notebook familiarity
    op_fpr_path = output_dir / "tpr_at_target_fpr.csv"
    pauc_fpr_path = output_dir / "partial_auc_low_fpr.csv"
    signal_eff_path = output_dir / "signal_efficiency_by_sample_at_target_fpr.csv"
    export_path = output_dir / "onnx_export_status.csv"
    parity_path = output_dir / "onnx_parity.csv"
    skipped_path = output_dir / "skipped_models.csv"
    plot_payload_path = output_dir / args.plot_pkl_name

    existing_summary = load_existing_table(summary_path)
    existing_op_tpr = load_existing_table(op_tpr_path)
    existing_op_fpr = load_existing_table(op_fpr_path)
    existing_pauc_fpr = load_existing_table(pauc_fpr_path)
    existing_signal_eff = load_existing_table(signal_eff_path)
    existing_export = load_existing_table(export_path)
    existing_parity = load_existing_table(parity_path)
    existing_skipped = load_existing_table(skipped_path)

    done_model_ids = (
        set(existing_summary["model_id"].astype(str))
        if ("model_id" in existing_summary.columns and not existing_summary.empty)
        else set()
    )
    done_onnx_ids = (
        set(existing_export.loc[existing_export["exported"] == True, "model_id"].astype(str))
        if ("model_id" in existing_export.columns and "exported" in existing_export.columns and not existing_export.empty)
        else set()
    )
    plot_entries_by_id = load_existing_plot_entries(plot_payload_path)

    ckpt_paths = discover_checkpoints(args.ckpt_glob)
    signal_sample_names = sample_names_from_patterns(args.signal_data_globs)
    background_sample_names = sample_names_from_patterns(args.background_data_globs)
    best_model_detail_samples = list(dict.fromkeys(["__all__", *signal_sample_names]))

    # IMPORTANT: the train/val split indices were made for the full --data-glob
    # file ordering.  Therefore the H5EventDataset used for evaluation must be
    # built from that same full file list.  The explicit signal/background globs
    # below are sample definitions for reporting; they are not used to rebuild a
    # smaller dataset, because that would shift the split indices and could mix
    # training events into the validation evaluation.
    input_paths = resolve_input_paths(args.data_glob)
    explicit_sample_input_paths: list[Path] = []
    if not args.no_explicit_dv_samples:
        explicit_sample_input_paths = resolve_input_paths_from_patterns(
            [*args.signal_data_globs, *args.background_data_globs],
            label="explicit signal/background DisplacedVertex globs",
        )

    print(
        f"[data] evaluation uses {len(input_paths)} files from --data-glob for split-compatible validation indices",
        flush=True,
    )
    if explicit_sample_input_paths:
        print(
            f"[data] reporting {len(signal_sample_names)} signal sample(s) and "
            f"{len(background_sample_names)} background sample(s) from "
            f"{len(explicit_sample_input_paths)} matched explicit sample file(s)",
            flush=True,
        )
    target_tprs = list(args.target_tprs)
    target_fprs = list(args.target_fprs)
    partial_auc_fpr_max = list(args.partial_auc_fpr_max)

    summary_rows: list[dict] = []
    op_tpr_rows: list[dict] = []
    op_fpr_rows: list[dict] = []
    pauc_fpr_rows: list[dict] = []
    signal_eff_rows: list[dict] = []
    export_rows: list[dict] = []
    parity_rows: list[dict] = []
    skipped_rows: list[dict] = []
    actual_num_validation_events = None
    num_loaded_models = 0

    for ckpt_path in ckpt_paths:
        model_id = stable_model_id_from_path(ckpt_path)
        model_has_plot = model_id in plot_entries_by_id
        model_has_required_samples = plot_entry_has_samples(
            plot_entries_by_id.get(model_id),
            best_model_detail_samples,
        )
        model_done = model_id in done_model_ids and model_has_plot and model_has_required_samples
        if model_done and not args.force_recompute:
            print(f"Skipping already processed ONNX model: {ckpt_path}", flush=True)
            continue
        if model_id in done_model_ids and model_has_plot and not model_has_required_samples and not args.force_recompute:
            available = sorted(plot_entries_by_id.get(model_id, {}).get("samples", {}).keys())
            print(
                f"Recomputing {ckpt_path}: cached plot_payload is missing requested samples. "
                f"Available cached samples: {available}",
                flush=True,
            )

        try:
            model, ckpt = build_displaced_vertex_model_from_checkpoint(ckpt_path, internal_dtype=torch.float32, args=args)
            num_loaded_models += 1
            run_group, _, display_name = checkpoint_display_parts(ckpt_path, ckpt)
            layer_type = str(_ckpt_get(ckpt, "layer_type", _ckpt_get(ckpt, "model_type", "binary_graph_classifier")))
            item = LoadedCheckpoint(ckpt_path, run_group, display_name, layer_type, model, ckpt)
        except Exception as exc:
            skipped_rows.append({"ckpt_path": str(ckpt_path), "error": repr(exc)})
            release_memory()
            continue

        print(f"Processing {item.display_name}...", flush=True)
        model_dir = output_dir / sanitize_name(item.display_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = model_dir / f"{sanitize_name(item.display_name)}.onnx"

        xdim = int(_ckpt_get(item.ckpt, "xdim", 7))
        edim = int(_ckpt_get(item.ckpt, "edim", 5))
        feature_metadata = infer_input_feature_metadata(xdim, edim)
        norm_node, norm_edge, stats_json, norm_kind, norm_clip = resolve_normalization_for_checkpoint(item.ckpt, args)
        metadata_for_onnx = {
            **feature_metadata,
            "task": "displaced_vertex_classification",
            "model_type": str(_ckpt_get(item.ckpt, "model_type", "binary_graph_classifier")),
            "normalize_node_features": bool(norm_node),
            "normalize_edge_features": bool(norm_edge),
            "feature_stats_json": str(stats_json) if stats_json is not None else "",
            "feature_norm_kind": str(norm_kind),
            "feature_norm_clip": float(norm_clip),
            "feature_norm_in_model": bool(norm_node or norm_edge),
            "requires_n_muon_nodes_input": True,
            "single_output_mode": str(args.single_output_mode),
            "score_transform": "sigmoid(logit); use --single-output-mode prob only for probability-output ONNX models",
        }

        export_info = {
            "model_id": model_id,
            "display_name": item.display_name,
            "run_group": item.run_group,
            "layer_type": item.layer_type,
            "exported": False,
            "reason": "",
            "onnx_path": str(onnx_path),
        }

        if args.force_recompute or model_id not in done_onnx_ids or not onnx_path.exists():
            dummy_x = torch.randn(int(args.onnx_dummy_nodes), xdim, dtype=torch.float32)
            dummy_src = torch.randint(0, int(args.onnx_dummy_nodes), (int(args.onnx_dummy_edges),), dtype=torch.long)
            dummy_dst = torch.randint(0, int(args.onnx_dummy_nodes), (int(args.onnx_dummy_edges),), dtype=torch.long)
            dummy_edge_index = torch.stack([dummy_src, dummy_dst], dim=0)
            dummy_edge_attr = torch.randn(int(args.onnx_dummy_edges), edim, dtype=torch.float32)
            dummy_n_muon_nodes = torch.tensor([max(1, int(args.onnx_dummy_nodes) // 2)], dtype=torch.long)
            export_core = export_model_to_onnx(
                item.model,
                onnx_path,
                dummy_x=dummy_x,
                dummy_edge_index=dummy_edge_index,
                dummy_edge_attr=dummy_edge_attr,
                dummy_n_muon_nodes=dummy_n_muon_nodes,
                onnx_opset=args.onnx_opset,
                metadata=metadata_for_onnx,
            )
            export_info.update(export_core)
        else:
            export_info["exported"] = True
            export_info["reason"] = "already exported"

        if onnx_path.exists():
            export_info.update(inspect_onnx_artifact(onnx_path))
        export_rows.append(export_info)

        # ONNX inference no longer needs the PyTorch model.
        item.model = None
        release_memory()

        if not export_info.get("exported", False) or not onnx_path.exists():
            print(f"  ONNX export failed for {item.display_name}, skipping metrics.", flush=True)
            print(f"  reason: {export_info.get('reason', '')}", flush=True)
            (model_dir / "metadata.json").write_text(
                json.dumps(metadata_from_checkpoint(item, args, feature_metadata, export_info), indent=2),
                encoding="utf-8",
            )
            del item
            release_memory()
            continue

        hist_state, lat_ort, providers, eval_meta = evaluate_onnx_model_on_files(
            onnx_path,
            input_paths,
            ckpt=item.ckpt,
            args=args,
            no_cuda=args.no_cuda,
            intra_threads=args.ort_intra_threads,
            inter_threads=args.ort_inter_threads,
            single_output_mode=args.single_output_mode,
            metric_bins=args.metric_bins,
        )
        actual_num_validation_events = int(len(lat_ort))

        metadata = metadata_from_checkpoint(item, args, feature_metadata, export_info, eval_meta=eval_meta)
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        threshold_ckpt = _ckpt_get(item.ckpt, "threshold", args.plot_threshold)
        target_fpr_ckpt = _ckpt_get(item.ckpt, "target_fpr", np.nan)
        best_monitor = _ckpt_get(item.ckpt, "best_monitor", None)

        append_metric_rows_from_hist(
            summary_rows=summary_rows,
            op_tpr_rows=op_tpr_rows,
            op_fpr_rows=op_fpr_rows,
            pauc_fpr_rows=pauc_fpr_rows,
            target_tprs=target_tprs,
            target_fprs=target_fprs,
            partial_auc_fpr_max=partial_auc_fpr_max,
            model_id=model_id,
            display_name=item.display_name,
            run_group=item.run_group,
            layer_type=item.layer_type,
            ckpt_path=str(item.ckpt_path),
            threshold_ckpt=threshold_ckpt,
            target_fpr_ckpt=target_fpr_ckpt,
            best_monitor=best_monitor,
            hist_state=hist_state,
            lat_ort=lat_ort,
            providers=providers,
            export_info=export_info,
            eval_meta=eval_meta,
        )

        append_signal_efficiency_rows_from_hist(
            signal_efficiency_rows=signal_eff_rows,
            target_fprs=target_fprs,
            model_id=model_id,
            display_name=item.display_name,
            run_group=item.run_group,
            layer_type=item.layer_type,
            hist_state=hist_state,
        )

        plot_entries_by_id[model_id] = plot_model_payload_from_hist(
            model_id=model_id,
            display_name=item.display_name,
            run_group=item.run_group,
            layer_type=item.layer_type,
            ckpt_path=str(item.ckpt_path),
            onnx_path=str(onnx_path),
            hist_state=hist_state,
            lat_ort=lat_ort,
            providers=providers,
            target_tprs=target_tprs,
            target_fprs=target_fprs,
            default_threshold=args.plot_threshold,
            include_threshold_scan=(not args.no_threshold_scan_in_pkl),
        )

        if args.save_pytorch_parity:
            parity_rows.append({
                "model_id": model_id,
                "display_name": item.display_name,
                "run_group": item.run_group,
                "layer_type": item.layer_type,
                "providers": ",".join(providers),
                "mean_event_latency_ms_pytorch": np.nan,
                "mean_event_latency_ms_onnxruntime": float(np.mean(lat_ort)),
                "max_abs_prob_diff_pytorch_vs_ort": np.nan,
                "mean_abs_prob_diff_pytorch_vs_ort": np.nan,
                "labels_match": False,
                "parity_error": "disabled because streaming histogram metrics do not retain full probability arrays",
            })

        del item, hist_state, lat_ort, providers
        release_memory()

    summary_df = upsert_rows(existing_summary, summary_rows, ["model_id"])
    op_tpr_df = upsert_rows(existing_op_tpr, op_tpr_rows, ["model_id", "target_tpr"])
    op_fpr_df = upsert_rows(existing_op_fpr, op_fpr_rows, ["model_id", "target_fpr"])
    pauc_fpr_df = upsert_rows(existing_pauc_fpr, pauc_fpr_rows, ["model_id", "fpr_max"])
    signal_eff_df = upsert_rows(existing_signal_eff, signal_eff_rows, ["model_id", "sample", "target_fpr"])
    if not signal_eff_df.empty and "sample" in signal_eff_df.columns:
        signal_eff_df = signal_eff_df[signal_eff_df["sample"].astype(str).isin(signal_sample_names)].copy()
    export_df = upsert_rows(existing_export, export_rows, ["model_id"])
    parity_df = upsert_rows(existing_parity, parity_rows, ["model_id"])
    skipped_df = upsert_rows(existing_skipped, skipped_rows, ["ckpt_path"])

    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
    if not op_tpr_df.empty:
        op_tpr_df.to_csv(op_tpr_path, index=False)
    if not op_fpr_df.empty:
        op_fpr_df.to_csv(op_fpr_path, index=False)
    if not pauc_fpr_df.empty:
        pauc_fpr_df.to_csv(pauc_fpr_path, index=False)
    if not signal_eff_df.empty:
        signal_eff_df.to_csv(signal_eff_path, index=False)
    if not export_df.empty:
        export_df.to_csv(export_path, index=False)
    if not parity_df.empty:
        parity_df.to_csv(parity_path, index=False)
    if not skipped_df.empty:
        skipped_df.to_csv(skipped_path, index=False)

    plot_model_entries = sorted(
        plot_entries_by_id.values(),
        key=lambda entry: (str(entry.get("run_group", "")), str(entry.get("display_name", ""))),
    )
    plot_payload = {
        "format": "displaced_vertex_results_tune_plot_payload_v1",
        "description": (
            "Compact plotting payload with histogram-derived ROC curves, confusion matrices, "
            "target-FPR and target-TPR operating points, optional threshold scans, "
            "and a global validation significance scan S/sqrt(S+B)."
        ),
        "target_tprs": target_tprs,
        "target_fprs": target_fprs,
        "partial_auc_fpr_max": partial_auc_fpr_max,
        "signal_samples": signal_sample_names,
        "background_samples": background_sample_names,
        "best_model_detail_samples": best_model_detail_samples,
        "evaluation_input_files_are_from_full_data_glob": True,
        "explicit_sample_input_files": [str(p) for p in explicit_sample_input_paths],
        "significance_definition": "S/sqrt(S+B), using cumulative validation counts for score >= threshold",
        "signal_efficiency_definition": (
            "Signal efficiencies are computed per sample on validation events only. "
            "Validation indices are interpreted using the full --data-glob file ordering; "
            "the explicit sample globs are used only to label/report separate samples. "
            "For target-FPR plots, the threshold is selected on the merged validation sample (__all__) "
            "and then applied to each signal sample separately."
        ),
        "metric_bins": int(args.metric_bins),
        "plot_threshold": float(args.plot_threshold),
        "contains_raw_scores": False,
        "contains_raw_labels": False,
        "models": plot_model_entries,
    }
    pd.to_pickle(plot_payload, plot_payload_path)

    run_summary = {
        "ckpt_glob": list(args.ckpt_glob),
        "data_glob": args.data_glob,
        "signal_data_globs": list(args.signal_data_globs),
        "background_data_globs": list(args.background_data_globs),
        "explicit_dv_samples_enabled": bool(not args.no_explicit_dv_samples),
        "signal_samples": signal_sample_names,
        "background_samples": background_sample_names,
        "best_model_detail_samples": best_model_detail_samples,
        "input_files": [str(p) for p in input_paths],
        "evaluation_input_files_are_from_full_data_glob": True,
        "explicit_sample_input_files": [str(p) for p in explicit_sample_input_paths],
        "split_file": str(Path(args.split_file).expanduser().resolve()) if args.split_file is not None else None,
        "split_key": args.split_key,
        "output_dir": str(output_dir),
        "num_matched_checkpoints": int(len(ckpt_paths)),
        "num_loaded_models": int(num_loaded_models),
        "num_skipped_models": int(len(skipped_rows)),
        "num_validation_events": int(actual_num_validation_events if actual_num_validation_events is not None else 0),
        "num_models_already_done": int(len(done_model_ids)),
        "num_models_newly_evaluated": int(len(summary_rows)),
        "export_onnx": True,
        "run_ort_validation": True,
        "onnx_installed": bool(onnx is not None),
        "onnxruntime_installed": bool(ort is not None),
        "metrics_source": "onnxruntime_hist",
        "single_output_mode": args.single_output_mode,
        "target_tprs": target_tprs,
        "target_fprs": target_fprs,
        "partial_auc_fpr_max": partial_auc_fpr_max,
        "signal_efficiency_csv_path": str(signal_eff_path),
        "metric_bins": int(args.metric_bins),
        "plot_payload_path": str(plot_payload_path),
        "plot_threshold": float(args.plot_threshold),
        "threshold_scan_in_pkl": bool(not args.no_threshold_scan_in_pkl),
        "num_plot_payload_models": int(len(plot_model_entries)),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Saved/updated ONNX-based DisplacedVertex results in:", output_dir, flush=True)
    print("Saved/updated plotting payload:", plot_payload_path, flush=True)


if __name__ == "__main__":
    main()
