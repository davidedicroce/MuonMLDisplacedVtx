#!/usr/bin/env python3
"""
DisplacedVertex_preproc.py

Compute trainer-compatible raw-feature normalization statistics from classifier H5 graph files,
and optionally write normalized H5 copies.

Expected event structure:
  /events/<id>/
      x            [N, 7]
      edge_index   [2, E]
      edge_attr    [E, 5]
      y or labels  [1]
  attrs:
      n_muon_nodes
      n_calo_nodes
      label

Raw node feature convention:
    x[:, :] = [r_pos, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]

Raw edge feature convention:
    edge_attr[:, :] = [d_energy_like, d_phi, d_eta, cos_angle, same_sector]

Optional track block (DisplacedVertex_converter.py with a track tree):
    track_x[:, :]         = TRACK_FEATURE_NAMES        -> stats "tr_standard" / "tr_robust", computed after the
                            model-side asinh transform of the heavy-tailed columns (TRACK_FEATURE_ASINH_SCALES)
    track_edge_attr[:, :] = same 5 edge features       -> stats "track_edge_standard" / "track_edge_robust"
    (only written when the inputs contain track nodes; the muon/calo/edge keys are unchanged)

Optional segment blocks (DisplacedVertex_converter.py --segment-edges / --tracklets):
    seg_edge_attr[:, :]      = SEGMENT_EDGE_FEATURE_NAMES  -> "seg_edge_standard" / "seg_edge_robust"
    tracklet_x[:, :]         = TRACKLET_FEATURE_NAMES      -> "tl_standard" / "tl_robust"
    tracklet_edge_attr[:, :] = TRACKLET_EDGE_FEATURE_NAMES -> "tracklet_edge_standard" / "tracklet_edge_robust"
With --segment-chi2-feature the muon-node column 5 is chi2/nDoF: its "mu_*" stats (key "energy_like") then describe
chi2/nDoF, and meta.muon_node_feature_names records the convention (all input files must agree).

All scanned arrays are kept in memory (~170 kB per event for the trigger-level mu=200 graphs): use
--max-events-per-file (e.g. 200) to take an unbiased subset across all files.

Example: compute stats only
python DisplacedVertex_preproc.py \
    --input-glob "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/*.h5" \
    --stats-json "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json"

Example: compute stats and write normalized copies
python DisplacedVertex_preproc.py \
    --input-glob "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/*.h5" \
    --stats-json "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json" \
    --write-normalized-h5 \
    --output-dir "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs_normalized" \
    --node-norm-kind standard \
    --edge-norm-kind standard \
    --normalize-node-features \
    --normalize-edge-features
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


RAW_NODE_FEATURE_NAMES = [
    "r_pos",
    "theta_pos",
    "phi_pos",
    "theta_dir",
    "phi_dir",
    "energy_like",
    "nCells_or_DoF",
]

EDGE_FEATURE_NAMES = [
    "d_energy_like",
    "d_phi",
    "d_eta",
    "cos_angle",
    "same_sector",
]

# Track feature schema and the model-side asinh transform, shared with the converter and the trainer.
# The track stats ("tr_standard"/"tr_robust") are computed AFTER the transform, because the model
# normalizes the transformed values.
from dv_converter_utils import (  # noqa: E402
    DV_NODE_FEATURE_NAMES,
    SEGMENT_EDGE_FEATURE_NAMES,
    TRACKLET_EDGE_FEATURE_NAMES,
    TRACKLET_FEATURE_NAMES,
    TRACK_FEATURE_NAMES,
    apply_track_feature_transform,
    is_packed_part,
    read_packed_meta,
    track_feature_asinh_scale_vector,
)
TRACK_FEATURE_ASINH_SCALES = track_feature_asinh_scale_vector(TRACK_FEATURE_NAMES)


def _decode_h5_attr(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    return str(v)


def _safe_read_dataset(group, name: str, file_path: str, event_key: str):
    try:
        return group[name][...]
    except Exception as e:
        raise RuntimeError(
            f"Failed to read dataset {name!r} in file={file_path} event={event_key}: "
            f"{type(e).__name__}: {e}"
        ) from e


def _read_label(group, file_path: str, event_key: str) -> float:
    if "y" in group:
        y = group["y"][...]
    elif "labels" in group:
        y = group["labels"][...]
    elif "label" in group.attrs:
        y = np.asarray([group.attrs["label"]], dtype=np.float32)
    else:
        raise RuntimeError(f"Missing label y/labels/attr in {file_path} /events/{event_key}")
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size != 1 or y[0] not in (0.0, 1.0):
        raise RuntimeError(f"Expected scalar label 0/1 in {file_path} /events/{event_key}, got {y}")
    return float(y[0])


def _safe_get_attr(group, name: str, file_path: str, event_key: str):
    try:
        return group.attrs[name]
    except Exception as e:
        raise RuntimeError(
            f"Failed to read attribute {name!r} in file={file_path} event={event_key}: "
            f"{type(e).__name__}: {e}"
        ) from e


def _safe_concat(chunks: List[np.ndarray], shape_tail: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if not chunks:
        return np.zeros((0,) + shape_tail, dtype=dtype)
    return np.concatenate(chunks, axis=0).astype(dtype, copy=False)


def _compute_stats(arr: np.ndarray, names: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(f"Expected shape [N, {len(names)}], got {arr.shape}")

    standard = {}
    robust = {}
    for i, name in enumerate(names):
        x = arr[:, i]
        x = x[np.isfinite(x)]
        if x.size == 0:
            mean = 0.0
            std = 1.0
            median = 0.0
            iqr = 1.0
        else:
            mean = float(np.mean(x))
            std = float(np.std(x))
            if not np.isfinite(std) or std < 1e-12:
                std = 1.0
            q25 = float(np.percentile(x, 25))
            q50 = float(np.percentile(x, 50))
            q75 = float(np.percentile(x, 75))
            iqr = q75 - q25
            if not np.isfinite(iqr) or iqr < 1e-12:
                iqr = 1.0
            median = q50
        standard[name] = {"mean": mean, "std": std}
        robust[name] = {"median": median, "iqr": iqr}
    return standard, robust


def _apply_standardize(arr: np.ndarray, stats: Dict[str, Dict[str, float]], names: List[str], clip: Optional[float]) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    for i, name in enumerate(names):
        mean = float(stats[name]["mean"])
        std = max(float(stats[name]["std"]), 1e-12)
        out[:, i] = (out[:, i] - mean) / std
    if clip is not None and clip > 0:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32, copy=False)


def _apply_robust(arr: np.ndarray, stats: Dict[str, Dict[str, float]], names: List[str], clip: Optional[float]) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    for i, name in enumerate(names):
        median = float(stats[name]["median"])
        iqr = max(float(stats[name]["iqr"]), 1e-12)
        out[:, i] = (out[:, i] - median) / iqr
    if clip is not None and clip > 0:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32, copy=False)


def _normalize_array(
    arr: np.ndarray,
    *,
    names: List[str],
    standard_stats: Dict[str, Dict[str, float]],
    robust_stats: Dict[str, Dict[str, float]],
    kind: str,
    clip: Optional[float],
) -> np.ndarray:
    if kind == "standard":
        return _apply_standardize(arr, standard_stats, names, clip)
    if kind == "robust":
        return _apply_robust(arr, robust_stats, names, clip)
    raise ValueError(f"Unknown normalization kind: {kind}")


def _copy_attrs(src, dst):
    for k in src.attrs.keys():
        dst.attrs[k] = src.attrs[k]


def _check_track_feature_names(f, h5_path: str):
    if "track_feature_names" not in f.attrs:
        return
    names = json.loads(_decode_h5_attr(f.attrs["track_feature_names"]))
    if list(names) != TRACK_FEATURE_NAMES:
        raise RuntimeError(f"{h5_path}: track_feature_names {names} differ from the expected {TRACK_FEATURE_NAMES}")


def _check_block_feature_names(f, h5_path: str, attr: str, expected: List[str]):
    if attr in f.attrs:
        names = json.loads(_decode_h5_attr(f.attrs[attr]))
        if list(names) != list(expected):
            raise RuntimeError(f"{h5_path}: {attr} {names} differ from the expected {expected}")


def _muon_node_feature_names(f) -> List[str]:
    """Muon-node feature names of a file (older files: the shared DV_NODE_FEATURE_NAMES)."""
    if "muon_node_feature_names" in f.attrs:
        return list(json.loads(_decode_h5_attr(f.attrs["muon_node_feature_names"])))
    return list(DV_NODE_FEATURE_NAMES)


class _PackedGroup(dict):
    """One event of a packed part with the interface of an H5 event group used here (datasets + attrs)."""

    def __init__(self, data, attrs):
        super().__init__(data)
        self.attrs = attrs


class _PackedEvents:
    def __init__(self, part_dir: str, meta: dict):
        self.part_dir = part_dir
        self.blocks = list(meta["blocks"])
        self._keys = [str(k) for k in np.load(os.path.join(part_dir, "event_key.npy"))]
        self._row = {k: i for i, k in enumerate(self._keys)}
        self._arr = {b: np.load(os.path.join(part_dir, f"{b}.npy"), mmap_mode="r") for b in self.blocks}
        self._ptr = {b: np.load(os.path.join(part_dir, f"{b}_ptr.npy")) for b in self.blocks}
        self._ev = {k: np.load(os.path.join(part_dir, f"{k}.npy"))
                    for k in ("y", "n_muon_nodes", "dataset_name", "root_file")}

    def keys(self):
        return list(self._keys)

    def __getitem__(self, key):
        j = self._row[str(key)]
        data = {"y": np.asarray([self._ev["y"][j]], dtype=np.float32)}
        for b in self.blocks:
            s, e = int(self._ptr[b][j]), int(self._ptr[b][j + 1])
            data[b] = np.array(self._arr[b][:, s:e] if b.endswith("edge_index") else self._arr[b][s:e])
        attrs = {"n_muon_nodes": int(self._ev["n_muon_nodes"][j]), "dataset_name": str(self._ev["dataset_name"][j]),
                 "root_file": str(self._ev["root_file"][j]), "label": int(self._ev["y"][j])}
        return _PackedGroup(data, attrs)


class _PackedFile:
    """A packed part (DisplacedVertex_converter.py --output-format packed) seen like the H5 file: attrs + events."""

    def __init__(self, part_dir: str):
        meta = read_packed_meta(part_dir)
        self.attrs = dict(meta.get("file_attrs") or {})
        self.attrs.setdefault("muon_node_feature_names", json.dumps(meta.get("muon_node_feature_names")
                                                                     or list(DV_NODE_FEATURE_NAMES)))
        self._events = _PackedEvents(part_dir, meta)

    def __contains__(self, key):
        return key == "events"

    def __getitem__(self, key):
        if key != "events":
            raise KeyError(key)
        return self._events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _open_part(path: str):
    """H5 file or packed part."""
    return _PackedFile(path) if is_packed_part(path) else h5py.File(path, "r")


def collect_stats_from_h5(h5_paths: List[str], max_events: int = -1, max_events_per_file: int = -1):
    mu_chunks = []
    ca_chunks = []
    edge_chunks = []
    tr_chunks = []
    tr_edge_chunks = []
    seg_edge_chunks = []
    tl_chunks = []
    tl_edge_chunks = []
    muon_names = None
    labels = []
    dataset_names = []
    skipped_files = []
    skipped_events = []
    total_events = 0

    for h5_path in h5_paths:
        print(f"[i] scanning {h5_path}", flush=True)
        try:
            with _open_part(h5_path) as f:
                if "events" not in f:
                    continue
                _check_track_feature_names(f, h5_path)
                _check_block_feature_names(f, h5_path, "segment_edge_feature_names", SEGMENT_EDGE_FEATURE_NAMES)
                _check_block_feature_names(f, h5_path, "tracklet_feature_names", TRACKLET_FEATURE_NAMES)
                _check_block_feature_names(f, h5_path, "tracklet_edge_feature_names", TRACKLET_EDGE_FEATURE_NAMES)
                file_muon_names = _muon_node_feature_names(f)
                if muon_names is None:
                    muon_names = file_muon_names
                elif file_muon_names != muon_names:
                    raise RuntimeError(f"{h5_path}: muon_node_feature_names {file_muon_names} differ from the other "
                                       f"inputs {muon_names} (mixed --segment-chi2-feature settings)")
                event_keys = sorted(f["events"].keys())
                if max_events_per_file > 0:
                    event_keys = event_keys[:max_events_per_file]
                for ek in event_keys:
                    try:
                        g = f["events"][ek]
                        x = _safe_read_dataset(g, "x", h5_path, ek).astype(np.float32, copy=False)
                        edge_attr = _safe_read_dataset(g, "edge_attr", h5_path, ek).astype(np.float32, copy=False)
                        label = _read_label(g, h5_path, ek)

                        if "n_muon_nodes" not in g.attrs:
                            raise RuntimeError(f"Missing attribute 'n_muon_nodes' in {h5_path} /events/{ek}")
                        n_mu = int(_safe_get_attr(g, "n_muon_nodes", h5_path, ek))
                        n_tot = x.shape[0]
                        if n_mu < 0 or n_mu > n_tot:
                            raise RuntimeError(
                                f"Invalid n_muon_nodes={n_mu} for x.shape[0]={n_tot} in {h5_path} /events/{ek}"
                            )

                        mu_x = x[:n_mu]
                        ca_x = x[n_mu:]
                        if mu_x.shape[0] > 0:
                            mu_chunks.append(mu_x)
                        if ca_x.shape[0] > 0:
                            ca_chunks.append(ca_x)
                        if edge_attr.shape[0] > 0:
                            edge_chunks.append(edge_attr)
                        if "track_x" in g and g["track_x"].shape[0] > 0:
                            tr_chunks.append(apply_track_feature_transform(
                                _safe_read_dataset(g, "track_x", h5_path, ek), TRACK_FEATURE_ASINH_SCALES))
                        if "track_edge_attr" in g and g["track_edge_attr"].shape[0] > 0:
                            tr_edge_chunks.append(
                                _safe_read_dataset(g, "track_edge_attr", h5_path, ek).astype(np.float32, copy=False)
                            )
                        for name, chunks in (("seg_edge_attr", seg_edge_chunks), ("tracklet_x", tl_chunks),
                                             ("tracklet_edge_attr", tl_edge_chunks)):
                            if name in g and g[name].shape[0] > 0:
                                chunks.append(_safe_read_dataset(g, name, h5_path, ek).astype(np.float32, copy=False))
                        labels.append(label)
                        dataset_names.append(_decode_h5_attr(g.attrs.get("dataset_name", "unknown")))

                        total_events += 1
                        if max_events > 0 and total_events >= max_events:
                            break
                    except Exception as e:
                        msg = f"{h5_path} :: /events/{ek} :: {type(e).__name__}: {e}"
                        skipped_events.append(msg)
                        print(f"[warn] skipping unreadable event: {msg}", flush=True)
                        continue
        except Exception as e:
            msg = f"{h5_path} :: {type(e).__name__}: {e}"
            skipped_files.append(msg)
            print(f"[warn] skipping unreadable file: {msg}", flush=True)
            continue

        if max_events > 0 and total_events >= max_events:
            break

    mu_arr = _safe_concat(mu_chunks, (len(RAW_NODE_FEATURE_NAMES),), dtype=np.float32)
    ca_arr = _safe_concat(ca_chunks, (len(RAW_NODE_FEATURE_NAMES),), dtype=np.float32)
    edge_arr = _safe_concat(edge_chunks, (len(EDGE_FEATURE_NAMES),), dtype=np.float32)
    tr_arr = _safe_concat(tr_chunks, (len(TRACK_FEATURE_NAMES),), dtype=np.float32)
    tr_edge_arr = _safe_concat(tr_edge_chunks, (len(EDGE_FEATURE_NAMES),), dtype=np.float32)
    seg_edge_arr = _safe_concat(seg_edge_chunks, (len(SEGMENT_EDGE_FEATURE_NAMES),), dtype=np.float32)
    tl_arr = _safe_concat(tl_chunks, (len(TRACKLET_FEATURE_NAMES),), dtype=np.float32)
    tl_edge_arr = _safe_concat(tl_edge_chunks, (len(TRACKLET_EDGE_FEATURE_NAMES),), dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)

    n_pos = int(np.count_nonzero(labels_arr == 1.0))
    n_neg = int(np.count_nonzero(labels_arr == 0.0))
    print(f"[i] scanned events: {total_events}", flush=True)
    print(f"[i] labels:      signal={n_pos}, background={n_neg}", flush=True)
    print(f"[i] mu nodes:    {mu_arr.shape}", flush=True)
    print(f"[i] calo nodes:  {ca_arr.shape}", flush=True)
    print(f"[i] edge attrs:  {edge_arr.shape}", flush=True)
    print(f"[i] track nodes: {tr_arr.shape}", flush=True)
    print(f"[i] track edges: {tr_edge_arr.shape}", flush=True)
    print(f"[i] segment-segment edges: {seg_edge_arr.shape}", flush=True)
    print(f"[i] tracklet nodes: {tl_arr.shape}, tracklet edges: {tl_edge_arr.shape}", flush=True)
    print(f"[i] skipped files:  {len(skipped_files)}", flush=True)
    print(f"[i] skipped events: {len(skipped_events)}", flush=True)

    mu_standard, mu_robust = _compute_stats(mu_arr, RAW_NODE_FEATURE_NAMES)
    ca_standard, ca_robust = _compute_stats(ca_arr, RAW_NODE_FEATURE_NAMES)
    edge_standard, edge_robust = _compute_stats(edge_arr, EDGE_FEATURE_NAMES)
    has_tracks = tr_arr.shape[0] > 0
    if has_tracks:
        tr_standard, tr_robust = _compute_stats(tr_arr, TRACK_FEATURE_NAMES)
        tr_edge_standard, tr_edge_robust = _compute_stats(tr_edge_arr, EDGE_FEATURE_NAMES)

    dataset_summary = {}
    for name, label in zip(dataset_names, labels):
        row = dataset_summary.setdefault(str(name), {"n_events": 0, "n_signal": 0, "n_background": 0})
        row["n_events"] += 1
        if int(label) == 1:
            row["n_signal"] += 1
        else:
            row["n_background"] += 1

    payload = {
        "mu_standard": mu_standard,
        "ca_standard": ca_standard,
        "edge_standard": edge_standard,
        "mu_robust": mu_robust,
        "ca_robust": ca_robust,
        "edge_robust": edge_robust,
        "label_summary": {
            "n_signal": n_pos,
            "n_background": n_neg,
            "positive_fraction": float(n_pos / max(len(labels_arr), 1)),
            "pos_weight_auto": float(n_neg / max(n_pos, 1)) if n_pos > 0 else None,
            "by_dataset": dataset_summary,
        },
        "meta": {
            "task": "graph_classification",
            "node_feature_names": RAW_NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "label_names": ["background", "signal"],
            "n_events_scanned": int(total_events),
            "n_mu_nodes": int(mu_arr.shape[0]),
            "n_ca_nodes": int(ca_arr.shape[0]),
            "n_edges": int(edge_arr.shape[0]),
            "source_h5_paths": [str(Path(p).resolve()) for p in h5_paths],
            "skipped_files": skipped_files,
            "skipped_events": skipped_events,
            "schema": "raw_classifier",
            "max_events_per_file": int(max_events_per_file),
        },
    }
    if has_tracks:
        payload["tr_standard"] = tr_standard
        payload["tr_robust"] = tr_robust
        payload["track_edge_standard"] = tr_edge_standard
        payload["track_edge_robust"] = tr_edge_robust
        payload["meta"]["track_feature_names"] = TRACK_FEATURE_NAMES
        # tr_* stats describe asinh(x / scale) for the columns with scale > 0 (see dv_converter_utils)
        payload["meta"]["track_feature_asinh_scales"] = [float(s) for s in TRACK_FEATURE_ASINH_SCALES]
        payload["meta"]["tr_stats_after_asinh"] = True
        payload["meta"]["n_tr_nodes"] = int(tr_arr.shape[0])
        payload["meta"]["n_track_edges"] = int(tr_edge_arr.shape[0])
    payload["meta"]["muon_node_feature_names"] = muon_names if muon_names is not None else list(DV_NODE_FEATURE_NAMES)
    if seg_edge_arr.shape[0] > 0:
        payload["seg_edge_standard"], payload["seg_edge_robust"] = _compute_stats(seg_edge_arr, SEGMENT_EDGE_FEATURE_NAMES)
        payload["meta"]["segment_edge_feature_names"] = SEGMENT_EDGE_FEATURE_NAMES
        payload["meta"]["n_seg_edges"] = int(seg_edge_arr.shape[0])
    if tl_arr.shape[0] > 0:
        payload["tl_standard"], payload["tl_robust"] = _compute_stats(tl_arr, TRACKLET_FEATURE_NAMES)
        payload["meta"]["tracklet_feature_names"] = TRACKLET_FEATURE_NAMES
        payload["meta"]["n_tl_nodes"] = int(tl_arr.shape[0])
    if tl_edge_arr.shape[0] > 0:
        payload["tracklet_edge_standard"], payload["tracklet_edge_robust"] = _compute_stats(
            tl_edge_arr, TRACKLET_EDGE_FEATURE_NAMES)
        payload["meta"]["tracklet_edge_feature_names"] = TRACKLET_EDGE_FEATURE_NAMES
        payload["meta"]["n_tracklet_edges"] = int(tl_edge_arr.shape[0])
    return payload


def write_json(payload: Dict, out_json: str):
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] wrote stats JSON: {out_path}", flush=True)


def write_normalized_h5_copies(
    h5_paths: List[str],
    output_dir: str,
    stats_payload: Dict,
    *,
    normalize_node_features: bool,
    normalize_edge_features: bool,
    node_norm_kind: str,
    edge_norm_kind: str,
    clip: Optional[float],
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_standard = stats_payload["mu_standard"]
    mu_robust = stats_payload["mu_robust"]
    ca_standard = stats_payload["ca_standard"]
    ca_robust = stats_payload["ca_robust"]
    edge_standard = stats_payload["edge_standard"]
    edge_robust = stats_payload["edge_robust"]
    tr_standard = stats_payload.get("tr_standard")
    tr_robust = stats_payload.get("tr_robust")
    tr_edge_standard = stats_payload.get("track_edge_standard")
    tr_edge_robust = stats_payload.get("track_edge_robust")

    for src_path in h5_paths:
        src_name = os.path.basename(src_path)
        dst_path = out_dir / src_name
        print(f"[i] writing normalized copy {dst_path}", flush=True)

        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_attrs(src, dst)
            if "events" not in src:
                continue
            dst_events = dst.create_group("events")
            for ek in sorted(src["events"].keys()):
                sg = src["events"][ek]
                dg = dst_events.create_group(ek)
                _copy_attrs(sg, dg)

                x = sg["x"][...].astype(np.float32, copy=False)
                edge_attr = sg["edge_attr"][...].astype(np.float32, copy=False)
                n_mu = int(sg.attrs["n_muon_nodes"])
                mu_x = x[:n_mu]
                ca_x = x[n_mu:]

                if normalize_node_features:
                    if mu_x.shape[0] > 0:
                        mu_x = _normalize_array(
                            mu_x,
                            names=RAW_NODE_FEATURE_NAMES,
                            standard_stats=mu_standard,
                            robust_stats=mu_robust,
                            kind=node_norm_kind,
                            clip=clip,
                        )
                    if ca_x.shape[0] > 0:
                        ca_x = _normalize_array(
                            ca_x,
                            names=RAW_NODE_FEATURE_NAMES,
                            standard_stats=ca_standard,
                            robust_stats=ca_robust,
                            kind=node_norm_kind,
                            clip=clip,
                        )
                    x = np.concatenate([mu_x, ca_x], axis=0).astype(np.float32, copy=False)

                if normalize_edge_features and edge_attr.shape[0] > 0:
                    edge_attr = _normalize_array(
                        edge_attr,
                        names=EDGE_FEATURE_NAMES,
                        standard_stats=edge_standard,
                        robust_stats=edge_robust,
                        kind=edge_norm_kind,
                        clip=clip,
                    )

                normalized = {"x": x, "edge_attr": edge_attr}
                if normalize_node_features and tr_standard is not None and "track_x" in sg and sg["track_x"].shape[0] > 0:
                    normalized["track_x"] = _normalize_array(
                        apply_track_feature_transform(sg["track_x"][...], TRACK_FEATURE_ASINH_SCALES),
                        names=TRACK_FEATURE_NAMES,
                        standard_stats=tr_standard,
                        robust_stats=tr_robust,
                        kind=node_norm_kind,
                        clip=clip,
                    )
                if (normalize_edge_features and tr_edge_standard is not None and "track_edge_attr" in sg
                        and sg["track_edge_attr"].shape[0] > 0):
                    normalized["track_edge_attr"] = _normalize_array(
                        sg["track_edge_attr"][...].astype(np.float32, copy=False),
                        names=EDGE_FEATURE_NAMES,
                        standard_stats=tr_edge_standard,
                        robust_stats=tr_edge_robust,
                        kind=edge_norm_kind,
                        clip=clip,
                    )

                for name in sg.keys():
                    data = normalized[name] if name in normalized else sg[name][...]
                    dg.create_dataset(name, data=data, compression="gzip", compression_opts=4)

            dst.attrs["preprocessed_from"] = str(Path(src_path).resolve())
            dst.attrs["node_normalized"] = bool(normalize_node_features)
            dst.attrs["edge_normalized"] = bool(normalize_edge_features)
            dst.attrs["node_norm_kind"] = str(node_norm_kind)
            dst.attrs["edge_norm_kind"] = str(edge_norm_kind)
            dst.attrs["feature_norm_clip"] = -1.0 if clip is None else float(clip)
    print(f"[done] wrote normalized H5 files to: {out_dir}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute raw-feature stats from DV classifier H5 files and optionally write normalized H5 copies."
    )
    ap.add_argument("--input-glob", required=True, help="Glob for the input H5 files or packed parts (*.packed)")
    ap.add_argument("--stats-json", required=True, help="Output JSON path for computed stats")
    ap.add_argument("--max-events", type=int, default=-1, help="Limit number of events to scan (-1 = all)")
    ap.add_argument("--max-events-per-file", type=int, default=-1,
                    help="Scan at most this many events of each file (-1 = all): an unbiased subset across samples, "
                         "unlike --max-events which stops after the first files in sorted order")
    ap.add_argument("--write-normalized-h5", action="store_true", default=False,
                    help="Also write normalized H5 copies")
    ap.add_argument("--output-dir", default=None,
                    help="Directory for normalized H5 files; required if --write-normalized-h5")
    ap.add_argument("--normalize-node-features", action="store_true", default=False)
    ap.add_argument("--normalize-edge-features", action="store_true", default=False)
    ap.add_argument("--node-norm-kind", default="standard", choices=["standard", "robust"])
    ap.add_argument("--edge-norm-kind", default="standard", choices=["standard", "robust"])
    ap.add_argument("--clip", type=float, default=-1.0,
                    help="Optional absolute clip after normalization; <=0 disables clipping")
    return ap.parse_args()


def main():
    args = parse_args()
    h5_paths = sorted(glob.glob(args.input_glob))
    if not h5_paths:
        raise SystemExit(f"No H5 files matched: {args.input_glob}")
    print(f"[i] found {len(h5_paths)} H5 files", flush=True)
    stats_payload = collect_stats_from_h5(h5_paths, max_events=args.max_events,
                                          max_events_per_file=args.max_events_per_file)
    write_json(stats_payload, args.stats_json)
    if args.write_normalized_h5:
        if any(is_packed_part(p) for p in h5_paths):
            raise SystemExit("--write-normalized-h5 needs H5 inputs (the model normalizes packed inputs itself).")
        if args.output_dir is None:
            raise SystemExit("--output-dir is required with --write-normalized-h5")
        clip = None if args.clip <= 0 else float(args.clip)
        write_normalized_h5_copies(
            h5_paths,
            args.output_dir,
            stats_payload,
            normalize_node_features=args.normalize_node_features,
            normalize_edge_features=args.normalize_edge_features,
            node_norm_kind=args.node_norm_kind,
            edge_norm_kind=args.edge_norm_kind,
            clip=clip,
        )


if __name__ == "__main__":
    main()
