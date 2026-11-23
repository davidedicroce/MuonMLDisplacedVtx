#!/usr/bin/env python3
"""
dv_converter_utils.py

Shared utilities for DisplacedVertex converter scripts:
  DisplacedVertex_converter_cartesian.py
  DisplacedVertex_converter_cylindrical.py
  DisplacedVertex_converter_polar.py
  DisplacedVertex.py

Contains:
  - Required branch lists
  - Generic ROOT reading helpers
  - Geometry and coordinate conversion helpers
  - Event selection functions
  - Raw muon/calo data collectors
  - Edge building functions
  - HDF5 writing helpers
  - Shared argparse setup
  - Shared converter main loop
  - DisplacedVertex graph-classification conversion helpers
  - Lazy HDF5 dataset and collate helpers for DisplacedVertex graphs
"""

import os
import re
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
# uproot is imported lazily in _open_tree_by_name: only the ROOT -> HDF5 conversion needs it, while the
# trainer imports this module for the shared feature definitions (TRACK_FEATURE_NAMES, ...).
uproot = None
import h5py


# -------------------------------------------------------
# Required branch lists
# -------------------------------------------------------

REQUIRED_MUON_BRANCHES = [
    "segmentDirectionX",
    "segmentDirectionY",
    "segmentDirectionZ",
    "segmentPositionX",
    "segmentPositionY",
    "segmentPositionZ",
    "segment_numberDoF",
    "CommonEventHash",
    "bucket_hasTruth",
    "bucket_chamberIndex",
    "bucket_layers",
    "bucket_sector",
    "bucket_segments",
]
# Needed only for the segment-chi2 feature and the tracklet nodes (all MuonBucketDump versions have it)
OPTIONAL_MUON_BRANCHES = ["segment_chiSquared"]

REQUIRED_CALO_BRANCHES = [
    "CommonEventHash",
    "tower_directionX",
    "tower_directionY",
    "tower_directionZ",
    "tower_energy_mev",
    "tower_eta",
    "tower_nCells",
    "tower_phi",
]

REQUIRED_VERTEX_BRANCHES = [
    "CommonEventHash",
    "truthMuonVertexPositionX",
    "truthMuonVertexPositionY",
    "truthMuonVertexPositionZ",
]

# Inner-tracker tracks, as dumped by IDTrackVertexDumperAlg into TrigFSTrackDump (trigger full-scan FTF),
# TrigRoITrackDump (trigger FTF in cones around the segments) or IDTrackDump (offline): same branches.
# The dumper keeps tracks with pT > --trigFSTrackMinPtGeV within dR < --trackSegmentDR (0.3) of a segment.
_TRACK_VECTOR_BRANCHES = [
    "track_pt",
    "track_eta",
    "track_phi",
    "track_d0",
    "track_z0",
    "track_vz",
    "track_d0Err",
    "track_z0Err",
    "track_charge",
    "track_chiSquared",
    "track_numberDoF",
    "track_nPixelHits",
    "track_nStripHits",
]
REQUIRED_TRACK_BRANCHES = ["CommonEventHash", *_TRACK_VECTOR_BRANCHES, "hsPV_z"]

# Feature conventions of the classification graphs (shared with DVInferenceToolBase in Athena)
DV_NODE_FEATURE_NAMES = ["r_pos", "theta_pos", "phi_pos", "theta_dir", "phi_dir", "energy_like", "nCells_or_DoF"]
DV_EDGE_FEATURE_NAMES = ["d_energy_like", "d_phi", "d_eta", "cos_angle", "same_sector"]
# Track nodes: perigee parameters; z0 relative to the hard-scatter vertex (first PriVtx, hsPV_z)
TRACK_FEATURE_NAMES = [
    "theta_dir",        # polar angle at the perigee
    "phi_dir",          # azimuth at the perigee
    "log_pt",           # ln(pT / GeV)
    "q_over_pt",        # charge / pT [1/GeV]
    "d0",               # signed transverse impact parameter [mm]
    "d0_sig",           # d0 / sigma(d0)
    "dz0_sintheta",     # (z0 + vz - z_PV) * sin(theta) [mm]
    "dz0_sig",          # (z0 + vz - z_PV) / sigma(z0)
    "n_pixel_hits",
    "n_strip_hits",
    "chi2_ndof",        # chi2 / max(nDoF, 1)
]
# Model-side input transform of the heavy-tailed track features: x -> asinh(x / scale) (0 = unchanged).
# The H5 files and the Athena graph builder keep the raw values; the transform is part of the model
# (DisplacedVertexGNN, embedded in the ONNX export) and the preprocessing computes the track
# normalization statistics after it. Changing it requires rerunning DisplacedVertex_preproc.py.
TRACK_FEATURE_ASINH_SCALES = {
    "d0": 0.05,             # mm
    "d0_sig": 1.0,
    "dz0_sintheta": 0.5,    # mm
    "dz0_sig": 1.0,
    "chi2_ndof": 1.0,
}


def track_feature_asinh_scale_vector(names=None) -> np.ndarray:
    """TRACK_FEATURE_ASINH_SCALES as a vector aligned with ``names`` (default TRACK_FEATURE_NAMES)."""
    names = TRACK_FEATURE_NAMES if names is None else list(names)
    return np.asarray([float(TRACK_FEATURE_ASINH_SCALES.get(n, 0.0)) for n in names], dtype=np.float32)


TRACK_FEATURE_ASINH_ABS_MAX = 1.0e6  # |x / scale| cap before the asinh (asinh(1e6) = 14.5)


def apply_track_feature_transform(track_x, scales=None) -> np.ndarray:
    """numpy version of the model-side transform, same formula as dv_training_utils.onnx_safe_asinh:
    asinh(x / scale) = sign * log(|.| + sqrt(|.|^2 + 1)) with |x / scale| capped at 1e6, where scale > 0; else x."""
    track_x = np.asarray(track_x, dtype=np.float32)
    scales = track_feature_asinh_scale_vector() if scales is None else np.asarray(scales, dtype=np.float32)
    out = track_x.copy()
    cols = np.nonzero(scales > 0)[0]
    if out.size and cols.size:
        v = out[:, cols] / scales[cols]
        z = np.minimum(np.abs(v), np.float32(TRACK_FEATURE_ASINH_ABS_MAX))
        out[:, cols] = np.sign(v) * np.log(z + np.sqrt(z * z + np.float32(1.0)))
    return out


DV_NODE_TYPE_MUON, DV_NODE_TYPE_CALO, DV_NODE_TYPE_TRACK, DV_NODE_TYPE_TRACKLET = 0, 1, 2, 3

# Optional segment-quality feature of the muon nodes (converter --segment-chi2-feature): slot 5 of x
# ("energy_like", always 0 for segments otherwise) holds chi2 / max(nDoF, 1) of the segment instead.
# The muon-node energy_like used by the edge features stays 0.
MUON_SLOT5_FEATURE_DEFAULT = "energy_like"
MUON_SLOT5_FEATURE_CHI2 = "chi2_ndof"

# Optional segment-segment edges (converter --segment-edges): all muon-node pairs with deltaR < --segment-edge-max-dr,
# both directions; src -> dst features (chord = straight line between the two segment positions in the r-z plane).
SEGMENT_EDGE_FEATURE_NAMES = [
    "d_eta",            # eta(dst) - eta(src), segment positions
    "d_phi",            # phi(dst) - phi(src), wrapped
    "cos_angle",        # cosine between the segment directions
    "chord_angle",      # max over the two segments of the r-z angle between segment direction and chord [rad];
                        # 0 for chords shorter than SEGMENT_CHORD_MIN_LENGTH_MM
    "chord_length_m",   # r-z distance between the segment positions [m]
    "same_layer",       # same MS station layer (Inner / Middle / Outer / Extended / BarrelExtended)
    "same_sector",      # same MS sector (bucket sector)
]
SEGMENT_CHORD_MIN_LENGTH_MM = 1.0

# Optional MS tracklet nodes (converter --tracklets): connected components of the "chain" links between
# segments in different station layers (deltaR < --tracklet-max-dr, chord_angle < --tracklet-max-chord-angle)
# spanning >= --tracklet-min-layers layers. Each tracklet is linked to its segments, and the tracklets are
# fully connected among themselves. "inner" / "outer" member: smallest / largest distance from the origin.
TRACKLET_FEATURE_NAMES = [
    "eta",              # mean eta of the member positions
    "phi",              # circular mean phi of the member positions
    "theta_chord",      # polar angle of the inner -> outer member chord
    "n_segments",
    "n_layers",         # distinct station layers
    "d_inner_m",        # distance of the inner member from the origin [m]
    "d_outer_m",        # distance of the outer member from the origin [m]
    "max_chord_angle",  # largest chord_angle of the links inside the tracklet [rad]
    "chord_rz_ip_m",    # r-z distance between the origin and the inner -> outer chord line [m]
    "sum_ndof",         # sum of the member nDoF
    "mean_chi2_ndof",   # mean member chi2 / max(nDoF, 1)
]
TRACKLET_EDGE_FEATURE_NAMES = [
    "d_eta",            # eta(dst) - eta(src)
    "d_phi",            # phi(dst) - phi(src), wrapped
    "cos_angle",        # segment direction . tracklet chord (segment edges) / chord . chord (tracklet pairs)
    "is_tracklet_pair", # 0: segment <-> tracklet edge, 1: tracklet <-> tracklet edge
]

# MS station layer of a chamber index, Muon::MuonStationIndex::toLayerIndex(ChIndex):
# Inner 0 (BI, EI, CSC), Middle 1, Outer 2, Extended 3 (EE), BarrelExtended 4 (BEE); unknown chamber -1
_CHAMBER_TO_STATION_LAYER = {0: 0, 1: 0, 7: 0, 8: 0, 15: 0, 16: 0, 2: 1, 3: 1, 9: 1, 10: 1,
                             4: 2, 5: 2, 11: 2, 12: 2, 13: 3, 14: 3, 6: 4}


def station_layer_from_chamber_index(chamber_index) -> np.ndarray:
    """MS station layer (Muon::MuonStationIndex::toLayerIndex) of bucket chamber indices; -1 if unknown."""
    ch = np.asarray(chamber_index, dtype=np.int64).ravel()
    return np.asarray([_CHAMBER_TO_STATION_LAYER.get(int(c), -1) for c in ch], dtype=np.int64)


# -------------------------------------------------------
# Generic ROOT helpers
# -------------------------------------------------------

def _normalize_keys(keys):
    return {re.sub(r"\[.*\]/[A-Za-z]$", "", k): k for k in keys}


def _open_tree_by_name(root_file: str, tree_name: str):
    global uproot
    if uproot is None:
        try:
            import uproot as _uproot
        except ImportError as exc:
            raise ImportError("uproot is required to read ROOT files. Install it in the conversion environment.") from exc
        uproot = _uproot
    f = uproot.open(root_file)
    if tree_name not in f:
        candidates = [k.split(";")[0] for k in f.keys()]
        raise ValueError(
            f"Tree '{tree_name}' not found in '{root_file}'. "
            f"Available objects: {candidates}"
        )
    obj = f[tree_name]
    try:
        _ = obj.num_entries
    except Exception as e:
        raise ValueError(f"Object '{tree_name}' in '{root_file}' is not a TTree.") from e
    return obj


def _flatten_event_hash(x):
    arr = np.asarray(x).ravel()
    if arr.size == 0:
        return None
    return tuple(int(v) for v in arr[:2])


def _safe_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _wrap_phi(dphi):
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _phi_to_sector(phi, sector_mod):
    """Legacy calo phi bin in [0, sector_mod), starting at phi=-pi. It does NOT follow the MS sector
    numbering of the muon nodes (bucket_sector, 1-16, sector 1 centred at phi=0), so a muon-calo
    same_sector built from it is always 0. Kept to reproduce the old datasets (--calo-sector-convention legacy)."""
    phi = np.asarray(phi, dtype=np.float32)
    phi01 = (phi + np.pi) / (2.0 * np.pi)
    sec = np.floor(phi01 * sector_mod).astype(np.int64)
    sec = np.clip(sec, 0, sector_mod - 1)
    return sec


# MS sector geometry of Muon::MuonSectorMapping (MuonDetDescrUtils): 16 sectors, sector 1 centred at phi=0,
# odd (large) sectors +-0.6*pi/8, even (small) ones +-0.4*pi/8, plus an overlap of 0.1*pi/8 on each side.
_MS_PI8 = np.pi / 8.0
_MS_LARGE_HALF_WIDTH = 0.6 * _MS_PI8
_MS_SMALL_HALF_WIDTH = 0.4 * _MS_PI8
_MS_OVERLAP = 0.1 * _MS_PI8


def _ms_sector_phi(sector):
    """MuonSectorMapping::sectorPhi: central phi of the MS sector [1-16]."""
    sector = np.asarray(sector, dtype=np.float64)
    return np.where(sector < 10, np.pi * (sector - 1.0) / 8.0, -np.pi * (2.0 - (sector - 1.0) / 8.0))


def _ms_inside_sector(sector, phi):
    """MuonSectorMapping::insideSector (sector width including the overlap)."""
    half_width = np.where(np.asarray(sector) % 2 == 1, _MS_LARGE_HALF_WIDTH, _MS_SMALL_HALF_WIDTH) + _MS_OVERLAP
    dphi = _wrap_phi(np.asarray(phi, dtype=np.float64) - _ms_sector_phi(sector))
    return (dphi >= -half_width) & (dphi <= half_width)


def ms_sector_from_phi(phi):
    """MS sector number [1-16] of an azimuth, identical to MuonR4::ExpandedSector{phi}.msSector() as used by
    DVInferenceToolBase for the calo-tower nodes: MuonSectorMapping::getSectors(phi) and, in an overlap region,
    the first of the two sectors (the previous one in phi; 16 in the 16/1 overlap)."""
    phi = _wrap_phi(np.asarray(phi, dtype=np.float64))
    # MuonSectorMapping::getSector
    val = (phi + _MS_LARGE_HALF_WIDTH) / _MS_PI8
    val = np.where(val < 0.0, val + 16.0, val)
    slice_idx = np.trunc(val / 2.0).astype(np.int64)
    sector = 2 * slice_idx + 1 + ((val - 2.0 * slice_idx) > 1.2).astype(np.int64)
    # MuonSectorMapping::getSectors: [previous (if inside), sector (if inside), next (if inside)]; take the first
    previous = np.where(sector != 1, sector - 1, 16)
    following = np.where(sector != 16, sector + 1, 1)
    out = np.where(_ms_inside_sector(previous, phi), previous,
                   np.where(_ms_inside_sector(sector, phi), sector, following))
    return out.astype(np.int64)


CALO_SECTOR_CONVENTIONS = ("cpp", "legacy")


def _calo_sector(phi, sector_mod: int = 16, convention: str = "cpp"):
    """Sector of a calo-tower (or track) node for the same_sector edge feature.
    cpp:    MS sector number, comparable with the muon bucket_sector (as in DVInferenceToolBase)
    legacy: _phi_to_sector (old converter; same_sector is then always 0 for muon-calo edges)"""
    if convention == "cpp":
        return ms_sector_from_phi(phi)
    if convention == "legacy":
        return _phi_to_sector(phi, sector_mod=sector_mod)
    raise ValueError(f"Unknown calo sector convention {convention!r}; expected one of {CALO_SECTOR_CONVENTIONS}")


def _read_tree(root_file: str, tree_name: str, required_branches, optional_branches=()):
    """Read the required (and the available optional) branches of a tree, indexed by CommonEventHash."""
    tree = _open_tree_by_name(root_file, tree_name)
    clean = _normalize_keys(list(tree.keys()))
    missing = [k for k in required_branches if k not in clean]
    if missing:
        raise ValueError(f"Tree '{tree_name}' missing branches: {missing}")

    branches = list(required_branches) + [k for k in optional_branches if k in clean and k not in required_branches]
    arrays = tree.arrays([clean[k] for k in branches], library="np")
    td = {k: arrays[clean[k]] for k in branches}

    evh_arr = td["CommonEventHash"]
    event_keys = []
    ev_to_idx = defaultdict(list)

    for i in range(len(evh_arr)):
        key = _flatten_event_hash(evh_arr[i])
        if key is None:
            continue
        event_keys.append(key)
        ev_to_idx[key].append(i)

    seen = set()
    unique_keys = []
    for k in event_keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return td, ev_to_idx, unique_keys


def _dataset_name_from_root_path(root_path: str) -> str:
    """
    Build a compact dataset label from the ROOT file location.

    Priority:
      1) parent directory name (common production layout)
      2) ROOT file stem

    If the name starts with known prefixes like "MuonBucketDump_",
    the prefix is removed so the label is concise.
    """
    p = Path(root_path)
    raw = p.parent.name if p.parent.name else p.stem

    for prefix in ("MuonBucketDump_", "MuonSegmentDump_"):
        if raw.startswith(prefix) and len(raw) > len(prefix):
            return raw[len(prefix):]

    return raw


# -------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------

def eta_to_theta(eta):
    return 2.0 * np.arctan(np.exp(-eta))


def direction_from_eta_phi(eta, phi):
    theta = eta_to_theta(eta)
    st = np.sin(theta)
    return np.array(
        [st * np.cos(phi), st * np.sin(phi), np.cos(theta)],
        dtype=np.float32,
    )


def first_intersection_with_envelope(eta, phi, r_max, z_max):
    """
    Ray from origin in direction (eta, phi), intersected with:
      - barrel cylinder r = r_max
      - endcap planes z = +/- z_max
    Returns (x, y, z) in mm for the first positive intersection, or None.
    """
    u = direction_from_eta_phi(eta, phi)
    ux, uy, uz = u
    candidates = []

    ur = np.hypot(ux, uy)
    if ur > 0:
        t_barrel = r_max / ur
        z_barrel = t_barrel * uz
        if np.abs(z_barrel) <= z_max:
            candidates.append(t_barrel)

    if np.abs(uz) > 0:
        t_endcap = z_max / np.abs(uz)
        x_end = t_endcap * ux
        y_end = t_endcap * uy
        r_end = np.hypot(x_end, y_end)
        if r_end <= r_max:
            candidates.append(t_endcap)

    if not candidates:
        return None

    positives = [tc for tc in candidates if tc > 0]
    if not positives:
        return None

    t = min(positives)
    x, y, z = t * u
    return float(x), float(y), float(z)


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


# -------------------------------------------------------
# Coordinate conversion helpers
# -------------------------------------------------------

def _cartesian_to_atlas_position_polar(x, y, z):
    """Convert Cartesian position to ATLAS spherical (r, theta, phi)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    rho = np.hypot(x, y)
    r = np.sqrt(x * x + y * y + z * z).astype(np.float32)
    theta = np.arctan2(rho, z).astype(np.float32)
    phi = np.arctan2(y, x).astype(np.float32)
    return r, theta, phi


def _cartesian_to_cylindrical(x, y, z):
    """Convert Cartesian position to cylindrical (rho, phi, z)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    rho = np.hypot(x, y).astype(np.float32)
    phi = np.arctan2(y, x).astype(np.float32)
    return rho, phi, z.astype(np.float32)


def _cartesian_to_atlas_direction_angles(dx, dy, dz):
    """Convert Cartesian direction vector to ATLAS angular (theta, phi)."""
    d = _safe_normalize(np.stack([dx, dy, dz], axis=1).astype(np.float32))
    dxu, dyu, dzu = d[:, 0], d[:, 1], d[:, 2]
    rho = np.hypot(dxu, dyu)
    theta = np.arctan2(rho, dzu).astype(np.float32)
    phi = np.arctan2(dyu, dxu).astype(np.float32)
    return theta, phi, d.astype(np.float32)


# -------------------------------------------------------
# Event selection
# -------------------------------------------------------

def _event_passes_vertex_envelope(vertex_td, idxs, vertex_r_max_mm, vertex_z_max_mm):
    """Keep event only if ALL truth vertices are inside the envelope."""
    all_x, all_y, all_z = [], [], []
    for i in idxs:
        xs = np.asarray(vertex_td["truthMuonVertexPositionX"][i]).ravel()
        ys = np.asarray(vertex_td["truthMuonVertexPositionY"][i]).ravel()
        zs = np.asarray(vertex_td["truthMuonVertexPositionZ"][i]).ravel()
        n = min(len(xs), len(ys), len(zs))
        if n == 0:
            continue
        all_x.extend(xs[:n].tolist())
        all_y.extend(ys[:n].tolist())
        all_z.extend(zs[:n].tolist())

    if len(all_x) == 0:
        return False

    x = np.asarray(all_x, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.float32)
    z = np.asarray(all_z, dtype=np.float32)
    r = np.sqrt(x * x + y * y)
    # Keep event only if truth vertices are:
    #   - outside the beam pipe region: rho > 0.3 m = 30 mm
    #   - inside the requested envelope cuts 
    min_vertex_r_mm = 30.0
    inside = (r > min_vertex_r_mm) & (r <= vertex_r_max_mm) & (np.abs(z) <= vertex_z_max_mm)

    return bool(np.all(inside))


def _event_has_min_segments_and_truth(muon_td, idxs, min_segments=2, require_truth=False, min_truth=2):
    """
    Require at least `min_segments` muon segments.
    If `require_truth`, also require at least `min_truth` bucket_hasTruth flags.
    """
    total_segments = 0
    total_true_hastruth = 0
    for i in idxs:
        segx = np.asarray(muon_td["segmentPositionX"][i]).ravel()
        segy = np.asarray(muon_td["segmentPositionY"][i]).ravel()
        segz = np.asarray(muon_td["segmentPositionZ"][i]).ravel()
        dirx = np.asarray(muon_td["segmentDirectionX"][i]).ravel()
        diry = np.asarray(muon_td["segmentDirectionY"][i]).ravel()
        dirz = np.asarray(muon_td["segmentDirectionZ"][i]).ravel()
        dof = np.asarray(muon_td["segment_numberDoF"][i]).ravel()
        nseg = min(len(segx), len(segy), len(segz), len(dirx), len(diry), len(dirz), len(dof))
        total_segments += int(nseg)
        if require_truth:
            hastruth = np.asarray(muon_td["bucket_hasTruth"][i]).ravel()
            if len(hastruth) > 0:
                total_true_hastruth += int(np.count_nonzero(hastruth.astype(bool)))

    if total_segments < min_segments:
        return False
    if require_truth and total_true_hastruth < min_truth:
        return False
    return True


# -------------------------------------------------------
# Raw data collectors (coordinate-agnostic)
# -------------------------------------------------------

def _collect_muon_raw(td, idxs):
    """
    Collect raw muon segment data from MuonBucketDump entries.

    Returns a dict with raw arrays needed to assemble node features,
    or None if no segments are found:
        x_mm, y_mm, z_mm  : segment positions in mm
        dx, dy, dz         : direction components (unnormalized)
        segment_dof        : degrees of freedom per segment
        bucket_sector, bucket_chamber, bucket_layers, bucket_seg : bucket metadata
    """
    x_mm_list, y_mm_list, z_mm_list = [], [], []
    dx_list, dy_list, dz_list = [], [], []
    segment_dof_list = []
    segment_chi2_list = []
    has_chi2 = "segment_chiSquared" in td
    bucket_seg_list, bucket_sector_list, bucket_chamber_list, bucket_layers_list = [], [], [], []

    for i in idxs:
        segx = np.asarray(td["segmentPositionX"][i]).ravel()
        segy = np.asarray(td["segmentPositionY"][i]).ravel()
        segz = np.asarray(td["segmentPositionZ"][i]).ravel()
        dirx = np.asarray(td["segmentDirectionX"][i]).ravel()
        diry = np.asarray(td["segmentDirectionY"][i]).ravel()
        dirz = np.asarray(td["segmentDirectionZ"][i]).ravel()
        dof = np.asarray(td["segment_numberDoF"][i]).ravel()
        chi2 = np.asarray(td["segment_chiSquared"][i]).ravel() if has_chi2 else np.full(len(dof), np.nan)

        nseg = min(len(segx), len(segy), len(segz), len(dirx), len(diry), len(dirz), len(dof), len(chi2))
        if nseg == 0:
            continue

        seg_count_val = np.asarray(td["bucket_segments"][i]).ravel()
        sec_val = np.asarray(td["bucket_sector"][i]).ravel()
        chamber_val = np.asarray(td["bucket_chamberIndex"][i]).ravel()
        layer_val = np.asarray(td["bucket_layers"][i]).ravel()

        seg_count = int(seg_count_val[0]) if len(seg_count_val) > 0 else -1
        sec = int(sec_val[0]) if len(sec_val) > 0 else -1
        chamber = int(chamber_val[0]) if len(chamber_val) > 0 else -1
        layer = int(layer_val[0]) if len(layer_val) > 0 else -1

        for j in range(nseg):
            x_mm_list.append(float(segx[j]))
            y_mm_list.append(float(segy[j]))
            z_mm_list.append(float(segz[j]))
            dx_list.append(float(dirx[j]))
            dy_list.append(float(diry[j]))
            dz_list.append(float(dirz[j]))
            segment_dof_list.append(float(dof[j]))
            segment_chi2_list.append(float(chi2[j]))
            bucket_seg_list.append(float(seg_count))
            bucket_sector_list.append(sec)
            bucket_chamber_list.append(chamber)
            bucket_layers_list.append(layer)

    if len(x_mm_list) == 0:
        return None

    return {
        "x_mm": np.asarray(x_mm_list, dtype=np.float32),
        "y_mm": np.asarray(y_mm_list, dtype=np.float32),
        "z_mm": np.asarray(z_mm_list, dtype=np.float32),
        "dx": np.asarray(dx_list, dtype=np.float32),
        "dy": np.asarray(dy_list, dtype=np.float32),
        "dz": np.asarray(dz_list, dtype=np.float32),
        "segment_dof": np.asarray(segment_dof_list, dtype=np.float32),
        "segment_chi2": np.asarray(segment_chi2_list, dtype=np.float32),  # NaN without segment_chiSquared
        "bucket_seg": np.asarray(bucket_seg_list, dtype=np.float32),
        "bucket_sector": np.asarray(bucket_sector_list, dtype=np.int64),
        "bucket_chamber": np.asarray(bucket_chamber_list, dtype=np.int64),
        "bucket_layers": np.asarray(bucket_layers_list, dtype=np.int64),
    }


def _collect_calo_filtered(
    td,
    idxs,
    seg_eta_list,
    seg_phi_list,
    sector_mod,
    min_tower_energy_mev,
    max_tower_segment_dr,
    calo_r_max_mm,
    calo_z_max_mm,
    sector_convention="legacy",
):
    """
    Filter calorimeter towers (energy cut, ΔR cut, envelope intersection).

    Returns a dict with filtered raw tower data, or None if no towers pass:
        tower_energy, tower_eta, tower_phi, tower_ncells
        dx, dy, dz  : direction components
        tower_xyz_m : intersection point with calorimeter envelope [m]
        tower_min_dr : minimum ΔR to any muon segment
        sector       : phi sector index (see _calo_sector for the conventions)
    """
    if len(seg_eta_list) == 0:
        return None

    seg_eta_arr = np.asarray(seg_eta_list, dtype=np.float32)
    seg_phi_arr = np.asarray(seg_phi_list, dtype=np.float32)

    tower_energy_list, tower_eta_list, tower_phi_list, tower_ncells_list = [], [], [], []
    dx_list, dy_list, dz_list = [], [], []
    tower_xyz_m_list = []
    tower_min_dr_list = []

    for i in idxs:
        tdx = np.asarray(td["tower_directionX"][i]).ravel()
        tdy = np.asarray(td["tower_directionY"][i]).ravel()
        tdz = np.asarray(td["tower_directionZ"][i]).ravel()
        teta = np.asarray(td["tower_eta"][i]).ravel()
        tphi = np.asarray(td["tower_phi"][i]).ravel()
        tene = np.asarray(td["tower_energy_mev"][i]).ravel()
        tnc = np.asarray(td["tower_nCells"][i]).ravel()

        ntow = min(len(tdx), len(tdy), len(tdz), len(teta), len(tphi), len(tene), len(tnc))
        if ntow == 0:
            continue

        for j in range(ntow):
            e = float(tene[j])
            if e < min_tower_energy_mev:
                continue

            eta = float(teta[j])
            phi = float(tphi[j])

            dphi = delta_phi(phi, seg_phi_arr)
            deta = eta - seg_eta_arr
            dr_all = np.hypot(deta, dphi)
            dr_min = float(np.min(dr_all))

            if dr_min >= max_tower_segment_dr:
                continue

            pos = first_intersection_with_envelope(
                eta=eta, phi=phi, r_max=calo_r_max_mm, z_max=calo_z_max_mm,
            )
            if pos is None:
                continue

            x_mm, y_mm, z_mm = pos
            tower_energy_list.append(e)
            tower_eta_list.append(eta)
            tower_phi_list.append(phi)
            tower_ncells_list.append(float(tnc[j]))
            dx_list.append(float(tdx[j]))
            dy_list.append(float(tdy[j]))
            dz_list.append(float(tdz[j]))
            tower_xyz_m_list.append([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0])
            tower_min_dr_list.append(dr_min)

    if len(tower_energy_list) == 0:
        return None

    tower_energy = np.asarray(tower_energy_list, dtype=np.float32)
    tower_phi = np.asarray(tower_phi_list, dtype=np.float32)

    return {
        "tower_energy": tower_energy,
        "tower_eta": np.asarray(tower_eta_list, dtype=np.float32),
        "tower_phi": tower_phi,
        "tower_ncells": np.asarray(tower_ncells_list, dtype=np.float32),
        "dx": np.asarray(dx_list, dtype=np.float32),
        "dy": np.asarray(dy_list, dtype=np.float32),
        "dz": np.asarray(dz_list, dtype=np.float32),
        "tower_xyz_m": np.asarray(tower_xyz_m_list, dtype=np.float32),
        "tower_min_dr": np.asarray(tower_min_dr_list, dtype=np.float32),
        "sector": _calo_sector(tower_phi, sector_mod=sector_mod, convention=sector_convention).astype(np.int64),
    }


# -------------------------------------------------------
# Edge building
# -------------------------------------------------------

def build_edges_segment_tower_by_dr(phi, eta, node_type, max_tower_segment_dr):
    """
    Directed edges between segment (node_type=0) and tower (node_type=1)
    nodes for pairs with ΔR < max_tower_segment_dr.
    """
    mu_idx = np.where(node_type == 0)[0]
    ca_idx = np.where(node_type == 1)[0]

    if len(mu_idx) == 0 or len(ca_idx) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    src_list, dst_list = [], []
    mu_eta = eta[mu_idx]
    mu_phi = phi[mu_idx]
    ca_eta = eta[ca_idx]
    ca_phi = phi[ca_idx]

    for local_m, global_m in enumerate(mu_idx):
        dphi = delta_phi(mu_phi[local_m], ca_phi)
        deta = mu_eta[local_m] - ca_eta
        dr = np.hypot(deta, dphi)
        matched = np.where(dr < max_tower_segment_dr)[0]
        for local_c in matched:
            global_c = ca_idx[local_c]
            src_list.append(global_m)
            dst_list.append(global_c)
            src_list.append(global_c)
            dst_list.append(global_m)

    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    return np.stack(
        [np.asarray(src_list, dtype=np.int64), np.asarray(dst_list, dtype=np.int64)],
        axis=0,
    )


def edge_features(energy_like, phi, eta, dir_u, sector, node_type, edge_index):
    """Compute 5 edge features: [d_energy, d_phi, d_eta, cos_angle, same_sector]."""
    if edge_index.shape[1] == 0:
        return np.zeros((0, 5), dtype=np.float32)

    src = edge_index[0]
    dst = edge_index[1]
    d_energy_like = (energy_like[dst] - energy_like[src]).reshape(-1, 1).astype(np.float32)
    d_phi = _wrap_phi(phi[dst] - phi[src]).reshape(-1, 1).astype(np.float32)
    d_eta = (eta[dst] - eta[src]).reshape(-1, 1).astype(np.float32)
    cosang = np.sum(dir_u[src] * dir_u[dst], axis=1, keepdims=True).astype(np.float32)
    same_sector = (sector[src] == sector[dst]).astype(np.float32).reshape(-1, 1)
    return np.concatenate([d_energy_like, d_phi, d_eta, cosang, same_sector], axis=1).astype(np.float32)


# -------------------------------------------------------
# HDF5 writing helpers
# -------------------------------------------------------

def _write_event_group(
    g,
    event_hash,
    x,
    edge_index,
    edge_attr,
    y_vertex,
    phi,
    eta,
    energy_like,
    dir_u,
    sector,
    n_muon_nodes,
    n_calo_nodes,
    muon_xyz_m=None,
    muon_bucket=None,
    tower_xyz_m=None,
    tower_min_dr=None,
    dataset_name=None,
):
    g.attrs["event_hash"] = np.asarray(event_hash, dtype=np.int64)
    g.attrs["n_muon_nodes"] = int(n_muon_nodes)
    g.attrs["n_calo_nodes"] = int(n_calo_nodes)
    if dataset_name is not None:
        g.attrs["dataset_name"] = str(dataset_name)

    g.create_dataset("x", data=x, compression="gzip", compression_opts=4)
    g.create_dataset("edge_index", data=edge_index, compression="gzip", compression_opts=4)
    g.create_dataset("edge_attr", data=edge_attr, compression="gzip", compression_opts=4)
    g.create_dataset("y_vertex", data=y_vertex, compression="gzip", compression_opts=4)
    g.create_dataset("phi", data=phi, compression="gzip", compression_opts=4)
    g.create_dataset("eta", data=eta, compression="gzip", compression_opts=4)
    g.create_dataset("energy_like", data=energy_like, compression="gzip", compression_opts=4)
    g.create_dataset("dir_u", data=dir_u, compression="gzip", compression_opts=4)
    g.create_dataset("sector", data=sector, compression="gzip", compression_opts=4)

    if muon_xyz_m is not None:
        g.create_dataset("muon_xyz_m", data=muon_xyz_m, compression="gzip", compression_opts=4)
    if muon_bucket is not None:
        g.create_dataset("muon_bucket", data=muon_bucket, compression="gzip", compression_opts=4)
    if tower_xyz_m is not None:
        g.create_dataset("tower_xyz_m", data=tower_xyz_m, compression="gzip", compression_opts=4)
    if tower_min_dr is not None:
        g.create_dataset("tower_min_dr", data=tower_min_dr, compression="gzip", compression_opts=4)


def _open_new_part(output_dir: Path, output_name: str, part_idx: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_name}_part{part_idx:04d}.h5"
    h5 = h5py.File(out_path, "w")
    h5.attrs["n_events_written"] = 0
    h5.create_group("events")
    return h5, out_path


# -------------------------------------------------------
# Shared argparse setup
# -------------------------------------------------------

def add_converter_args(ap):
    """Add all shared converter arguments to an ArgumentParser."""
    ap.add_argument("--input-dir", required=True, help="Directory with ROOT files")
    ap.add_argument("--pattern", default="*.root", help="Glob pattern (default: *.root)")
    ap.add_argument("--output-dir", required=True, help="Directory for output H5 files")
    ap.add_argument("--output-name", required=True, help="Base name for output files (without _partXXXX.h5)")
    ap.add_argument("--max-events", type=int, default=-1, help="Global cap across all ROOT files (-1 = all)")
    ap.add_argument("--events-per-part", type=int, default=10000, help="Max graphs per output H5 part")
    ap.add_argument("--sector-mod", type=int, default=16, help="Number of sectors for calo phi->sector mapping")
    ap.add_argument("--min-tower-energy-mev", type=float, default=1000.0)
    ap.add_argument("--max-tower-segment-dr", type=float, default=0.4)
    ap.add_argument("--calo-r-max-mm", type=float, default=4250.0)
    ap.add_argument("--calo-z-max-mm", type=float, default=6500.0)
    ap.add_argument("--vertex-r-max-mm", type=float, required=True,
                    help="Keep event only if all truth vertices satisfy r <= this")
    ap.add_argument("--vertex-z-max-mm", type=float, required=True,
                    help="Keep event only if all truth vertices satisfy |z| <= this")
    ap.add_argument("--isMC", dest="isMC", action="store_true", default=True,
                    help="Require at least 2 true bucket_hasTruth values per event (default: True)")
    ap.add_argument("--isData", dest="isMC", action="store_false",
                    help="Disable MC-specific bucket_hasTruth requirement")
    ap.add_argument("--allow-single-modality", action="store_true",
                    help="Keep events even if only muon or only calo survives")
    return ap


# -------------------------------------------------------
# Shared converter main loop
# -------------------------------------------------------

def run_converter_main_loop(args, build_vertex_target_fn, build_muon_nodes_fn, build_calo_nodes_fn):
    """
    Main processing loop shared by all DisplacedVertex converters.

    Args:
        args: parsed argparse namespace (from add_converter_args)
        build_vertex_target_fn: callable(vertex_td, idxs) -> np.ndarray or None
        build_muon_nodes_fn: callable(muon_td, idxs) -> dict or None
        build_calo_nodes_fn: callable(calo_td, idxs, **kwargs) -> dict or None
            Must accept keyword args: sector_mod, min_tower_energy_mev,
            max_tower_segment_dr, calo_r_max_mm, calo_z_max_mm,
            seg_eta_list, seg_phi_list
    """
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No ROOT files matched: {os.path.join(args.input_dir, args.pattern)}")

    output_dir = Path(args.output_dir)
    output_name = args.output_name
    part_idx = 1
    h5, out_path = _open_new_part(output_dir, output_name, part_idx)
    events_grp = h5["events"]
    print(f"[i] writing {out_path}")

    total_written = 0
    written_in_part = 0
    skipped = 0

    for root_path in files:
        print(f"[i] reading {root_path}")
        dataset_name = _dataset_name_from_root_path(root_path)

        try:
            mu_td, mu_ev_to_idx, mu_keys = _read_tree(root_path, "MuonBucketDump", REQUIRED_MUON_BRANCHES)
            ca_td, ca_ev_to_idx, ca_keys = _read_tree(root_path, "CaloDump", REQUIRED_CALO_BRANCHES)
            vx_td, vx_ev_to_idx, vx_keys = _read_tree(root_path, "MuonVertexDump", REQUIRED_VERTEX_BRANCHES)
        except Exception as e:
            print(f"[!] skip file (failed to read trees): {root_path} :: {e}")
            continue

        mu_set = set(mu_keys)
        ca_set = set(ca_keys)

        if args.allow_single_modality:
            event_keys = [k for k in vx_keys if (k in mu_set) or (k in ca_set)]
        else:
            event_keys = [k for k in vx_keys if (k in mu_set) and (k in ca_set)]

        for evh in event_keys:
            if args.max_events > 0 and total_written >= args.max_events:
                h5.attrs["skipped_empty_or_too_small"] = skipped
                h5.close()
                print(f"[done] reached --max-events={args.max_events}; wrote {total_written} graphs")
                return

            if written_in_part >= args.events_per_part:
                h5.attrs["skipped_empty_or_too_small"] = skipped
                h5.close()
                part_idx += 1
                h5, out_path = _open_new_part(output_dir, output_name, part_idx)
                events_grp = h5["events"]
                print(f"[i] writing {out_path}")
                written_in_part = 0
                skipped = 0

            vx_idxs = np.asarray(vx_ev_to_idx[evh], dtype=np.int64)

            if not _event_passes_vertex_envelope(
                vx_td, vx_idxs,
                vertex_r_max_mm=args.vertex_r_max_mm,
                vertex_z_max_mm=args.vertex_z_max_mm,
            ):
                skipped += 1
                continue

            if evh not in mu_ev_to_idx:
                skipped += 1
                continue

            mu_idxs = np.asarray(mu_ev_to_idx[evh], dtype=np.int64)
            if not _event_has_min_segments_and_truth(
                mu_td, mu_idxs, min_segments=2, require_truth=args.isMC, min_truth=2,
            ):
                skipped += 1
                continue

            y_vertex = build_vertex_target_fn(vx_td, vx_idxs)
            if y_vertex is None:
                skipped += 1
                continue

            mu_nodes = build_muon_nodes_fn(mu_td, mu_idxs)

            ca_nodes = None
            if evh in ca_ev_to_idx and mu_nodes is not None:
                ca_nodes = build_calo_nodes_fn(
                    ca_td,
                    np.asarray(ca_ev_to_idx[evh], dtype=np.int64),
                    sector_mod=args.sector_mod,
                    min_tower_energy_mev=args.min_tower_energy_mev,
                    max_tower_segment_dr=args.max_tower_segment_dr,
                    calo_r_max_mm=args.calo_r_max_mm,
                    calo_z_max_mm=args.calo_z_max_mm,
                    seg_eta_list=mu_nodes["eta"],
                    seg_phi_list=mu_nodes["phi"],
                )

            if (mu_nodes is None) and (ca_nodes is None):
                skipped += 1
                continue

            if (not args.allow_single_modality) and ((mu_nodes is None) or (ca_nodes is None)):
                skipped += 1
                continue

            pieces, phi_pieces, eta_pieces = [], [], []
            energy_like_pieces, dir_pieces, sector_pieces, type_pieces = [], [], [], []
            n_muon_nodes = 0
            n_calo_nodes = 0
            muon_xyz_m = muon_bucket = tower_xyz_m = tower_min_dr = None

            if mu_nodes is not None:
                pieces.append(mu_nodes["x"])
                phi_pieces.append(mu_nodes["phi"])
                eta_pieces.append(mu_nodes["eta"])
                energy_like_pieces.append(mu_nodes["energy_like"])
                dir_pieces.append(mu_nodes["dir_u"])
                sector_pieces.append(mu_nodes["sector"])
                type_pieces.append(mu_nodes["node_type"])
                n_muon_nodes = mu_nodes["x"].shape[0]
                muon_xyz_m = mu_nodes["muon_xyz_m"]
                muon_bucket = mu_nodes["muon_bucket"]

            if ca_nodes is not None:
                pieces.append(ca_nodes["x"])
                phi_pieces.append(ca_nodes["phi"])
                eta_pieces.append(ca_nodes["eta"])
                energy_like_pieces.append(ca_nodes["energy_like"])
                dir_pieces.append(ca_nodes["dir_u"])
                sector_pieces.append(ca_nodes["sector"])
                type_pieces.append(ca_nodes["node_type"])
                n_calo_nodes = ca_nodes["x"].shape[0]
                tower_xyz_m = ca_nodes["tower_xyz_m"]
                tower_min_dr = ca_nodes["tower_min_dr"]

            x = np.concatenate(pieces, axis=0).astype(np.float32)
            phi = np.concatenate(phi_pieces, axis=0).astype(np.float32)
            eta = np.concatenate(eta_pieces, axis=0).astype(np.float32)
            energy_like = np.concatenate(energy_like_pieces, axis=0).astype(np.float32)
            dir_u = np.concatenate(dir_pieces, axis=0).astype(np.float32)
            sector = np.concatenate(sector_pieces, axis=0).astype(np.int64)
            node_type = np.concatenate(type_pieces, axis=0).astype(np.int64)

            if x.shape[0] < 2:
                skipped += 1
                continue

            edge_index = build_edges_segment_tower_by_dr(
                phi=phi, eta=eta, node_type=node_type,
                max_tower_segment_dr=args.max_tower_segment_dr,
            )

            if edge_index.shape[1] == 0:
                skipped += 1
                continue

            edge_attr = edge_features(
                energy_like=energy_like, phi=phi, eta=eta,
                dir_u=dir_u, sector=sector, node_type=node_type,
                edge_index=edge_index,
            )

            g = events_grp.create_group(f"{total_written:07d}")
            _write_event_group(
                g=g,
                event_hash=evh,
                x=x,
                edge_index=edge_index.astype(np.int64),
                edge_attr=edge_attr.astype(np.float32),
                y_vertex=y_vertex.astype(np.float32),
                phi=phi,
                eta=eta,
                energy_like=energy_like,
                dir_u=dir_u,
                sector=sector,
                n_muon_nodes=n_muon_nodes,
                n_calo_nodes=n_calo_nodes,
                muon_xyz_m=muon_xyz_m,
                muon_bucket=muon_bucket,
                tower_xyz_m=tower_xyz_m,
                tower_min_dr=tower_min_dr,
                dataset_name=dataset_name,
            )

            total_written += 1
            written_in_part += 1
            h5.attrs["n_events_written"] = int(h5.attrs["n_events_written"]) + 1

    h5.attrs["skipped_empty_or_too_small"] = skipped
    h5.close()
    print(f"[done] wrote {total_written} graphs across all files")
# ===========================================================================
# DisplacedVertex graph classification utilities
# ===========================================================================

# The functions below are intentionally additive: the regression converters above
# keep their original behaviour, while the new DisplacedVertex.py classifier can
# reuse the same ROOT reading, node-building, edge-building, and HDF5 machinery.

def _lazy_import_torch():
    """Import torch only when dataset/collate helpers need it."""
    try:
        import torch
        return torch
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "PyTorch is required for DisplacedVertexGraphDataset/collate helpers, "
            "but it could not be imported. ROOT -> HDF5 conversion does not need torch."
        ) from exc


DEFAULT_DV_SIGNAL_FILENAME_PATTERNS = [
    "*a_mumu_*.root",
    "*Haa4mu_*.root",
]


def _normalize_signal_filename_patterns(signal_filename_patterns=None) -> list[str]:
    """Return signal filename patterns as a concrete list."""
    if signal_filename_patterns is None:
        return list(DEFAULT_DV_SIGNAL_FILENAME_PATTERNS)
    if isinstance(signal_filename_patterns, str):
        return [signal_filename_patterns]
    return list(signal_filename_patterns)


def _filename_matches_signal_patterns(
    root_path: str,
    signal_filename_patterns=None,
) -> bool:
    """Return True when the ROOT basename matches any configured signal pattern."""
    import fnmatch

    basename = os.path.basename(root_path)
    signal_filename_patterns = _normalize_signal_filename_patterns(signal_filename_patterns)
    return any(fnmatch.fnmatch(basename, pat) for pat in signal_filename_patterns)

# Backward-compatible alias for older code paths.
_filename_matches_signal_pattern = _filename_matches_signal_patterns

def _event_vertex_rhos_mm(vertex_td, idxs) -> np.ndarray:
    """Return all truth-vertex rho values for an event in millimetres."""
    if vertex_td is None or idxs is None:
        return np.zeros((0,), dtype=np.float32)

    rhos = []
    for i in idxs:
        xs = np.asarray(vertex_td["truthMuonVertexPositionX"][i]).ravel()
        ys = np.asarray(vertex_td["truthMuonVertexPositionY"][i]).ravel()
        n = min(len(xs), len(ys))
        if n == 0:
            continue
        for j in range(n):
            x = xs[j]
            y = ys[j]
            if x is None or y is None:
                continue
            x = float(x)
            y = float(y)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            rhos.append(np.hypot(x, y))

    return np.asarray(rhos, dtype=np.float32)


def compute_displaced_vertex_graph_label(
    root_path: str,
    vertex_td=None,
    vx_idxs=None,
    signal_filename_patterns=None,
    signal_r_min_mm: float = 800.0,
    signal_r_max_mm: float = 8000.0,
) -> tuple[int, np.ndarray]:
    """
    Compute the graph-level binary label for one event.

    Label definition:
      - files whose basename does not match any configured signal pattern are background (0)
      - matching files are signal (1) only when at least one truth vertex has
        ``signal_r_min_mm < rho < signal_r_max_mm``

    Returns ``(label, vertex_rhos_mm)``. Boundaries are strict to match
    "higher than 800 mm and lower than 8000 mm".
    """
    rhos_mm = _event_vertex_rhos_mm(vertex_td, vx_idxs)

    if not _filename_matches_signal_patterns(root_path, signal_filename_patterns):
        return 0, rhos_mm

    in_window = (rhos_mm > float(signal_r_min_mm)) & (rhos_mm < float(signal_r_max_mm))
    return int(np.any(in_window)), rhos_mm


def _eta_from_xyz_mm(x_mm, y_mm, z_mm) -> np.ndarray:
    """Compute pseudorapidity from Cartesian positions in millimetres."""
    x_mm = np.asarray(x_mm, dtype=np.float32)
    y_mm = np.asarray(y_mm, dtype=np.float32)
    z_mm = np.asarray(z_mm, dtype=np.float32)
    r_xy = np.hypot(x_mm, y_mm)
    eta = np.empty_like(r_xy, dtype=np.float32)
    mask = r_xy > 0
    eta[mask] = np.arcsinh(z_mm[mask] / r_xy[mask])
    eta[~mask] = np.sign(z_mm[~mask]) * 1.0e6
    return eta.astype(np.float32)


def build_displaced_vertex_muon_nodes_cylindrical(muon_td, idxs, slot5_feature: str = MUON_SLOT5_FEATURE_DEFAULT):
    """
    Build muon-segment nodes using the cylindrical converter feature convention.

    Node features:
        [r, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]

    For muon segments, ``energy_like`` is zero and ``nCells_or_DoF`` is the
    segment number of degrees of freedom. With slot5_feature = MUON_SLOT5_FEATURE_CHI2 the
    feature column 5 holds chi2 / max(nDoF, 1) instead (the energy_like array used by the
    edge features stays 0).

    The returned dict also carries the raw per-segment geometry of the segment-segment edges and
    tracklet nodes (float64 positions in mm and unit directions, station layer, nDoF, chi2/nDoF).
    """
    if slot5_feature not in (MUON_SLOT5_FEATURE_DEFAULT, MUON_SLOT5_FEATURE_CHI2):
        raise ValueError(f"Unknown muon slot-5 feature {slot5_feature!r}")
    raw = _collect_muon_raw(muon_td, idxs)
    if raw is None:
        return None

    x_mm, y_mm, z_mm = raw["x_mm"], raw["y_mm"], raw["z_mm"]
    pos_m = np.stack([x_mm, y_mm, z_mm], axis=1).astype(np.float32) / 1000.0

    r_pos, theta_pos, phi_pos = _cartesian_to_atlas_position_polar(
        pos_m[:, 0], pos_m[:, 1], pos_m[:, 2]
    )
    theta_dir, phi_dir, dir_u = _cartesian_to_atlas_direction_angles(
        raw["dx"], raw["dy"], raw["dz"]
    )

    phi = np.arctan2(y_mm, x_mm).astype(np.float32)
    eta = _eta_from_xyz_mm(x_mm, y_mm, z_mm)
    energy_like = np.zeros(len(x_mm), dtype=np.float32)
    ncells_or_dof = raw["segment_dof"].astype(np.float32)
    chi2_ndof = (raw["segment_chi2"].astype(np.float64) / np.maximum(raw["segment_dof"].astype(np.float64), 1.0))
    if slot5_feature == MUON_SLOT5_FEATURE_CHI2:
        if not np.all(np.isfinite(chi2_ndof)):
            raise ValueError("The segment-chi2 feature needs the MuonBucketDump branch segment_chiSquared.")
        slot5 = chi2_ndof.astype(np.float32)
    else:
        slot5 = energy_like

    x = np.stack(
        [r_pos, theta_pos, phi_pos, theta_dir, phi_dir, slot5, ncells_or_dof],
        axis=1,
    ).astype(np.float32)

    seg_pos_mm = np.stack([x_mm, y_mm, z_mm], axis=1).astype(np.float64)
    seg_dir = np.stack([raw["dx"], raw["dy"], raw["dz"]], axis=1).astype(np.float64)
    seg_dir_norm = np.linalg.norm(seg_dir, axis=1, keepdims=True)
    seg_dir = seg_dir / np.where(seg_dir_norm > 0, seg_dir_norm, 1.0)

    return {
        "x": x,
        "phi": phi,
        "eta": eta,
        "energy_like": energy_like,
        "dir_u": dir_u,
        "sector": raw["bucket_sector"].astype(np.int64),
        "node_type": np.zeros(len(x_mm), dtype=np.int64),
        "muon_xyz_m": pos_m,
        "muon_bucket": np.stack(
            [
                raw["bucket_chamber"].astype(np.int64),
                raw["bucket_layers"].astype(np.int64),
                raw["bucket_sector"].astype(np.int64),
                raw["bucket_seg"].astype(np.int64),
            ],
            axis=1,
        ),
        # raw segment geometry for the segment-segment edges / tracklets (not written to the H5 files)
        "seg_pos_mm": seg_pos_mm,
        "seg_dir": seg_dir,
        "seg_layer": station_layer_from_chamber_index(raw["bucket_chamber"]),
        "seg_ndof": raw["segment_dof"].astype(np.float64),
        "seg_chi2_ndof": chi2_ndof,
    }


def build_displaced_vertex_calo_nodes_cylindrical(
    calo_td,
    idxs,
    sector_mod: int,
    min_tower_energy_mev: float,
    max_tower_segment_dr: float,
    calo_r_max_mm: float,
    calo_z_max_mm: float,
    seg_eta_list,
    seg_phi_list,
    sector_convention: str = "cpp",
):
    """
    Build calorimeter tower nodes using the cylindrical converter convention.

    Towers are filtered by energy, ΔR to muon segments, and the calorimeter
    envelope intersection, matching the previous cylindrical converter logic.
    The tower sector follows ``sector_convention`` (default "cpp": MS sector number, as in
    DVInferenceToolBase; "legacy": the old phi bin).
    """
    raw = _collect_calo_filtered(
        calo_td,
        idxs,
        seg_eta_list,
        seg_phi_list,
        sector_mod,
        min_tower_energy_mev,
        max_tower_segment_dr,
        calo_r_max_mm,
        calo_z_max_mm,
        sector_convention=sector_convention,
    )
    if raw is None:
        return None

    tower_xyz_m = raw["tower_xyz_m"]
    r_pos, theta_pos, phi_pos = _cartesian_to_atlas_position_polar(
        tower_xyz_m[:, 0], tower_xyz_m[:, 1], tower_xyz_m[:, 2]
    )
    theta_dir, phi_dir, dir_u = _cartesian_to_atlas_direction_angles(
        raw["dx"], raw["dy"], raw["dz"]
    )

    x = np.stack(
        [
            r_pos,
            theta_pos,
            phi_pos,
            theta_dir,
            phi_dir,
            raw["tower_energy"],
            raw["tower_ncells"],
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "x": x,
        "phi": raw["tower_phi"].astype(np.float32),
        "eta": raw["tower_eta"].astype(np.float32),
        "energy_like": raw["tower_energy"].astype(np.float32),
        "dir_u": dir_u,
        "sector": raw["sector"].astype(np.int64),
        "node_type": np.ones(len(raw["tower_energy"]), dtype=np.int64),
        "tower_xyz_m": tower_xyz_m.astype(np.float32),
        "tower_min_dr": raw["tower_min_dr"].astype(np.float32),
    }


def assemble_displaced_vertex_graph(
    mu_nodes,
    calo_nodes,
    max_tower_segment_dr: float,
    require_edges: bool = False,
):
    """Concatenate node dictionaries and build edge_index/edge_attr."""
    if (mu_nodes is None) and (calo_nodes is None):
        return None

    pieces, phi_pieces, eta_pieces = [], [], []
    energy_like_pieces, dir_pieces, sector_pieces, type_pieces = [], [], [], []
    n_muon_nodes = 0
    n_calo_nodes = 0
    muon_xyz_m = muon_bucket = tower_xyz_m = tower_min_dr = None

    if mu_nodes is not None:
        pieces.append(mu_nodes["x"])
        phi_pieces.append(mu_nodes["phi"])
        eta_pieces.append(mu_nodes["eta"])
        energy_like_pieces.append(mu_nodes["energy_like"])
        dir_pieces.append(mu_nodes["dir_u"])
        sector_pieces.append(mu_nodes["sector"])
        type_pieces.append(mu_nodes["node_type"])
        n_muon_nodes = int(mu_nodes["x"].shape[0])
        muon_xyz_m = mu_nodes.get("muon_xyz_m")
        muon_bucket = mu_nodes.get("muon_bucket")

    if calo_nodes is not None:
        pieces.append(calo_nodes["x"])
        phi_pieces.append(calo_nodes["phi"])
        eta_pieces.append(calo_nodes["eta"])
        energy_like_pieces.append(calo_nodes["energy_like"])
        dir_pieces.append(calo_nodes["dir_u"])
        sector_pieces.append(calo_nodes["sector"])
        type_pieces.append(calo_nodes["node_type"])
        n_calo_nodes = int(calo_nodes["x"].shape[0])
        tower_xyz_m = calo_nodes.get("tower_xyz_m")
        tower_min_dr = calo_nodes.get("tower_min_dr")

    x = np.concatenate(pieces, axis=0).astype(np.float32)
    if x.shape[0] == 0:
        return None

    phi = np.concatenate(phi_pieces, axis=0).astype(np.float32)
    eta = np.concatenate(eta_pieces, axis=0).astype(np.float32)
    energy_like = np.concatenate(energy_like_pieces, axis=0).astype(np.float32)
    dir_u = np.concatenate(dir_pieces, axis=0).astype(np.float32)
    sector = np.concatenate(sector_pieces, axis=0).astype(np.int64)
    node_type = np.concatenate(type_pieces, axis=0).astype(np.int64)

    edge_index = build_edges_segment_tower_by_dr(
        phi=phi,
        eta=eta,
        node_type=node_type,
        max_tower_segment_dr=max_tower_segment_dr,
    ).astype(np.int64)

    if require_edges and edge_index.shape[1] == 0:
        return None

    edge_attr = edge_features(
        energy_like=energy_like,
        phi=phi,
        eta=eta,
        dir_u=dir_u,
        sector=sector,
        node_type=node_type,
        edge_index=edge_index,
    ).astype(np.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "phi": phi,
        "eta": eta,
        "energy_like": energy_like,
        "dir_u": dir_u,
        "sector": sector,
        "node_type": node_type,
        "n_muon_nodes": n_muon_nodes,
        "n_calo_nodes": n_calo_nodes,
        "muon_xyz_m": muon_xyz_m,
        "muon_bucket": muon_bucket,
        "tower_xyz_m": tower_xyz_m,
        "tower_min_dr": tower_min_dr,
    }


def build_displaced_vertex_track_nodes(
    track_td,
    idxs,
    seg_eta_list,
    seg_phi_list,
    min_track_pt_mev: float = 2000.0,
    max_track_segment_dr: float = 0.3,
    max_track_dz0_sintheta_mm: float = -1.0,
):
    """
    Build inner-tracker track nodes (TRACK_FEATURE_NAMES) from a TrigFSTrackDump-like tree.

    Selection (to be mirrored by the Athena graph builder):
      - pT >= min_track_pt_mev
      - deltaR(track perigee eta/phi, muon-node eta/phi) < max_track_segment_dr for at least one muon node
        (the dumper already applied its own cone, dR < 0.3 by default, so larger values have no effect)
      - optionally |(z0 + vz - z_PV) sin(theta)| <= max_track_dz0_sintheta_mm (< 0: no cut)
    Without a hard-scatter vertex (hsPV_z = -999) the z0 features are 0, i.e. the track counts as compatible with
    the vertex, as in the prompt-track veto (MlMuonDVtxTrkVetoHypoTool).

    Returns None when no track passes, else a dict with the node features ``x`` [T, 11] and the
    eta/phi/energy_like (= pT in MeV)/dir_u/sector arrays used to build the track-segment edges.
    """
    if track_td is None or idxs is None or len(idxs) == 0:
        return None
    seg_eta = np.asarray(seg_eta_list, dtype=np.float32)
    seg_phi = np.asarray(seg_phi_list, dtype=np.float32)
    if seg_eta.size == 0:
        return None

    cols = {b: [] for b in _TRACK_VECTOR_BRANCHES}
    pv_z = None
    for i in idxs:
        per_branch = {b: np.asarray(track_td[b][i]).ravel() for b in _TRACK_VECTOR_BRANCHES}
        n = min(len(v) for v in per_branch.values())
        for b, v in per_branch.items():
            cols[b].append(v[:n].astype(np.float64))
        if pv_z is None:
            z = np.asarray(track_td["hsPV_z"][i], dtype=np.float64).ravel()
            if z.size and np.isfinite(z[0]) and z[0] > -998.0:
                pv_z = float(z[0])
    trk = {b: np.concatenate(v) if v else np.zeros(0, dtype=np.float64) for b, v in cols.items()}
    if trk["track_pt"].size == 0:
        return None

    eta32 = trk["track_eta"].astype(np.float32)
    phi32 = trk["track_phi"].astype(np.float32)
    dr = np.hypot(eta32[:, None] - seg_eta[None, :], _wrap_phi(phi32[:, None] - seg_phi[None, :]))
    min_dr = dr.min(axis=1).astype(np.float32)

    theta = 2.0 * np.arctan(np.exp(-trk["track_eta"]))
    dz = trk["track_z0"] + trk["track_vz"] - pv_z if pv_z is not None else np.zeros_like(trk["track_pt"])
    dz0_sintheta = dz * np.sin(theta)

    keep = (trk["track_pt"] >= min_track_pt_mev) & (min_dr < max_track_segment_dr)
    if max_track_dz0_sintheta_mm is not None and max_track_dz0_sintheta_mm >= 0:
        keep &= np.abs(dz0_sintheta) <= max_track_dz0_sintheta_mm
    if not np.any(keep):
        return None

    pt_gev = trk["track_pt"][keep] / 1000.0
    d0 = trk["track_d0"][keep]
    d0_err = trk["track_d0Err"][keep]
    z0_err = trk["track_z0Err"][keep]
    th = theta[keep]
    ph = trk["track_phi"][keep]
    x = np.stack(
        [
            th,
            ph,
            np.log(pt_gev),
            trk["track_charge"][keep] / pt_gev,
            d0,
            np.divide(d0, d0_err, out=np.zeros_like(d0), where=d0_err > 0),
            dz0_sintheta[keep],
            np.divide(dz[keep], z0_err, out=np.zeros_like(d0), where=z0_err > 0),
            trk["track_nPixelHits"][keep],
            trk["track_nStripHits"][keep],
            trk["track_chiSquared"][keep] / np.maximum(trk["track_numberDoF"][keep], 1.0),
        ],
        axis=1,
    ).astype(np.float32)
    dir_u = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1).astype(np.float32)

    return {
        "x": x,
        "eta": eta32[keep],
        "phi": phi32[keep],
        "energy_like": trk["track_pt"][keep].astype(np.float32),
        "dir_u": dir_u,
        "sector": ms_sector_from_phi(phi32[keep]),
        "node_type": np.full(int(keep.sum()), DV_NODE_TYPE_TRACK, dtype=np.int64),
        "min_dr": min_dr[keep],
        "has_pv": pv_z is not None,
    }


def build_edges_track_segment_by_dr(mu_eta, mu_phi, trk_eta, trk_phi, max_track_segment_dr, track_offset):
    """
    Directed muon-segment <-> track edges for pairs with deltaR < max_track_segment_dr, both directions,
    muon-major order like build_edges_segment_tower_by_dr. Track node t has the global index track_offset + t,
    i.e. the track nodes are appended after the muon and calo nodes of ``x``.
    """
    mu_eta = np.asarray(mu_eta, dtype=np.float32)
    mu_phi = np.asarray(mu_phi, dtype=np.float32)
    trk_eta = np.asarray(trk_eta, dtype=np.float32)
    trk_phi = np.asarray(trk_phi, dtype=np.float32)
    if mu_eta.size == 0 or trk_eta.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    dr = np.hypot(mu_eta[:, None] - trk_eta[None, :], _wrap_phi(mu_phi[:, None] - trk_phi[None, :]))
    mu_idx, trk_idx = np.nonzero(dr < max_track_segment_dr)
    trk_idx = trk_idx.astype(np.int64) + int(track_offset)
    src = np.empty(2 * mu_idx.size, dtype=np.int64)
    dst = np.empty(2 * mu_idx.size, dtype=np.int64)
    src[0::2], dst[0::2] = mu_idx, trk_idx
    src[1::2], dst[1::2] = trk_idx, mu_idx
    return np.stack([src, dst], axis=0)


def attach_track_nodes_to_graph(graph: dict, track_nodes, max_track_segment_dr: float) -> dict:
    """
    Add the track block to an assembled muon+calo graph without changing x/edge_index/edge_attr:
      track_x          [T, len(TRACK_FEATURE_NAMES)]
      track_edge_index [2, E_t]  global indices, track node t = x.shape[0] + t
      track_edge_attr  [E_t, 5]  same features as edge_attr (energy_like of a track = pT [MeV];
                                 same_sector compares the MS sector of the track phi with the bucket sector)
    The full graph is concat(x-block, track_x-block), concat(edge_index, track_edge_index) and
    concat(edge_attr, track_edge_attr); node types follow the order muon | calo | track.
    """
    n_mu = int(graph["n_muon_nodes"])
    n_mc = int(graph["x"].shape[0])
    n_feat = len(TRACK_FEATURE_NAMES)
    if track_nodes is None:
        graph.update(
            {
                "track_x": np.zeros((0, n_feat), dtype=np.float32),
                "track_edge_index": np.zeros((2, 0), dtype=np.int64),
                "track_edge_attr": np.zeros((0, len(DV_EDGE_FEATURE_NAMES)), dtype=np.float32),
                "track_eta": np.zeros((0,), dtype=np.float32),
                "track_phi": np.zeros((0,), dtype=np.float32),
                "track_min_dr": np.zeros((0,), dtype=np.float32),
                "n_track_nodes": 0,
            }
        )
        return graph

    track_edge_index = build_edges_track_segment_by_dr(
        graph["eta"][:n_mu], graph["phi"][:n_mu], track_nodes["eta"], track_nodes["phi"],
        max_track_segment_dr=max_track_segment_dr, track_offset=n_mc,
    )
    track_edge_attr = edge_features(
        energy_like=np.concatenate([graph["energy_like"], track_nodes["energy_like"]]).astype(np.float32),
        phi=np.concatenate([graph["phi"], track_nodes["phi"]]).astype(np.float32),
        eta=np.concatenate([graph["eta"], track_nodes["eta"]]).astype(np.float32),
        dir_u=np.concatenate([graph["dir_u"], track_nodes["dir_u"]], axis=0).astype(np.float32),
        sector=np.concatenate([graph["sector"], track_nodes["sector"]]).astype(np.int64),
        node_type=np.concatenate([graph["node_type"], track_nodes["node_type"]]).astype(np.int64),
        edge_index=track_edge_index,
    ).astype(np.float32)
    graph.update(
        {
            "track_x": track_nodes["x"].astype(np.float32),
            "track_edge_index": track_edge_index.astype(np.int64),
            "track_edge_attr": track_edge_attr,
            "track_eta": track_nodes["eta"].astype(np.float32),
            "track_phi": track_nodes["phi"].astype(np.float32),
            "track_min_dr": track_nodes["min_dr"].astype(np.float32),
            "n_track_nodes": int(track_nodes["x"].shape[0]),
        }
    )
    return graph


def _segment_pair_geometry(mu_nodes: dict, a: np.ndarray, b: np.ndarray):
    """
    r-z chord geometry of the segment pairs (a[k], b[k]), float64 (mirrored by DVInferenceToolBase):
      chord       = (r_b - r_a, z_b - z_a) between the segment positions, r = sqrt(x^2 + y^2)
      chord_angle = max(angle(chord, d_a), angle(chord, d_b)), angle = acos(|unit chord . unit d_rz|) in [0, pi/2],
                    d_rz = ((x dx + y dy) / r, dz) the segment direction in the r-z plane; 0 for chords shorter
                    than SEGMENT_CHORD_MIN_LENGTH_MM
    Returns (chord_angle [rad], chord_length [mm]).
    """
    pos = mu_nodes["seg_pos_mm"]
    d = mu_nodes["seg_dir"]
    r = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2]
    safe_r = np.where(r > 0.0, r, 1.0)
    d_r = np.where(r > 0.0, (pos[:, 0] * d[:, 0] + pos[:, 1] * d[:, 1]) / safe_r, np.hypot(d[:, 0], d[:, 1]))
    d_z = d[:, 2]
    d_norm = np.hypot(d_r, d_z)
    d_norm = np.where(d_norm > 0.0, d_norm, 1.0)
    d_r, d_z = d_r / d_norm, d_z / d_norm

    c_r = r[b] - r[a]
    c_z = z[b] - z[a]
    length = np.hypot(c_r, c_z)
    ok = length >= SEGMENT_CHORD_MIN_LENGTH_MM
    safe_len = np.where(ok, length, 1.0)
    c_r, c_z = c_r / safe_len, c_z / safe_len
    ang_a = np.arccos(np.clip(np.abs(c_r * d_r[a] + c_z * d_z[a]), 0.0, 1.0))
    ang_b = np.arccos(np.clip(np.abs(c_r * d_r[b] + c_z * d_z[b]), 0.0, 1.0))
    return np.where(ok, np.maximum(ang_a, ang_b), 0.0), length


def _muon_pairs_within_dr(eta, phi, max_dr: float):
    """Muon-node pairs (i < j, row-major order) with deltaR(i, j) < max_dr (float32, as the other edge blocks)."""
    eta = np.asarray(eta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    a, b = np.triu_indices(eta.size, 1)
    if a.size == 0:
        return a.astype(np.int64), b.astype(np.int64)
    dr = np.hypot(eta[a] - eta[b], _wrap_phi(phi[a] - phi[b]))
    keep = dr < np.float32(max_dr)
    return a[keep].astype(np.int64), b[keep].astype(np.int64)


def build_segment_segment_edges(mu_nodes: dict, max_dr: float):
    """
    Segment-segment edges: all muon-node pairs with deltaR < max_dr, for each pair (i < j, row-major) the
    edges i -> j and j -> i. Returns (seg_edge_index [2, E] with muon-node indices, seg_edge_attr [E, 7]
    with SEGMENT_EDGE_FEATURE_NAMES).
    """
    n_feat = len(SEGMENT_EDGE_FEATURE_NAMES)
    a, b = _muon_pairs_within_dr(mu_nodes["eta"], mu_nodes["phi"], max_dr)
    if a.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, n_feat), dtype=np.float32)
    src = np.empty(2 * a.size, dtype=np.int64)
    dst = np.empty(2 * a.size, dtype=np.int64)
    src[0::2], dst[0::2] = a, b
    src[1::2], dst[1::2] = b, a

    chord_angle, chord_length = _segment_pair_geometry(mu_nodes, a, b)
    eta = np.asarray(mu_nodes["eta"], dtype=np.float32)
    phi = np.asarray(mu_nodes["phi"], dtype=np.float32)
    d = mu_nodes["seg_dir"]
    layer = np.asarray(mu_nodes["seg_layer"])
    sector = np.asarray(mu_nodes["sector"])
    attr = np.empty((src.size, n_feat), dtype=np.float32)
    attr[:, 0] = eta[dst] - eta[src]
    attr[:, 1] = _wrap_phi(phi[dst] - phi[src])
    attr[:, 2] = np.clip(np.sum(d[src] * d[dst], axis=1), -1.0, 1.0)
    attr[:, 3] = np.repeat(chord_angle, 2)
    attr[:, 4] = np.repeat(chord_length / 1000.0, 2)
    attr[:, 5] = (layer[src] == layer[dst]).astype(np.float32)
    attr[:, 6] = (sector[src] == sector[dst]).astype(np.float32)
    return np.stack([src, dst], axis=0), attr


def build_tracklets(mu_nodes: dict, max_dr: float, max_chord_angle: float, min_layers: int):
    """
    MS tracklets: connected components (union-find) of the links between muon nodes in different, known
    station layers with deltaR < max_dr and chord_angle < max_chord_angle, keeping components with at least
    min_layers distinct layers, ordered by their smallest muon-node index.
    Returns a list of (member muon-node indices ascending, largest link chord_angle).
    """
    layer = np.asarray(mu_nodes["seg_layer"])
    n = layer.size
    a, b = _muon_pairs_within_dr(mu_nodes["eta"], mu_nodes["phi"], max_dr)
    cand = (layer[a] != layer[b]) & (layer[a] >= 0) & (layer[b] >= 0)
    a, b = a[cand], b[cand]
    if a.size == 0:
        return []
    chord_angle, _ = _segment_pair_geometry(mu_nodes, a, b)
    link = chord_angle < max_chord_angle
    a, b, chord_angle = a[link], b[link], chord_angle[link]

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(a.tolist(), b.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    max_angle = {}
    for i, ang in zip(a.tolist(), chord_angle.tolist()):
        r = int(roots[i])
        max_angle[r] = max(max_angle.get(r, 0.0), ang)

    tracklets = []
    for r in sorted(max_angle):  # the root is the smallest member index
        members = np.nonzero(roots == r)[0].astype(np.int64)
        if np.unique(layer[members]).size >= int(min_layers):
            tracklets.append((members, float(max_angle[r])))
    return tracklets


def build_tracklet_block(mu_nodes: dict, tracklets, n_graph_nodes: int):
    """
    Tracklet nodes (TRACKLET_FEATURE_NAMES) and edges (TRACKLET_EDGE_FEATURE_NAMES): for each tracklet k
    (in order) and member i (ascending) the edges i -> t_k and t_k -> i, then t_k1 -> t_k2 for all
    k1 != k2 (row-major). Tracklet node k has the index n_graph_nodes + k (n_graph_nodes = muon + calo nodes),
    the same convention as the track block.
    """
    n_feat = len(TRACKLET_FEATURE_NAMES)
    n_edge_feat = len(TRACKLET_EDGE_FEATURE_NAMES)
    if not tracklets:
        return (np.zeros((0, n_feat), dtype=np.float32), np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, n_edge_feat), dtype=np.float32))
    eta = np.asarray(mu_nodes["eta"], dtype=np.float32)
    phi = np.asarray(mu_nodes["phi"], dtype=np.float32)
    pos = mu_nodes["seg_pos_mm"]
    d = mu_nodes["seg_dir"]
    layer = np.asarray(mu_nodes["seg_layer"])
    chi2_ndof = mu_nodes["seg_chi2_ndof"]
    if not np.all(np.isfinite(chi2_ndof)):
        raise ValueError("The tracklet nodes need the MuonBucketDump branch segment_chiSquared.")
    dist = np.linalg.norm(pos, axis=1)
    rho = np.hypot(pos[:, 0], pos[:, 1])

    K = len(tracklets)
    tx = np.zeros((K, n_feat), dtype=np.float64)
    t_eta = np.zeros(K, dtype=np.float32)
    t_phi = np.zeros(K, dtype=np.float32)
    t_dir = np.zeros((K, 3), dtype=np.float64)
    for k, (m, max_angle) in enumerate(tracklets):
        inner = int(m[np.argmin(dist[m])])
        outer = int(m[np.argmax(dist[m])])
        chord = pos[outer] - pos[inner]
        chord_len = np.linalg.norm(chord)
        t_dir[k] = chord / chord_len if chord_len > 0.0 else np.array([0.0, 0.0, 1.0])
        c_r, c_z = rho[outer] - rho[inner], pos[outer, 2] - pos[inner, 2]
        rz_len = np.hypot(c_r, c_z)
        rz_ip = abs(rho[inner] * c_z - pos[inner, 2] * c_r) / rz_len if rz_len >= SEGMENT_CHORD_MIN_LENGTH_MM else 0.0
        e_mean = float(np.mean(eta[m].astype(np.float64)))
        p_mean = float(np.arctan2(np.sum(np.sin(phi[m].astype(np.float64))), np.sum(np.cos(phi[m].astype(np.float64)))))
        tx[k] = [e_mean, p_mean, np.arctan2(np.hypot(chord[0], chord[1]), chord[2]), m.size,
                 np.unique(layer[m]).size, dist[inner] / 1000.0, dist[outer] / 1000.0, max_angle,
                 rz_ip / 1000.0, float(np.sum(mu_nodes["seg_ndof"][m])), float(np.mean(chi2_ndof[m]))]
        t_eta[k], t_phi[k] = np.float32(e_mean), np.float32(p_mean)

    src, dst, attr = [], [], []
    for k, (m, _) in enumerate(tracklets):
        t = n_graph_nodes + k
        for i in m.tolist():
            cosang = float(np.clip(np.dot(d[i], t_dir[k]), -1.0, 1.0))
            src += [i, t]
            dst += [t, i]
            attr.append([t_eta[k] - eta[i], _wrap_phi(t_phi[k] - phi[i]), cosang, 0.0])
            attr.append([eta[i] - t_eta[k], _wrap_phi(phi[i] - t_phi[k]), cosang, 0.0])
    for k1 in range(K):
        for k2 in range(K):
            if k1 == k2:
                continue
            src.append(n_graph_nodes + k1)
            dst.append(n_graph_nodes + k2)
            attr.append([t_eta[k2] - t_eta[k1], _wrap_phi(t_phi[k2] - t_phi[k1]),
                         float(np.clip(np.dot(t_dir[k1], t_dir[k2]), -1.0, 1.0)), 1.0])
    return (tx.astype(np.float32), np.asarray([src, dst], dtype=np.int64).reshape(2, -1),
            np.asarray(attr, dtype=np.float32).reshape(-1, n_edge_feat))


def attach_segment_blocks_to_graph(graph: dict, mu_nodes: dict, segment_edge_max_dr: float = -1.0,
                                   tracklet_max_dr: float = -1.0, tracklet_max_chord_angle: float = 0.2,
                                   tracklet_min_layers: int = 2) -> dict:
    """
    Add the optional segment blocks to an assembled graph without changing x/edge_index/edge_attr:
      seg_edge_index      [2, Es]  muon-node indices          (segment_edge_max_dr > 0)
      seg_edge_attr       [Es, 7]  SEGMENT_EDGE_FEATURE_NAMES
      tracklet_x          [K, 11]  TRACKLET_FEATURE_NAMES     (tracklet_max_dr > 0)
      tracklet_edge_index [2, Ek]  muon-node indices and tracklet k = x.shape[0] + k
      tracklet_edge_attr  [Ek, 4]  TRACKLET_EDGE_FEATURE_NAMES
    """
    if segment_edge_max_dr is not None and segment_edge_max_dr > 0:
        graph["seg_edge_index"], graph["seg_edge_attr"] = build_segment_segment_edges(mu_nodes, segment_edge_max_dr)
    if tracklet_max_dr is not None and tracklet_max_dr > 0:
        tracklets = build_tracklets(mu_nodes, tracklet_max_dr, tracklet_max_chord_angle, tracklet_min_layers)
        tx, tei, tea = build_tracklet_block(mu_nodes, tracklets, int(graph["x"].shape[0]))
        graph.update({"tracklet_x": tx, "tracklet_edge_index": tei, "tracklet_edge_attr": tea,
                      "n_tracklet_nodes": int(tx.shape[0])})
    return graph


def _try_read_optional_tree(root_path: str, tree_name: str, required_branches):
    """Read an optional tree; return ``(None, {}, [])`` when absent/incompatible."""
    try:
        return _read_tree(root_path, tree_name, required_branches)
    except Exception as exc:
        print(f"[w] optional tree '{tree_name}' unavailable in {root_path}: {exc}")
        return None, defaultdict(list), []


def iter_displaced_vertex_classification_samples(
    root_path: str,
    muon_tree_name: str = "MuonBucketDump",
    calo_tree_name: str = "CaloDump",
    vertex_tree_name: str = "MuonVertexDump",
    signal_filename_patterns=None,
    signal_r_min_mm: float = 800.0,
    signal_r_max_mm: float = 8000.0,
    sector_mod: int = 16,
    min_tower_energy_mev: float = 1000.0,
    max_tower_segment_dr: float = 0.4,
    calo_r_max_mm: float = 4250.0,
    calo_z_max_mm: float = 6500.0,
    min_segments: int = 1,
    require_edges: bool = False,
    max_events: int = -1,
    track_tree_name: str = "TrigFSTrackDump",
    min_track_pt_mev: float = 2000.0,
    max_track_segment_dr: float = 0.3,
    max_track_dz0_sintheta_mm: float = -1.0,
    calo_sector_convention: str = "cpp",
    segment_chi2_feature: bool = True,
    segment_edge_max_dr: float = 0.4,
    tracklet_max_dr: float = 0.4,
    tracklet_max_chord_angle: float = 0.2,
    tracklet_min_layers: int = 2,
):
    """
    Yield one graph-classification sample per ROOT event that has usable nodes.

    Each yielded sample contains numpy arrays with keys:
      x, edge_index, edge_attr, y, labels, phi, eta, energy_like, dir_u,
      sector, node_type, and optional diagnostic arrays.
    With a track tree (track_tree_name, "" = no tracks) it also holds the track block
    (track_x, track_edge_index, track_edge_attr, n_track_nodes, ...; see attach_track_nodes_to_graph).
    The track block never changes x/edge_index/edge_attr, so graphs stay readable by muon+calo models.
    Segment blocks (on by default, see attach_segment_blocks_to_graph; False / <= 0 switches them off):
      segment_chi2_feature  muon-node feature 5 = chi2 / max(nDoF, 1) (changes x)
      segment_edge_max_dr   > 0: segment-segment edges (seg_edge_index, seg_edge_attr)
      tracklet_max_dr       > 0: MS tracklet nodes (tracklet_x, tracklet_edge_index, tracklet_edge_attr), built from
                            the links with tracklet_max_chord_angle, kept with >= tracklet_min_layers station layers
    """
    signal_filename_patterns = _normalize_signal_filename_patterns(signal_filename_patterns)
    mu_td, mu_ev_to_idx, mu_keys = _read_tree(root_path, muon_tree_name, REQUIRED_MUON_BRANCHES,
                                              optional_branches=OPTIONAL_MUON_BRANCHES)
    needs_chi2 = bool(segment_chi2_feature) or (tracklet_max_dr is not None and tracklet_max_dr > 0)
    if needs_chi2 and "segment_chiSquared" not in mu_td:
        raise ValueError(f"{root_path}: the segment-chi2 feature / tracklets need the {muon_tree_name} branch "
                         "segment_chiSquared.")
    slot5_feature = MUON_SLOT5_FEATURE_CHI2 if segment_chi2_feature else MUON_SLOT5_FEATURE_DEFAULT
    ca_td, ca_ev_to_idx, _ = _try_read_optional_tree(root_path, calo_tree_name, REQUIRED_CALO_BRANCHES)
    vx_td, vx_ev_to_idx, _ = _try_read_optional_tree(root_path, vertex_tree_name, REQUIRED_VERTEX_BRANCHES)
    tr_td, tr_ev_to_idx = None, {}
    if track_tree_name:
        tr_td, tr_ev_to_idx, _ = _try_read_optional_tree(root_path, track_tree_name, REQUIRED_TRACK_BRANCHES)

    dataset_name = _dataset_name_from_root_path(root_path)
    is_signal_file = _filename_matches_signal_patterns(root_path, signal_filename_patterns)
    if is_signal_file and vx_td is None:
        print(
            f"[w] {os.path.basename(root_path)} matches one of {signal_filename_patterns}, "
            "but no usable vertex tree was found; signal-candidate events will be skipped."
        )

    yielded = 0
    for evh in mu_keys:
        if max_events > 0 and yielded >= max_events:
            break

        mu_idxs = np.asarray(mu_ev_to_idx[evh], dtype=np.int64)
        if min_segments > 0 and not _event_has_min_segments_and_truth(
            mu_td,
            mu_idxs,
            min_segments=min_segments,
            require_truth=False,
            min_truth=0,
        ):
            continue

        vx_idxs = (
            np.asarray(vx_ev_to_idx[evh], dtype=np.int64)
            if vx_td is not None and evh in vx_ev_to_idx
            else None
        )
        label, vertex_rhos_mm = compute_displaced_vertex_graph_label(
            root_path=root_path,
            vertex_td=vx_td,
            vx_idxs=vx_idxs,
            signal_filename_patterns=signal_filename_patterns,
            signal_r_min_mm=signal_r_min_mm,
            signal_r_max_mm=signal_r_max_mm,
        )

        # For signal-pattern files, skip only this event/graph and continue processing the rest of the ROOT file.
        if is_signal_file and label != 1:
            continue

        mu_nodes = build_displaced_vertex_muon_nodes_cylindrical(mu_td, mu_idxs, slot5_feature=slot5_feature)
        if mu_nodes is None:
            continue

        calo_nodes = None
        if ca_td is not None and evh in ca_ev_to_idx:
            calo_nodes = build_displaced_vertex_calo_nodes_cylindrical(
                ca_td,
                np.asarray(ca_ev_to_idx[evh], dtype=np.int64),
                sector_mod=sector_mod,
                min_tower_energy_mev=min_tower_energy_mev,
                max_tower_segment_dr=max_tower_segment_dr,
                calo_r_max_mm=calo_r_max_mm,
                calo_z_max_mm=calo_z_max_mm,
                seg_eta_list=mu_nodes["eta"],
                seg_phi_list=mu_nodes["phi"],
                sector_convention=calo_sector_convention,
            )

        graph = assemble_displaced_vertex_graph(
            mu_nodes=mu_nodes,
            calo_nodes=calo_nodes,
            max_tower_segment_dr=max_tower_segment_dr,
            require_edges=require_edges,
        )
        if graph is None:
            continue

        if track_tree_name:
            track_nodes = None
            if tr_td is not None and evh in tr_ev_to_idx:
                track_nodes = build_displaced_vertex_track_nodes(
                    tr_td,
                    np.asarray(tr_ev_to_idx[evh], dtype=np.int64),
                    seg_eta_list=mu_nodes["eta"],
                    seg_phi_list=mu_nodes["phi"],
                    min_track_pt_mev=min_track_pt_mev,
                    max_track_segment_dr=max_track_segment_dr,
                    max_track_dz0_sintheta_mm=max_track_dz0_sintheta_mm,
                )
            attach_track_nodes_to_graph(graph, track_nodes, max_track_segment_dr=max_track_segment_dr)

        attach_segment_blocks_to_graph(
            graph, mu_nodes,
            segment_edge_max_dr=segment_edge_max_dr,
            tracklet_max_dr=tracklet_max_dr,
            tracklet_max_chord_angle=tracklet_max_chord_angle,
            tracklet_min_layers=tracklet_min_layers,
        )

        y = np.asarray([label], dtype=np.float32)
        sample = dict(graph)
        sample.update(
            {
                "y": y,
                "labels": y.copy(),
                "event_hash": np.asarray(evh, dtype=np.int64),
                "dataset_name": dataset_name,
                "root_file": os.path.basename(root_path),
                "is_signal_file": np.asarray([int(is_signal_file)], dtype=np.int8),
                "vertex_rho_mm": vertex_rhos_mm.astype(np.float32),
            }
        )
        yielded += 1
        yield sample


def _write_dv_classification_event_group(g, sample: dict) -> None:
    """Write one DisplacedVertex graph-classification sample into an HDF5 group."""
    event_hash = sample.get("event_hash")
    if event_hash is not None:
        g.attrs["event_hash"] = np.asarray(event_hash, dtype=np.int64)
    g.attrs["label"] = int(np.asarray(sample["y"]).ravel()[0])
    g.attrs["n_muon_nodes"] = int(sample.get("n_muon_nodes", 0))
    g.attrs["n_calo_nodes"] = int(sample.get("n_calo_nodes", 0))
    if "n_track_nodes" in sample:
        g.attrs["n_track_nodes"] = int(sample["n_track_nodes"])
    if "n_tracklet_nodes" in sample:
        g.attrs["n_tracklet_nodes"] = int(sample["n_tracklet_nodes"])
    if sample.get("dataset_name") is not None:
        g.attrs["dataset_name"] = str(sample["dataset_name"])
    if sample.get("root_file") is not None:
        g.attrs["root_file"] = str(sample["root_file"])

    required = ["x", "edge_index", "edge_attr", "y", "labels"]
    for key in required:
        g.create_dataset(key, data=sample[key], compression="gzip", compression_opts=4)

    for key in ("phi", "eta", "energy_like", "dir_u", "sector", "node_type", "is_signal_file", "vertex_rho_mm"):
        if key in sample and sample[key] is not None:
            g.create_dataset(key, data=sample[key], compression="gzip", compression_opts=4)

    for key in ("muon_xyz_m", "muon_bucket", "tower_xyz_m", "tower_min_dr"):
        if key in sample and sample[key] is not None:
            g.create_dataset(key, data=sample[key], compression="gzip", compression_opts=4)

    # Track block (absent when the conversion runs without a track tree)
    for key in ("track_x", "track_edge_index", "track_edge_attr", "track_eta", "track_phi", "track_min_dr"):
        if key in sample and sample[key] is not None:
            g.create_dataset(key, data=sample[key], compression="gzip", compression_opts=4)

    # Optional segment blocks (--segment-edges, --tracklets)
    for key in ("seg_edge_index", "seg_edge_attr", "tracklet_x", "tracklet_edge_index", "tracklet_edge_attr"):
        if key in sample and sample[key] is not None:
            g.create_dataset(key, data=sample[key], compression="gzip", compression_opts=4)


# ---------------------------------------------------------------------------------------------------------------------
# Packed parts (converter --output-format packed, the default; pack_DisplacedVertex_dataset.py for H5 parts): the
# training arrays of one part as uncompressed .npy files in <stem>.packed/, memory-mapped by the training
# (dv_training_utils.PackedEventDataset): an event is a few array slices instead of ~12 gzip H5 datasets.
#   <block>.npy + <block>_ptr.npy   blocks concatenated over the events (edge indices [2, E] along axis 1, int32),
#                                   event e = ptr[e]:ptr[e+1]
#   y.npy, n_muon_nodes.npy, event_key.npy, dataset_name.npy, root_file.npy   per event
#   meta.json                       format, n_events, blocks, converter_config, muon_node_feature_names, file_attrs
# Event keys and order are those of the H5 part (/events/0000000, ...): split files of the H5 parts stay valid.
# The H5 diagnostics (phi, eta, vertex_rho_mm, muon_xyz_m, track_eta, ...) are not packed: the event viewers need
# --output-format h5 or both.
# ---------------------------------------------------------------------------------------------------------------------
DV_PACKED_FORMAT = "dv_packed_v1"
DV_PACKED_SUFFIX = ".packed"
DV_PACKED_BLOCKS = {  # name -> columns of a feature block (None for the [2, E] edge indices)
    "x": len(DV_NODE_FEATURE_NAMES), "edge_index": None, "edge_attr": len(DV_EDGE_FEATURE_NAMES),
    "track_x": len(TRACK_FEATURE_NAMES), "track_edge_index": None, "track_edge_attr": len(DV_EDGE_FEATURE_NAMES),
    "seg_edge_index": None, "seg_edge_attr": len(SEGMENT_EDGE_FEATURE_NAMES),
    "tracklet_x": len(TRACKLET_FEATURE_NAMES), "tracklet_edge_index": None,
    "tracklet_edge_attr": len(TRACKLET_EDGE_FEATURE_NAMES),
}
DV_PACKED_OPTIONAL = {
    "use_tracks": ("track_x", "track_edge_index", "track_edge_attr"),
    "use_segment_edges": ("seg_edge_index", "seg_edge_attr"),
    "use_tracklets": ("tracklet_x", "tracklet_edge_index", "tracklet_edge_attr"),
}
DV_OUTPUT_FORMATS = ("packed", "h5", "both")


def is_packed_part(path) -> bool:
    return os.path.isfile(os.path.join(str(path), "meta.json"))


def data_part_stem(path) -> str:
    """Part name without .h5 / .packed: an H5 part and its packed twin compare equal."""
    name = os.path.basename(str(path).rstrip("/"))
    for suffix in (DV_PACKED_SUFFIX, ".h5"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_packed_meta(part_dir) -> dict:
    import json
    with open(os.path.join(str(part_dir), "meta.json")) as f:
        return json.load(f)


def _as_numpy(v):
    return v.numpy() if hasattr(v, "numpy") and not isinstance(v, np.ndarray) else np.asarray(v)


class PackedPartWriter:
    """Stream events into one packed part (block data appended to raw files, assembled into .npy files on close; the
    part appears atomically by a rename). Events are dicts with the blocks (numpy arrays or tensors) plus y,
    n_muon_nodes, event_key, dataset_name, root_file; the blocks of the first event fix those of the part."""

    def __init__(self, part_dir):
        import shutil
        self.part_dir = str(part_dir).rstrip("/")
        self.tmp = f"{self.part_dir}.tmp{os.getpid()}"
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.makedirs(self.tmp)
        self.blocks = None
        self._cols = {}
        self._files = {}
        self._lens = {}
        self._per_event = {"y": [], "n_muon_nodes": [], "event_key": [], "dataset_name": [], "root_file": []}

    def _raw(self, name: str) -> str:
        return os.path.join(self.tmp, name + ".raw")

    def add(self, ev: dict) -> None:
        present = [b for b in DV_PACKED_BLOCKS if ev.get(b) is not None]
        if self.blocks is None:
            self.blocks = present
            for b in self.blocks:
                self._lens[b] = []
                if DV_PACKED_BLOCKS[b] is None:
                    self._files[b] = (open(self._raw(b + ".r0"), "wb"), open(self._raw(b + ".r1"), "wb"))
                else:
                    a = _as_numpy(ev[b])
                    self._cols[b] = int(a.shape[1]) if a.ndim == 2 and a.shape[1] > 0 else DV_PACKED_BLOCKS[b]
                    self._files[b] = (open(self._raw(b), "wb"),)
        elif present != self.blocks:
            raise RuntimeError(f"event {ev.get('event_key')}: blocks {present} differ from the part's {self.blocks}")
        for b in self.blocks:
            a = _as_numpy(ev[b])
            if DV_PACKED_BLOCKS[b] is None:
                # same layout as H5EventDataset: edge_index transposed if stored [E, 2], the others reshaped
                if b == "edge_index" and a.ndim == 2 and a.shape[0] != 2:
                    a = a.T
                a = a.reshape(2, -1).astype(np.int64)
                if a.size and (a.min() < 0 or a.max() > np.iinfo(np.int32).max):
                    raise RuntimeError(f"{b} of event {ev.get('event_key')} does not fit int32")
                a = a.astype(np.int32)
                a[0].tofile(self._files[b][0])
                a[1].tofile(self._files[b][1])
                self._lens[b].append(a.shape[1])
            else:
                a = np.ascontiguousarray(a.astype(np.float32).reshape(-1, self._cols[b]))
                a.tofile(self._files[b][0])
                self._lens[b].append(a.shape[0])
        self._per_event["y"].append(float(np.asarray(_as_numpy(ev["y"])).reshape(-1)[0]))
        self._per_event["n_muon_nodes"].append(int(np.asarray(_as_numpy(ev["n_muon_nodes"])).reshape(-1)[0]))
        self._per_event["event_key"].append(str(ev["event_key"]))
        for k in ("dataset_name", "root_file"):
            v = ev.get(k)
            self._per_event[k].append("unknown" if v is None else str(v))

    def _assemble(self, b: str) -> None:
        n = int(np.sum(self._lens[b], dtype=np.int64))
        np.save(os.path.join(self.tmp, f"{b}_ptr.npy"),
                np.concatenate([[0], np.cumsum(self._lens[b], dtype=np.int64)]).astype(np.int64))
        path = os.path.join(self.tmp, f"{b}.npy")
        if DV_PACKED_BLOCKS[b] is None:
            out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(2, n))
            for r in (0, 1):
                if n:
                    out[r] = np.memmap(self._raw(f"{b}.r{r}"), dtype=np.int32, mode="r", shape=(n,))
                os.remove(self._raw(f"{b}.r{r}"))
        else:
            out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, self._cols[b]))
            if n:
                out[:] = np.memmap(self._raw(b), dtype=np.float32, mode="r", shape=(n, self._cols[b]))
            os.remove(self._raw(b))
        out.flush()
        del out

    def close(self, meta: dict | None = None) -> dict:
        import json
        import shutil
        for fhs in self._files.values():
            for fh in fhs:
                fh.close()
        for b in self.blocks or []:
            self._assemble(b)
        np.save(os.path.join(self.tmp, "y.npy"), np.asarray(self._per_event["y"], dtype=np.float32))
        np.save(os.path.join(self.tmp, "n_muon_nodes.npy"), np.asarray(self._per_event["n_muon_nodes"], dtype=np.int64))
        for k in ("event_key", "dataset_name", "root_file"):
            np.save(os.path.join(self.tmp, f"{k}.npy"), np.asarray(self._per_event[k], dtype=str))
        out_meta = {"format": DV_PACKED_FORMAT, "n_events": len(self._per_event["y"]), "blocks": list(self.blocks or [])}
        out_meta.update(meta or {})
        with open(os.path.join(self.tmp, "meta.json"), "w") as f:
            json.dump(out_meta, f, indent=1)
        shutil.rmtree(self.part_dir, ignore_errors=True)
        os.replace(self.tmp, self.part_dir)
        return out_meta

    def abort(self) -> None:
        import shutil
        for fhs in self._files.values():
            for fh in fhs:
                fh.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


def iter_packed_events(part_dir, blocks=None, max_events: int = -1):
    """Events of a packed part as numpy dicts (blocks + y, n_muon_nodes, event_key, dataset_name, root_file)."""
    meta = read_packed_meta(part_dir)
    blocks = list(meta["blocks"]) if blocks is None else [b for b in blocks if b in meta["blocks"]]
    arr = {b: np.load(os.path.join(str(part_dir), f"{b}.npy"), mmap_mode="r") for b in blocks}
    ptr = {b: np.load(os.path.join(str(part_dir), f"{b}_ptr.npy")) for b in blocks}
    per_event = {k: np.load(os.path.join(str(part_dir), f"{k}.npy"))
                 for k in ("y", "n_muon_nodes", "event_key", "dataset_name", "root_file")}
    n = int(meta["n_events"]) if max_events < 0 else min(int(meta["n_events"]), int(max_events))
    for e in range(n):
        ev = {k: v[e] for k, v in per_event.items()}
        for b in blocks:
            s, t = int(ptr[b][e]), int(ptr[b][e + 1])
            ev[b] = np.array(arr[b][:, s:t] if DV_PACKED_BLOCKS[b] is None else arr[b][s:t])
        yield ev


def dv_output_paths(output_path: str, output_format: str) -> tuple:
    """(H5 path or None, packed part or None) of a conversion output, from either kind of path."""
    if output_format not in DV_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {DV_OUTPUT_FORMATS}, got {output_format!r}")
    base = os.path.join(os.path.dirname(str(output_path)), data_part_stem(output_path))
    h5_path = (str(output_path) if str(output_path).endswith(".h5") else base + ".h5") if output_format != "packed" else None
    packed_path = base + DV_PACKED_SUFFIX if output_format != "h5" else None
    return h5_path, packed_path


def save_displaced_vertex_samples_to_hdf5(samples: list[dict], hdf5_path: str) -> None:
    """Save an in-memory list of DisplacedVertex classification samples to HDF5."""
    out_dir = os.path.dirname(hdf5_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(hdf5_path, "w") as h5:
        h5.attrs["task"] = "graph_classification"
        h5.attrs["n_events_written"] = len(samples)
        events_grp = h5.create_group("events")
        for i, sample in enumerate(samples):
            _write_dv_classification_event_group(events_grp.create_group(f"{i:07d}"), sample)


def _dv_file_attrs(root_path: str, kwargs: dict) -> tuple:
    """(file attributes, converter config, muon-node feature names) of a conversion: the self-describing schema."""
    import inspect
    import json
    config = inspect.signature(iter_displaced_vertex_classification_samples).bind(root_path, **kwargs)
    config.apply_defaults()
    config = {k: v for k, v in config.arguments.items() if k != "root_path"}
    attrs = {
        "source_root": os.path.abspath(root_path),
        "task": "graph_classification",
        "label_definition": (
            "0=background for non-signal-pattern files; "
            "1=signal for signal-pattern files if any truth vertex satisfies "
            "r_min_mm < rho < r_max_mm; signal-pattern events outside this window are skipped"
        ),
        "node_feature_names": json.dumps(DV_NODE_FEATURE_NAMES),
        "edge_feature_names": json.dumps(DV_EDGE_FEATURE_NAMES),
        "calo_sector_convention": str(config["calo_sector_convention"]),
    }
    if config["track_tree_name"]:
        attrs["track_feature_names"] = json.dumps(TRACK_FEATURE_NAMES)
        attrs["track_edge_feature_names"] = json.dumps(DV_EDGE_FEATURE_NAMES)
    muon_names = list(DV_NODE_FEATURE_NAMES)
    if config["segment_chi2_feature"]:
        muon_names[5] = MUON_SLOT5_FEATURE_CHI2
    attrs["muon_node_feature_names"] = json.dumps(muon_names)
    if config["segment_edge_max_dr"] is not None and config["segment_edge_max_dr"] > 0:
        attrs["segment_edge_feature_names"] = json.dumps(SEGMENT_EDGE_FEATURE_NAMES)
    if config["tracklet_max_dr"] is not None and config["tracklet_max_dr"] > 0:
        attrs["tracklet_feature_names"] = json.dumps(TRACKLET_FEATURE_NAMES)
        attrs["tracklet_edge_feature_names"] = json.dumps(TRACKLET_EDGE_FEATURE_NAMES)
    config_json = {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()
                   if isinstance(v, (str, int, float, bool, list, tuple, type(None)))}
    attrs["converter_config"] = json.dumps(config_json)
    return attrs, config_json, muon_names


def convert_displaced_vertex_root_file(
    root_path: str,
    output_path: str,
    overwrite: bool = False,
    output_format: str = "h5",
    **kwargs,
) -> tuple[int, int, int]:
    """
    Convert one ROOT file to one HDF5 file (output_format="h5"), one packed part ("packed": <stem>.packed beside
    output_path) or both, in one pass over the ROOT file.

    Returns ``(n_written, n_signal, n_background)``.
    """
    h5_path, packed_path = dv_output_paths(output_path, output_format)
    outputs = [p for p in (h5_path, packed_path) if p is not None]
    if not overwrite and all(os.path.exists(p) for p in outputs):
        print(f"Skipping {os.path.basename(root_path)} → already exists: {', '.join(outputs)}")
        return 0, 0, 0
    out_dir = os.path.dirname(outputs[0])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    attrs, config_json, muon_names = _dv_file_attrs(root_path, kwargs)
    tmp_path = f"{h5_path}.tmp" if h5_path else None
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

    n_written = 0
    n_signal = 0
    n_background = 0
    h5 = None
    packer = None
    try:
        if h5_path:
            h5 = h5py.File(tmp_path, "w")
            for k, v in attrs.items():
                h5.attrs[k] = v
            events_grp = h5.create_group("events")
        if packed_path:
            packer = PackedPartWriter(packed_path)

        for sample in iter_displaced_vertex_classification_samples(root_path, **kwargs):
            label = int(np.asarray(sample["y"]).ravel()[0])
            if label == 1:
                n_signal += 1
            else:
                n_background += 1
            key = f"{n_written:07d}"
            if h5 is not None:
                _write_dv_classification_event_group(events_grp.create_group(key), sample)
            if packer is not None:
                packer.add({**sample, "event_key": key})
            n_written += 1

        if h5 is not None:
            h5.attrs["n_events_written"] = int(n_written)
            h5.attrs["n_signal"] = int(n_signal)
            h5.attrs["n_background"] = int(n_background)
            h5.close()
            h5 = None
            os.replace(tmp_path, h5_path)
        if packer is not None:
            st = os.stat(root_path)
            packer.close(meta={
                "source": os.path.basename(root_path), "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns,
                "converter_config": config_json, "muon_node_feature_names": muon_names, "file_attrs": attrs,
                "n_signal": int(n_signal), "n_background": int(n_background),
            })
            packer = None
        print(
            f"Saved {n_written} graphs to {', '.join(outputs)} "
            f"(signal={n_signal}, background={n_background})"
        )
        return n_written, n_signal, n_background

    except BaseException:
        if h5 is not None:
            h5.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if packer is not None:
            packer.abort()
        raise


def add_dv_classification_converter_args(ap):
    """Add command-line options for ``DisplacedVertex.py``."""
    ap.add_argument("--dir-in", "--input-dir", dest="dir_in", required=True, help="Directory with ROOT files")
    ap.add_argument("--dir-out", "--output-dir", dest="dir_out", required=True, help="Directory for the per-file outputs (packed parts and/or H5)")
    ap.add_argument("--pattern", default="*.root", help="Input ROOT glob pattern inside dir-in")
    ap.add_argument("--overwrite", action="store_true", help="Re-create HDF5 files even when they already exist")
    ap.add_argument("--output-format", choices=DV_OUTPUT_FORMATS, default="packed",
                    help="packed (default): <stem>.packed parts, the fast training input (train with --data-glob "
                         "\"<dir-out>/*.packed\"; splitter, preproc and evaluation read them too); h5: the gzip H5 "
                         "parts with the diagnostics (phi, eta, vertex_rho_mm, ...) needed by the event viewers; "
                         "both: H5 and packed in one pass")
    ap.add_argument("--max-events-per-file", type=int, default=-1, help="Optional cap per ROOT file (-1 = all usable events)")

    ap.add_argument("--muon-tree-name", default="MuonBucketDump")
    ap.add_argument("--calo-tree-name", default="CaloDump")
    ap.add_argument("--vertex-tree-name", default="MuonVertexDump")

    ap.add_argument("--signal-filename-patterns", "--signal-filename-pattern", nargs="+", default=DEFAULT_DV_SIGNAL_FILENAME_PATTERNS,
        help="One or more fnmatch patterns treated as signal candidates.",)
    
    ap.add_argument("--signal-r-min-mm", type=float, default=800.0)
    ap.add_argument("--signal-r-max-mm", type=float, default=8000.0)

    ap.add_argument("--sector-mod", type=int, default=16, help="Number of phi bins of the legacy calo sector convention")
    ap.add_argument("--calo-sector-convention", choices=CALO_SECTOR_CONVENTIONS, default="cpp",
                    help="Sector of the calo nodes for the same_sector edge feature. cpp (default): MS sector number "
                         "as in DVInferenceToolBase (comparable with the muon bucket sector); legacy: old phi bin "
                         "(same_sector always 0 for muon-calo edges, reproduces the old datasets)")
    ap.add_argument("--min-tower-energy-mev", type=float, default=1000.0)
    ap.add_argument("--max-tower-segment-dr", type=float, default=0.4)
    ap.add_argument("--calo-r-max-mm", type=float, default=4250.0)
    ap.add_argument("--calo-z-max-mm", type=float, default=6500.0)
    ap.add_argument("--min-segments", type=int, default=1, help="Minimum muon segments required to keep an event")
    ap.add_argument("--require-edges", action="store_true", help="Skip graphs with no segment↔tower edges")

    ap.add_argument("--track-tree-name", default="TrigFSTrackDump",
                    help="Inner-tracker track tree (TrigFSTrackDump, TrigRoITrackDump or IDTrackDump; same branches). "
                         "Empty string: no track block. Files without the tree get an empty track block.")
    ap.add_argument("--min-track-pt-mev", type=float, default=2000.0,
                    help="Minimum track pT for a track node (the dumper keeps pT > 1 GeV)")
    ap.add_argument("--max-track-segment-dr", type=float, default=0.3,
                    help="Track nodes and track-segment edges for deltaR(track, segment) below this value "
                         "(the dumper cone is dR < 0.3: larger values have no effect)")
    ap.add_argument("--max-track-dz0sintheta-mm", type=float, default=-1.0,
                    help="Keep only tracks with |(z0 + vz - z_PV) sin(theta)| <= this value (< 0: no cut, default)")

    seg = ap.add_argument_group("segment blocks (all on by default; the --no-* options reproduce the older graphs)")
    seg.add_argument("--no-segment-chi2-feature", dest="segment_chi2_feature", action="store_false", default=True,
                     help="Keep muon-node feature 5 at 0 instead of the segment chi2 / max(nDoF, 1)")
    seg.add_argument("--no-segment-edges", dest="segment_edges", action="store_false", default=True,
                     help="No segment-segment edges (seg_edge_index, seg_edge_attr; deltaR < --segment-edge-max-dr)")
    seg.add_argument("--segment-edge-max-dr", type=float, default=0.4)
    seg.add_argument("--no-tracklets", dest="tracklets", action="store_false", default=True,
                     help="No MS tracklet nodes (tracklet_x, tracklet_edge_index, tracklet_edge_attr)")
    seg.add_argument("--tracklet-max-dr", type=float, default=0.4, help="deltaR of a tracklet link")
    seg.add_argument("--tracklet-max-chord-angle", type=float, default=0.2,
                     help="Maximum r-z angle between the segment directions and the chord of a tracklet link [rad]")
    seg.add_argument("--tracklet-min-layers", type=int, default=2, help="Minimum distinct station layers of a tracklet")
    return ap


def segment_block_kwargs_from_args(args) -> dict:
    """Converter keyword arguments of the optional segment blocks from add_dv_classification_converter_args options."""
    return {
        "segment_chi2_feature": bool(args.segment_chi2_feature),
        "segment_edge_max_dr": float(args.segment_edge_max_dr) if args.segment_edges else -1.0,
        "tracklet_max_dr": float(args.tracklet_max_dr) if args.tracklets else -1.0,
        "tracklet_max_chord_angle": float(args.tracklet_max_chord_angle),
        "tracklet_min_layers": int(args.tracklet_min_layers),
    }


def run_dv_directory_conversion(args) -> None:
    """Run Bucket_converter-style per-file conversion with skip-existing logic."""
    files = sorted(glob.glob(os.path.join(args.dir_in, args.pattern)))
    os.makedirs(args.dir_out, exist_ok=True)
    print(f"Found {len(files)} ROOT files in {args.dir_in}")

    total_written = 0
    total_signal = 0
    total_background = 0
    failed = 0

    for root_path in files:
        file_name = os.path.basename(root_path)
        sample_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(args.dir_out, f"{sample_name}.h5")
        outputs = [q for q in dv_output_paths(output_path, args.output_format) if q is not None]
        if all(os.path.exists(q) for q in outputs) and not args.overwrite:
            print(f"Skipping {file_name} → already exists: {', '.join(outputs)}")
            continue

        print(f"Processing {file_name}")
        try:
            n_written, n_signal, n_background = convert_displaced_vertex_root_file(
                root_path=root_path,
                output_path=output_path,
                overwrite=args.overwrite,
                output_format=args.output_format,
                muon_tree_name=args.muon_tree_name,
                calo_tree_name=args.calo_tree_name,
                vertex_tree_name=args.vertex_tree_name,
                signal_filename_patterns=args.signal_filename_patterns,
                signal_r_min_mm=args.signal_r_min_mm,
                signal_r_max_mm=args.signal_r_max_mm,
                sector_mod=args.sector_mod,
                min_tower_energy_mev=args.min_tower_energy_mev,
                max_tower_segment_dr=args.max_tower_segment_dr,
                calo_r_max_mm=args.calo_r_max_mm,
                calo_z_max_mm=args.calo_z_max_mm,
                min_segments=args.min_segments,
                require_edges=args.require_edges,
                max_events=args.max_events_per_file,
                track_tree_name=args.track_tree_name,
                min_track_pt_mev=args.min_track_pt_mev,
                max_track_segment_dr=args.max_track_segment_dr,
                max_track_dz0_sintheta_mm=args.max_track_dz0sintheta_mm,
                calo_sector_convention=args.calo_sector_convention,
                **segment_block_kwargs_from_args(args),
            )
            total_written += n_written
            total_signal += n_signal
            total_background += n_background
        except Exception as exc:
            failed += 1
            print(f"Failed processing {file_name}: {exc}")

    print(
        "All files processed. "
        f"graphs={total_written}, signal={total_signal}, background={total_background}, failed_files={failed}"
    )


def load_displaced_vertex_graphs_from_root(root_path: str, **kwargs) -> list[dict]:
    """Build all usable DisplacedVertex classification graphs from one ROOT file in memory."""
    return list(iter_displaced_vertex_classification_samples(root_path, **kwargs))


class DisplacedVertexDataset:
    """
    ROOT-backed in-memory dataset, similar in spirit to ``BucketsDataset``.

    Call ``_load_data()`` to populate ``data_list`` with one dictionary per graph.
    If PyTorch is installed, ``__getitem__`` returns tensors for model-ready arrays.
    """

    def __init__(self, root_file: str | None = None, **kwargs):
        self.root_file = root_file
        self.kwargs = kwargs
        self.data_list: list[dict] = []

    def _load_data(self) -> list[dict]:
        if not self.root_file:
            raise ValueError("No ROOT file provided.")
        self.data_list = load_displaced_vertex_graphs_from_root(self.root_file, **self.kwargs)
        return self.data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        sample = self.data_list[idx]
        try:
            torch = _lazy_import_torch()
        except ImportError:
            return sample

        return _dv_sample_to_torch(sample, torch)


def _as_hdf5_path_list(hdf5_paths) -> list[str]:
    if isinstance(hdf5_paths, (str, os.PathLike)):
        hdf5_paths = [str(hdf5_paths)]
    paths = []
    for path in hdf5_paths:
        path = str(path)
        matches = sorted(glob.glob(path))
        paths.extend(matches if matches else [path])
    return paths


def _dv_sample_to_torch(sample: dict, torch):
    """Convert a numpy sample dictionary to torch tensors for training."""
    out = {
        "x": torch.tensor(sample["x"], dtype=torch.float32),
        "features": torch.tensor(sample["x"], dtype=torch.float32),
        "edge_index": torch.tensor(sample["edge_index"], dtype=torch.long),
        "edge_attr": torch.tensor(sample["edge_attr"], dtype=torch.float32),
        "y": torch.tensor(sample["y"], dtype=torch.float32),
        "labels": torch.tensor(sample.get("labels", sample["y"]), dtype=torch.float32),
    }
    if out["edge_index"].dim() == 2 and out["edge_index"].shape[0] != 2:
        out["edge_index"] = out["edge_index"].t().contiguous()

    for key in ("phi", "eta", "energy_like", "dir_u", "sector", "node_type"):
        if key in sample:
            dtype = torch.long if key in ("sector", "node_type") else torch.float32
            out[key] = torch.tensor(sample[key], dtype=dtype)
    return out


class H5DisplacedVertexGraphDataset:
    """Lazy-loading PyTorch-style Dataset backed by one or more DV HDF5 files."""

    def __init__(self, hdf5_paths):
        self.hdf5_paths = _as_hdf5_path_list(hdf5_paths)
        self.sample_map: list[tuple[int, str]] = []
        self._files = [None] * len(self.hdf5_paths)

        for file_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, "r") as f:
                group = f["events"] if "events" in f else f
                keys = sorted(k for k in group.keys())
                self.sample_map.extend((file_idx, key) for key in keys)

        self.length = len(self.sample_map)

    def _get_file(self, file_idx: int):
        if self._files[file_idx] is None:
            self._files[file_idx] = h5py.File(self.hdf5_paths[file_idx], "r")
        return self._files[file_idx]

    def __getitem__(self, idx: int) -> dict:
        torch = _lazy_import_torch()
        file_idx, sample_key = self.sample_map[idx]
        f = self._get_file(file_idx)
        group = f["events"][sample_key] if "events" in f else f[sample_key]

        sample = {key: group[key][()] for key in group.keys() if isinstance(group[key], h5py.Dataset)}
        if "y" not in sample and "labels" in sample:
            sample["y"] = sample["labels"]
        if "labels" not in sample and "y" in sample:
            sample["labels"] = sample["y"]
        return _dv_sample_to_torch(sample, torch)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return self.length

    def __del__(self):
        for f in getattr(self, "_files", []):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass


def load_displaced_vertex_graphs_from_hdf5(hdf5_paths, max_events: int | None = None) -> list[dict]:
    """Load DV HDF5 graph samples into memory as numpy arrays."""
    data_list = []
    for path in _as_hdf5_path_list(hdf5_paths):
        with h5py.File(path, "r") as f:
            group = f["events"] if "events" in f else f
            for key in sorted(group.keys()):
                g = group[key]
                sample = {name: g[name][()] for name in g.keys() if isinstance(g[name], h5py.Dataset)}
                if "y" not in sample and "labels" in sample:
                    sample["y"] = sample["labels"]
                if "labels" not in sample and "y" in sample:
                    sample["labels"] = sample["y"]
                data_list.append(sample)
                if max_events is not None and len(data_list) >= max_events:
                    return data_list
    return data_list


def displaced_vertex_collate_fn(batch: list) -> dict:
    """Collate variable-size DV graphs into a PyTorch mini-batch dictionary."""
    torch = _lazy_import_torch()

    x_list = [item["x"] for item in batch]
    edge_index_list = [item["edge_index"] for item in batch]
    edge_attr_list = [item["edge_attr"] for item in batch]
    y_list = [item["y"].view(-1) for item in batch]

    node_counts = [x.size(0) for x in x_list]
    node_offsets = torch.tensor([0] + node_counts[:-1], dtype=torch.long).cumsum(dim=0)

    shifted_edges = []
    shifted_attrs = []
    for i, edge_index in enumerate(edge_index_list):
        if edge_index.numel() == 0:
            continue
        shifted_edges.append(edge_index + node_offsets[i])
        shifted_attrs.append(edge_attr_list[i])

    if shifted_edges:
        edge_index = torch.cat(shifted_edges, dim=1)
        edge_attr = torch.cat(shifted_attrs, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_dim = edge_attr_list[0].shape[1] if edge_attr_list and edge_attr_list[0].dim() == 2 else 5
        edge_attr = torch.zeros((0, edge_dim), dtype=torch.float32)

    batch_vec = torch.cat(
        [torch.full((count,), i, dtype=torch.long) for i, count in enumerate(node_counts)],
        dim=0,
    )

    return {
        "x": torch.cat(x_list, dim=0),
        "features": torch.cat(x_list, dim=0),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": torch.cat(y_list, dim=0),
        "labels": torch.cat(y_list, dim=0),
        "batch": batch_vec,
    }


def validate_displaced_vertex_hdf5_files(paths) -> None:
    """Print a warning for DV HDF5 groups missing required classifier datasets."""
    required = ("x", "edge_index", "edge_attr", "y")
    for path in _as_hdf5_path_list(paths):
        print(f"Checking {path}")
        with h5py.File(path, "r") as f:
            group = f["events"] if "events" in f else f
            for key in sorted(group.keys()):
                g = group[key]
                missing = [name for name in required if name not in g]
                if missing:
                    print(f"  ❌ Missing {missing} in {key} of {path}")


def get_num_workers() -> int:
    """Return a sensible number of DataLoader worker processes."""
    import multiprocessing

    num_cpus = multiprocessing.cpu_count()
    print(f"Detected {num_cpus} CPU cores.")
    return num_cpus

# -------------------------------------------------------
# Classifier-chain HDF5 convenience helpers
# -------------------------------------------------------

def read_displaced_vertex_classifier_labels(hdf5_paths):
    """
    Return labels and lightweight metadata from one or more classifier HDF5 files.

    This helper is intentionally schema-tolerant: labels can be stored as a dataset
    named ``y`` or ``labels``, or as an event-group attribute named ``label``.
    """
    paths = _as_hdf5_path_list(hdf5_paths)
    labels = []
    dataset_names = []
    root_files = []
    event_refs = []
    for file_idx, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            group = f["events"] if "events" in f else f
            for key in sorted(group.keys()):
                g = group[key]
                if "y" in g:
                    y = g["y"][...]
                elif "labels" in g:
                    y = g["labels"][...]
                elif "label" in g.attrs:
                    y = np.asarray([g.attrs["label"]], dtype=np.float32)
                else:
                    raise RuntimeError(f"Missing classifier label in {path} /events/{key}")
                y = np.asarray(y, dtype=np.float32).reshape(-1)
                if y.size != 1 or y[0] not in (0.0, 1.0):
                    raise RuntimeError(f"Expected scalar label 0/1 in {path} /events/{key}, got {y}")
                labels.append(float(y[0]))
                dataset_names.append(_decode_h5_string(g.attrs.get("dataset_name", "unknown")))
                root_files.append(_decode_h5_string(g.attrs.get("root_file", os.path.basename(path))))
                event_refs.append((file_idx, key))
    return {
        "labels": np.asarray(labels, dtype=np.float32),
        "dataset_names": np.asarray(dataset_names, dtype=object),
        "root_files": np.asarray(root_files, dtype=object),
        "event_refs": np.asarray(event_refs, dtype=object),
        "h5_paths": np.asarray([str(Path(p).resolve()) for p in paths], dtype=object),
    }


def _decode_h5_string(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    return str(v)


def summarize_displaced_vertex_classifier_h5_files(hdf5_paths) -> dict:
    """Return event and label counts for classifier HDF5 files."""
    meta = read_displaced_vertex_classifier_labels(hdf5_paths)
    labels = meta["labels"]
    n_signal = int(np.count_nonzero(labels == 1.0))
    n_background = int(np.count_nonzero(labels == 0.0))
    by_dataset = {}
    for name, label in zip(meta["dataset_names"], labels):
        row = by_dataset.setdefault(str(name), {"n_events": 0, "n_signal": 0, "n_background": 0})
        row["n_events"] += 1
        if int(label) == 1:
            row["n_signal"] += 1
        else:
            row["n_background"] += 1
    return {
        "n_events": int(labels.size),
        "n_signal": n_signal,
        "n_background": n_background,
        "positive_fraction": float(n_signal / max(labels.size, 1)),
        "pos_weight_auto": float(n_background / max(n_signal, 1)) if n_signal > 0 else None,
        "by_dataset": by_dataset,
    }
