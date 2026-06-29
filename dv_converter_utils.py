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
try:
    import uproot
except ImportError:  # ROOT reading needs uproot, but HDF5 dataset helpers can still be imported.
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


# -------------------------------------------------------
# Generic ROOT helpers
# -------------------------------------------------------

def _normalize_keys(keys):
    return {re.sub(r"\[.*\]/[A-Za-z]$", "", k): k for k in keys}


def _open_tree_by_name(root_file: str, tree_name: str):
    if uproot is None:
        raise ImportError("uproot is required to read ROOT files. Install it in the conversion environment.")
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
    phi = np.asarray(phi, dtype=np.float32)
    phi01 = (phi + np.pi) / (2.0 * np.pi)
    sec = np.floor(phi01 * sector_mod).astype(np.int64)
    sec = np.clip(sec, 0, sector_mod - 1)
    return sec


def _read_tree(root_file: str, tree_name: str, required_branches):
    tree = _open_tree_by_name(root_file, tree_name)
    clean = _normalize_keys(list(tree.keys()))
    missing = [k for k in required_branches if k not in clean]
    if missing:
        raise ValueError(f"Tree '{tree_name}' missing branches: {missing}")

    arrays = tree.arrays([clean[k] for k in required_branches], library="np")
    td = {k: arrays[clean[k]] for k in required_branches}

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
    bucket_seg_list, bucket_sector_list, bucket_chamber_list, bucket_layers_list = [], [], [], []

    for i in idxs:
        segx = np.asarray(td["segmentPositionX"][i]).ravel()
        segy = np.asarray(td["segmentPositionY"][i]).ravel()
        segz = np.asarray(td["segmentPositionZ"][i]).ravel()
        dirx = np.asarray(td["segmentDirectionX"][i]).ravel()
        diry = np.asarray(td["segmentDirectionY"][i]).ravel()
        dirz = np.asarray(td["segmentDirectionZ"][i]).ravel()
        dof = np.asarray(td["segment_numberDoF"][i]).ravel()

        nseg = min(len(segx), len(segy), len(segz), len(dirx), len(diry), len(dirz), len(dof))
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
):
    """
    Filter calorimeter towers (energy cut, ΔR cut, envelope intersection).

    Returns a dict with filtered raw tower data, or None if no towers pass:
        tower_energy, tower_eta, tower_phi, tower_ncells
        dx, dy, dz  : direction components
        tower_xyz_m : intersection point with calorimeter envelope [m]
        tower_min_dr : minimum ΔR to any muon segment
        sector       : phi sector index
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
        "sector": _phi_to_sector(tower_phi, sector_mod=sector_mod).astype(np.int64),
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


def build_displaced_vertex_muon_nodes_cylindrical(muon_td, idxs):
    """
    Build muon-segment nodes using the cylindrical converter feature convention.

    Node features:
        [r, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]

    For muon segments, ``energy_like`` is zero and ``nCells_or_DoF`` is the
    segment number of degrees of freedom.
    """
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

    x = np.stack(
        [r_pos, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, ncells_or_dof],
        axis=1,
    ).astype(np.float32)

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
):
    """
    Build calorimeter tower nodes using the cylindrical converter convention.

    Towers are filtered by energy, ΔR to muon segments, and the calorimeter
    envelope intersection, matching the previous cylindrical converter logic.
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
):
    """
    Yield one graph-classification sample per ROOT event that has usable nodes.

    Each yielded sample contains numpy arrays with keys:
      x, edge_index, edge_attr, y, labels, phi, eta, energy_like, dir_u,
      sector, node_type, and optional diagnostic arrays.
    """
    signal_filename_patterns = _normalize_signal_filename_patterns(signal_filename_patterns)
    mu_td, mu_ev_to_idx, mu_keys = _read_tree(root_path, muon_tree_name, REQUIRED_MUON_BRANCHES)
    ca_td, ca_ev_to_idx, _ = _try_read_optional_tree(root_path, calo_tree_name, REQUIRED_CALO_BRANCHES)
    vx_td, vx_ev_to_idx, _ = _try_read_optional_tree(root_path, vertex_tree_name, REQUIRED_VERTEX_BRANCHES)

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

        mu_nodes = build_displaced_vertex_muon_nodes_cylindrical(mu_td, mu_idxs)
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
            )

        graph = assemble_displaced_vertex_graph(
            mu_nodes=mu_nodes,
            calo_nodes=calo_nodes,
            max_tower_segment_dr=max_tower_segment_dr,
            require_edges=require_edges,
        )
        if graph is None:
            continue

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


def convert_displaced_vertex_root_file(
    root_path: str,
    output_path: str,
    overwrite: bool = False,
    **kwargs,
) -> tuple[int, int, int]:
    """
    Convert one ROOT file to one HDF5 file.

    Returns ``(n_written, n_signal, n_background)``.
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"Skipping {os.path.basename(root_path)} → already exists: {output_path}")
        return 0, 0, 0

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = f"{output_path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    n_written = 0
    n_signal = 0
    n_background = 0

    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["source_root"] = os.path.abspath(root_path)
            h5.attrs["task"] = "graph_classification"
            h5.attrs["label_definition"] = (
                "0=background for non-signal-pattern files; "
                "1=signal for signal-pattern files if any truth vertex satisfies "
                "r_min_mm < rho < r_max_mm; signal-pattern events outside this window are skipped"
            )
            events_grp = h5.create_group("events")

            for sample in iter_displaced_vertex_classification_samples(root_path, **kwargs):
                label = int(np.asarray(sample["y"]).ravel()[0])
                if label == 1:
                    n_signal += 1
                else:
                    n_background += 1

                g = events_grp.create_group(f"{n_written:07d}")
                _write_dv_classification_event_group(g, sample)
                n_written += 1

            h5.attrs["n_events_written"] = int(n_written)
            h5.attrs["n_signal"] = int(n_signal)
            h5.attrs["n_background"] = int(n_background)

        os.replace(tmp_path, output_path)
        print(
            f"Saved {n_written} graphs to {output_path} "
            f"(signal={n_signal}, background={n_background})"
        )
        return n_written, n_signal, n_background

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def add_dv_classification_converter_args(ap):
    """Add command-line options for ``DisplacedVertex.py``."""
    ap.add_argument("--dir-in", "--input-dir", dest="dir_in", required=True, help="Directory with ROOT files")
    ap.add_argument("--dir-out", "--output-dir", dest="dir_out", required=True, help="Directory for per-file HDF5 outputs")
    ap.add_argument("--pattern", default="*.root", help="Input ROOT glob pattern inside dir-in")
    ap.add_argument("--overwrite", action="store_true", help="Re-create HDF5 files even when they already exist")
    ap.add_argument("--max-events-per-file", type=int, default=-1, help="Optional cap per ROOT file (-1 = all usable events)")

    ap.add_argument("--muon-tree-name", default="MuonBucketDump")
    ap.add_argument("--calo-tree-name", default="CaloDump")
    ap.add_argument("--vertex-tree-name", default="MuonVertexDump")

    ap.add_argument("--signal-filename-patterns", "--signal-filename-pattern", nargs="+", default=DEFAULT_DV_SIGNAL_FILENAME_PATTERNS,
        help="One or more fnmatch patterns treated as signal candidates.",)
    
    ap.add_argument("--signal-r-min-mm", type=float, default=800.0)
    ap.add_argument("--signal-r-max-mm", type=float, default=8000.0)

    ap.add_argument("--sector-mod", type=int, default=16)
    ap.add_argument("--min-tower-energy-mev", type=float, default=1000.0)
    ap.add_argument("--max-tower-segment-dr", type=float, default=0.4)
    ap.add_argument("--calo-r-max-mm", type=float, default=4250.0)
    ap.add_argument("--calo-z-max-mm", type=float, default=6500.0)
    ap.add_argument("--min-segments", type=int, default=1, help="Minimum muon segments required to keep an event")
    ap.add_argument("--require-edges", action="store_true", help="Skip graphs with no segment↔tower edges")
    return ap


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

        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {file_name} → already exists: {output_path}")
            continue

        print(f"Processing {file_name}")
        try:
            n_written, n_signal, n_background = convert_displaced_vertex_root_file(
                root_path=root_path,
                output_path=output_path,
                overwrite=args.overwrite,
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
