#!/usr/bin/env python3

"""
DisplacedVertex_converter.py

Adapted to match the logic used in the event viewer:

- event selection:
    keep only events where ALL truth vertices are inside the envelope
      r <= --vertex-r-max-mm
      |z| <= --vertex-z-max-mm

- segment-level event requirement:
    keep only events with at least 2 muon segments
    and, if --isMC, at least 2 true bucket_hasTruth flags

- muon nodes:
    all segments from all matching MuonBucketDump entries

- calo nodes:
    all towers from all matching CaloDump entries, but only if:
      tower_energy_mev >= --min-tower-energy-mev
      min ΔR(tower, any segment_position_projection) < --max-tower-segment-dr
      and the tower ray intersects the calorimeter envelope

- edges:
    directed segment<->tower edges for pairs with ΔR < --max-tower-segment-dr

Target:
  y_vertex = first valid truth vertex [x, y, z] in meters

Node features:
  [ pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy_like, nCells_or_DoF ]

where:
  - for muon nodes:
      pos_*          = segment position [m]
      dir_*          = normalized segment direction
      energy_like    = 0
      nCells_or_DoF  = segment_numberDoF

  - for calo nodes:
      pos_*          = tower intersection point with calorimeter envelope [m]
      dir_*          = normalized tower direction
      energy_like    = tower_energy_mev
      nCells_or_DoF  = tower_nCells
      
Edge features:
  [ d_energy_like, d_phi_wrapped, d_eta, cos_angle, same_sector ]
  
Example:
python -u DisplacedVertex_converter.py \
  --input-dir hdd_data/ \
  --pattern "MuonBucketDump_H*/outputs/MuonBucketDump_group.det-muon.*root" \
  --output-dir ./data \
  --output-name displaced_vertex_dataset \
  --vertex-r-max-mm 8000.0 \
  --vertex-z-max-mm 12000.0 \
  --calo-r-max-mm 4250 \
  --calo-z-max-mm 6500 \
  --min-tower-energy-mev 1000 \
  --max-tower-segment-dr 0.4
"""

import argparse
import os
import re
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import uproot
import h5py


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


# -------------------------------------------------
# Generic ROOT helpers
# -------------------------------------------------

def _normalize_keys(keys):
    return {re.sub(r"\[.*\]/[A-Za-z]$", "", k): k for k in keys}


def _open_tree_by_name(root_file: str, tree_name: str):
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


# -------------------------------------------------
# Viewer-matching geometry helpers
# -------------------------------------------------

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

    Returns (x, y, z) in mm for the first positive intersection.
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


# -------------------------------------------------
# Event selection
# -------------------------------------------------

def _event_passes_vertex_envelope(vertex_td, idxs, vertex_r_max_mm, vertex_z_max_mm):
    """
    Keep event only if ALL truth vertices are inside the envelope,
    matching the event viewer logic.
    """
    all_x = []
    all_y = []
    all_z = []

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
    inside = (r <= vertex_r_max_mm) & (np.abs(z) <= vertex_z_max_mm)
    return bool(np.all(inside))


def _build_vertex_target(vertex_td, idxs):
    """
    Preserve original dataset contract: one target per event.
    Use the first valid truth vertex, in meters.
    """
    for i in idxs:
        xs = np.asarray(vertex_td["truthMuonVertexPositionX"][i]).ravel()
        ys = np.asarray(vertex_td["truthMuonVertexPositionY"][i]).ravel()
        zs = np.asarray(vertex_td["truthMuonVertexPositionZ"][i]).ravel()

        n = min(len(xs), len(ys), len(zs))
        if n == 0:
            continue

        for j in range(n):
            x = xs[j]
            y = ys[j]
            z = zs[j]
            if x is None or y is None or z is None:
                continue
            return (np.asarray([x, y, z], dtype=np.float32) / 1000.0).astype(np.float32)

    return None


def _event_has_min_segments_and_truth(muon_td, idxs, min_segments=2, require_truth=False, min_truth=2):
    """
    Require at least `min_segments` muon segments in the event.

    If `require_truth` is True, also require at least `min_truth`
    true bucket_hasTruth values across the matching MuonBucketDump entries.
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


# -------------------------------------------------
# Node builders (viewer-matching)
# -------------------------------------------------
def _build_muon_nodes(td, idxs):
    """
    Expand ALL segments from matching MuonBucketDump entries.

    Node features:
      [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy_like, nCells_or_DoF]

    for muon nodes:
      - pos_* in meters from segment position
      - dir_* normalized segment direction
      - energy_like = 0
      - nCells_or_DoF = segment_numberDoF
    """
    x_mm, y_mm, z_mm = [], [], []
    dx, dy, dz = [], [], []
    segment_dof = []

    bucket_seg = []
    bucket_sector = []
    bucket_chamber = []
    bucket_layers = []

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
            x_mm.append(float(segx[j]))
            y_mm.append(float(segy[j]))
            z_mm.append(float(segz[j]))

            dx.append(float(dirx[j]))
            dy.append(float(diry[j]))
            dz.append(float(dirz[j]))

            segment_dof.append(float(dof[j]))

            bucket_seg.append(float(seg_count))
            bucket_sector.append(sec)
            bucket_chamber.append(chamber)
            bucket_layers.append(layer)

    if len(x_mm) == 0:
        return None

    x_mm = np.asarray(x_mm, dtype=np.float32)
    y_mm = np.asarray(y_mm, dtype=np.float32)
    z_mm = np.asarray(z_mm, dtype=np.float32)

    pos_m = np.stack([x_mm, y_mm, z_mm], axis=1).astype(np.float32) / 1000.0
    dir_u = _safe_normalize(np.stack([dx, dy, dz], axis=1).astype(np.float32))

    phi = np.arctan2(y_mm, x_mm).astype(np.float32)
    r_xy = np.hypot(x_mm, y_mm)
    eta = np.empty_like(r_xy, dtype=np.float32)
    mask = r_xy > 0
    eta[mask] = np.arcsinh(z_mm[mask] / r_xy[mask])
    eta[~mask] = np.sign(z_mm[~mask]) * 1.0e6

    energy_like = np.zeros(len(x_mm), dtype=np.float32)
    ncells_or_dof = np.asarray(segment_dof, dtype=np.float32)
    node_type = np.zeros(len(x_mm), dtype=np.int64)

    x = np.concatenate(
        [
            pos_m,
            dir_u,
            energy_like[:, None],
            ncells_or_dof[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "x": x,
        "phi": phi,
        "eta": eta,
        "energy_like": energy_like,
        "dir_u": dir_u.astype(np.float32),
        "sector": np.asarray(bucket_sector, dtype=np.int64),
        "node_type": node_type.astype(np.int64),  # internal only
        "muon_xyz_m": pos_m.astype(np.float32),
        "muon_bucket": np.stack(
            [
                np.asarray(bucket_chamber, dtype=np.int64),
                np.asarray(bucket_layers, dtype=np.int64),
                np.asarray(bucket_sector, dtype=np.int64),
                np.asarray(bucket_seg, dtype=np.int64),
            ],
            axis=1,
        ),
    }


def _build_calo_nodes(
    td,
    idxs,
    sector_mod,
    min_tower_energy_mev,
    max_tower_segment_dr,
    calo_r_max_mm,
    calo_z_max_mm,
    seg_eta_list,
    seg_phi_list,
):
    """
    Expand ALL towers and filter exactly like the event viewer:
      - E >= threshold
      - min ΔR to any segment position projection < threshold
      - must intersect calorimeter envelope

    Node features:
      [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy_like, nCells_or_DoF]
    """
    tower_energy = []
    tower_eta = []
    tower_phi = []
    tower_ncells = []
    dx, dy, dz = [], [], []
    tower_xyz_m = []
    tower_min_dr = []

    if len(seg_eta_list) == 0:
        return None

    seg_eta_arr = np.asarray(seg_eta_list, dtype=np.float32)
    seg_phi_arr = np.asarray(seg_phi_list, dtype=np.float32)

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
                eta=eta,
                phi=phi,
                r_max=calo_r_max_mm,
                z_max=calo_z_max_mm,
            )
            if pos is None:
                continue

            x_mm, y_mm, z_mm = pos

            tower_energy.append(e)
            tower_eta.append(eta)
            tower_phi.append(phi)
            tower_ncells.append(float(tnc[j]))
            dx.append(float(tdx[j]))
            dy.append(float(tdy[j]))
            dz.append(float(tdz[j]))
            tower_xyz_m.append([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0])
            tower_min_dr.append(dr_min)

    if len(tower_energy) == 0:
        return None

    tower_energy = np.asarray(tower_energy, dtype=np.float32)
    tower_eta = np.asarray(tower_eta, dtype=np.float32)
    tower_phi = np.asarray(tower_phi, dtype=np.float32)
    tower_ncells = np.asarray(tower_ncells, dtype=np.float32)
    tower_xyz_m = np.asarray(tower_xyz_m, dtype=np.float32)

    dir_u = _safe_normalize(np.stack([dx, dy, dz], axis=1).astype(np.float32))
    sector = _phi_to_sector(tower_phi, sector_mod=sector_mod)
    node_type = np.ones_like(tower_energy, dtype=np.int64)

    x = np.concatenate(
        [
            tower_xyz_m,
            dir_u,
            tower_energy[:, None],
            tower_ncells[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "x": x,
        "phi": tower_phi.astype(np.float32),
        "eta": tower_eta.astype(np.float32),
        "energy_like": tower_energy.astype(np.float32),
        "dir_u": dir_u.astype(np.float32),
        "sector": sector.astype(np.int64),
        "node_type": node_type.astype(np.int64),  # internal only
        "tower_energy_mev": tower_energy.astype(np.float32),
        "tower_nCells": tower_ncells.astype(np.float32),
        "tower_xyz_m": tower_xyz_m.astype(np.float32),
        "tower_min_dr": np.asarray(tower_min_dr, dtype=np.float32),
    }
    
# -------------------------------------------------
# Edge building (viewer-matching)
# -------------------------------------------------

def build_edges_segment_tower_by_dr(phi, eta, node_type, max_tower_segment_dr):
    """
    Directed edges only between:
      segment -> tower
      tower   -> segment

    where ΔR(segment_position_projection, tower_eta_phi) < threshold.

    This matches the same tower/segment association logic used in the viewer.
    """
    mu_idx = np.where(node_type == 0)[0]
    ca_idx = np.where(node_type == 1)[0]

    if len(mu_idx) == 0 or len(ca_idx) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    src_list = []
    dst_list = []

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
        [
            np.asarray(src_list, dtype=np.int64),
            np.asarray(dst_list, dtype=np.int64),
        ],
        axis=0,
    )


def edge_features(energy_like, phi, eta, dir_u, sector, node_type, edge_index):
    if edge_index.shape[1] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    src = edge_index[0]
    dst = edge_index[1]

    d_energy_like = (energy_like[dst] - energy_like[src]).reshape(-1, 1).astype(np.float32)
    d_phi = _wrap_phi(phi[dst] - phi[src]).reshape(-1, 1).astype(np.float32)
    d_eta = (eta[dst] - eta[src]).reshape(-1, 1).astype(np.float32)
    cosang = np.sum(dir_u[src] * dir_u[dst], axis=1, keepdims=True).astype(np.float32)
    same_sector = (sector[src] == sector[dst]).astype(np.float32).reshape(-1, 1)

    return np.concatenate(
        [d_energy_like, d_phi, d_eta, cosang, same_sector], axis=1,).astype(np.float32)


# -------------------------------------------------
# H5 writing helpers
# -------------------------------------------------

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
):
    g.attrs["event_hash"] = np.asarray(event_hash, dtype=np.int64)
    g.attrs["n_muon_nodes"] = int(n_muon_nodes)
    g.attrs["n_calo_nodes"] = int(n_calo_nodes)

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


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory with ROOT files")
    ap.add_argument("--pattern", default="*.root", help="Glob pattern (default: *.root)")
    ap.add_argument("--output-dir", required=True, help="Directory for output H5 files")
    ap.add_argument("--output-name", required=True, help="Base name for output files (without _partXXXX.h5)")
    ap.add_argument("--max-events", type=int, default=-1, help="Global cap across all ROOT files (-1 = all)")
    ap.add_argument("--events-per-part", type=int, default=10000, help="Max graphs per output H5 part")
    ap.add_argument("--sector-mod", type=int, default=16, help="Number of sectors for calo phi->sector mapping")

    # Viewer-matching cuts
    ap.add_argument("--min-tower-energy-mev", type=float, default=1000.0)
    ap.add_argument("--max-tower-segment-dr", type=float, default=0.4)
    ap.add_argument("--calo-r-max-mm", type=float, default=4250.0)
    ap.add_argument("--calo-z-max-mm", type=float, default=6500.0)

    # Vertex envelope used for event filtering
    ap.add_argument("--vertex-r-max-mm", type=float, required=True,
                    help="Keep event only if all truth vertices satisfy r <= this")
    ap.add_argument("--vertex-z-max-mm", type=float, required=True,
                    help="Keep event only if all truth vertices satisfy |z| <= this")
    ap.add_argument("--isMC", dest="isMC", action="store_true", default=True, 
                    help="If set, require at least 2 true bucket_hasTruth values per event (default: True)",)
    ap.add_argument("--isData", dest="isMC", action="store_false", help="Disable MC-specific bucket_hasTruth requirement",)

    ap.add_argument("--allow-single-modality", action="store_true",
        help="If set, keep events even if only muon or only calo survives. "
             "Note: with viewer-like tower matching, calo needs segments to survive.",
    )

    args = ap.parse_args()

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
                vx_td,
                vx_idxs,
                vertex_r_max_mm=args.vertex_r_max_mm,
                vertex_z_max_mm=args.vertex_z_max_mm,
            ):
                skipped += 1
                continue

            if evh not in mu_ev_to_idx:
                skipped += 1
                continue

            mu_idxs = np.asarray(mu_ev_to_idx[evh], dtype=np.int64)
            if not _event_has_min_segments_and_truth(mu_td, mu_idxs, min_segments=2, require_truth=args.isMC, min_truth=2,):
                skipped += 1
                continue

            y_vertex = _build_vertex_target(vx_td, vx_idxs)
            if y_vertex is None:
                skipped += 1
                continue

            mu_nodes = None
            if evh in mu_ev_to_idx:
                mu_nodes = _build_muon_nodes(mu_td, mu_idxs)

            ca_nodes = None
            if evh in ca_ev_to_idx and evh in mu_ev_to_idx and mu_nodes is not None:
                ca_nodes = _build_calo_nodes(
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

            pieces = []
            phi_pieces = []
            eta_pieces = []
            energy_like_pieces = []
            dir_pieces = []
            sector_pieces = []
            type_pieces = []

            n_muon_nodes = 0
            n_calo_nodes = 0
            muon_xyz_m = None
            muon_bucket = None
            tower_xyz_m = None
            tower_min_dr = None

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

            n_nodes = x.shape[0]
            if n_nodes < 2:
                skipped += 1
                continue

            edge_index = build_edges_segment_tower_by_dr(
                phi=phi,
                eta=eta,
                node_type=node_type,
                max_tower_segment_dr=args.max_tower_segment_dr,
            )

            if edge_index.shape[1] == 0:
                skipped += 1
                continue

            edge_attr = edge_features(
                energy_like=energy_like,
                phi=phi,
                eta=eta,
                dir_u=dir_u,
                sector=sector,
                node_type=node_type,
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
            )

            total_written += 1
            written_in_part += 1
            h5.attrs["n_events_written"] = int(h5.attrs["n_events_written"]) + 1

    h5.attrs["skipped_empty_or_too_small"] = skipped
    h5.close()
    print(f"[done] wrote {total_written} graphs across all files")


if __name__ == "__main__":
    main()