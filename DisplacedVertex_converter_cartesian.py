#!/usr/bin/env python3
"""
DisplacedVertex_converter_cartesian.py

Node features: [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy_like, nDoF]
Target: y_vertex = [x, y, z]  (mm -> m)
"""

import argparse
import numpy as np

from dv_converter_utils import (
    _collect_muon_raw,
    _collect_calo_filtered,
    _safe_normalize,
    add_converter_args,
    run_converter_main_loop,
)


def _build_vertex_target(vertex_td, idxs):
    """Return [x, y, z] in metres (first valid truth vertex)."""
    for i in idxs:
        xs = np.asarray(vertex_td["truthMuonVertexPositionX"][i]).ravel()
        ys = np.asarray(vertex_td["truthMuonVertexPositionY"][i]).ravel()
        zs = np.asarray(vertex_td["truthMuonVertexPositionZ"][i]).ravel()
        n = min(len(xs), len(ys), len(zs))
        if n == 0:
            continue
        for j in range(n):
            x, y, z = xs[j], ys[j], zs[j]
            if x is None or y is None or z is None:
                continue
            return (np.asarray([x, y, z], dtype=np.float32) / 1000.0).astype(np.float32)
    return None


def _build_muon_nodes(td, idxs):
    """Muon node features: [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, 0, dof]."""
    raw = _collect_muon_raw(td, idxs)
    if raw is None:
        return None

    x_mm, y_mm, z_mm = raw["x_mm"], raw["y_mm"], raw["z_mm"]
    pos_m = np.stack([x_mm, y_mm, z_mm], axis=1).astype(np.float32) / 1000.0
    dir_u = _safe_normalize(
        np.stack([raw["dx"], raw["dy"], raw["dz"]], axis=1).astype(np.float32)
    )

    r_xy = np.hypot(x_mm, y_mm)
    phi = np.arctan2(y_mm, x_mm).astype(np.float32)
    eta = np.empty_like(r_xy, dtype=np.float32)
    mask = r_xy > 0
    eta[mask] = np.arcsinh(z_mm[mask] / r_xy[mask])
    eta[~mask] = np.sign(z_mm[~mask]) * 1.0e6

    energy_like = np.zeros(len(x_mm), dtype=np.float32)
    ncells_or_dof = raw["segment_dof"].astype(np.float32)

    x = np.concatenate(
        [pos_m, dir_u, energy_like[:, None], ncells_or_dof[:, None]], axis=1
    ).astype(np.float32)

    return {
        "x": x,
        "phi": phi,
        "eta": eta,
        "energy_like": energy_like,
        "dir_u": dir_u,
        "sector": raw["bucket_sector"],
        "node_type": np.zeros(len(x_mm), dtype=np.int64),
        "muon_xyz_m": pos_m,
        "muon_bucket": np.stack(
            [
                raw["bucket_chamber"],
                raw["bucket_layers"],
                raw["bucket_sector"],
                raw["bucket_seg"].astype(np.int64),
            ],
            axis=1,
        ),
    }


def _build_calo_nodes(
    td, idxs, sector_mod, min_tower_energy_mev,
    max_tower_segment_dr, calo_r_max_mm, calo_z_max_mm,
    seg_eta_list, seg_phi_list,
):
    """Calo node features: [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, energy, nCells]."""
    raw = _collect_calo_filtered(
        td, idxs, seg_eta_list, seg_phi_list,
        sector_mod, min_tower_energy_mev, max_tower_segment_dr,
        calo_r_max_mm, calo_z_max_mm,
    )
    if raw is None:
        return None

    dir_u = _safe_normalize(
        np.stack([raw["dx"], raw["dy"], raw["dz"]], axis=1).astype(np.float32)
    )
    tower_xyz_m = raw["tower_xyz_m"]

    x = np.concatenate(
        [tower_xyz_m, dir_u, raw["tower_energy"][:, None], raw["tower_ncells"][:, None]], axis=1
    ).astype(np.float32)

    return {
        "x": x,
        "phi": raw["tower_phi"],
        "eta": raw["tower_eta"],
        "energy_like": raw["tower_energy"],
        "dir_u": dir_u,
        "sector": raw["sector"],
        "node_type": np.ones(len(raw["tower_energy"]), dtype=np.int64),
        "tower_xyz_m": tower_xyz_m,
        "tower_min_dr": raw["tower_min_dr"],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Convert ROOT -> HDF5 graphs (Cartesian node features)"
    )
    add_converter_args(ap)
    args = ap.parse_args()
    run_converter_main_loop(args, _build_vertex_target, _build_muon_nodes, _build_calo_nodes)


if __name__ == "__main__":
    main()
