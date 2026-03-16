#!/usr/bin/env python3
"""
DisplacedVertex_cylindrical_converter.py

Node features: [r, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]
Target: y_vertex = [rho, phi, z]  (m, rad, m)

Example:
python -u DisplacedVertex_cylindrical_converter.py \
    --input-dir hdd_data/ \
    --pattern "MuonBucketDump_H*/outputs/MuonBucketDump_group.det-muon.*root" \
    --output-dir ./data_cylindrical \
    --output-name displaced_vertex_dataset \
    --vertex-r-max-mm 8000.0 \
    --vertex-z-max-mm 12000.0 \
    --calo-r-max-mm 4250 \
    --calo-z-max-mm 6500 \
    --min-tower-energy-mev 1000 \
    --max-tower-segment-dr 0.4
"""

import argparse
import numpy as np

from dv_converter_utils import (
    _collect_muon_raw,
    _collect_calo_filtered,
    _cartesian_to_atlas_position_polar,
    _cartesian_to_atlas_direction_angles,
    _cartesian_to_cylindrical,
    add_converter_args,
    run_converter_main_loop,
)


def _build_vertex_target(vertex_td, idxs):
    """Return [rho, phi, z] in cylindrical coordinates (metres/radians)."""
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
            x_m = np.float32(x) / 1000.0
            y_m = np.float32(y) / 1000.0
            z_m = np.float32(z) / 1000.0
            rho, phi, zc = _cartesian_to_cylindrical(x_m, y_m, z_m)
            return np.asarray([rho, phi, zc], dtype=np.float32)
    return None


def _build_muon_nodes(td, idxs):
    raw = _collect_muon_raw(td, idxs)
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

    r_xy = np.hypot(x_mm, y_mm)
    phi = np.arctan2(y_mm, x_mm).astype(np.float32)
    eta = np.empty_like(r_xy, dtype=np.float32)
    mask = r_xy > 0
    eta[mask] = np.arcsinh(z_mm[mask] / r_xy[mask])
    eta[~mask] = np.sign(z_mm[~mask]) * 1.0e6

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
    raw = _collect_calo_filtered(
        td, idxs, seg_eta_list, seg_phi_list,
        sector_mod, min_tower_energy_mev, max_tower_segment_dr,
        calo_r_max_mm, calo_z_max_mm,
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
        [r_pos, theta_pos, phi_pos, theta_dir, phi_dir, raw["tower_energy"], raw["tower_ncells"]],
        axis=1,
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
        description="Convert ROOT -> HDF5 graphs (cylindrical target, spherical node coords)"
    )
    add_converter_args(ap)
    args = ap.parse_args()
    run_converter_main_loop(args, _build_vertex_target, _build_muon_nodes, _build_calo_nodes)


if __name__ == "__main__":
    main()
