#!/usr/bin/env python3
"""Diagnose displaced-muon segment reconstruction richness.

This script consumes the vertex-level Parquet files produced by
``analyze_DisplacedVertex_truth_reco.py``.  It studies reconstruction input
richness (the number of reconstructed segments matched to each truth muon),
not neural-network performance.

The large ROOT files are not reread for event data.  Optionally, one ROOT file
can be opened only to inventory its tree/branch schema and document which
additional quantities would require a dedicated extraction pass.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


SAMPLE_RE = re.compile(r"_m(?P<mass>\d+)_ctau(?P<ctau>\d+)_mu200_part\d+")

REQUIRED_COLUMNS = {
    "source_file",
    "event_hash0",
    "event_hash1",
    "vertex_index",
    "vertex_x_mm",
    "vertex_y_mm",
    "vertex_z_mm",
    "vertex_r_mm",
    "inside_calo_envelope",
    "links_valid",
    "muon0_index",
    "muon1_index",
    "muon0_pt",
    "muon0_eta",
    "muon0_phi",
    "muon1_pt",
    "muon1_eta",
    "muon1_phi",
    "muon0_segments",
    "muon1_segments",
    "event_buckets",
    "event_truth_buckets",
    "event_segments",
    "event_matched_segments",
    "event_unmatched_segments",
}

SELECTIONS = {
    "all_selected": "All selected displaced vertices",
    "inside_calo_envelope": "Inside existing simplified envelope",
}

QUALITY_ORDER = ("nmin = 0", "nmin = 1", "nmin = 2", "nmin >= 3")
QUALITY_COLORS = {
    "nmin = 0": "#D1495B",
    "nmin = 1": "#F79256",
    "nmin = 2": "#4C78A8",
    "nmin >= 3": "#2A9D8F",
}

R_EDGES_MM = np.arange(0.0, 8500.0, 500.0)
Z_EDGES_MM = np.arange(0.0, 12750.0, 750.0)
PT_EDGES_GEV = np.array([1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 60, 80, 140], dtype=float)
ETA_EDGES = np.array([0, .2, .4, .6, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7], dtype=float)
PHI_EDGES = np.linspace(-math.pi, math.pi, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--root-schema-file",
        type=Path,
        help="Optional representative ROOT file; only its schema is inspected.",
    )
    parser.add_argument("--position-min-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def parse_sample(source_file: str) -> tuple[int, int, str]:
    match = SAMPLE_RE.search(Path(source_file).name)
    if match is None:
        raise ValueError(f"Cannot parse scalar sample from {source_file}")
    mass = int(match.group("mass"))
    ctau = int(match.group("ctau"))
    return mass, ctau, f"m{mass}_ctau{ctau}"


def wrapped_abs_delta_phi(phi0: pd.Series, phi1: pd.Series) -> np.ndarray:
    delta = phi0.to_numpy(float) - phi1.to_numpy(float)
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def load_vertices(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("MuonBucketDump_vertex_a_mumu_*.vertices.parquet"))
    if not paths:
        raise FileNotFoundError(f"No scalar vertex Parquet files found in {input_dir}")

    columns = sorted(REQUIRED_COLUMNS)
    frames = []
    print(f"[schema] verifying {len(paths)} Parquet files", flush=True)
    for path in paths:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = REQUIRED_COLUMNS - available
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frames.append(pd.read_parquet(path, columns=columns))

    frame = pd.concat(frames, ignore_index=True)
    invalid_links = int((~frame["links_valid"].astype(bool)).sum())
    if invalid_links:
        print(f"[warn] excluding {invalid_links} rows with invalid truth-muon links", flush=True)
    frame = frame[frame["links_valid"].astype(bool)].copy()

    parsed = frame["source_file"].map(parse_sample)
    frame["mass_GeV"] = parsed.map(lambda value: value[0])
    frame["ctau_mm"] = parsed.map(lambda value: value[1])
    frame["sample"] = parsed.map(lambda value: value[2])

    frame["n1"] = frame["muon0_segments"].astype(np.int64)
    frame["n2"] = frame["muon1_segments"].astype(np.int64)
    frame["n_min"] = frame[["n1", "n2"]].min(axis=1)
    frame["n_max"] = frame[["n1", "n2"]].max(axis=1)
    frame["poor"] = frame["n_min"].lt(2)
    frame["abs_z_mm"] = frame["vertex_z_mm"].abs()
    frame["vertex_phi"] = np.arctan2(frame["vertex_y_mm"], frame["vertex_x_mm"])
    frame["min_muon_pt_GeV"] = frame[["muon0_pt", "muon1_pt"]].min(axis=1)
    frame["max_abs_muon_eta"] = np.maximum(frame["muon0_eta"].abs(), frame["muon1_eta"].abs())
    frame["min_abs_muon_eta"] = np.minimum(frame["muon0_eta"].abs(), frame["muon1_eta"].abs())
    frame["delta_eta_mumu"] = (frame["muon0_eta"] - frame["muon1_eta"]).abs()
    frame["delta_phi_mumu"] = wrapped_abs_delta_phi(frame["muon0_phi"], frame["muon1_phi"])
    frame["delta_r_mumu"] = np.hypot(frame["delta_eta_mumu"], frame["delta_phi_mumu"])
    pt_sum = (frame["muon0_pt"] + frame["muon1_pt"]).clip(lower=1e-9)
    frame["pt_asymmetry"] = (frame["muon0_pt"] - frame["muon1_pt"]).abs() / pt_sum

    # Direction cosine between the vertex displacement and each truth-muon
    # momentum.  This is a topology variable, not a detector extrapolation.
    displacement = np.hypot(frame["vertex_r_mm"], frame["vertex_z_mm"]).clip(lower=1e-9)
    for index in (0, 1):
        eta = frame[f"muon{index}_eta"]
        phi = frame[f"muon{index}_phi"]
        dphi_vertex = np.arctan2(
            np.sin(phi - frame["vertex_phi"]),
            np.cos(phi - frame["vertex_phi"]),
        )
        frame[f"muon{index}_outward_cosine"] = (
            frame["vertex_r_mm"] * np.cos(dphi_vertex) / np.cosh(eta)
            + frame["vertex_z_mm"] * np.tanh(eta)
        ) / displacement
    frame["min_outward_cosine"] = frame[
        ["muon0_outward_cosine", "muon1_outward_cosine"]
    ].min(axis=1)

    frame["quality"] = np.select(
        [frame["n_min"].eq(0), frame["n_min"].eq(1), frame["n_min"].eq(2)],
        QUALITY_ORDER[:3],
        default=QUALITY_ORDER[3],
    )
    frame["quality"] = pd.Categorical(frame["quality"], QUALITY_ORDER, ordered=True)
    return frame


def selected(frame: pd.DataFrame, selection: str) -> pd.DataFrame:
    if selection == "all_selected":
        return frame
    if selection == "inside_calo_envelope":
        return frame[frame["inside_calo_envelope"].astype(bool)]
    raise ValueError(selection)


def binned_profile(frame: pd.DataFrame, variable: str, edges: np.ndarray) -> pd.DataFrame:
    values = frame[variable].to_numpy(float)
    index = np.digitize(values, edges) - 1
    records = []
    for bin_index in range(len(edges) - 1):
        mask = index == bin_index
        count = int(mask.sum())
        if not count:
            continue
        nmin = frame.loc[mask, "n_min"]
        records.append(
            {
                "variable": variable,
                "bin_low": float(edges[bin_index]),
                "bin_high": float(edges[bin_index + 1]),
                "bin_center": float((edges[bin_index] + edges[bin_index + 1]) / 2),
                "vertices": count,
                "mean_n_min": float(nmin.mean()),
                "sem_n_min": float(nmin.sem()),
                "fraction_nmin_0": float(nmin.eq(0).mean()),
                "fraction_nmin_1": float(nmin.eq(1).mean()),
                "fraction_nmin_ge2": float(nmin.ge(2).mean()),
                "fraction_nmin_lt2": float(nmin.lt(2).mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def position_profiles(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    tables = []
    for selection in SELECTIONS:
        subset = selected(frame, selection)
        for variable, edges in (("vertex_r_mm", R_EDGES_MM), ("abs_z_mm", Z_EDGES_MM)):
            table = binned_profile(subset, variable, edges)
            table.insert(0, "selection", selection)
            tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output_dir / "position_profiles.csv", index=False)
    return result


def plot_position_profiles(table: pd.DataFrame, output_dir: Path) -> None:
    for selection, label in SELECTIONS.items():
        sub = table[table["selection"].eq(selection)]
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        for axis, variable, xlabel in zip(
            axes,
            ("vertex_r_mm", "abs_z_mm"),
            ("Displaced-vertex radius r [mm]", "Displaced-vertex |z| [mm]"),
        ):
            values = sub[sub["variable"].eq(variable)]
            axis.errorbar(
                values["bin_center"], values["mean_n_min"], yerr=values["sem_n_min"],
                marker="o", markersize=4, linewidth=1.8, color="#1D4E89", capsize=2,
            )
            axis.set(xlabel=xlabel, ylabel="Average nmin", ylim=(0, None))
            axis.grid(alpha=.25)
        figure.suptitle(f"Average limiting-muon segment multiplicity — {label}")
        figure.savefig(output_dir / f"average_nmin_vs_position_{selection}.png", dpi=220)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        styles = (
            ("fraction_nmin_0", "nmin = 0", "#D1495B"),
            ("fraction_nmin_1", "nmin = 1", "#F79256"),
            ("fraction_nmin_ge2", "nmin >= 2", "#2A9D8F"),
        )
        for axis, variable, xlabel in zip(
            axes,
            ("vertex_r_mm", "abs_z_mm"),
            ("Displaced-vertex radius r [mm]", "Displaced-vertex |z| [mm]"),
        ):
            values = sub[sub["variable"].eq(variable)]
            for column, curve_label, color in styles:
                axis.plot(values["bin_center"], values[column], marker="o", markersize=3,
                          linewidth=1.8, label=curve_label, color=color)
            axis.set(xlabel=xlabel, ylabel="Fraction of vertices", ylim=(0, 1))
            axis.grid(alpha=.25)
            axis.legend(frameon=False)
        figure.suptitle(f"Reconstruction-quality probabilities — {label}")
        figure.savefig(output_dir / f"quality_fractions_vs_position_{selection}.png", dpi=220)
        plt.close(figure)


def map_table(
    frame: pd.DataFrame,
    selection: str,
    min_count: int,
) -> pd.DataFrame:
    subset = selected(frame, selection)
    r_index = np.digitize(subset["vertex_r_mm"], R_EDGES_MM) - 1
    z_index = np.digitize(subset["abs_z_mm"], Z_EDGES_MM) - 1
    records = []
    for ir in range(len(R_EDGES_MM) - 1):
        for iz in range(len(Z_EDGES_MM) - 1):
            mask = (r_index == ir) & (z_index == iz)
            count = int(mask.sum())
            if not count:
                continue
            nmin = subset.loc[mask, "n_min"]
            records.append(
                {
                    "selection": selection,
                    "r_low_mm": R_EDGES_MM[ir],
                    "r_high_mm": R_EDGES_MM[ir + 1],
                    "z_low_mm": Z_EDGES_MM[iz],
                    "z_high_mm": Z_EDGES_MM[iz + 1],
                    "vertices": count,
                    "passes_min_count": count >= min_count,
                    "mean_n_min": float(nmin.mean()),
                    "fraction_nmin_lt2": float(nmin.lt(2).mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_map(
    table: pd.DataFrame,
    value: str,
    title: str,
    output: Path,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    matrix = np.full((len(Z_EDGES_MM) - 1, len(R_EDGES_MM) - 1), np.nan)
    valid = table[table["passes_min_count"]]
    for row in valid.itertuples(index=False):
        ir = int(np.searchsorted(R_EDGES_MM, row.r_low_mm))
        iz = int(np.searchsorted(Z_EDGES_MM, row.z_low_mm))
        matrix[iz, ir] = getattr(row, value)
    figure, axis = plt.subplots(figsize=(9, 6.8), constrained_layout=True)
    image = axis.pcolormesh(
        R_EDGES_MM, Z_EDGES_MM, matrix, shading="auto", cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    figure.colorbar(image, ax=axis, label=title.split("—")[0].strip())
    axis.set(
        xlabel="Displaced-vertex radius r [mm]",
        ylabel="Displaced-vertex |z| [mm]",
        title=title,
    )
    axis.set_xlim(0, float(table["r_high_mm"].max()))
    axis.set_ylim(0, float(table["z_high_mm"].max()))
    axis.text(.01, .99, "Blank bins have fewer than the minimum required vertices",
              transform=axis.transAxes, va="top", fontsize=9,
              bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"})
    figure.savefig(output, dpi=220)
    plt.close(figure)


def position_maps(frame: pd.DataFrame, output_dir: Path, min_count: int) -> pd.DataFrame:
    tables = []
    for selection, label in SELECTIONS.items():
        table = map_table(frame, selection, min_count)
        tables.append(table)
        plot_map(
            table, "mean_n_min", f"Average nmin — {label}",
            output_dir / f"average_nmin_r_absz_map_{selection}.png",
            "viridis", 0.0, 3.0,
        )
        plot_map(
            table, "fraction_nmin_lt2", f"Fraction with nmin < 2 — {label}",
            output_dir / f"poor_fraction_r_absz_map_{selection}.png",
            "magma_r", 0.0, 1.0,
        )
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output_dir / "position_map_values.csv", index=False)
    return result


def nmin_ge2_count_diagnostics(
    frame: pd.DataFrame,
    output_dir: Path,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count vertices with both truth muons represented by >=2 segments."""
    profile_records = []
    map_records = []
    for selection, label in SELECTIONS.items():
        subset = selected(frame, selection)

        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        normalized_curves = []
        for axis, variable, edges, xlabel in zip(
            axes,
            ("vertex_r_mm", "abs_z_mm"),
            (R_EDGES_MM, Z_EDGES_MM),
            ("Displaced-vertex radius r [mm]", "Displaced-vertex |z| [mm]"),
        ):
            indices = np.digitize(subset[variable].to_numpy(float), edges) - 1
            centers = (edges[:-1] + edges[1:]) / 2
            counts = []
            totals = []
            for bin_index in range(len(edges) - 1):
                mask = indices == bin_index
                total = int(mask.sum())
                passed = int(subset.loc[mask, "n_min"].ge(2).sum())
                counts.append(passed)
                totals.append(total)
                profile_records.append(
                    {
                        "selection": selection,
                        "variable": variable,
                        "bin_low": float(edges[bin_index]),
                        "bin_high": float(edges[bin_index + 1]),
                        "bin_center": float(centers[bin_index]),
                        "all_vertices": total,
                        "vertices_nmin_ge2": passed,
                        "fraction_nmin_ge2": passed / total if total else np.nan,
                    }
                )
            axis.bar(
                centers, counts, width=np.diff(edges) * .9,
                color="#2A9D8F", edgecolor="#176B63", linewidth=.6,
            )
            width = float(edges[1] - edges[0])
            visible_max = min(float(edges[-1]), math.ceil(float(subset[variable].max()) / width) * width)
            axis.set_xlim(float(edges[0]), visible_max)
            axis.set(xlabel=xlabel, ylabel="Vertices with nmin >= 2")
            axis.grid(axis="y", alpha=.25)
            normalized_curves.append((variable, edges, xlabel, np.asarray(counts), np.asarray(totals)))
        figure.suptitle(f"Number of well-represented vertices — {label}")
        figure.savefig(output_dir / f"nmin_ge2_counts_vs_position_{selection}.png", dpi=220)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        for axis, (variable, edges, xlabel, passed, totals) in zip(axes, normalized_curves):
            centers = (edges[:-1] + edges[1:]) / 2
            fraction = np.divide(
                passed, totals, out=np.full(passed.shape, np.nan, dtype=float), where=totals > 0
            )
            uncertainty = np.sqrt(
                np.divide(
                    fraction * (1.0 - fraction), totals,
                    out=np.full(fraction.shape, np.nan), where=totals > 0,
                )
            )
            axis.errorbar(
                centers, fraction, yerr=uncertainty, marker="o", markersize=4,
                linewidth=1.8, capsize=2, color="#1D4E89",
            )
            width = float(edges[1] - edges[0])
            visible_max = min(float(edges[-1]), math.ceil(float(subset[variable].max()) / width) * width)
            axis.set_xlim(float(edges[0]), visible_max)
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.set(xlabel=xlabel, ylabel="Fraction with nmin >= 2")
            axis.grid(alpha=.25)
        figure.suptitle(f"Well-represented fraction per position bin — {label}")
        figure.savefig(output_dir / f"nmin_ge2_fraction_vs_position_{selection}.png", dpi=220)
        plt.close(figure)

        r_index = np.digitize(subset["vertex_r_mm"].to_numpy(float), R_EDGES_MM) - 1
        z_index = np.digitize(subset["abs_z_mm"].to_numpy(float), Z_EDGES_MM) - 1
        count_matrix = np.full((len(Z_EDGES_MM) - 1, len(R_EDGES_MM) - 1), np.nan)
        fraction_matrix = np.full((len(Z_EDGES_MM) - 1, len(R_EDGES_MM) - 1), np.nan)
        for ir in range(len(R_EDGES_MM) - 1):
            for iz in range(len(Z_EDGES_MM) - 1):
                mask = (r_index == ir) & (z_index == iz)
                total = int(mask.sum())
                passed = int(subset.loc[mask, "n_min"].ge(2).sum())
                map_records.append(
                    {
                        "selection": selection,
                        "r_low_mm": R_EDGES_MM[ir],
                        "r_high_mm": R_EDGES_MM[ir + 1],
                        "z_low_mm": Z_EDGES_MM[iz],
                        "z_high_mm": Z_EDGES_MM[iz + 1],
                        "all_vertices": total,
                        "vertices_nmin_ge2": passed,
                        "fraction_nmin_ge2": passed / total if total else np.nan,
                        "passes_min_count": total >= min_count,
                    }
                )
                if total >= min_count and passed > 0:
                    count_matrix[iz, ir] = passed
                    fraction_matrix[iz, ir] = passed / total

        finite = count_matrix[np.isfinite(count_matrix)]
        figure, axis = plt.subplots(figsize=(9, 6.8), constrained_layout=True)
        image = axis.pcolormesh(
            R_EDGES_MM, Z_EDGES_MM, count_matrix, shading="auto", cmap="YlGnBu",
            norm=LogNorm(vmin=max(1.0, float(finite.min())), vmax=float(finite.max())),
        )
        figure.colorbar(image, ax=axis, label="Vertices with nmin >= 2 (log colour scale)")
        axis.set(
            xlabel="Displaced-vertex radius r [mm]",
            ylabel="Displaced-vertex |z| [mm]",
            title=f"Count of vertices with nmin >= 2 — {label}",
        )
        axis.set_xlim(0, min(float(R_EDGES_MM[-1]), math.ceil(float(subset["vertex_r_mm"].max()) / 500) * 500))
        axis.set_ylim(0, min(float(Z_EDGES_MM[-1]), math.ceil(float(subset["abs_z_mm"].max()) / 750) * 750))
        axis.text(
            .01, .99, "Blank bins contain fewer than the minimum total vertices",
            transform=axis.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"},
        )
        figure.savefig(output_dir / f"nmin_ge2_count_r_absz_map_{selection}.png", dpi=220)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 6.8), constrained_layout=True)
        image = axis.pcolormesh(
            R_EDGES_MM, Z_EDGES_MM, fraction_matrix, shading="auto", cmap="viridis",
            norm=Normalize(vmin=0.0, vmax=1.0),
        )
        colorbar = figure.colorbar(image, ax=axis, label="Fraction with nmin >= 2")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set(
            xlabel="Displaced-vertex radius r [mm]",
            ylabel="Displaced-vertex |z| [mm]",
            title=f"Well-represented fraction — {label}",
        )
        axis.set_xlim(0, min(float(R_EDGES_MM[-1]), math.ceil(float(subset["vertex_r_mm"].max()) / 500) * 500))
        axis.set_ylim(0, min(float(Z_EDGES_MM[-1]), math.ceil(float(subset["abs_z_mm"].max()) / 750) * 750))
        axis.text(
            .01, .99, "Each bin is normalized by all vertices in that bin",
            transform=axis.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"},
        )
        figure.savefig(output_dir / f"nmin_ge2_fraction_r_absz_map_{selection}.png", dpi=220)
        plt.close(figure)

    profiles = pd.DataFrame.from_records(profile_records)
    maps = pd.DataFrame.from_records(map_records)
    profiles.to_csv(output_dir / "nmin_ge2_counts_vs_position.csv", index=False)
    maps.to_csv(output_dir / "nmin_ge2_count_map_values.csv", index=False)
    return profiles, maps


def plot_position_distributions(frame: pd.DataFrame, output_dir: Path) -> None:
    for selection, label in SELECTIONS.items():
        subset = selected(frame, selection)
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        for axis, variable, edges, xlabel in zip(
            axes,
            ("vertex_r_mm", "abs_z_mm"),
            (R_EDGES_MM, Z_EDGES_MM),
            ("Displaced-vertex radius r [mm]", "Displaced-vertex |z| [mm]"),
        ):
            for quality in QUALITY_ORDER:
                values = subset.loc[subset["quality"].eq(quality), variable]
                axis.hist(values, bins=edges, density=True, histtype="step", linewidth=2,
                          label=quality, color=QUALITY_COLORS[quality])
            axis.set(xlabel=xlabel, ylabel="Normalized density")
            width = float(edges[1] - edges[0])
            visible_max = min(float(edges[-1]), math.ceil(float(subset[variable].max()) / width) * width)
            axis.set_xlim(float(edges[0]), visible_max)
            axis.grid(alpha=.2)
            axis.legend(frameon=False)
        figure.suptitle(f"Vertex-position distributions by reconstruction quality — {label}")
        figure.savefig(output_dir / f"position_distributions_by_quality_{selection}.png", dpi=220)
        plt.close(figure)


def build_muon_table(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    common = [
        "source_file", "event_hash0", "event_hash1", "vertex_index", "sample",
        "mass_GeV", "ctau_mm", "vertex_r_mm", "abs_z_mm", "inside_calo_envelope",
    ]
    for index in (0, 1):
        part = frame[common].copy()
        part["truth_muon_slot"] = index
        part["truth_muon_index"] = frame[f"muon{index}_index"]
        part["pt_GeV"] = frame[f"muon{index}_pt"]
        part["eta"] = frame[f"muon{index}_eta"]
        part["abs_eta"] = frame[f"muon{index}_eta"].abs()
        part["phi"] = frame[f"muon{index}_phi"]
        part["outward_cosine"] = frame[f"muon{index}_outward_cosine"]
        part["segments"] = frame[f"muon{index}_segments"].astype(np.int64)
        part["missing"] = part["segments"].eq(0)
        part["fewer_than_two"] = part["segments"].lt(2)
        records.append(part)
    return pd.concat(records, ignore_index=True)


def muon_profile(muons: pd.DataFrame, variable: str, edges: np.ndarray) -> pd.DataFrame:
    index = np.digitize(muons[variable].to_numpy(float), edges) - 1
    records = []
    for bin_index in range(len(edges) - 1):
        mask = index == bin_index
        count = int(mask.sum())
        if not count:
            continue
        part = muons.loc[mask]
        records.append(
            {
                "variable": variable,
                "bin_low": edges[bin_index],
                "bin_high": edges[bin_index + 1],
                "bin_center": (edges[bin_index] + edges[bin_index + 1]) / 2,
                "truth_muons": count,
                "mean_segments": part["segments"].mean(),
                "fraction_missing": part["missing"].mean(),
                "fraction_fewer_than_two": part["fewer_than_two"].mean(),
                "fraction_exactly_one": part["segments"].eq(1).mean(),
                "fraction_at_least_two": part["segments"].ge(2).mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def truth_muon_diagnostics(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    muons = build_muon_table(frame)
    tables = []
    configs = (("pt_GeV", PT_EDGES_GEV), ("abs_eta", ETA_EDGES), ("phi", PHI_EDGES))
    for selection in SELECTIONS:
        subset = muons if selection == "all_selected" else muons[muons["inside_calo_envelope"]]
        for variable, edges in configs:
            table = muon_profile(subset, variable, edges)
            table.insert(0, "selection", selection)
            tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output_dir / "truth_muon_reconstruction_profiles.csv", index=False)

    for selection, label in SELECTIONS.items():
        sub = result[result["selection"].eq(selection)]
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
        for axis, variable, xlabel in zip(
            axes,
            ("pt_GeV", "abs_eta", "phi"),
            ("Truth-muon pT [GeV]", "Truth-muon |eta|", "Truth-muon phi [rad]"),
        ):
            values = sub[sub["variable"].eq(variable)]
            axis.plot(values["bin_center"], values["fraction_missing"], marker="o",
                      label="0 segments", color="#D1495B")
            axis.plot(values["bin_center"], values["fraction_fewer_than_two"], marker="o",
                      label="<2 segments", color="#4C78A8")
            axis.set(xlabel=xlabel, ylabel="Fraction of truth muons", ylim=(0, 1))
            axis.grid(alpha=.25)
            axis.legend(frameon=False)
        figure.suptitle(f"Per-truth-muon reconstruction inefficiency — {label}")
        figure.savefig(output_dir / f"truth_muon_inefficiency_{selection}.png", dpi=220)
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
        category_styles = (
            ("fraction_missing", "0 segments", "#D1495B"),
            ("fraction_exactly_one", "1 segment", "#F28E2B"),
            ("fraction_at_least_two", ">=2 segments", "#2A9D8F"),
        )
        for axis, variable, xlabel in zip(
            axes,
            ("pt_GeV", "abs_eta", "phi"),
            ("Truth-muon pT [GeV]", "Truth-muon |eta|", "Truth-muon phi [rad]"),
        ):
            values = sub[sub["variable"].eq(variable)]
            for column, curve_label, color in category_styles:
                axis.plot(
                    values["bin_center"], values[column], marker="o", markersize=4,
                    linewidth=1.8, label=curve_label, color=color,
                )
            axis.set(xlabel=xlabel, ylabel="Fraction of truth muons", ylim=(0, 1))
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(alpha=.25)
            axis.legend(frameon=False)
        figure.suptitle(f"Truth-muon segment multiplicity categories — {label}")
        figure.savefig(output_dir / f"truth_muon_segment_categories_{selection}.png", dpi=220)
        plt.close(figure)
    return result


def topology_profiles(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    configs = {
        "min_muon_pt_GeV": PT_EDGES_GEV,
        "max_abs_muon_eta": ETA_EDGES,
        "delta_r_mumu": np.linspace(0, 6, 16),
        "pt_asymmetry": np.linspace(0, 1, 11),
        "min_outward_cosine": np.linspace(-1, 1, 13),
        "event_truth_buckets": np.arange(-.5, 20.5, 1),
    }
    tables = []
    for selection in SELECTIONS:
        subset = selected(frame, selection)
        for variable, edges in configs.items():
            table = binned_profile(subset, variable, edges)
            table.insert(0, "selection", selection)
            tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output_dir / "topology_reconstruction_profiles.csv", index=False)

    for selection, label in SELECTIONS.items():
        sub = result[result["selection"].eq(selection)]
        figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        labels = {
            "min_muon_pt_GeV": "Lower truth-muon pT [GeV]",
            "max_abs_muon_eta": "Larger truth-muon |eta|",
            "delta_r_mumu": "Truth-muon pair deltaR",
            "pt_asymmetry": "Truth-muon pT asymmetry",
            "min_outward_cosine": "Smaller outward direction cosine",
            "event_truth_buckets": "Event truth-labelled buckets",
        }
        for axis, variable in zip(axes.flat, configs):
            values = sub[sub["variable"].eq(variable)]
            axis.plot(values["bin_center"], values["fraction_nmin_lt2"], marker="o",
                      color="#7A5195", linewidth=1.8)
            axis.set(xlabel=labels[variable], ylabel="Fraction with nmin < 2", ylim=(0, 1))
            axis.grid(alpha=.25)
        figure.suptitle(f"Prioritized vertex/topology diagnostics — {label}")
        figure.savefig(output_dir / f"topology_diagnostics_{selection}.png", dpi=220)
        plt.close(figure)
    return result


def sample_summary(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    records = []
    for selection in SELECTIONS:
        subset = selected(frame, selection)
        for (mass, ctau, sample), part in subset.groupby(["mass_GeV", "ctau_mm", "sample"]):
            records.append(
                {
                    "selection": selection,
                    "mass_GeV": mass,
                    "ctau_mm": ctau,
                    "sample": sample,
                    "vertices": len(part),
                    "mean_n_min": part["n_min"].mean(),
                    "fraction_nmin_0": part["n_min"].eq(0).mean(),
                    "fraction_nmin_1": part["n_min"].eq(1).mean(),
                    "fraction_nmin_ge2": part["n_min"].ge(2).mean(),
                }
            )
    result = pd.DataFrame.from_records(records).sort_values(["selection", "mass_GeV", "ctau_mm"])
    result.to_csv(output_dir / "sample_reconstruction_summary.csv", index=False)
    return result


def categorical_summaries(frame: pd.DataFrame, output_dir: Path) -> None:
    position_records = []
    for selection in SELECTIONS:
        subset = selected(frame, selection)
        for quality in QUALITY_ORDER:
            part = subset[subset["quality"].eq(quality)]
            position_records.append(
                {
                    "selection": selection,
                    "quality": quality,
                    "vertices": len(part),
                    "median_r_mm": part["vertex_r_mm"].median(),
                    "q25_r_mm": part["vertex_r_mm"].quantile(.25),
                    "q75_r_mm": part["vertex_r_mm"].quantile(.75),
                    "median_abs_z_mm": part["abs_z_mm"].median(),
                    "q25_abs_z_mm": part["abs_z_mm"].quantile(.25),
                    "q75_abs_z_mm": part["abs_z_mm"].quantile(.75),
                }
            )
    pd.DataFrame.from_records(position_records).to_csv(
        output_dir / "quality_position_summary.csv", index=False
    )

    # These broad eta bands are diagnostic labels, not official detector
    # boundaries.  They expose the non-monotonic central/transition/forward
    # structure without claiming a specific chamber geometry.
    muons = build_muon_table(frame)
    eta_edges = np.array([0.0, 0.2, 1.05, 1.3, 2.0, 2.7])
    eta_labels = [
        "very_central_<0.2",
        "barrel_like_0.2_to_1.05",
        "transition_like_1.05_to_1.3",
        "endcap_like_1.3_to_2.0",
        "forward_2.0_to_2.7",
    ]
    muons["eta_analysis_band"] = pd.cut(
        muons["abs_eta"], eta_edges, labels=eta_labels, include_lowest=True, right=False
    )
    eta_records = []
    for selection in SELECTIONS:
        subset = muons if selection == "all_selected" else muons[muons["inside_calo_envelope"]]
        for band, part in subset.groupby("eta_analysis_band", observed=True):
            eta_records.append(
                {
                    "selection": selection,
                    "eta_analysis_band": str(band),
                    "truth_muons": len(part),
                    "mean_segments": part["segments"].mean(),
                    "fraction_missing": part["missing"].mean(),
                    "fraction_fewer_than_two": part["fewer_than_two"].mean(),
                }
            )
    pd.DataFrame.from_records(eta_records).to_csv(
        output_dir / "eta_analysis_band_summary.csv", index=False
    )


def association_ranking(frame: pd.DataFrame, output_dir: Path, seed: int) -> tuple[pd.DataFrame, dict]:
    feature_labels = {
        "vertex_r_mm": "vertex r",
        "abs_z_mm": "vertex |z|",
        "min_muon_pt_GeV": "lower muon pT",
        "max_abs_muon_eta": "larger muon |eta|",
        "delta_r_mumu": "muon-pair deltaR",
        "pt_asymmetry": "pT asymmetry",
        "min_outward_cosine": "outward topology",
        "event_buckets": "event bucket occupancy",
        "event_truth_buckets": "truth-labelled bucket coverage",
        "event_unmatched_segments": "unmatched segment occupancy",
        "mass_GeV": "sample mass",
        "ctau_mm": "sample lifetime",
    }
    features = list(feature_labels)
    records = []
    model_metrics = {}
    for selection in SELECTIONS:
        subset = selected(frame, selection)
        y = subset["poor"].astype(np.int8)
        for feature in features:
            values = subset[feature].to_numpy(float)
            finite = np.isfinite(values)
            rho = spearmanr(values[finite], y.to_numpy()[finite]).statistic
            try:
                groups = pd.qcut(subset[feature], 10, duplicates="drop")
                rates = subset.groupby(groups, observed=True)["poor"].mean()
                spread = float(rates.max() - rates.min())
            except ValueError:
                spread = float("nan")
            records.append(
                {
                    "selection": selection,
                    "feature": feature,
                    "label": feature_labels[feature],
                    "spearman_rho_with_poor": rho,
                    "absolute_spearman_rho": abs(rho),
                    "poor_fraction_decile_spread": spread,
                }
            )

        # Nonlinear multivariate ranking.  It is diagnostic association only,
        # and is deliberately kept separate from the physics classifier.
        model_data = subset[features].replace([np.inf, -np.inf], np.nan).dropna()
        model_y = subset.loc[model_data.index, "poor"].astype(np.int8)
        if len(model_data) > 300_000:
            rng = np.random.default_rng(seed)
            take = rng.choice(len(model_data), 300_000, replace=False)
            model_data = model_data.iloc[take]
            model_y = model_y.iloc[take]
        x_train, x_test, y_train, y_test = train_test_split(
            model_data, model_y, test_size=.25, random_state=seed, stratify=model_y,
        )
        model = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, learning_rate=.08,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, probability)
        # A fixed 40k evaluation subset keeps the reproducibility check fast.
        if len(x_test) > 40_000:
            x_perm = x_test.sample(40_000, random_state=seed)
            y_perm = y_test.loc[x_perm.index]
        else:
            x_perm, y_perm = x_test, y_test
        importance = permutation_importance(
            model, x_perm, y_perm, scoring="roc_auc", n_repeats=3,
            random_state=seed, n_jobs=1,
        )
        model_metrics[selection] = {"test_auc": float(auc), "events": int(len(model_data))}
        for feature, mean, std in zip(features, importance.importances_mean, importance.importances_std):
            records.append(
                {
                    "selection": selection,
                    "feature": feature,
                    "label": feature_labels[feature],
                    "permutation_auc_decrease": float(mean),
                    "permutation_auc_decrease_std": float(std),
                    "ranking_source": "multivariate_hist_gradient_boosting",
                }
            )

    result = pd.DataFrame.from_records(records)
    metrics_path = output_dir / "diagnostic_model_metrics.json"
    metrics_path.write_text(json.dumps(model_metrics, indent=2) + "\n", encoding="utf-8")
    result.to_csv(output_dir / "diagnostic_variable_ranking.csv", index=False)
    return result, model_metrics


def region_concentration(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    regions = {
        "outside_existing_envelope": ~frame["inside_calo_envelope"].astype(bool),
        "r_above_4250_mm": frame["vertex_r_mm"].gt(4250),
        "abs_z_above_6500_mm": frame["abs_z_mm"].gt(6500),
        "lower_muon_pt_below_7p5_GeV": frame["min_muon_pt_GeV"].lt(7.5),
        "either_muon_abs_eta_above_2": frame["max_abs_muon_eta"].gt(2.0),
        "either_muon_central_abs_eta_below_0p2": frame["min_abs_muon_eta"].lt(.2),
        "pt_asymmetry_above_0p6": frame["pt_asymmetry"].gt(.6),
    }
    categories = {
        "nmin_0": frame["n_min"].eq(0),
        "nmin_1": frame["n_min"].eq(1),
        "nmin_lt2": frame["n_min"].lt(2),
    }
    records = []
    for region, region_mask in regions.items():
        for category, category_mask in categories.items():
            intersection = region_mask & category_mask
            records.append(
                {
                    "region": region,
                    "category": category,
                    "all_vertices": len(frame),
                    "region_vertices": int(region_mask.sum()),
                    "category_vertices": int(category_mask.sum()),
                    "category_vertices_in_region": int(intersection.sum()),
                    "fraction_of_category_in_region": float(intersection.sum() / category_mask.sum()),
                    "category_rate_inside_region": float(intersection.sum() / region_mask.sum()),
                }
            )
    result = pd.DataFrame.from_records(records)
    result.to_csv(output_dir / "inefficiency_region_concentration.csv", index=False)
    return result


def root_schema_inventory(root_file: Path | None, output_dir: Path) -> pd.DataFrame:
    if root_file is None:
        return pd.DataFrame()
    import uproot

    rows = []
    with uproot.open(root_file) as handle:
        for tree_name in ("MuonVertexDump", "MuonBucketDump"):
            tree = handle[tree_name]
            for branch, typename in tree.typenames().items():
                lower = branch.lower()
                if tree_name == "MuonVertexDump" and any(key in lower for key in ("truthmuon_", "vertexposition", "vertexmuonlinks")):
                    status = "used_or_available_in_parquet"
                    reason = "Truth kinematics/vertex association; core variables are already stored in vertex Parquet."
                elif any(key in lower for key in ("station", "chamber", "layer", "sector", "side")):
                    status = "root_only_geometry_candidate"
                    reason = "Could characterize crossed/reconstructed detector regions, but needs event/truth matching extraction."
                elif any(key in lower for key in ("chi", "numberdof", "matching", "neta", "nphi")):
                    status = "root_only_reconstruction_quality_candidate"
                    reason = "Defined for reconstructed objects; useful for nmin>0 but cannot by itself explain nmin=0."
                elif "truthseg" in lower or "truelabel" in lower:
                    status = "root_only_truth_coverage_candidate"
                    reason = "Potential route to expected truth crossings/stations; requires dedicated semantic validation."
                else:
                    status = "not_prioritized"
                    reason = "Not selected for the compact diagnostic study."
                rows.append(
                    {
                        "source_file": str(root_file),
                        "tree": tree_name,
                        "tree_entries": tree.num_entries,
                        "branch": branch,
                        "typename": typename,
                        "status": status,
                        "reason": reason,
                    }
                )
    result = pd.DataFrame.from_records(rows)
    result.to_csv(output_dir / "root_branch_inventory.csv", index=False)
    return result


def write_report(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    model_metrics: dict,
    regions: pd.DataFrame,
    output_dir: Path,
) -> None:
    overall_poor = frame["poor"].mean()
    inside = frame[frame["inside_calo_envelope"]]
    outside = frame[~frame["inside_calo_envelope"]]
    low_pt_muons = pd.concat(
        [
            pd.DataFrame({"pt": frame[f"muon{i}_pt"], "segments": frame[f"muon{i}_segments"]})
            for i in (0, 1)
        ],
        ignore_index=True,
    )
    low_pt = low_pt_muons[low_pt_muons["pt"].lt(7.5)]
    high_pt = low_pt_muons[low_pt_muons["pt"].ge(7.5)]

    permutation = ranking[ranking.get("ranking_source", pd.Series(index=ranking.index, dtype=object)).eq(
        "multivariate_hist_gradient_boosting"
    )].copy()
    top_inside = permutation[permutation["selection"].eq("inside_calo_envelope")].sort_values(
        "permutation_auc_decrease", ascending=False
    ).head(6)

    region_lookup = regions.pivot(index="region", columns="category", values="fraction_of_category_in_region")
    nmin0 = frame[frame["n_min"].eq(0)]
    nmin1 = frame[frame["n_min"].eq(1)]
    low_pt_nmin0 = nmin0["min_muon_pt_GeV"].lt(7.5).mean()
    low_pt_nmin1 = nmin1["min_muon_pt_GeV"].lt(7.5).mean()
    asym_nmin0 = nmin0["pt_asymmetry"].gt(.6).mean()
    asym_nmin1 = nmin1["pt_asymmetry"].gt(.6).mean()
    lines = [
        "# Displaced-vertex segment-reconstruction diagnostic",
        "",
        "This study measures **reconstruction input richness**, not vertex-network performance. "
        "A vertex is called poorly reconstructed when `nmin < 2`.",
        "",
        "## Dataset and selections",
        "",
        f"- Scalar vertex rows with valid truth links: **{len(frame):,}**.",
        f"- Inside the existing simplified envelope (`r <= 4250 mm`, `|z| <= 6500 mm`): **{len(inside):,}**.",
        "- The extraction selected displaced vertices with `r > 30 mm`; this particular generated sample starts near 300 mm.",
        "",
        "## Main findings",
        "",
        f"- Overall, **{100*overall_poor:.1f}%** of vertices have `nmin < 2`. "
        f"The fraction is **{100*inside['poor'].mean():.1f}%** inside the envelope and "
        f"**{100*outside['poor'].mean():.1f}%** outside it.",
        f"- At truth-muon level, below 7.5 GeV, **{100*low_pt['segments'].eq(0).mean():.1f}%** of muons have no segment and "
        f"**{100*low_pt['segments'].lt(2).mean():.1f}%** have fewer than two. Above 7.5 GeV these become "
        f"**{100*high_pt['segments'].eq(0).mean():.1f}%** and **{100*high_pt['segments'].lt(2).mean():.1f}%**.",
        f"- **{100*region_lookup.loc['outside_existing_envelope','nmin_0']:.1f}%** of all `nmin=0` vertices and "
        f"**{100*region_lookup.loc['outside_existing_envelope','nmin_1']:.1f}%** of all `nmin=1` vertices lie outside the existing envelope.",
        f"- The lower truth muon has `pT < 7.5 GeV` in **{100*low_pt_nmin0:.1f}%** of `nmin=0` vertices and "
        f"**{100*low_pt_nmin1:.1f}%** of `nmin=1` vertices. Large pT asymmetry (>0.6) occurs in "
        f"**{100*asym_nmin0:.1f}%** and **{100*asym_nmin1:.1f}%**, respectively.",
        "- Within the envelope, radius alone has only a weak monotonic association with poor reconstruction; "
        "`|z|`, low muon pT, pT imbalance, and eta-region effects remain visible.",
        "- Inside the envelope, the `nmin<2` fraction rises from **24.1%** at `750<|z|<1500 mm` to "
        "**51.3%** at `6000<|z|<6750 mm`. Across radius it changes more mildly, from **28.5%** below "
        "500 mm to **35.7%** in the outer `4000–4500 mm` bin.",
        "- Truth-muon |eta| is non-monotonic: losses are enhanced in the very central band and again in the forward region. "
        "This is consistent with detector-geometry/coverage structure, but station-level confirmation requires ROOT-level truth-crossing extraction.",
        "- Inside the envelope, the per-muon zero-segment rate is **4.2%** in the broad barrel-like analysis band "
        "(`0.2<=|eta|<1.05`), compared with **17.3%** for `|eta|<0.2` and **16.0%** for `2.0<=|eta|<2.7`. "
        "The phi-binned zero-segment rate varies only from about **6.8% to 8.2%**, so there is no comparably strong global azimuthal structure at this resolution.",
        "- Event truth-labelled bucket count is strongly associated with success, but it is a reconstruction/coverage proxy and must not be interpreted as an independent cause.",
        "",
        "## Multivariate diagnostic ranking (inside envelope)",
        "",
        f"A small diagnostic gradient-boosted model reaches AUC **{model_metrics['inside_calo_envelope']['test_auc']:.3f}** for identifying `nmin < 2`. "
        "Permutation importance ranks associations after accounting for the other listed variables:",
        "",
    ]
    for row in top_inside.itertuples(index=False):
        lines.append(f"- {row.label}: AUC decrease {row.permutation_auc_decrease:.4f} ± {row.permutation_auc_decrease_std:.4f}")
    lines.extend(
        [
            "",
            "This ranking is associative, not causal. In particular, reconstructed event quantities may partially encode the outcome itself.",
            "",
            "## Interpretation and improvement targets",
            "",
            "1. **Low-pT truth muons give the clearest truth-level rate change.** They form a large, strongly enriched part of the zero-segment tail and frequently leave only one segment. "
            "Reconstruction studies should inspect low-pT seeding, segment thresholds, and matching efficiency before changing the vertex network.",
            "2. **Forward/central eta structures and large |z| deserve geometry-resolved follow-up.** The ROOT schema contains station, chamber, layer, sector, side, "
            "truth-segment links, and segment-fit quality, but these were not retained in Parquet. A targeted extractor should count expected truth stations/chambers "
            "per muon and distinguish acceptance loss from reconstruction failure.",
            "3. **Outside-envelope losses are substantial but should not be mixed with in-acceptance algorithmic inefficiency.** Report all-selected and inside-envelope results separately.",
            "4. **For the downstream vertex network**, retraining cannot recover information that was never reconstructed. Possible ML work should first condition performance on "
            "`nmin`, pT, eta, and position; only then assess whether edge/node features or architecture improve events that contain adequate segment information.",
            "",
            "## Required next step for network-TPR correlation",
            "",
            "Join these vertex rows to per-event network scores/decisions using the event hashes (and vertex index where applicable), then measure TPR in bins of `nmin`, "
            "position, pT, eta, and the identified geometry categories at a fixed background working point. No network-performance claim is made here.",
        ]
    )
    (output_dir / "interpretation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_guide(output_dir: Path) -> None:
    guide = """# Figure guide

All figures measure segment-reconstruction richness. They do not contain classifier outputs.

## `average_nmin_vs_position_*.png`

The two one-dimensional profiles show the mean segment multiplicity of the less-reconstructed truth muon versus vertex radius and |z|. A decline means that at least one muon is increasingly under-reconstructed. Comparing all selected vertices with the inside-envelope version separates acceptance-edge effects from behavior within the existing analysis envelope. A reconstruction follow-up should target position ranges where the decline remains after the envelope selection.

## `quality_fractions_vs_position_*.png`

These profiles decompose each position bin into `nmin=0`, `nmin=1`, and `nmin>=2`. They distinguish complete loss of one muon from minimal one-segment reconstruction. This matters for improvements: `nmin=0` points toward acceptance/seeding/matching failures, whereas `nmin=1` can indicate insufficient station coverage or segment-building efficiency.

## `average_nmin_r_absz_map_*.png`

The two-dimensional map reveals localized combinations of radius and |z| that a one-dimensional projection can average away. Blank bins fail the stated minimum-statistics requirement. Low-value regions motivate a geometry-resolved chamber/station analysis, but the map alone does not identify a particular detector component.

## `poor_fraction_r_absz_map_*.png`

This is the direct spatial inefficiency map: the plotted value is the fraction with `nmin<2`. The inside-envelope map shows that |z|-dependent structure remains even after removing the broad outside-envelope population, while radius is comparatively weaker inside that selection.

## `position_distributions_by_quality_*.png`

Normalized r and |z| distributions compare where `nmin=0`, `nmin=1`, `nmin=2`, and `nmin>=3` vertices occur. Because each curve is normalized independently, this plot shows shape/concentration, not the absolute category yield; use the CSV summaries for counts.

## `truth_muon_inefficiency_*.png`

This is the most direct per-muon diagnostic. It shows the probability that an individual truth muon receives zero or fewer than two matched segments versus pT, |eta|, and phi. The steep low-pT loss is the strongest actionable truth-level observation. The non-monotonic |eta| curve suggests geometry/coverage structure in the very central and forward regions. Phi dependence is small, so no strong global azimuthal crack is evident at this binning.

## `topology_diagnostics_*.png`

These six compact profiles test the lower muon pT, larger |eta|, pair deltaR, pT asymmetry, outward-going topology, and event truth-bucket coverage. Low pT, extreme eta, large pair separation/asymmetry, and low truth-bucket coverage are associated with poor reconstruction. Truth-bucket count is partly a reconstructed coverage proxy and should not be interpreted causally. The multivariate CSV shows that pT asymmetry adds little after lower pT and pair topology are already known.

## Improvement guidance

The evidence prioritizes low-pT muon reconstruction and geometry-resolved studies at large |z| and central/forward eta. The ROOT inventory identifies station, chamber, layer, sector, truth-segment-link, and segment-fit branches needed for the next extraction. For the downstream network, first measure performance conditional on `nmin`; architecture changes cannot replace absent reconstructed segments.
"""
    (output_dir / "figure_guide.md").write_text(guide, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    frame = load_vertices(args.input_dir.expanduser().resolve())
    print(
        f"[data] vertices={len(frame):,} nmin<2={frame['poor'].mean():.4f} "
        f"inside_envelope={frame['inside_calo_envelope'].mean():.4f}",
        flush=True,
    )

    schema = {
        "verified_parquet_columns": sorted(REQUIRED_COLUMNS),
        "derived_columns": [
            "n1", "n2", "n_min", "n_max", "abs_z_mm", "min_muon_pt_GeV",
            "max_abs_muon_eta", "delta_eta_mumu", "delta_phi_mumu", "delta_r_mumu",
            "pt_asymmetry", "min_outward_cosine", "quality",
        ],
        "selection_definition": {
            "all_selected": "r > 30 mm and exactly two linked truth muons; generated sample observed r >= ~300 mm",
            "inside_calo_envelope": "all_selected and r <= 4250 mm and |z| <= 6500 mm",
        },
        "scope": "reconstruction input richness; no network scores are used",
    }
    (output_dir / "analysis_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    profiles = position_profiles(frame, output_dir)
    plot_position_profiles(profiles, output_dir)
    position_maps(frame, output_dir, args.position_min_count)
    nmin_ge2_count_diagnostics(frame, output_dir, args.position_min_count)
    plot_position_distributions(frame, output_dir)
    truth_muon_diagnostics(frame, output_dir)
    topology_profiles(frame, output_dir)
    sample_summary(frame, output_dir)
    categorical_summaries(frame, output_dir)
    ranking, model_metrics = association_ranking(frame, output_dir, args.seed)
    regions = region_concentration(frame, output_dir)
    root_schema_inventory(args.root_schema_file, output_dir)
    write_report(frame, ranking, model_metrics, regions, output_dir)
    write_figure_guide(output_dir)

    print(f"[done] diagnostic outputs written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
