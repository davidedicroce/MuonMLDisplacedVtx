#!/usr/bin/env python3
"""Study reconstructed-segment multiplicity for the two truth muons per vertex.

This analysis consumes the vertex-level Parquet files produced by
analyze_DisplacedVertex_truth_reco.py.  It studies reconstruction input
richness only; it does not read or evaluate any classifier output.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


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
    "muon0_index",
    "muon1_index",
    "muon0_segments",
    "muon1_segments",
}
SELECTIONS = (
    ("all_selected", "All selected displaced vertices"),
    ("inside_calo_envelope", "Inside converter calorimeter envelope"),
)
NMIN_BIN_ORDER = ("0", "1", "2", "3", "4+")
NMIN_BIN_COLORS = {
    "0": "#E76F51",
    "1": "#F4A261",
    "2": "#E9C46A",
    "3": "#55A868",
    "4+": "#2A9D8F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument("--examples-per-case", type=int, default=3)
    parser.add_argument("--position-min-count", type=int, default=100)
    parser.add_argument("--envelope-r-max-mm", type=float, default=4250.0)
    parser.add_argument("--envelope-z-max-mm", type=float, default=6500.0)
    return parser.parse_args()


def parse_sample(source_file: str) -> tuple[int, int, str]:
    match = SAMPLE_RE.search(Path(source_file).name)
    if match is None:
        raise ValueError(f"Cannot parse mass and ctau from {source_file}")
    mass = int(match.group("mass"))
    ctau = int(match.group("ctau"))
    return mass, ctau, f"m{mass}_ctau{ctau}"


def load_vertices(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("MuonBucketDump_vertex_a_mumu_*.vertices.parquet"))
    if not paths:
        raise FileNotFoundError(f"No scalar vertex Parquet files found in {input_dir}")

    print(f"[schema] checking {len(paths)} vertex-level Parquet files")
    frames = []
    columns = sorted(REQUIRED_COLUMNS)
    for path in paths:
        frame = pd.read_parquet(path)
        available = set(frame.columns)
        missing = REQUIRED_COLUMNS - available
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frames.append(frame[columns])

    vertices = pd.concat(frames, ignore_index=True)
    parsed = vertices["source_file"].map(parse_sample)
    vertices["mass_GeV"] = parsed.map(lambda item: item[0])
    vertices["ctau_mm"] = parsed.map(lambda item: item[1])
    vertices["sample"] = parsed.map(lambda item: item[2])

    vertices["n1"] = vertices["muon0_segments"].astype(np.int64)
    vertices["n2"] = vertices["muon1_segments"].astype(np.int64)
    vertices["n_min"] = vertices[["n1", "n2"]].min(axis=1)
    vertices["n_max"] = vertices[["n1", "n2"]].max(axis=1)
    vertices["n_total"] = vertices["n1"] + vertices["n2"]
    vertices["n_imbalance"] = (vertices["n1"] - vertices["n2"]).abs()
    vertices["n_min_bin"] = pd.Categorical(
        np.select(
            [
                vertices["n_min"] == 0,
                vertices["n_min"] == 1,
                vertices["n_min"] == 2,
                vertices["n_min"] == 3,
            ],
            ["0", "1", "2", "3"],
            default="4+",
        ),
        categories=NMIN_BIN_ORDER,
        ordered=True,
    )
    return vertices


def selection_frame(vertices: pd.DataFrame, selection: str) -> pd.DataFrame:
    if selection == "all_selected":
        return vertices
    if selection == "inside_calo_envelope":
        return vertices[vertices["inside_calo_envelope"]]
    raise ValueError(f"Unknown selection: {selection}")


def add_group_metadata(
    frame: pd.DataFrame,
    *,
    selection: str,
    mass: int | None = None,
    ctau: int | None = None,
    sample: str = "all_scalar_samples",
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "sample", sample)
    result.insert(0, "ctau_mm", ctau)
    result.insert(0, "mass_GeV", mass)
    result.insert(0, "selection", selection)
    return result


def exact_pair_table(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame.groupby(["n_min", "n_max"], observed=True)
        .size()
        .rename("vertices")
        .reset_index()
        .sort_values(["vertices", "n_min", "n_max"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    counts["rank"] = np.arange(1, len(counts) + 1)
    counts["fraction"] = counts["vertices"] / len(frame)
    counts["percent"] = 100.0 * counts["fraction"]
    return counts[["rank", "n_min", "n_max", "vertices", "fraction", "percent"]]


def nmin_table(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame["n_min_bin"]
        .value_counts(sort=False)
        .reindex(NMIN_BIN_ORDER, fill_value=0)
        .rename_axis("n_min_bin")
        .rename("vertices")
        .reset_index()
    )
    counts["fraction"] = counts["vertices"] / len(frame)
    counts["percent"] = 100.0 * counts["fraction"]
    return counts


def richness_table(frame: pd.DataFrame) -> pd.DataFrame:
    criteria = (
        ("both_missing", "n1 = 0 and n2 = 0", frame["n_min"].eq(0) & frame["n_max"].eq(0)),
        ("one_missing", "n_min = 0 and n_max > 0", frame["n_min"].eq(0) & frame["n_max"].gt(0)),
        (
            "both_present_minimally",
            "n_min = 1",
            frame["n_min"].eq(1),
        ),
        (
            "both_at_least_two",
            "n_min >= 2",
            frame["n_min"].ge(2),
        ),
        (
            "both_at_least_three",
            "n_min >= 3 (subset of n_min >= 2)",
            frame["n_min"].ge(3),
        ),
    )
    rows = []
    for name, definition, mask in criteria:
        count = int(mask.sum())
        rows.append(
            {
                "classification": name,
                "definition": definition,
                "vertices": count,
                "fraction": count / len(frame),
                "percent": 100.0 * count / len(frame),
            }
        )
    return pd.DataFrame(rows)


def build_tables(vertices: pd.DataFrame, output_dir: Path, top_pairs: int) -> None:
    overall_pairs = []
    sample_pairs = []
    overall_nmin = []
    sample_nmin = []
    overall_richness = []
    sample_richness = []

    sample_groups = list(
        vertices.groupby(["mass_GeV", "ctau_mm", "sample"], sort=True)
    )
    for selection, _ in SELECTIONS:
        selected = selection_frame(vertices, selection)
        overall_pairs.append(
            add_group_metadata(exact_pair_table(selected), selection=selection)
        )
        overall_nmin.append(add_group_metadata(nmin_table(selected), selection=selection))
        overall_richness.append(
            add_group_metadata(richness_table(selected), selection=selection)
        )

        for (mass, ctau, sample), group in sample_groups:
            selected_group = selection_frame(group, selection)
            metadata = {
                "selection": selection,
                "mass": int(mass),
                "ctau": int(ctau),
                "sample": sample,
            }
            sample_pairs.append(
                add_group_metadata(exact_pair_table(selected_group), **metadata)
            )
            sample_nmin.append(
                add_group_metadata(nmin_table(selected_group), **metadata)
            )
            sample_richness.append(
                add_group_metadata(richness_table(selected_group), **metadata)
            )

    overall_pairs_frame = pd.concat(overall_pairs, ignore_index=True)
    sample_pairs_frame = pd.concat(sample_pairs, ignore_index=True)
    overall_pairs_frame.to_csv(output_dir / "exact_segment_pairs_overall.csv", index=False)
    sample_pairs_frame.to_csv(output_dir / "exact_segment_pairs_by_sample.csv", index=False)
    overall_pairs_frame[overall_pairs_frame["rank"] <= top_pairs].to_csv(
        output_dir / "top_exact_segment_pairs_overall.csv", index=False
    )
    sample_pairs_frame[sample_pairs_frame["rank"] <= top_pairs].to_csv(
        output_dir / "top_exact_segment_pairs_by_sample.csv", index=False
    )
    pd.concat(overall_nmin, ignore_index=True).to_csv(
        output_dir / "n_min_distribution_overall.csv", index=False
    )
    pd.concat(sample_nmin, ignore_index=True).to_csv(
        output_dir / "n_min_distribution_by_sample.csv", index=False
    )
    pd.concat(overall_richness, ignore_index=True).to_csv(
        output_dir / "segment_richness_summary_overall.csv", index=False
    )
    pd.concat(sample_richness, ignore_index=True).to_csv(
        output_dir / "segment_richness_summary_by_sample.csv", index=False
    )


def pair_matrix(
    frame: pd.DataFrame,
    max_n_min: int,
    max_n_max: int,
) -> np.ndarray:
    matrix = np.zeros((max_n_min + 1, max_n_max + 1), dtype=np.int64)
    pairs = frame.groupby(["n_min", "n_max"], observed=True).size()
    for (n_min, n_max), count in pairs.items():
        matrix[int(n_min), int(n_max)] = int(count)
    return matrix


def draw_pair_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    norm: LogNorm,
    annotate: bool,
):
    masked = np.ma.masked_where(matrix <= 0, matrix)
    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=norm,
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("$n_{max}$")
    ax.set_ylabel("$n_{min}$")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(-0.5, matrix.shape[0] - 0.5)
    if annotate:
        threshold = max(1, int(matrix.sum() * 0.002))
        for n_min, n_max in zip(*np.where(matrix >= threshold)):
            ax.text(
                n_max,
                n_min,
                f"{matrix[n_min, n_max]:,}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[n_min, n_max] < matrix.max() / 8 else "black",
            )
    return image


def plot_pair_heatmaps(vertices: pd.DataFrame, output_dir: Path) -> None:
    max_n_min = int(vertices["n_min"].max())
    max_n_max = int(vertices["n_max"].max())
    global_norm = LogNorm(vmin=1, vmax=int(vertices.groupby(["n_min", "n_max"]).size().max()))

    for selection, selection_label in SELECTIONS:
        selected = selection_frame(vertices, selection)
        matrix = pair_matrix(selected, max_n_min, max_n_max)
        figure, ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
        image = draw_pair_heatmap(
            ax,
            matrix,
            title=f"{selection_label}: exact truth-muon segment-count pairs",
            norm=global_norm,
            annotate=True,
        )
        colorbar = figure.colorbar(image, ax=ax)
        colorbar.set_label("Number of displaced vertices (log scale)")
        figure.savefig(
            output_dir / f"segment_pair_heatmap_overall_{selection}.png", dpi=220
        )
        plt.close(figure)

    for selection, selection_label in SELECTIONS:
        figure, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
        image = None
        for ax, ((mass, ctau, _), group) in zip(
            axes.flat,
            vertices.groupby(["mass_GeV", "ctau_mm", "sample"], sort=True),
        ):
            selected = selection_frame(group, selection)
            matrix = pair_matrix(selected, max_n_min, max_n_max)
            image = draw_pair_heatmap(
                ax,
                matrix,
                title=f"$m_a={mass}$ GeV, $c\\tau={ctau}$ mm",
                norm=global_norm,
                annotate=False,
            )
        figure.suptitle(
            f"{selection_label}: exact segment-count pairs by scalar sample",
            fontsize=16,
            weight="bold",
        )
        colorbar = figure.colorbar(image, ax=axes, shrink=0.82)
        colorbar.set_label("Number of displaced vertices (common log scale)")
        figure.savefig(
            output_dir / f"segment_pair_heatmaps_by_sample_{selection}.png", dpi=200
        )
        plt.close(figure)


def plot_nmin(vertices: pd.DataFrame, output_dir: Path) -> None:
    width = 0.36
    x = np.arange(len(NMIN_BIN_ORDER))
    figure, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for index, (selection, selection_label) in enumerate(SELECTIONS):
        selected = selection_frame(vertices, selection)
        table = nmin_table(selected)
        values = table["percent"].to_numpy()
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=selection_label,
            color="#457B9D" if index == 0 else "#2A9D8F",
        )
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)
    ax.set_xticks(x, NMIN_BIN_ORDER)
    ax.set_xlabel("$n_{min}=\\min(n_1,n_2)$")
    ax.set_ylabel("Fraction of displaced vertices [%]")
    ax.set_title(
        "Minimum matched-segment multiplicity across the two truth muons",
        fontsize=14,
        weight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    figure.savefig(output_dir / "n_min_distribution_overall.png", dpi=220)
    plt.close(figure)

    for selection, selection_label in SELECTIONS:
        records = []
        for (mass, ctau, _), group in vertices.groupby(
            ["mass_GeV", "ctau_mm", "sample"], sort=True
        ):
            table = nmin_table(selection_frame(group, selection))
            records.append((mass, ctau, table.set_index("n_min_bin")["percent"]))

        figure, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
        bottom = np.zeros(len(records))
        y = np.arange(len(records))
        labels = [f"$m_a={mass}$ GeV, $c\\tau={ctau}$ mm" for mass, ctau, _ in records]
        for category in NMIN_BIN_ORDER:
            values = np.array([series.loc[category] for _, _, series in records])
            ax.barh(
                y,
                values,
                left=bottom,
                color=NMIN_BIN_COLORS[category],
                label=category,
                edgecolor="white",
                linewidth=0.6,
            )
            for row_index, value in enumerate(values):
                if value >= 5:
                    ax.text(
                        bottom[row_index] + value / 2,
                        row_index,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            bottom += values
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Fraction of displaced vertices [%]")
        ax.set_title(
            f"{selection_label}: $n_{{min}}$ distribution by scalar sample",
            fontsize=14,
            weight="bold",
        )
        ax.legend(
            title="$n_{min}$",
            ncols=5,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
        )
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        figure.savefig(
            output_dir / f"n_min_distribution_by_sample_{selection}.png", dpi=220
        )
        plt.close(figure)


def plot_position_map(
    vertices: pd.DataFrame,
    output_dir: Path,
    *,
    min_count: int,
    envelope_r_max_mm: float,
    envelope_z_max_mm: float,
) -> None:
    r_edges = np.linspace(0, 5000, 26)
    z_edges = np.linspace(0, 7500, 26)
    radius = vertices["vertex_r_mm"].to_numpy(float)
    abs_z = np.abs(vertices["vertex_z_mm"].to_numpy(float))
    passed_values = vertices["n_min"].ge(2).to_numpy(float)
    total, _, _ = np.histogram2d(abs_z, radius, bins=(z_edges, r_edges))
    passed, _, _ = np.histogram2d(
        abs_z, radius, bins=(z_edges, r_edges), weights=passed_values
    )
    fraction = np.divide(
        passed,
        total,
        out=np.full_like(passed, np.nan, dtype=float),
        where=total >= min_count,
    )

    figure, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        r_edges,
        z_edges,
        100.0 * fraction,
        cmap="viridis",
        vmin=0,
        vmax=100,
        shading="flat",
    )
    colorbar = figure.colorbar(mesh, ax=ax)
    colorbar.set_label("Vertices with $n_{min}\\geq2$ [%]")
    ax.add_patch(
        Rectangle(
            (0, 0),
            envelope_r_max_mm,
            envelope_z_max_mm,
            fill=False,
            linestyle="--",
            linewidth=1.8,
            edgecolor="white",
            label="Converter calorimeter envelope",
        )
    )
    ax.set_xlabel("Displaced-vertex radius $r$ [mm]")
    ax.set_ylabel("Displaced-vertex $|z|$ [mm]")
    ax.set_title(
        "Fraction with at least two matched segments for each truth muon",
        fontsize=14,
        weight="bold",
    )
    ax.legend(loc="lower right")
    figure.savefig(output_dir / "n_min_ge2_position_map.png", dpi=220)
    plt.close(figure)


def manual_examples(
    vertices: pd.DataFrame,
    output_dir: Path,
    examples_per_case: int,
) -> None:
    cases = (
        ("n1_n2_0_0", vertices["n1"].eq(0) & vertices["n2"].eq(0)),
        ("n1_n2_0_1", vertices["n1"].eq(0) & vertices["n2"].eq(1)),
        ("n1_n2_1_1", vertices["n1"].eq(1) & vertices["n2"].eq(1)),
        ("n1_n2_1_3", vertices["n1"].eq(1) & vertices["n2"].eq(3)),
        ("both_ge3", vertices["n1"].ge(3) & vertices["n2"].ge(3)),
    )
    columns = [
        "source_file",
        "sample",
        "mass_GeV",
        "ctau_mm",
        "event_hash0",
        "event_hash1",
        "vertex_index",
        "muon0_index",
        "muon1_index",
        "n1",
        "n2",
        "n_min",
        "n_max",
        "n_total",
        "n_imbalance",
        "vertex_x_mm",
        "vertex_y_mm",
        "vertex_z_mm",
        "vertex_r_mm",
        "inside_calo_envelope",
    ]
    examples = []
    for case, mask in cases:
        candidates = vertices.loc[mask, columns].sort_values(
            ["mass_GeV", "ctau_mm", "event_hash0", "event_hash1", "vertex_index"]
        )
        # Prefer examples from different signal configurations so the manual
        # checks are not all taken from the first file in lexical order.
        selected = (
            candidates.groupby("sample", sort=False)
            .head(1)
            .head(examples_per_case)
            .copy()
        )
        selected.insert(0, "requested_case", case)
        examples.append(selected)
    result = pd.concat(examples, ignore_index=True)
    result.to_csv(output_dir / "manual_examples.csv", index=False)

    lines = [
        "Manual segment-multiplicity examples",
        "This reports reconstruction input richness, not classifier performance.",
        "",
    ]
    for row in result.itertuples(index=False):
        line = (
            f"[{row.requested_case}] sample={row.sample} "
            f"event_hash=({row.event_hash0}, {row.event_hash1}) "
            f"vertex_index={row.vertex_index} "
            f"truth_muons=({row.muon0_index}, {row.muon1_index}) "
            f"segments=({row.n1}, {row.n2}) "
            f"position_mm=(x={row.vertex_x_mm:.2f}, y={row.vertex_y_mm:.2f}, "
            f"z={row.vertex_z_mm:.2f}, r={row.vertex_r_mm:.2f}) "
            f"inside_calo_envelope={row.inside_calo_envelope}"
        )
        lines.append(line)
        print(line)
    (output_dir / "manual_examples.txt").write_text("\n".join(lines) + "\n")


def write_dataset_summary(vertices: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for selection, selection_label in SELECTIONS:
        selected = selection_frame(vertices, selection)
        rows.append(
            {
                "selection": selection,
                "selection_label": selection_label,
                "vertices": len(selected),
                "n_min_min": int(selected["n_min"].min()),
                "n_min_max": int(selected["n_min"].max()),
                "n_max_min": int(selected["n_max"].min()),
                "n_max_max": int(selected["n_max"].max()),
                "mean_n_min": float(selected["n_min"].mean()),
                "mean_n_max": float(selected["n_max"].mean()),
                "mean_n_total": float(selected["n_total"].mean()),
                "mean_n_imbalance": float(selected["n_imbalance"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "dataset_segment_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vertices = load_vertices(args.input_dir)
    print("[schema] verified columns:")
    for column in sorted(REQUIRED_COLUMNS):
        print(f"  {column}")
    print(
        f"[data] vertices={len(vertices):,} "
        f"n_min=[{vertices['n_min'].min()}, {vertices['n_min'].max()}] "
        f"n_max=[{vertices['n_max'].min()}, {vertices['n_max'].max()}]"
    )

    build_tables(vertices, args.output_dir, args.top_pairs)
    write_dataset_summary(vertices, args.output_dir)
    plot_pair_heatmaps(vertices, args.output_dir)
    plot_nmin(vertices, args.output_dir)
    plot_position_map(
        vertices,
        args.output_dir,
        min_count=args.position_min_count,
        envelope_r_max_mm=args.envelope_r_max_mm,
        envelope_z_max_mm=args.envelope_z_max_mm,
    )
    manual_examples(vertices, args.output_dir, args.examples_per_case)
    print(f"[done] wrote segment-multiplicity results to {args.output_dir}")


if __name__ == "__main__":
    main()
