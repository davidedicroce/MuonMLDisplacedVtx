#!/usr/bin/env python3
"""Create supervisor-ready truth/reconstruction completeness plots and tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


CATEGORIES = ("2/2", "1/2", "0/2")
COLORS = {"2/2": "#2A9D8F", "1/2": "#F4A261", "0/2": "#E76F51"}
SAMPLE_RE = re.compile(r"_m(?P<mass>\d+)_ctau(?P<ctau>\d+)_mu200_part\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Scalar displaced-muon signal sample")
    return parser.parse_args()


def sample_from_summary(summary: dict) -> tuple[int, int, str]:
    name = Path(summary["source_file"]).name
    match = SAMPLE_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot parse mass/ctau from {name}")
    mass = int(match.group("mass"))
    ctau = int(match.group("ctau"))
    return mass, ctau, f"$m_a={mass}$ GeV, $c\\tau={ctau}$ mm"


def load_summary_table(input_dir: Path) -> pd.DataFrame:
    grouped: dict[tuple[int, int, str], dict[str, int]] = {}
    paths = sorted(input_dir.glob("MuonBucketDump_vertex_a_mumu_*.summary.json"))
    if not paths:
        raise FileNotFoundError(f"No per-file summary JSON files in {input_dir}")

    for path in paths:
        summary = json.loads(path.read_text())
        key = sample_from_summary(summary)
        row = grouped.setdefault(
            key,
            {
                "files": 0,
                "vertices": 0,
                "calo_envelope_vertices": 0,
                **{f"all_{cat}": 0 for cat in CATEGORIES},
                **{f"envelope_{cat}": 0 for cat in CATEGORIES},
            },
        )
        row["files"] += 1
        row["vertices"] += int(summary["displaced_vertices"])
        row["calo_envelope_vertices"] += int(
            summary["calo_envelope_displaced_vertices"]
        )
        for category in CATEGORIES:
            row[f"all_{category}"] += int(summary["category_counts"][category])
            row[f"envelope_{category}"] += int(
                summary["calo_envelope_category_counts"][category]
            )

    records = []
    for (mass, ctau, label), values in grouped.items():
        record = {"mass_GeV": mass, "ctau_mm": ctau, "sample": label, **values}
        for prefix, denominator in (
            ("all", values["vertices"]),
            ("envelope", values["calo_envelope_vertices"]),
        ):
            for category in CATEGORIES:
                record[f"{prefix}_{category}_fraction"] = (
                    values[f"{prefix}_{category}"] / denominator
                    if denominator
                    else np.nan
                )
        records.append(record)
    return pd.DataFrame(records).sort_values(["mass_GeV", "ctau_mm"])


def add_stacked_bars(
    ax: plt.Axes,
    labels: list[str],
    fractions: dict[str, np.ndarray],
    *,
    annotate: bool,
) -> None:
    left = np.zeros(len(labels), dtype=float)
    y = np.arange(len(labels))
    for category in CATEGORIES:
        values = np.asarray(fractions[category], dtype=float)
        ax.barh(
            y,
            100.0 * values,
            left=100.0 * left,
            color=COLORS[category],
            label=category,
            height=0.68,
            edgecolor="white",
            linewidth=0.7,
        )
        if annotate:
            for index, value in enumerate(values):
                if value >= 0.035:
                    ax.text(
                        100.0 * (left[index] + value / 2),
                        index,
                        f"{100 * value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#17202A",
                    )
        left += values
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fraction of displaced vertices [%]")
    ax.grid(axis="x", color="#D5D8DC", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def plot_overview(table: pd.DataFrame, output_dir: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 7.8), sharey=True)
    labels = [
        f"$m_a={row.mass_GeV}$ GeV, $c\\tau={row.ctau_mm}$ mm"
        for row in table.itertuples()
    ]
    for ax, prefix, panel_title in (
        (axes[0], "all", "All generated displaced vertices"),
        (
            axes[1],
            "envelope",
            "Vertices inside the converter calorimeter envelope",
        ),
    ):
        add_stacked_bars(
            ax,
            labels,
            {
                category: table[f"{prefix}_{category}_fraction"].to_numpy()
                for category in CATEGORIES
            },
            annotate=True,
        )
        ax.set_title(panel_title, fontsize=12, weight="bold")
        ax.invert_yaxis()

    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        title="Truth muons with ≥1 matched segment",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncols=3,
        frameon=False,
    )
    figure.suptitle(
        f"{title}: truth-muon segment reconstruction completeness",
        fontsize=15,
        weight="bold",
        y=0.99,
    )
    figure.subplots_adjust(
        left=0.20, right=0.985, top=0.82, bottom=0.10, wspace=0.055
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"truth_reco_completeness_by_sample.{suffix}", dpi=220)
    plt.close(figure)


def plot_overall(table: pd.DataFrame, output_dir: Path, title: str) -> None:
    totals = {}
    for prefix, denominator_col in (
        ("all", "vertices"),
        ("envelope", "calo_envelope_vertices"),
    ):
        denominator = int(table[denominator_col].sum())
        totals[prefix] = {
            category: int(table[f"{prefix}_{category}"].sum()) / denominator
            for category in CATEGORIES
        }

    figure, ax = plt.subplots(figsize=(10.5, 4.4))
    add_stacked_bars(
        ax,
        ["All displaced vertices", "Inside calorimeter envelope"],
        {
            category: np.array(
                [totals["all"][category], totals["envelope"][category]]
            )
            for category in CATEGORIES
        },
        annotate=True,
    )
    ax.invert_yaxis()
    ax.set_title(
        f"{title}: overall reconstruction completeness",
        fontsize=14,
        weight="bold",
    )
    ax.legend(
        title="Truth muons with ≥1 matched segment",
        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.43),
        frameon=False,
    )
    figure.subplots_adjust(left=0.22, right=0.98, top=0.85, bottom=0.31)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"truth_reco_completeness_overall.{suffix}", dpi=220)
    plt.close(figure)


def plot_position_map(input_dir: Path, output_dir: Path, title: str) -> None:
    paths = sorted(input_dir.glob("*.vertices.parquet"))
    frames = [
        pd.read_parquet(
            path,
            columns=["vertex_r_mm", "vertex_z_mm", "completeness_category"],
        )
        for path in paths
    ]
    vertices = pd.concat(frames, ignore_index=True)
    r_edges = np.linspace(0, 5000, 26)
    z_edges = np.linspace(0, 7500, 26)
    complete = (vertices["completeness_category"] == "2/2").to_numpy(float)
    radius = vertices["vertex_r_mm"].to_numpy(float)
    abs_z = np.abs(vertices["vertex_z_mm"].to_numpy(float))
    total, _, _ = np.histogram2d(abs_z, radius, bins=(z_edges, r_edges))
    passed, _, _ = np.histogram2d(
        abs_z, radius, bins=(z_edges, r_edges), weights=complete
    )
    fraction = np.divide(
        passed,
        total,
        out=np.full_like(passed, np.nan, dtype=float),
        where=total >= 100,
    )

    figure, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        r_edges,
        z_edges,
        100.0 * fraction,
        cmap="viridis",
        vmin=40,
        vmax=100,
        shading="flat",
    )
    colorbar = figure.colorbar(mesh, ax=ax)
    colorbar.set_label("Vertices with both truth muons reconstructed (2/2) [%]")
    ax.add_patch(
        Rectangle(
            (0, 0),
            4250,
            6500,
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
        f"{title}: 2/2 reconstruction fraction across decay position",
        fontsize=14,
        weight="bold",
    )
    ax.legend(loc="lower right", frameon=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"truth_reco_completeness_position_map.{suffix}", dpi=220)
    plt.close(figure)


def write_tables(table: pd.DataFrame, output_dir: Path) -> None:
    export = table.copy()
    for prefix in ("all", "envelope"):
        for category in CATEGORIES:
            export[f"{prefix}_{category}_percent"] = (
                100.0 * export[f"{prefix}_{category}_fraction"]
            )
    export.to_csv(output_dir / "truth_reco_completeness_by_sample.csv", index=False)

    display = export[
        [
            "mass_GeV",
            "ctau_mm",
            "files",
            "vertices",
            "all_2/2_percent",
            "all_1/2_percent",
            "all_0/2_percent",
            "calo_envelope_vertices",
            "envelope_2/2_percent",
            "envelope_1/2_percent",
            "envelope_0/2_percent",
        ]
    ].copy()
    display.columns = [
        "Mass [GeV]",
        "cτ [mm]",
        "Files",
        "All vertices",
        "All 2/2 [%]",
        "All 1/2 [%]",
        "All 0/2 [%]",
        "Inside calo envelope",
        "Envelope 2/2 [%]",
        "Envelope 1/2 [%]",
        "Envelope 0/2 [%]",
    ]
    percent_columns = [column for column in display if "[%]" in column]
    display[percent_columns] = display[percent_columns].round(2)
    headers = [str(column) for column in display.columns]
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        markdown_lines.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )
    (output_dir / "truth_reco_completeness_by_sample.md").write_text(
        "\n".join(markdown_lines) + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = load_summary_table(args.input_dir)
    write_tables(table, args.output_dir)
    plot_overall(table, args.output_dir, args.title)
    plot_overview(table, args.output_dir, args.title)
    plot_position_map(args.input_dir, args.output_dir, args.title)
    print(f"Created figures and tables in {args.output_dir}")


if __name__ == "__main__":
    main()
