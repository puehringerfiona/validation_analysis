from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_sensitivity_analysis as base


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "stats" / "sensitivity_analysis" / "aot550_parent_compare"
DEFAULT_PLOT_ROOT = Path(__file__).resolve().parent / "aot550_parent_compare" / "combined_six_panel_heatmaps"
FILTER_VARIANT = "hard_filter_retain"

SOURCES = [
    ("ACOLITE", "parent", "acolite", DEFAULT_RESULTS_ROOT / "acolite" / "parent" / FILTER_VARIANT / "ACOLITE"),
    ("ACOLITE", "aot550", "acolite", DEFAULT_RESULTS_ROOT / "acolite" / "aot550" / FILTER_VARIANT / "ACOLITE"),
    (
        "ACOLITE+T-Mart",
        "parent",
        "acolite_tmart",
        DEFAULT_RESULTS_ROOT / "acolite_tmart" / "parent" / FILTER_VARIANT / "ACOLITE_TMart",
    ),
    (
        "ACOLITE+T-Mart",
        "aot550",
        "acolite_tmart",
        DEFAULT_RESULTS_ROOT / "acolite_tmart" / "aot550" / FILTER_VARIANT / "ACOLITE_TMart",
    ),
    (
        "ACOLITE+RAdCor",
        "parent",
        "acolite_radcor",
        DEFAULT_RESULTS_ROOT / "parent" / FILTER_VARIANT / "ACOLITE_RAdCor",
    ),
    (
        "ACOLITE+RAdCor",
        "aot550",
        "acolite_radcor",
        DEFAULT_RESULTS_ROOT / "aot_550" / FILTER_VARIANT / "ACOLITE_RAdCor",
    ),
]


def load_all_matchups() -> pd.DataFrame:
    frames = []
    for processor_label, variant, processor, root in SOURCES:
        if not root.exists():
            continue
        frame = base.load_matchups(root, processor, with_context=True)
        if frame.empty:
            continue
        frame["processor_label"] = processor_label
        frame["variant"] = variant
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def matrix_for(subset: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    subset = subset.dropna(subset=["msa_pair_percent", "insitu_timestamp_utc", "distance_to_shore_km", "wavelength_nm"])
    if subset.empty:
        return pd.DataFrame(), np.array([])
    ordered_points = (
        subset[["row_index", "insitu_timestamp_utc", "distance_to_shore_km"]]
        .drop_duplicates()
        .sort_values(["insitu_timestamp_utc", "row_index"])
        .reset_index(drop=True)
    )
    ordered_points["measurement_order"] = ordered_points.index
    subset = subset.merge(ordered_points[["row_index", "measurement_order"]], on="row_index", how="left")
    matrix = subset.pivot_table(index="measurement_order", columns="wavelength_nm", values="msa_pair_percent", aggfunc="median")
    distances = ordered_points.set_index("measurement_order").reindex(matrix.index)["distance_to_shore_km"].to_numpy(float)
    return matrix, distances


def save_combined_heatmaps(matchups: pd.DataFrame, output_root: Path, complete_only: bool = False) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    group_columns = ["campaign", "sensor", "selected_transects", "transect_nr"]
    columns = ["ACOLITE", "ACOLITE+T-Mart", "ACOLITE+RAdCor"]
    rows = ["parent", "aot550"]
    for keys, group in matchups.groupby(group_columns, dropna=False):
        campaign, sensor, selected_transects, transect_nr = keys
        panel_data = {}
        values = []
        for variant in rows:
            for processor_label in columns:
                subset = group[(group["variant"].eq(variant)) & (group["processor_label"].eq(processor_label))]
                matrix, distances = matrix_for(subset)
                panel_data[(variant, processor_label)] = (matrix, distances)
                if not matrix.empty:
                    finite = matrix.to_numpy(dtype=float)
                    finite = finite[np.isfinite(finite)]
                    if finite.size:
                        values.append(finite)
        if complete_only and any(panel_data[(variant, processor_label)][0].empty for variant in rows for processor_label in columns):
            continue
        if not values:
            continue
        all_values = np.concatenate(values)
        vmax = max(1.0, min(100.0, float(np.nanpercentile(all_values, 98))))
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), sharex=False, sharey=False)
        last_image = None
        for row_index, variant in enumerate(rows):
            for col_index, processor_label in enumerate(columns):
                ax = axes[row_index, col_index]
                matrix, distances = panel_data[(variant, processor_label)]
                ax.set_title(f"{processor_label} {variant}", fontsize=10)
                if matrix.empty:
                    ax.text(0.5, 0.5, "No product", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                last_image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
                ax.set_xticks(np.arange(len(matrix.columns)))
                ax.set_xticklabels([int(wl) for wl in matrix.columns], rotation=90, fontsize=8)
                y_positions = np.linspace(0, len(matrix.index) - 1, min(6, len(matrix.index)), dtype=int)
                ax.set_yticks(y_positions)
                ax.set_yticklabels([f"{distances[pos]:.2f}" for pos in y_positions], fontsize=8)
                if col_index == 0:
                    ax.set_ylabel("Distance to shore (km)")
                if row_index == 1:
                    ax.set_xlabel("Wavelength (nm)")
        title_transect = f"{float(transect_nr):g}" if pd.notna(transect_nr) else str(selected_transects)
        fig.suptitle(f"{campaign} {sensor} transect {title_transect} MSA (%)", y=0.98)
        if last_image is not None:
            fig.colorbar(last_image, ax=axes.ravel().tolist(), label="MSA (%)", shrink=0.84)
        out = output_root / f"{base.safe_filename(campaign)}_{base.safe_filename(sensor)}_t{base.safe_filename(title_transect)}_combined_msa_heatmap.png"
        fig.savefig(out, dpi=240, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create six-panel parent/aot550 MSA heatmaps for ACOLITE-family products.")
    parser.add_argument("--plot-root", default=str(DEFAULT_PLOT_ROOT))
    parser.add_argument("--complete-only", action="store_true", help="Only write figures where all six panels have data.")
    args = parser.parse_args()
    matchups = load_all_matchups()
    if matchups.empty:
        raise SystemExit("No matchup outputs found for combined heatmaps.")
    outputs = save_combined_heatmaps(matchups, Path(args.plot_root), complete_only=args.complete_only)
    print(f"Wrote {len(outputs)} combined heatmaps under {args.plot_root}")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
