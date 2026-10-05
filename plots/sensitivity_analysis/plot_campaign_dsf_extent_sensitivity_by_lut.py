"""Recreate the campaign-level DSF extent-sensitivity-by-LUT figures for ACOLITE and
ACOLITE+T-Mart, from the raw per-transect/per-wavelength sensitivity CSVs.

The script that originally produced acolite_campaign_dsf_extent_sensitivity_by_lut.png
and acolite_tmart_campaign_dsf_extent_sensitivity_by_lut.png could not be found on disk
(checked plots/sensitivity_analysis, its obsolete/ subfolder, and a full-repo search).
This rebuilds the same figure layout (2 rows: MOD1/MOD2, 3 columns: median MSA, p90 MSA,
median spectral angle, one line per campaign) directly from the underlying data, using
Helvetica, larger fonts, and campaign colors matched to plot_ramses_measurements.py's
tab10 assignment for consistency with the in-situ Rrs plot.

Marker style is a toggle (MARKER_STYLE below, or --marker-style on the CLI):
  - "plain":    every campaign plotted with a circle marker (the original look).
  - "distinct": a different marker shape per campaign, as redundant encoding
                alongside the tab10 colors. Needed because the tab10 9-color
                set fails CVD separation badly (validated: worst adjacent
                pair BIE20240618/ZRH20231008 green-vs-orange, DeltaE 0.7
                under protanopia) - color alone doesn't distinguish all 9
                campaigns for colorblind readers. Marker sizes are per-shape
                corrected (MARKER_SIZE_SCALE), since matplotlib's marker
                glyphs are not visually equal-area at a shared markersize
                (e.g. a star or X reads larger than a circle or triangle at
                an identical value).
Each style writes to its own subfolder (LUT_dsf_extent/ for "plain",
LUT_dsf_extent/markers/ for "distinct") so neither overwrites the other.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_ROOT = Path(r"C:\Users\puehrifi\Documents\results\stats\sensitivity_analysis\hard_filter_retain")
# Figure output lives under LUT_dsf_extent/ (plain) or LUT_dsf_extent/markers/
# (distinct) - NOT directly under PLOT_ROOT, which is only the base for the
# interactive-chart JSON (see INTERACTIVE_JSON_PATH), matching where
# build_visualization_website.py's read_sensitivity_interactive_data() looks.
PLOT_ROOT = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\results\plots\sensitivity_analysis")
FIGURE_OUTPUT_DIR_BY_STYLE = {
    "plain": PLOT_ROOT / "LUT_dsf_extent",
    "distinct": PLOT_ROOT / "LUT_dsf_extent" / "markers",
}

# Default when run with no --marker-style flag.
DEFAULT_MARKER_STYLE = "distinct"

PROCESSORS = {
    "ACOLITE": "ACOLITE DSF",
    "ACOLITE_TMart": "T-Mart + ACOLITE DSF",
}

# Campaign order and colors match plot_ramses_measurements.py's CAMPAIGNS list, which
# assigns colors via matplotlib's tab10 colormap indexed by list position.
CAMPAIGNS_IN_COLOR_ORDER = [
    "CST20230610",
    "ZRH20231008",
    "BIE20240618",
    "WAL20240619",
    "CST20250303",
    "ZRH20260226",
    "ZRH20260227",
    "ZRH20260423",
    "ZRH20260430",
]
_TAB10 = plt.get_cmap("tab10")
CAMPAIGN_COLORS = {campaign: _TAB10(i % _TAB10.N) for i, campaign in enumerate(CAMPAIGNS_IN_COLOR_ORDER)}

# Distinct silhouettes (not just fill differences) so campaigns stay
# identifiable by shape alone even where the tab10 colors are not
# CVD-separable, and even in grayscale/print. Only used for marker_style="distinct".
CAMPAIGN_MARKERS = {
    "CST20230610": "o",  # circle
    "ZRH20231008": "s",  # square
    "BIE20240618": "^",  # triangle up
    "WAL20240619": "D",  # diamond
    "CST20250303": "v",  # triangle down
    "ZRH20260226": "P",  # filled plus
    "ZRH20260227": "X",  # filled x
    "ZRH20260423": "*",  # star
    "ZRH20260430": "p",  # pentagon
}

# matplotlib's marker glyphs are not equal-area at a shared markersize (a
# star or diamond reads larger, a plus/X reads smaller, than a circle at the
# same value); these per-shape multipliers even out the apparent size so no
# campaign looks more prominent purely from its marker's geometry. Only used
# for marker_style="distinct".
DISTINCT_BASE_MARKERSIZE = 3.2
MARKER_SIZE_SCALE = {
    "o": 1.0,
    "s": 0.85,
    "^": 1.0,
    "D": 0.8,
    "v": 1.0,
    "P": 1.15,
    "X": 1.15,
    "*": 1.4,
    "p": 0.95,
}
PLAIN_MARKERSIZE = 1.8

EXTENTS_KM = [0.5, 1, 3, 5, 7, 10, 15, 30]
LUTS = ["MOD1", "MOD2"]

# Figure is authored at its final print width (16 cm) so font sizes below are the
# actual on-page point sizes, with no further scaling expected when it is placed in
# the document. Smallest labels (tick labels) are set to 9 pt.
FIGURE_WIDTH_CM = 16.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / 2.54
FIGURE_HEIGHT_IN = FIGURE_WIDTH_IN * (13.2 / 16.0)
SAVE_DPI = 800

# plot_processor()'s 3-column layout (left/right margins + wspace below,
# matching its own fig.subplots_adjust call) sets the per-axis width every
# other figure in this module should match. The SB grid below has only 2
# columns, so hitting the same physical axis width means solving for a
# narrower SB_FIGURE_WIDTH_IN (keeping the same absolute left/right margins,
# in inches, as its original 0.11/0.99 fractions at FIGURE_WIDTH_IN) rather
# than reusing FIGURE_WIDTH_IN outright.
_PROCESSOR_NCOLS, _PROCESSOR_WSPACE = 3, 0.20
_PROCESSOR_LEFT, _PROCESSOR_RIGHT = 0.075, 0.99
_TARGET_AXIS_WIDTH_IN = (
    (_PROCESSOR_RIGHT - _PROCESSOR_LEFT)
    * FIGURE_WIDTH_IN
    / (_PROCESSOR_NCOLS + (_PROCESSOR_NCOLS - 1) * _PROCESSOR_WSPACE)
)

_SB_NCOLS, SB_WSPACE = 2, 0.10
_SB_LEFT_MARGIN_IN = 0.11 * FIGURE_WIDTH_IN
_SB_RIGHT_MARGIN_IN = (1 - 0.99) * FIGURE_WIDTH_IN
SB_FIGURE_WIDTH_IN = (
    _SB_LEFT_MARGIN_IN
    + _SB_RIGHT_MARGIN_IN
    + _TARGET_AXIS_WIDTH_IN * (_SB_NCOLS + (_SB_NCOLS - 1) * SB_WSPACE)
)
SB_LEFT = _SB_LEFT_MARGIN_IN / SB_FIGURE_WIDTH_IN
SB_RIGHT = 1 - _SB_RIGHT_MARGIN_IN / SB_FIGURE_WIDTH_IN

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 0.9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.titlepad": 4.0,
    }
)


def marker_and_size_for(campaign: str, marker_style: str) -> tuple[str, float]:
    if marker_style == "plain":
        return "o", PLAIN_MARKERSIZE
    marker = CAMPAIGN_MARKERS[campaign]
    return marker, DISTINCT_BASE_MARKERSIZE * MARKER_SIZE_SCALE[marker]


def snap_extent_km(physical_tile_extent: str) -> float | None:
    match = re.match(r"\s*([\d.]+)", str(physical_tile_extent))
    if not match:
        return None
    value = float(match.group(1))
    return min(EXTENTS_KM, key=lambda candidate: abs(candidate - value))


def read_csv_auto(path: Path) -> pd.DataFrame:
    header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    separator = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=separator)


def load_campaign_csv(processor_folder: str, campaign: str) -> pd.DataFrame | None:
    campaign_dir = RESULTS_ROOT / processor_folder / campaign
    matches = list(campaign_dir.glob("*_acolite_all_settings_by_transect_visible_wavelength.csv"))
    if not matches:
        return None
    frame = read_csv_auto(matches[0])
    if "lut" not in frame.columns:
        print(f"Skipping {matches[0]}: no 'lut' column found (columns: {list(frame.columns)[:5]}...)")
        return None
    frame = frame[frame["lut"].isin(LUTS)].copy()
    if frame.empty:
        return None
    frame["msa_percent"] = pd.to_numeric(frame["msa_percent"], errors="coerce")
    frame["sb_percent"] = pd.to_numeric(frame["sb_percent"], errors="coerce")
    frame["spectral_angle_deg_median"] = pd.to_numeric(frame["spectral_angle_deg_median"], errors="coerce")
    frame["extent_km"] = frame["physical_tile_extent"].map(snap_extent_km)
    frame["campaign"] = campaign
    return frame


def summarize_campaign(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lut, extent_km), group in frame.groupby(["lut", "extent_km"]):
        msa = group["msa_percent"].dropna()
        if msa.empty:
            continue
        # spectral_angle_deg_median repeats across wavelength rows for a given
        # sensor/transect/tile setting, so dedupe before taking the median.
        dedup = group.drop_duplicates(subset=["sensor", "selected_transects", "tile_dimensions"])
        rows.append(
            {
                "lut": lut,
                "extent_km": extent_km,
                "median_msa": msa.median(),
                "p90_msa": msa.quantile(0.9),
                "median_sb": group["sb_percent"].dropna().median(),
                "median_spectral_angle": dedup["spectral_angle_deg_median"].dropna().median(),
                "sensors": ",".join(sorted(group["sensor"].dropna().unique())),
            }
        )
    return pd.DataFrame(rows)


def build_campaign_summaries(processor_folder: str) -> dict[str, pd.DataFrame]:
    summaries = {}
    for campaign in CAMPAIGNS_IN_COLOR_ORDER:
        raw = load_campaign_csv(processor_folder, campaign)
        if raw is None:
            continue
        summary = summarize_campaign(raw)
        if not summary.empty:
            summaries[campaign] = summary
    return summaries


def compute_shared_ylimits(all_summaries: dict[str, dict[str, pd.DataFrame]]) -> dict[tuple[str, str], tuple[float, float]]:
    """Min/max per (lut, metric) pooled across every processor, so corresponding
    subplots in the ACOLITE and ACOLITE+T-Mart figures share the same y-range.
    Applied via an invisible autoscale hint in plot_processor (see there) rather
    than a hand-rolled set_ylim, so matplotlib's own margin/log-margin logic -
    already tuned per axis via rcParams - computes an identical window in both
    figures instead of two independently-eyeballed ones."""
    bounds: dict[tuple[str, str], tuple[float, float]] = {}
    metrics = ["median_msa", "p90_msa", "median_spectral_angle"]
    for lut in LUTS:
        for metric in metrics:
            values = []
            for summaries in all_summaries.values():
                for summary in summaries.values():
                    subset = summary[summary["lut"] == lut][metric].dropna()
                    if not subset.empty:
                        values.append(subset)
            if not values:
                continue
            pooled = pd.concat(values)
            bounds[(lut, metric)] = (float(pooled.min()), float(pooled.max()))
    return bounds


def plot_processor(
    summaries: dict[str, pd.DataFrame],
    processor_label: str,
    output_stub: str,
    marker_style: str,
    shared_ylimits: dict[tuple[str, str], tuple[float, float]],
):
    if not summaries:
        print(f"No data found for {processor_label}, skipping.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    # Fix the axes grid geometry up front so row-label positions (computed from
    # axes bounding boxes below) are correct; nothing after this call should move
    # the subplot grid.
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.87, wspace=0.20, hspace=0.30)
    # Short column titles (top row only) since subplots are ~1.8 in wide at this print
    # size; the LUT (MOD1/MOD2) is given once per row via a rotated row label instead
    # of being repeated inside every subplot title.
    metric_specs = [
        ("median_msa", "Median MSA [%] (log)", True),
        ("p90_msa", "p90 MSA [%] (log)", True),
        ("median_spectral_angle", "Median spectral angle [°]", False),
    ]

    x_positions = {extent: i for i, extent in enumerate(EXTENTS_KM)}

    for row, lut in enumerate(LUTS):
        for col, (metric, subtitle, log_scale) in enumerate(metric_specs):
            ax = axes[row, col]
            for campaign, summary in summaries.items():
                subset = summary[summary["lut"] == lut].sort_values("extent_km")
                subset = subset.dropna(subset=[metric])
                if subset.empty:
                    continue
                xs = [x_positions[e] for e in subset["extent_km"]]
                marker, markersize = marker_and_size_for(campaign, marker_style)
                plot_kwargs = {}
                if marker_style == "distinct":
                    plot_kwargs.update(markeredgecolor="white", markeredgewidth=0.3)
                ax.plot(
                    xs,
                    subset[metric],
                    marker=marker,
                    markersize=markersize,
                    color=CAMPAIGN_COLORS[campaign],
                    label=campaign,
                    **plot_kwargs,
                )
            bounds = shared_ylimits.get((lut, metric))
            if bounds is not None:
                # Invisible points at the cross-processor min/max: expands this
                # axes' data limits without drawing anything, so the normal
                # autoscale below (including its log-aware margin) lands on the
                # same window it would for the other processor's figure.
                ax.plot(x_positions[EXTENTS_KM[0]], bounds[0], alpha=0)
                ax.plot(x_positions[EXTENTS_KM[0]], bounds[1], alpha=0)
            ax.set_xticks(range(len(EXTENTS_KM)))
            ax.set_xticklabels([f"{e:g}" for e in EXTENTS_KM])
            if row == 1:
                ax.set_xlabel("DSF extent [km]")
            if log_scale:
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(subtitle)
            ax.grid(alpha=0.25)
        row_label_y = axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
        fig.text(0.012, row_label_y, lut, rotation=90, ha="left", va="center", fontsize=10, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for row in range(2):
            handles, labels = axes[row, 0].get_legend_handles_labels()
            if handles:
                break

    labelled_handles = []
    labelled_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        labelled_handles.append(handle)
        labelled_labels.append(label)

    fig.legend(
        labelled_handles,
        labelled_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    fig.suptitle("Campaign-Level DSF Extent Sensitivity by LUT", y=0.99, fontweight="bold")
    fig.text(0.5, 0.95, processor_label, ha="center", va="top", fontsize=10, fontstyle="italic")
    # No bbox_inches="tight" on save: the figure canvas must stay exactly
    # FIGURE_WIDTH_IN wide so the point sizes set above are the true on-page sizes
    # once this is placed at FIGURE_WIDTH_CM in the document, with no rescaling.

    output_dir = FIGURE_OUTPUT_DIR_BY_STYLE[marker_style]
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stub}_campaign_dsf_extent_sensitivity_by_lut.png"
    svg_path = output_dir / f"{output_stub}_campaign_dsf_extent_sensitivity_by_lut.svg"
    fig.savefig(png_path, dpi=SAVE_DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def plot_sb_grid(all_summaries: dict[str, dict[str, pd.DataFrame]], marker_style: str):
    """Standalone median-SB figure: 2 LUT rows x 2 processor columns (ACOLITE
    DSF, T-Mart+ACOLITE DSF), one campaign-colored line per panel as in
    plot_processor(). SB is signed (over-/underestimation), so unlike MSA/p90
    it stays on a linear axis with a y=0 reference line, and shares one y-range
    across all four panels via sharey (simpler than plot_processor()'s
    cross-figure invisible-point trick, since this is already a single figure)."""
    processor_specs = [("ACOLITE", "ACOLITE DSF", "acolite"), ("ACOLITE_TMart", "T-Mart + ACOLITE DSF", "acolite_tmart")]
    if not any(all_summaries.get(folder) for folder, _, _ in processor_specs):
        print("No data found for SB grid, skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(SB_FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), sharey=True)
    fig.subplots_adjust(left=SB_LEFT, right=SB_RIGHT, bottom=0.23, top=0.85, wspace=SB_WSPACE, hspace=0.30)
    x_positions = {extent: i for i, extent in enumerate(EXTENTS_KM)}

    for row, lut in enumerate(LUTS):
        for col, (folder, processor_label, _) in enumerate(processor_specs):
            ax = axes[row, col]
            summaries = all_summaries.get(folder, {})
            for campaign, summary in summaries.items():
                subset = summary[summary["lut"] == lut].sort_values("extent_km")
                subset = subset.dropna(subset=["median_sb"])
                if subset.empty:
                    continue
                xs = [x_positions[e] for e in subset["extent_km"]]
                marker, markersize = marker_and_size_for(campaign, marker_style)
                plot_kwargs = {}
                if marker_style == "distinct":
                    plot_kwargs.update(markeredgecolor="white", markeredgewidth=0.3)
                ax.plot(
                    xs,
                    subset["median_sb"],
                    marker=marker,
                    markersize=markersize,
                    color=CAMPAIGN_COLORS[campaign],
                    label=campaign,
                    **plot_kwargs,
                )
            ax.axhline(0, color="0.4", linewidth=0.7, linestyle="--", zorder=0)
            ax.set_xticks(range(len(EXTENTS_KM)))
            ax.set_xticklabels([f"{e:g}" for e in EXTENTS_KM])
            if row == 1:
                ax.set_xlabel("DSF extent [km]")
            if row == 0:
                ax.set_title(processor_label)
            ax.grid(alpha=0.25)
        row_label_y = axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
        fig.text(0.012, row_label_y, lut, rotation=90, ha="left", va="center", fontsize=10, fontweight="bold")

    axes[0, 0].set_ylabel("Median SB [%]")
    axes[1, 0].set_ylabel("Median SB [%]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for row in range(2):
            handles, labels = axes[row, 0].get_legend_handles_labels()
            if handles:
                break
    labelled_handles, labelled_labels, seen = [], [], set()
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        labelled_handles.append(handle)
        labelled_labels.append(label)
    fig.legend(labelled_handles, labelled_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005), frameon=False)
    # Split across two lines (suptitle + italic subtitle, same pattern as
    # plot_processor()'s processor_label): at SB_FIGURE_WIDTH_IN, the full
    # "... by LUT -- Median SB" title in one line no longer fits the
    # narrower canvas and gets clipped.
    fig.suptitle("Campaign-Level DSF Extent Sensitivity by LUT", y=0.99, fontweight="bold")
    fig.text(0.5, 0.95, "Median SB", ha="center", va="top", fontsize=10, fontstyle="italic")

    output_dir = FIGURE_OUTPUT_DIR_BY_STYLE[marker_style]
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "acolite_family_sb_by_lut.png"
    svg_path = output_dir / "acolite_family_sb_by_lut.svg"
    fig.savefig(png_path, dpi=SAVE_DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


INTERACTIVE_JSON_PATH = PLOT_ROOT / "sensitivity_dsf_extent_by_lut_data.json"

# Metric keys/labels match the site-wide MSA/SB/SAM taxonomy used elsewhere in
# build_visualization_website.py (display_metric()), so the interactive chart's
# toggle options line up with terminology used across the rest of the site.
INTERACTIVE_METRICS = {
    "msa": {"label": "Median MSA [%]", "field": "median_msa", "log": True},
    "sb": {"label": "Median SB [%]", "field": "median_sb", "log": False},
    "sam": {"label": "Median SAM [°]", "field": "median_spectral_angle", "log": False},
}


def export_interactive_json(path: Path = INTERACTIVE_JSON_PATH):
    payload = {
        "campaigns": CAMPAIGNS_IN_COLOR_ORDER,
        "campaignColors": {
            campaign: "#{:02x}{:02x}{:02x}".format(
                *(int(round(channel * 255)) for channel in CAMPAIGN_COLORS[campaign][:3])
            )
            for campaign in CAMPAIGNS_IN_COLOR_ORDER
        },
        # Same per-campaign marker shape as the "distinct" static figures
        # (CAMPAIGN_MARKERS above), so the interactive chart's own markers
        # carry the same colorblind-friendly redundant encoding rather than
        # plain colored dots. Shape codes match matplotlib's marker
        # vocabulary; renderSensitivityChart() maps these to SVG shapes.
        "campaignMarkers": CAMPAIGN_MARKERS,
        "extents": EXTENTS_KM,
        "luts": LUTS,
        "processors": list(PROCESSORS.keys()),
        "processorLabels": PROCESSORS,
        "metrics": INTERACTIVE_METRICS,
        "series": {},
    }
    for processor_folder in PROCESSORS:
        summaries = build_campaign_summaries(processor_folder)
        processor_series = []
        for campaign, summary in summaries.items():
            for lut in LUTS:
                subset = summary[summary["lut"] == lut].sort_values("extent_km")
                if subset.empty:
                    continue
                points = [
                    {
                        "extent": row.extent_km,
                        "median_msa": None if pd.isna(row.median_msa) else round(float(row.median_msa), 4),
                        "median_sb": None if pd.isna(row.median_sb) else round(float(row.median_sb), 4),
                        "median_spectral_angle": (
                            None if pd.isna(row.median_spectral_angle) else round(float(row.median_spectral_angle), 4)
                        ),
                    }
                    for row in subset.itertuples()
                ]
                processor_series.append({"campaign": campaign, "lut": lut, "points": points})
        payload["series"][processor_folder] = processor_series

    path.write_text(json.dumps(payload, indent=None, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marker-style",
        choices=["plain", "distinct"],
        default=DEFAULT_MARKER_STYLE,
        help=(
            "plain: one circle marker for every campaign (original look). "
            "distinct: a different marker shape per campaign, for colorblind "
            f"readers (see module docstring). Default: {DEFAULT_MARKER_STYLE}"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_summaries = {folder: build_campaign_summaries(folder) for folder in PROCESSORS}
    shared_ylimits = compute_shared_ylimits(all_summaries)
    plot_processor(all_summaries["ACOLITE"], "ACOLITE DSF", "acolite", args.marker_style, shared_ylimits)
    plot_processor(all_summaries["ACOLITE_TMart"], "T-Mart + ACOLITE DSF", "acolite_tmart", args.marker_style, shared_ylimits)
    plot_sb_grid(all_summaries, args.marker_style)
    export_interactive_json()


if __name__ == "__main__":
    main()
