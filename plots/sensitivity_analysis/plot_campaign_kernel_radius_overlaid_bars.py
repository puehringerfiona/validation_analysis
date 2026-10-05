"""Restyled rebuild of the radcor_plot_options/*_option5_grouped_kernel_overlaid_lut_bars
figures: grouped bars with the worse-performing LUT ghosted behind the better one, at
matching print/typography conventions to acolite_campaign_dsf_extent_sensitivity_by_lut.png
(16 cm width, Helvetica, 9 pt minimum labels).

LUT identity is encoded by hue (blue = MOD1, orange = MOD2). Kernel radius is an
ordered quantity, so instead of a second categorical channel (hatch) competing with
color, it is encoded as a lightness ramp within each LUT's own hue: light = 1 km,
base = 3 km, dark = 5 km.

No source script for the option5 figures could be found on disk, so this is rebuilt
from the raw per-campaign sensitivity CSVs. Zooming into the original PNGs showed the
ghost/foreground assignment is made PER (campaign, kernel radius) pair -- at some
radii MOD1 is drawn in front, at others MOD2 is, switching within the same campaign
group -- so "better" here means the lower (better) metric value at that specific
(campaign, radius) point, not a single campaign-wide LUT choice. Both LUTs' real
values are plotted (foreground bar + a wider, lower-opacity ghost bar behind it);
nothing is aggregated away.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator

RESULTS_ROOT = Path(r"C:\Users\puehrifi\Documents\results\stats\sensitivity_analysis\hard_filter_retain")
PLOT_ROOT = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\results\plots\sensitivity_analysis\LUT_kernel_radius")

PROCESSORS = {
    "ACOLITE_RAdCor": "ACOLITE TSDSF + RAdCor",
    "C2RCC_RAdCor": "RAdCor + C2RCC",
    "Polymer_RAdCor": "RAdCor + POLYMER",
}

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

KERNEL_RADII_KM = [1, 3, 5]
LUTS = ["MOD1", "MOD2"]
# First two fixed-order slots of the reference dataviz palette, validated CVD-safe.
LUT_COLORS = {"MOD1": "#2a78d6", "MOD2": "#eb6834"}
NEUTRAL_BASE = "#9a9a9a"

# Kernel radius is an ordered quantity, so it is encoded as a lightness ramp within
# each LUT's own hue (light = 1 km, base = 3 km, dark = 5 km) rather than hatch
# texture -- the sequential "one hue, light-to-dark" convention used for ordinal
# data, instead of adding a second categorical channel (hatch) that competes with
# color for attention.
RADIUS_SHADE_AMOUNT = {1: 0.35, 3: 0.0, 5: -0.35}


def shade_hex(hex_color: str, amount: float) -> str:
    """amount > 0 lightens toward white, amount < 0 darkens toward black."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    target = 255 if amount >= 0 else 0
    factor = abs(amount)
    r, g, b = (round(c + (target - c) * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


LUT_RADIUS_COLORS = {
    lut: {radius: shade_hex(LUT_COLORS[lut], RADIUS_SHADE_AMOUNT[radius]) for radius in KERNEL_RADII_KM}
    for lut in LUTS
}
NEUTRAL_RADIUS_COLORS = {radius: shade_hex(NEUTRAL_BASE, RADIUS_SHADE_AMOUNT[radius]) for radius in KERNEL_RADII_KM}

FIGURE_WIDTH_CM = 16.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / 2.54
TITLE_Y_OFFSET_IN = 0.26
SUBTITLE_Y_OFFSET_IN = 0.50
# Matches acolite_campaign_dsf_extent_sensitivity_by_lut.py's subtitle-to-axes-top
# gap: that figure is FIGURE_WIDTH_IN * (13.2/16.0) tall with subtitle at y=0.95 and
# axes top at y=0.87, i.e. a 0.08 * height gap.
_REFERENCE_FIGURE_HEIGHT_IN = (16.0 / 2.54) * (13.2 / 16.0)
SUBTITLE_TO_AXES_GAP_IN = 0.08 * _REFERENCE_FIGURE_HEIGHT_IN
TOP_MARGIN_IN = SUBTITLE_Y_OFFSET_IN + SUBTITLE_TO_AXES_GAP_IN
BOTTOM_MARGIN_IN = 1.25
PANEL_HEIGHT_IN = 1.4
FIGURE_HEIGHT_IN = TOP_MARGIN_IN + PANEL_HEIGHT_IN + BOTTOM_MARGIN_IN
SAVE_DPI = 800

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "ytick.minor.width": 0.4,
        "ytick.major.pad": 1.5,
        "ytick.minor.pad": 1.5,
        "axes.titlepad": 4.0,
    }
)


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
    if "lut" not in frame.columns or "kernel_radius" not in frame.columns:
        return None
    frame = frame[frame["lut"].isin(LUTS)].copy()
    if frame.empty:
        return None
    frame["msa_percent"] = pd.to_numeric(frame["msa_percent"], errors="coerce")
    frame["sb_percent"] = pd.to_numeric(frame["sb_percent"], errors="coerce")
    frame["spectral_angle_deg_median"] = pd.to_numeric(frame["spectral_angle_deg_median"], errors="coerce")
    frame["kernel_radius_km"] = frame["kernel_radius"].astype(str).str.extract(r"([\d.]+)").astype(float)
    frame["campaign"] = campaign
    return frame


def summarize_campaign(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lut, kernel_radius_km), group in frame.groupby(["lut", "kernel_radius_km"]):
        msa = group["msa_percent"].dropna()
        if msa.empty:
            continue
        dedup = group.drop_duplicates(subset=["sensor", "selected_transects", "tile_dimensions"])
        rows.append(
            {
                "lut": lut,
                "kernel_radius_km": kernel_radius_km,
                "median_msa": msa.median(),
                "p90_msa": msa.quantile(0.9),
                "median_sb": group["sb_percent"].dropna().median(),
                "median_spectral_angle": dedup["spectral_angle_deg_median"].dropna().median(),
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


METRIC_SPECS = [
    ("median_msa", "Median MSA [%] (log)", True),
    ("p90_msa", "p90 MSA [%] (log)", True),
    ("median_spectral_angle", "Median SAM [\u00b0]", False),
]


def compute_shared_ylimits(all_summaries: dict[str, dict[str, pd.DataFrame]]) -> dict[str, tuple[float, float]]:
    """Min/max per metric pooled across every processor, so the corresponding
    subplot (MSA, p90 MSA, SAM) is on the same y-range in all three processor
    figures. Applied via an invisible autoscale hint in plot_processor (see
    there), so matplotlib's own margin/log-margin logic computes an identical
    window in every figure instead of three independently-scaled ones."""
    bounds: dict[str, tuple[float, float]] = {}
    for metric, _, _ in METRIC_SPECS:
        values = [summary[metric].dropna() for summaries in all_summaries.values() for summary in summaries.values()]
        values = [v for v in values if not v.empty]
        if not values:
            continue
        pooled = pd.concat(values)
        bounds[metric] = (float(pooled.min()), float(pooled.max()))
    return bounds


def plot_processor(
    processor_folder: str,
    processor_label: str,
    output_stub: str,
    summaries: dict[str, pd.DataFrame],
    shared_ylimits: dict[str, tuple[float, float]],
):
    if not summaries:
        print(f"No data found for {processor_folder}, skipping.")
        return

    campaigns = [c for c in CAMPAIGNS_IN_COLOR_ORDER if c in summaries]
    n_campaigns = len(campaigns)
    slot_w = 1.0
    group_gap = 1.0
    group_w = slot_w * len(KERNEL_RADII_KM)
    group_starts = [i * (group_w + group_gap) for i in range(n_campaigns)]

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        top=1 - TOP_MARGIN_IN / FIGURE_HEIGHT_IN,
        bottom=BOTTOM_MARGIN_IN / FIGURE_HEIGHT_IN,
        wspace=0.14,
    )

    for col, (metric, title, log_scale) in enumerate(METRIC_SPECS):
        ax = axes[col]
        for gi, campaign in enumerate(campaigns):
            summary = summaries[campaign]
            for ri, radius in enumerate(KERNEL_RADII_KM):
                x_center = group_starts[gi] + ri * slot_w + slot_w / 2
                values = {}
                for lut in LUTS:
                    row = summary[(summary["lut"] == lut) & (summary["kernel_radius_km"] == radius)]
                    if not row.empty and pd.notna(row.iloc[0][metric]):
                        values[lut] = float(row.iloc[0][metric])
                if not values:
                    continue
                better_lut = min(values, key=values.get)
                if len(values) == 2:
                    worse_lut = max(values, key=values.get)
                    ax.bar(
                        x_center,
                        values[worse_lut],
                        width=slot_w * 0.82,
                        color=LUT_RADIUS_COLORS[worse_lut][radius],
                        alpha=0.30,
                        edgecolor=LUT_RADIUS_COLORS[worse_lut][radius],
                        linewidth=0.3,
                        zorder=2,
                    )
                ax.bar(
                    x_center,
                    values[better_lut],
                    width=slot_w * 0.55,
                    color=LUT_RADIUS_COLORS[better_lut][radius],
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=3,
                )
        bounds = shared_ylimits.get(metric)
        if bounds is not None:
            # Invisible points at the cross-processor min/max: expands this
            # axes' data limits without drawing anything, so the normal
            # autoscale below lands on the same window it would for the
            # other two processors' figures.
            hint_x = group_starts[0] + group_w / 2
            ax.plot(hint_x, bounds[0], alpha=0)
            ax.plot(hint_x, bounds[1], alpha=0)
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
            ax.tick_params(axis="y", which="minor", length=2.0)
        ax.tick_params(axis="y", which="major", length=3.5)
        ax.set_title(title)
        ax.set_xticks([start + group_w / 2 for start in group_starts])
        ax.set_xticklabels(campaigns, rotation=90, ha="center", va="top")
        ax.set_xlim(-group_gap / 2, group_starts[-1] + group_w + group_gap / 2)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    lut_handles = [Patch(facecolor=LUT_COLORS[lut], edgecolor="black", linewidth=0.4, label=lut) for lut in LUTS]
    radius_handles = [
        Patch(facecolor=NEUTRAL_RADIUS_COLORS[r], edgecolor="black", linewidth=0.4, label=f"{r} km")
        for r in KERNEL_RADII_KM
    ]
    ghost_handle = [Patch(facecolor="0.6", edgecolor="none", alpha=0.30, label="Worse LUT")]
    fig.legend(
        handles=lut_handles + radius_handles + ghost_handle,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Campaign-Level Kernel Radius Sensitivity by LUT",
        y=1 - TITLE_Y_OFFSET_IN / FIGURE_HEIGHT_IN,
        fontweight="bold",
    )
    fig.text(
        0.5,
        1 - SUBTITLE_Y_OFFSET_IN / FIGURE_HEIGHT_IN,
        processor_label,
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
    )

    png_path = PLOT_ROOT / f"{output_stub}_campaign_kernel_radius_overlaid_bars.png"
    svg_path = PLOT_ROOT / f"{output_stub}_campaign_kernel_radius_overlaid_bars.svg"
    fig.savefig(png_path, dpi=SAVE_DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def plot_sb_grid(all_summaries: dict[str, dict[str, pd.DataFrame]]):
    """Standalone median-SB figure: 2 LUT rows x 3 processor columns (ACOLITE
    TSDSF+RAdCor, RAdCor+C2RCC, RAdCor+POLYMER). Unlike plot_processor()'s MSA/
    p90/SAM panels, LUT is split across rows here rather than ghosted within
    one panel - SB is signed, so "better" would mean smaller |SB|, not smaller
    raw value, and overlaying two bars that can each go either side of zero
    reads far less clearly than just giving each LUT its own row. Bars keep
    the same radius-shaded-by-LUT-hue coloring as plot_processor(); one shared
    y-range (with a y=0 reference line) across all six panels."""
    processor_specs = [
        ("ACOLITE_RAdCor", "ACOLITE TSDSF + RAdCor"),
        ("C2RCC_RAdCor", "RAdCor + C2RCC"),
        ("Polymer_RAdCor", "RAdCor + POLYMER"),
    ]
    if not any(all_summaries.get(folder) for folder, _ in processor_specs):
        print("No data found for SB grid, skipping.")
        return

    all_campaigns = sorted(
        {c for folder, _ in processor_specs for c in all_summaries.get(folder, {})},
        key=CAMPAIGNS_IN_COLOR_ORDER.index,
    )
    n_campaigns = len(all_campaigns)
    slot_w = 1.0
    group_gap = 1.0
    group_w = slot_w * len(KERNEL_RADII_KM)
    group_starts = [i * (group_w + group_gap) for i in range(n_campaigns)]

    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN * 1.6), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.88, bottom=0.18, wspace=0.10, hspace=0.55)

    sb_bounds = None
    all_values = [
        summary[(summary["lut"] == lut)]["median_sb"].dropna()
        for folder, _ in processor_specs
        for summary in all_summaries.get(folder, {}).values()
        for lut in LUTS
    ]
    all_values = [v for v in all_values if not v.empty]
    if all_values:
        pooled = pd.concat(all_values)
        sb_bounds = (float(pooled.min()), float(pooled.max()))

    for row, lut in enumerate(LUTS):
        for col, (folder, processor_label) in enumerate(processor_specs):
            ax = axes[row, col]
            summaries = all_summaries.get(folder, {})
            for gi, campaign in enumerate(all_campaigns):
                summary = summaries.get(campaign)
                if summary is None:
                    continue
                for ri, radius in enumerate(KERNEL_RADII_KM):
                    x_center = group_starts[gi] + ri * slot_w + slot_w / 2
                    value_row = summary[(summary["lut"] == lut) & (summary["kernel_radius_km"] == radius)]
                    if value_row.empty or pd.isna(value_row.iloc[0]["median_sb"]):
                        continue
                    ax.bar(
                        x_center,
                        float(value_row.iloc[0]["median_sb"]),
                        width=slot_w * 0.7,
                        color=LUT_RADIUS_COLORS[lut][radius],
                        edgecolor="black",
                        linewidth=0.4,
                        zorder=3,
                    )
            if sb_bounds is not None:
                hint_x = group_starts[0] + group_w / 2
                ax.plot(hint_x, sb_bounds[0], alpha=0)
                ax.plot(hint_x, sb_bounds[1], alpha=0)
            ax.axhline(0, color="0.25", linewidth=0.7, linestyle="--", zorder=2)
            ax.tick_params(axis="y", which="major", length=3.5)
            if row == 0:
                ax.set_title(processor_label)
            ax.set_xticks([start + group_w / 2 for start in group_starts])
            ax.set_xticklabels(all_campaigns, rotation=90, ha="center", va="top")
            ax.set_xlim(-group_gap / 2, group_starts[-1] + group_w + group_gap / 2)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        row_label_y = axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
        fig.text(0.015, row_label_y, lut, rotation=90, ha="left", va="center", fontsize=10, fontweight="bold")

    axes[0, 0].set_ylabel("Median SB [%]", labelpad=8)
    axes[1, 0].set_ylabel("Median SB [%]", labelpad=8)

    radius_handles = [
        Patch(facecolor=NEUTRAL_RADIUS_COLORS[r], edgecolor="black", linewidth=0.4, label=f"{r} km")
        for r in KERNEL_RADII_KM
    ]
    fig.legend(
        handles=radius_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.suptitle("Campaign-Level Kernel Radius Sensitivity by LUT — Median SB", y=0.97, fontweight="bold")

    png_path = PLOT_ROOT / "radcor_family_sb_by_lut.png"
    svg_path = PLOT_ROOT / "radcor_family_sb_by_lut.svg"
    fig.savefig(png_path, dpi=SAVE_DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def main():
    processors = [
        ("ACOLITE_RAdCor", "ACOLITE TSDSF + RAdCor", "acolite_radcor"),
        ("C2RCC_RAdCor", "RAdCor + C2RCC", "c2rcc_radcor"),
        ("Polymer_RAdCor", "RAdCor + POLYMER", "polymer_radcor"),
    ]
    all_summaries = {folder: build_campaign_summaries(folder) for folder, _, _ in processors}
    shared_ylimits = compute_shared_ylimits(all_summaries)
    for folder, label, stub in processors:
        plot_processor(folder, label, stub, all_summaries[folder], shared_ylimits)
    plot_sb_grid(all_summaries)


if __name__ == "__main__":
    main()
