"""Consolidated settings-variability figures for the "Settings Sensitivity and
Processor Robustness" results section: replaces the nine separate per-campaign
box-and-swarm prototypes (hard_filter_retain/by_campaign/*.png) with a single
figure faceted by processor, campaign on the shared x-axis, no sensor split.

Two figures are produced, sharing the same panel layout (1 row x 5 processors,
same processor order, same x = campaign, same log y-axis) so they read as a pair:

1. box_swarm: full distribution of per-setting median MSA (one point per tested
   LUT x extent/kernel-radius combination) per campaign, as a box + swarm, with
   a star marking where that lake's manually-adopted primary-best setting lands
   in this specific campaign's distribution.
2. best_worst_dumbbell: just the best and worst tested setting per campaign,
   connected by a line -- the precise visual analogue of the worst/best ratio
   and percentage-point range figures reported in the results text (as opposed
   to the box's IQR, which is a different statistic).

Campaigns are colored with the same tab10 assignment used throughout the rest of
the thesis (plot_ramses_measurements.py's CAMPAIGNS order), so this figure reads
as part of the same visual language as the other campaign-colored plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIABILITY_ROOT = Path(r"C:\Users\puehrifi\Documents\results\plots\sensitivity_analysis\hard_filter_retain")
SETTINGS_ROOT = Path(r"C:\Users\puehrifi\Documents\results\stats\sensitivity_analysis\hard_filter_retain")
PLOT_ROOT = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\results\plots\sensitivity_analysis\variability")

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

# Processor order matches the settings-sensitivity ranking in the results text
# (decreasing median worst/best ratio): ACOLITE DSF > T-Mart+ACOLITE DSF >
# ACOLITE TSDSF+RAdCor > RAdCor+POLYMER > RAdCor+C2RCC. Naming follows processing
# order: T-Mart and RAdCor are applied before the downstream AC step.
PROCESSORS = [
    {
        "variability_label": "ACOLITE DSF",
        "display_label": "ACOLITE DSF",
        "best_settings_path": SETTINGS_ROOT / "ACOLITE" / "best_sensitivity_settings_by_lake_processor.csv",
        "setting_column": "extent_km",
    },
    {
        "variability_label": "T-Mart+ACOLITE DSF",
        "display_label": "T-Mart+\nACOLITE DSF",
        "best_settings_path": SETTINGS_ROOT / "ACOLITE_TMart" / "best_acolite_tmart_sensitivity_settings_by_lake.csv",
        "setting_column": "extent_km",
    },
    {
        "variability_label": "ACOLITE TSDSF+RAdCor",
        "display_label": "ACOLITE TSDSF\n+RAdCor",
        "best_settings_path": SETTINGS_ROOT / "ACOLITE_RAdCor" / "best_sensitivity_settings_by_lake_processor.csv",
        "setting_column": "kernel_radius",
    },
    {
        "variability_label": "RAdCor+POLYMER",
        "display_label": "RAdCor+\nPOLYMER",
        "best_settings_path": SETTINGS_ROOT / "Polymer_RAdCor" / "best_sensitivity_settings_by_lake_processor.csv",
        "setting_column": "kernel_radius",
    },
    {
        "variability_label": "RAdCor+C2RCC",
        "display_label": "RAdCor+\nC2RCC",
        "best_settings_path": SETTINGS_ROOT / "C2RCC_RAdCor" / "best_sensitivity_settings_by_lake_processor.csv",
        "setting_column": "kernel_radius",
    },
]

FIGURE_WIDTH_CM = 16.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / 2.54
FIGURE_HEIGHT_IN = FIGURE_WIDTH_IN * 0.60
SAVE_DPI = 450

TITLE_Y_OFFSET_IN = 0.24
LEGEND_Y_OFFSET_IN = 0.46
TOP_MARGIN_IN = 0.86
# Dumbbell figure keeps a legend row that the boxplot figure no longer has, so it
# needs extra clearance below the legend for the two-line column titles.
DUMBBELL_TOP_MARGIN_IN = 1.05
BOTTOM_MARGIN_IN = 1.05
LEFT_MARGIN_IN = 0.55

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.titlepad": 4.0,
    }
)


def lake_of(campaign: str) -> str:
    return campaign[:3]


def lighten_rgba(color, fraction: float):
    """Blend an RGBA color toward white by `fraction`, for a lighter box fill
    while the edge/swarm colors stay at the true, unblended campaign hue."""
    r, g, b, a = color
    return (r + (1 - r) * fraction, g + (1 - g) * fraction, b + (1 - b) * fraction, a)


# Matches rank_best_sensitivity_settings.py's COMMON_EXTENTS_KM / parse_extent_km:
# the best-settings tables store the snapped canonical extent (e.g. 7.0), while
# msa_variability_setting_summary.csv stores the raw physical tile extent (e.g.
# 7.02), so exact float comparison must snap first or every DSF-extent match fails.
COMMON_EXTENTS_KM = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0]


def snap_extent_km(value: float) -> float:
    return min(COMMON_EXTENTS_KM, key=lambda candidate: abs(candidate - value))


def load_setting_distributions() -> pd.DataFrame:
    frame = pd.read_csv(VARIABILITY_ROOT / "msa_variability_setting_summary.csv")
    return frame


def load_primary_best(spec: dict) -> pd.DataFrame:
    frame = pd.read_csv(spec["best_settings_path"])
    frame = frame[frame["is_primary_best"] == True].copy()  # noqa: E712
    return frame[["lake", "lut", spec["setting_column"]]]


def match_setting_value(setting_frame: pd.DataFrame, spec: dict, lut: str, setting_value) -> pd.DataFrame:
    column = spec["setting_column"]
    subset = setting_frame[setting_frame["lut"] == lut]
    if column == "kernel_radius":
        target = str(setting_value).strip()
        return subset[subset["kernel_radius"].astype(str).str.strip() == target]
    target = snap_extent_km(float(setting_value))
    snapped = pd.to_numeric(subset["extent_km"], errors="coerce").map(
        lambda v: snap_extent_km(v) if pd.notna(v) else np.nan
    )
    return subset[np.isclose(snapped, target)]


def build_panel_grid(top_margin_in: float = TOP_MARGIN_IN):
    fig, axes = plt.subplots(1, len(PROCESSORS), figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), sharey=True)
    fig.subplots_adjust(
        left=LEFT_MARGIN_IN / FIGURE_WIDTH_IN,
        right=0.995,
        top=1 - top_margin_in / FIGURE_HEIGHT_IN,
        bottom=BOTTOM_MARGIN_IN / FIGURE_HEIGHT_IN,
        wspace=0.12,
    )
    return fig, axes


def style_panel(ax, processor_label: str, show_ylabel: bool):
    ax.set_yscale("log")
    ax.set_title(processor_label, fontsize=8.5, linespacing=1.05)
    ax.set_xticks(range(len(CAMPAIGNS_IN_COLOR_ORDER)))
    ax.set_xticklabels(CAMPAIGNS_IN_COLOR_ORDER, rotation=90, ha="center", va="top")
    ax.set_xlim(-0.9, len(CAMPAIGNS_IN_COLOR_ORDER) - 1 + 0.9)
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Median MSA per setting [%] (log)")


def plot_box_swarm():
    setting_distributions = load_setting_distributions()
    fig, axes = build_panel_grid()

    rng = np.random.default_rng(0)
    for col, spec in enumerate(PROCESSORS):
        ax = axes[col]
        proc_frame = setting_distributions[setting_distributions["processor_label"] == spec["variability_label"]]
        best_frame = load_primary_best(spec)

        for xi, campaign in enumerate(CAMPAIGNS_IN_COLOR_ORDER):
            campaign_values = proc_frame[proc_frame["campaign"] == campaign]["setting_median_msa_percent"].dropna()
            if campaign_values.empty:
                continue
            color = CAMPAIGN_COLORS[campaign]
            box = ax.boxplot(
                campaign_values,
                positions=[xi],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
                boxprops={"facecolor": lighten_rgba(color, 0.55), "edgecolor": color, "linewidth": 1.1},
                whiskerprops={"color": color, "linewidth": 0.7},
                capprops={"color": color, "linewidth": 0.7},
            )
            jitter = rng.uniform(-0.18, 0.18, size=len(campaign_values))
            ax.scatter(
                xi + jitter,
                campaign_values,
                s=4,
                color=color,
                edgecolor="none",
                zorder=3,
            )

            lake_best = best_frame[best_frame["lake"] == lake_of(campaign)]
            if not lake_best.empty:
                lut = lake_best.iloc[0]["lut"]
                setting_value = lake_best.iloc[0][spec["setting_column"]]
                matched = match_setting_value(
                    proc_frame[proc_frame["campaign"] == campaign], spec, lut, setting_value
                )
                if not matched.empty:
                    ax.scatter(
                        [xi],
                        [matched.iloc[0]["setting_median_msa_percent"]],
                        marker="*",
                        s=55,
                        color="black",
                        edgecolor="white",
                        linewidth=0.4,
                        zorder=4,
                    )

        style_panel(ax, spec["display_label"], show_ylabel=(col == 0))

    fig.suptitle(
        "Performance Variability across Settings, by Campaign",
        y=1 - TITLE_Y_OFFSET_IN / FIGURE_HEIGHT_IN,
        fontweight="bold",
    )

    png_path = PLOT_ROOT / "msa_variability_by_processor_campaign_boxplot.png"
    svg_path = PLOT_ROOT / "msa_variability_by_processor_campaign_boxplot.svg"
    fig.savefig(png_path, dpi=SAVE_DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def plot_best_worst_dumbbell():
    summary = pd.read_csv(
        Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\results\plots\sensitivity_analysis\settings_variation_summary.csv")
    )
    processor_key_map = {
        "ACOLITE DSF": "ACOLITE",
        "T-Mart+ACOLITE DSF": "ACOLITE_TMart",
        "ACOLITE TSDSF+RAdCor": "ACOLITE_RAdCor",
        "RAdCor+POLYMER": "Polymer_RAdCor",
        "RAdCor+C2RCC": "C2RCC_RAdCor",
    }
    fig, axes = build_panel_grid(top_margin_in=DUMBBELL_TOP_MARGIN_IN)

    for col, spec in enumerate(PROCESSORS):
        ax = axes[col]
        proc_key = processor_key_map[spec["variability_label"]]
        proc_frame = summary[summary["processor"] == proc_key]

        for xi, campaign in enumerate(CAMPAIGNS_IN_COLOR_ORDER):
            row = proc_frame[proc_frame["campaign"] == campaign]
            if row.empty:
                continue
            row = row.iloc[0]
            color = CAMPAIGN_COLORS[campaign]
            ax.plot(
                [xi, xi],
                [row["best_median_msa"], row["worst_median_msa"]],
                color=color,
                linewidth=1.4,
                zorder=2,
                solid_capstyle="round",
            )
            ax.scatter([xi], [row["best_median_msa"]], color=color, edgecolor="black", linewidth=0.4, s=20, zorder=3)
            ax.scatter(
                [xi], [row["worst_median_msa"]], marker="^", color=color, edgecolor="black", linewidth=0.4, s=24, zorder=3
            )

        style_panel(ax, spec["display_label"], show_ylabel=(col == 0))
        if col == 0:
            ax.set_ylabel("Median MSA [%] (log)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="0.3", linestyle="none", markersize=5, label="Best tested setting"),
        plt.Line2D([0], [0], marker="^", color="0.3", linestyle="none", markersize=5.5, label="Worst tested setting"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - LEGEND_Y_OFFSET_IN / FIGURE_HEIGHT_IN),
        frameon=False,
        ncol=2,
    )
    fig.suptitle(
        "Best vs. Worst Tested Setting, by Campaign",
        y=1 - TITLE_Y_OFFSET_IN / FIGURE_HEIGHT_IN,
        fontweight="bold",
    )

    png_path = PLOT_ROOT / "msa_variability_by_processor_campaign_dumbbell.png"
    svg_path = PLOT_ROOT / "msa_variability_by_processor_campaign_dumbbell.svg"
    fig.savefig(png_path, dpi=SAVE_DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


OVERALL_POINT_COLOR = "#2a78d6"


def plot_overall_by_processor():
    """Single-panel companion to the per-campaign box+swarm: one box per processor,
    pooling every tested setting across all campaigns. Points are a single neutral
    color (not campaign-colored), since this figure's point is the pooled spread
    per processor, not which campaign drives it."""
    setting_distributions = load_setting_distributions()

    figure_height_in = FIGURE_WIDTH_IN * 0.55
    top_margin_in = 0.55
    bottom_margin_in = 0.85
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, figure_height_in))
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=1 - top_margin_in / figure_height_in,
        bottom=bottom_margin_in / figure_height_in,
    )

    rng = np.random.default_rng(0)
    for xi, spec in enumerate(PROCESSORS):
        proc_frame = setting_distributions[setting_distributions["processor_label"] == spec["variability_label"]]
        values = proc_frame["setting_median_msa_percent"].dropna()
        if values.empty:
            continue
        ax.boxplot(
            values,
            positions=[xi],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            boxprops={"facecolor": "0.85", "edgecolor": "0.25", "linewidth": 0.8},
            whiskerprops={"color": "0.25", "linewidth": 0.8},
            capprops={"color": "0.25", "linewidth": 0.8},
            zorder=2,
        )
        jitter = rng.uniform(-0.22, 0.22, size=len(values))
        ax.scatter(
            xi + jitter,
            values,
            s=5,
            color=OVERALL_POINT_COLOR,
            alpha=0.55,
            edgecolor="none",
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xticks(range(len(PROCESSORS)))
    ax.set_xticklabels([spec["display_label"] for spec in PROCESSORS], fontsize=9)
    ax.set_xlim(-0.6, len(PROCESSORS) - 1 + 0.6)
    ax.set_ylabel("Median MSA per setting [%] (log)")
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Performance Variability across Settings, Overall",
        y=1 - 0.30 / figure_height_in,
        fontweight="bold",
    )

    png_path = PLOT_ROOT / "msa_variability_overall_by_processor.png"
    svg_path = PLOT_ROOT / "msa_variability_overall_by_processor.svg"
    fig.savefig(png_path, dpi=SAVE_DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def main():
    plot_box_swarm()
    plot_best_worst_dumbbell()
    plot_overall_by_processor()


if __name__ == "__main__":
    main()
