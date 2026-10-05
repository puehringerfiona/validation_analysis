#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Sizing/typography matches plots/sensitivity_analysis/plot_campaign_dsf_extent_sensitivity_by_lut.py:
# figures are authored at their final 16 cm print width with absolute point sizes, so no
# rescaling is expected once placed in the document at that width, and no bbox_inches="tight"
# is used on save (that would crop below the authored width).
FIGURE_WIDTH_CM = 16.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / 2.54
SAVE_DPI = 450
ANNOTATION_FONTSIZE = 8

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
        "figure.titleweight": "bold",
        "axes.titlepad": 3.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

STATS_DIR = Path(__file__).resolve().parents[2] / "stats"
if str(STATS_DIR) not in sys.path:
    sys.path.insert(0, str(STATS_DIR))

TRANSECTS_MODULE_DIR = Path(
    r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\quality_control"
)
if str(TRANSECTS_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSECTS_MODULE_DIR))

from campaign_locations import location_for_campaign
from campaign_transects import get_campaign_transects


INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
ORIENTATION_OUTPUT_SUBDIR = "raw_rrs_spectra_by_raw_relative_azimuth"
DEFAULT_ORIENTATION_OUTPUT_ROOT = INSITU_ROOT / "all" / ORIENTATION_OUTPUT_SUBDIR
DEFAULT_OUTPUT_ROOT = INSITU_ROOT / "all" / "quality_control"

HARD_FILTER_COLUMN = "hard_filter_retain"
ELIGIBILITY_EXCLUDE_COLUMN = "eligibility_exclude"
RAW_RELATIVE_AZIMUTH_COLUMN = "raw_relative_azimuth_angle"
TRANSECT_COLUMN = "transect_nr"
RRS_FILTER_COLUMNS = (
    "rrs_required_nonfinite",
    "rrs_required_nonpositive",
    "rrs_negative_baseline_shift",
    "rrs_positive_baseline_shift",
)

# In situ QC exclusion criteria used to color the "removed" panel of the
# hard_filter_states figure. A row can fail more than one of these; the order
# below (matching the order the reasons are appended in 2_hard_filter.py /
# 5_run_qc_pipeline.py) determines which single reason it is attributed to
# for coloring. Rows that already failed eligibility (ELIGIBILITY_EXCLUDE_COLUMN)
# are excluded from this panel entirely rather than attributed to a reason here.
QC_EXCLUSION_REASON_GROUPS = {
    "geometry": {
        "label": "VAZ outside 75-150°",
        "columns": ("hard_filter_vaz_outside_75_150",),
        "color": "#4a3aa7",
    },
    "rrs_invalid": {
        "label": "required Rrs non-finite or non-positive",
        "columns": ("rrs_required_nonfinite", "rrs_required_nonpositive"),
        "color": "#eda100",
    },
    "negative_baseline": {
        "label": "negative baseline shift",
        "columns": ("rrs_negative_baseline_shift",),
        "color": "#e34948",
    },
    "positive_baseline": {
        "label": "positive baseline shift",
        "columns": ("rrs_positive_baseline_shift",),
        "color": "#e87ba4",
    },
    "raw_radiometry": {
        "label": "unstable/missing radiometry at 560 nm",
        "columns": ("hard_filter_raw_radiometry_nested_median_neighbour_560_180s_gt25pct_or_nonfinite",),
        "color": "#2a78d6",
    },
    "bottom_reflectance": {
        "label": "bottom reflectance",
        "columns": ("hard_filter_bottom_reflectance_exclude",),
        "color": "#008300",
    },
}
QC_EXCLUSION_EXTRA_COLUMNS = tuple(
    sorted(
        {
            column
            for config in QC_EXCLUSION_REASON_GROUPS.values()
            for column in config["columns"]
            if column not in RRS_FILTER_COLUMNS
        }
    )
)

ORIENTATION_GROUPS = {
    "solar": {
        "label": "solar orientation (75-150 deg)",
        "color": "#1f77b4",
        "range": (75.0, 150.0),
    },
    "antisolar": {
        "label": "antisolar orientation (210-285 deg)",
        "color": "#d62728",
        "range": (210.0, 285.0),
    },
    "other": {
        "label": "other / unavailable",
        "color": "#9aa0a6",
        "range": None,
    },
}

DEFAULT_WAVELENGTH_MIN_NM = 350.0
DEFAULT_WAVELENGTH_MAX_NM = 900.0

FILTER_FIGURES = {
    "isolated_rrs_filters": {
        "title": "isolated Rrs filters",
        "suptitle_suffix": "isolated Rrs filters",
        "modes": (
            {"title": "all measurements", "column": None, "retain_true": True},
            {"title": "Rrs filter retained", "column": "isolated_rrs_filters", "retain_true": True},
        ),
    },
    "hard_filter_states": {
        "title": "",
        "suptitle_suffix": "",
        "modes": (
            {"title": "all measurements", "column": None, "retain_true": True},
            {"title": "removed", "column": "qc_exclusion_reasons", "retain_true": True},
            {"title": "retained", "column": HARD_FILTER_COLUMN, "retain_true": True},
        ),
    },
}


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def orientation_series(angles: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(angles, errors="coerce")
    orientation = pd.Series("other", index=angles.index)
    solar = numeric.between(
        ORIENTATION_GROUPS["solar"]["range"][0],
        ORIENTATION_GROUPS["solar"]["range"][1],
        inclusive="both",
    )
    antisolar = numeric.between(
        ORIENTATION_GROUPS["antisolar"]["range"][0],
        ORIENTATION_GROUPS["antisolar"]["range"][1],
        inclusive="both",
    )
    orientation.loc[solar] = "solar"
    orientation.loc[antisolar] = "antisolar"
    return orientation


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", **kwargs)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).replace(".0", "") for column in df.columns]
    return df


def find_rrs_columns(
    df: pd.DataFrame,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
) -> tuple[list[str], np.ndarray]:
    rrs_info: list[tuple[float, str]] = []
    for column in df.columns:
        match = re.match(r"Rrs_(\d+\.?\d*)$", str(column))
        if match:
            wavelength = float(match.group(1))
            if wavelength_min_nm <= wavelength <= wavelength_max_nm:
                rrs_info.append((wavelength, column))

    if not rrs_info:
        raise ValueError("No Rrs columns found in the selected wavelength range.")

    rrs_info = sorted(rrs_info, key=lambda item: item[0])
    return [column for _, column in rrs_info], np.array([wavelength for wavelength, _ in rrs_info])


def load_metadata(
    path: Path,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    require_orientation: bool,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    df = normalize_columns(read_csv_auto(path))
    required_columns = [
        column
        for column in (
            HARD_FILTER_COLUMN,
            ELIGIBILITY_EXCLUDE_COLUMN,
            TRANSECT_COLUMN,
            *RRS_FILTER_COLUMNS,
            *QC_EXCLUSION_EXTRA_COLUMNS,
        )
    ]
    if require_orientation:
        required_columns.append(RAW_RELATIVE_AZIMUTH_COLUMN)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing required column(s): {', '.join(missing)}")

    rrs_columns, wavelengths = find_rrs_columns(
        df,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
    )
    df[rrs_columns] = df[rrs_columns].apply(pd.to_numeric, errors="coerce")
    for column in (HARD_FILTER_COLUMN, ELIGIBILITY_EXCLUDE_COLUMN, *RRS_FILTER_COLUMNS, *QC_EXCLUSION_EXTRA_COLUMNS):
        df[column] = parse_bool_series(df[column])
    df[TRANSECT_COLUMN] = pd.to_numeric(df[TRANSECT_COLUMN], errors="coerce")
    if require_orientation:
        df = df.copy()
        df[RAW_RELATIVE_AZIMUTH_COLUMN] = pd.to_numeric(
            df[RAW_RELATIVE_AZIMUTH_COLUMN],
            errors="coerce",
        )
        df["raw_relative_azimuth_orientation"] = orientation_series(
            df[RAW_RELATIVE_AZIMUTH_COLUMN]
        )
    return df, rrs_columns, wavelengths


def finite_ylim(
    matrix: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[float, float]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    lower = min(0.0, float(np.nanpercentile(finite, lower_percentile)))
    upper = float(np.nanpercentile(finite, upper_percentile))
    if lower == upper:
        upper = lower + 1e-12
    return lower, upper + (upper - lower) * 0.05


def campaign_metadata_files(campaign_dir: Path) -> list[Path]:
    campaign = campaign_dir.name
    qc_dir = campaign_dir / "visualization_analysis" / "quality_control"
    files = sorted(qc_dir.glob(f"{campaign}_metadata_with_Rrs_*_final_qc.csv"))
    files = [path for path in files if "_current_" not in path.name]
    return files


def campaign_dirs(insitu_root: Path, campaigns: list[str]) -> list[Path]:
    if campaigns:
        return [insitu_root / campaign for campaign in campaigns]
    return sorted(
        path
        for path in insitu_root.iterdir()
        if path.is_dir() and re.match(r"\d{8}_", path.name)
    )


def isolated_rrs_filter_retained(metadata: pd.DataFrame) -> pd.Series:
    exclude = pd.Series(False, index=metadata.index)
    for column in RRS_FILTER_COLUMNS:
        exclude = exclude | metadata[column]
    return ~exclude


def qc_exclusion_group_mask(metadata: pd.DataFrame, config: dict[str, object]) -> pd.Series:
    mask = pd.Series(False, index=metadata.index)
    for column in config["columns"]:
        mask = mask | metadata[column]
    return mask


def qc_exclusion_mask(metadata: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=metadata.index)
    for config in QC_EXCLUSION_REASON_GROUPS.values():
        mask = mask | qc_exclusion_group_mask(metadata, config)
    # Rows that already failed eligibility are excluded upstream of this QC
    # stage and are not attributed to any of the criteria above.
    return mask & ~metadata[ELIGIBILITY_EXCLUDE_COLUMN]


def qc_exclusion_reason(metadata: pd.DataFrame) -> pd.Series:
    reason = pd.Series("", index=metadata.index)
    assigned = pd.Series(False, index=metadata.index)
    for key, config in QC_EXCLUSION_REASON_GROUPS.items():
        matches = qc_exclusion_group_mask(metadata, config) & ~assigned
        reason.loc[matches] = key
        assigned = assigned | matches
    return reason


def qc_exclusion_legend_handles(subset: pd.DataFrame, present_keys: set[str]) -> list[Line2D]:
    # Counts are inclusive (how many removed rows satisfy each criterion, regardless of
    # whether it's the one that won the line's color, since a row can fail more than
    # one) and are shown next to each label instead of a separate in-panel text block.
    # Categories with zero matches are omitted.
    handles = []
    for key, config in QC_EXCLUSION_REASON_GROUPS.items():
        if key not in present_keys:
            continue
        count = int(qc_exclusion_group_mask(subset, config).sum())
        if count == 0:
            continue
        handles.append(
            Line2D([0], [0], color=config["color"], lw=1.5, label=f"{config['label']} (n={count})")
        )
    return handles


def campaign_transect_mask(metadata: pd.DataFrame, campaign: str) -> pd.Series:
    # The "retained" panel is meant to show the final analysis-ready dataset, which
    # excludes any transect not selected in campaign_transects.py (e.g. a pilot
    # transect that was walked but never adopted, or transect_nr == -1 for points
    # collected outside any defined transect) even if it otherwise passed hard_filter.
    try:
        transects = get_campaign_transects(campaign, require_metadata=False)
    except KeyError:
        return pd.Series(True, index=metadata.index)
    return metadata[TRANSECT_COLUMN].isin(transects)


def filtered_metadata(metadata: pd.DataFrame, mode: dict[str, object], campaign: str) -> pd.DataFrame:
    column = mode["column"]
    if column is None:
        return metadata.copy()
    if column == "isolated_rrs_filters":
        retained = isolated_rrs_filter_retained(metadata)
    elif column == "qc_exclusion_reasons":
        retained = qc_exclusion_mask(metadata)
    else:
        retained = metadata[column]
    if column == HARD_FILTER_COLUMN:
        retained = retained & campaign_transect_mask(metadata, campaign)
    if mode["retain_true"]:
        return metadata[retained].copy()
    return metadata[~retained].copy()


def orientation_count_text(metadata: pd.DataFrame) -> str:
    counts = metadata["raw_relative_azimuth_orientation"].value_counts()
    return "\n".join(
        [
            f"n={len(metadata)}",
            f"solar={int(counts.get('solar', 0))}",
            f"antisolar={int(counts.get('antisolar', 0))}",
            f"other={int(counts.get('other', 0))}",
        ]
    )


def orientation_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=config["color"],
            lw=1.5,
            label=config["label"],
        )
        for config in ORIENTATION_GROUPS.values()
    ]


def sensor_from_metadata_file(metadata_file: Path) -> str:
    match = re.search(r"_metadata_with_Rrs_(?P<sensor>.+)_final_qc\.csv$", metadata_file.name)
    return match.group("sensor") if match else "unknown_sensor"


def plot_campaign(
    campaign_dir: Path,
    metadata_file: Path,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    color_by_orientation: bool,
    output_root: Path | None,
) -> list[Path]:
    campaign = campaign_dir.name
    sensor = sensor_from_metadata_file(metadata_file)
    metadata, rrs_columns, wavelengths = load_metadata(
        metadata_file,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        require_orientation=color_by_orientation,
    )
    all_matrix = metadata[rrs_columns].to_numpy(dtype=float).T
    percentiles = (5.0, 95.0) if campaign in {"20260423_ZRH", "20260430_ZRH"} else (2.0, 98.0)
    y_limits = finite_ylim(
        all_matrix,
        lower_percentile=percentiles[0],
        upper_percentile=percentiles[1],
    )

    resolved_output_root = output_root if output_root is not None else DEFAULT_OUTPUT_ROOT
    output_dir = resolved_output_root / campaign
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for figure_key, figure_config in FILTER_FIGURES.items():
        modes = figure_config["modes"]
        has_qc_exclusion_panel = any(mode["column"] == "qc_exclusion_reasons" for mode in modes)
        needs_bottom_legend = color_by_orientation or (has_qc_exclusion_panel and not color_by_orientation)
        # The qc-exclusion legend wraps at ncol=2; the fixed 1.7in/0.95in margins below
        # were tuned for campaigns with <=4 exclusion categories present (2 legend rows).
        # A few campaigns hit 5 categories (3 rows), which the fixed margins are too
        # short for and the legend collides with the x-axis label. Extra rows beyond 2
        # get proportionally more room; campaigns with <=4 categories are unaffected.
        qc_legend_rows = 2
        if has_qc_exclusion_panel and not color_by_orientation:
            removed_subset = metadata[qc_exclusion_mask(metadata)]
            present_count = sum(
                1
                for config in QC_EXCLUSION_REASON_GROUPS.values()
                if int(qc_exclusion_group_mask(removed_subset, config).sum()) > 0
            )
            qc_legend_rows = max(1, -(-present_count // 2))
        extra_legend_rows = max(0, qc_legend_rows - 2)
        legend_row_height_in = 0.225
        panel_width_in = FIGURE_WIDTH_IN / len(modes)
        figure_height_in = (
            panel_width_in * 0.55
            + (1.7 if needs_bottom_legend else 0.9)
            + extra_legend_rows * legend_row_height_in
        )
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(modes),
            figsize=(FIGURE_WIDTH_IN, figure_height_in),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes)
        qc_exclusion_keys_present: set[str] = set()
        qc_exclusion_subset: pd.DataFrame | None = None

        for ax, mode in zip(axes, modes):
            subset = filtered_metadata(metadata, mode, campaign)
            if subset.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no measurements",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=ANNOTATION_FONTSIZE,
                )
            else:
                is_qc_exclusion_panel = mode["column"] == "qc_exclusion_reasons"
                if is_qc_exclusion_panel:
                    reason = qc_exclusion_reason(subset)
                    qc_exclusion_keys_present = set(reason[reason != ""].unique())
                    qc_exclusion_subset = subset
                    for reason_key, reason_config in QC_EXCLUSION_REASON_GROUPS.items():
                        group = subset[reason == reason_key]
                        if group.empty:
                            continue
                        matrix = group[rrs_columns].to_numpy(dtype=float).T
                        ax.plot(
                            wavelengths,
                            matrix,
                            color=reason_config["color"],
                            alpha=0.5,
                            linewidth=0.6,
                        )
                    matrix = subset[rrs_columns].to_numpy(dtype=float).T
                elif color_by_orientation:
                    for orientation_key, orientation_config in ORIENTATION_GROUPS.items():
                        group = subset[
                            subset["raw_relative_azimuth_orientation"] == orientation_key
                        ]
                        if group.empty:
                            continue
                        matrix = group[rrs_columns].to_numpy(dtype=float).T
                        ax.plot(
                            wavelengths,
                            matrix,
                            color=orientation_config["color"],
                            alpha=0.20 if orientation_key != "other" else 0.12,
                            linewidth=0.5,
                        )
                    matrix = subset[rrs_columns].to_numpy(dtype=float).T
                else:
                    matrix = subset[rrs_columns].to_numpy(dtype=float).T
                    ax.plot(
                        wavelengths,
                        matrix,
                        color="#2f6f9f",
                        alpha=0.18,
                        linewidth=0.5,
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    median = np.nanmedian(matrix, axis=1)
                ax.plot(
                    wavelengths,
                    median,
                    color="#0b1f33",
                    linewidth=1.2,
                    label="median",
                )
                if is_qc_exclusion_panel:
                    count_text = f"n={len(subset)}"
                elif color_by_orientation:
                    count_text = orientation_count_text(subset)
                else:
                    count_text = f"n={matrix.shape[1]}"
                ax.text(
                    0.98,
                    0.94,
                    count_text,
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=ANNOTATION_FONTSIZE,
                )

            ax.set_title(mode["title"])
            ax.set_xlabel("wavelength [nm]")
            ax.set_xlim(wavelength_min_nm, wavelength_max_nm)
            ax.set_ylim(*y_limits)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(True, alpha=0.25, linewidth=0.5)

        axes[0].set_ylabel(r"$R_{rs}$ [sr$^{-1}$]")
        title_parts = [location_for_campaign(campaign)]
        suffix = figure_config.get("suptitle_suffix")
        if suffix:
            title_parts.append(f"({suffix})")
        if color_by_orientation:
            title_parts.append("- raw relative azimuth orientation")
        # Margins are computed from fixed inch budgets (not fractions) so the title
        # block, panel titles, and legend keep the same absolute spacing regardless of
        # how tall this particular figure is (which varies with the number of panels).
        # No tight_layout/bbox_inches="tight" is used: the canvas must stay exactly
        # FIGURE_WIDTH_IN wide so the point sizes above are the true on-page sizes once
        # this is placed at FIGURE_WIDTH_CM in the document. Neither fig.suptitle() nor
        # fig.text() know about subplots_adjust("top"), so both y positions are set
        # explicitly from the same inch budget. Title structure matches the sensitivity
        # plots: bold main title, then the location/date line smaller and italic below
        # it, with less gap above that second line than below it.
        main_title_fontsize = plt.rcParams["figure.titlesize"]
        subtitle_fontsize = 10
        top_gap_in = 0.07
        main_title_height_in = main_title_fontsize * 1.2 / 72
        gap_main_to_subtitle_in = 0.05
        subtitle_height_in = subtitle_fontsize * 1.2 / 72
        gap_subtitle_to_panels_in = 0.09
        panel_title_height_in = 10 * 1.2 / 72  # one line at 10 pt
        titlepad_in = plt.rcParams["axes.titlepad"] / 72

        main_title_y = 1.0 - top_gap_in / figure_height_in
        subtitle_y = main_title_y - (main_title_height_in + gap_main_to_subtitle_in) / figure_height_in
        top_margin_in = (
            top_gap_in
            + main_title_height_in
            + gap_main_to_subtitle_in
            + subtitle_height_in
            + gap_subtitle_to_panels_in
            + panel_title_height_in
            + titlepad_in
        )
        bottom_margin_in = (0.95 if needs_bottom_legend else 0.5) + extra_legend_rows * legend_row_height_in
        fig.suptitle("In Situ $R_{rs}$ Spectra Quality Control", y=main_title_y)
        fig.text(
            0.5,
            subtitle_y,
            " ".join(title_parts),
            ha="center",
            va="top",
            fontsize=subtitle_fontsize,
            fontstyle="italic",
        )
        fig.subplots_adjust(
            left=0.75 / FIGURE_WIDTH_IN,
            right=1.0 - 0.15 / FIGURE_WIDTH_IN,
            top=1.0 - top_margin_in / figure_height_in,
            bottom=bottom_margin_in / figure_height_in,
            wspace=0.08,
        )
        if color_by_orientation:
            fig.legend(
                handles=orientation_legend_handles(),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=3,
                frameon=False,
            )
        elif has_qc_exclusion_panel:
            qc_legend_handles = (
                qc_exclusion_legend_handles(qc_exclusion_subset, qc_exclusion_keys_present)
                if qc_exclusion_subset is not None
                else []
            )
            if qc_legend_handles:
                fig.legend(
                    handles=qc_legend_handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.01),
                    ncol=2,
                    frameon=False,
                )
        filename_suffix = (
            f"raw_rrs_spectra_raw_relative_azimuth_orientation_{figure_key}.png"
            if color_by_orientation
            else f"raw_rrs_spectra_{figure_key}.png"
        )
        output_path = output_dir / f"{campaign}_{sensor}_{filename_suffix}"
        fig.savefig(output_path, dpi=SAVE_DPI)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot campaign-level raw Rrs spectra by QC filter state."
    )
    parser.add_argument("campaigns", nargs="*", help="Campaign IDs to process. Default: all.")
    parser.add_argument("--insitu-root", type=Path, default=INSITU_ROOT)
    parser.add_argument("--wavelength-min-nm", type=float, default=DEFAULT_WAVELENGTH_MIN_NM)
    parser.add_argument("--wavelength-max-nm", type=float, default=DEFAULT_WAVELENGTH_MAX_NM)
    parser.add_argument(
        "--color-by-raw-relative-azimuth",
        action="store_true",
        help="Color individual spectra by raw relative azimuth orientation.",
    )
    parser.add_argument(
        "--orientation-output-root",
        type=Path,
        default=DEFAULT_ORIENTATION_OUTPUT_ROOT,
        help="Output root used with --color-by-raw-relative-azimuth.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root used without --color-by-raw-relative-azimuth.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created = 0
    failures: list[tuple[str, Exception]] = []
    output_root = (
        args.orientation_output_root
        if args.color_by_raw_relative_azimuth
        else args.output_root
    )

    for campaign_dir in campaign_dirs(args.insitu_root, args.campaigns):
        metadata_files = campaign_metadata_files(campaign_dir)
        if not metadata_files:
            print(f"Skipping {campaign_dir.name}: no non-current final_qc metadata file found")
            continue
        for metadata_file in metadata_files:
            try:
                output_paths = plot_campaign(
                    campaign_dir,
                    metadata_file,
                    args.wavelength_min_nm,
                    args.wavelength_max_nm,
                    args.color_by_raw_relative_azimuth,
                    output_root,
                )
            except Exception as exc:
                failures.append((metadata_file.name, exc))
                print(f"FAILED {metadata_file.name}: {exc}")
            else:
                created += len(output_paths)
                for output_path in output_paths:
                    print(f"Saved {output_path}")

    print(f"Created {created} campaign/sensor Rrs spectra figure(s).")
    if failures:
        print("Failures:")
        for campaign, exc in failures:
            print(f"  {campaign}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
