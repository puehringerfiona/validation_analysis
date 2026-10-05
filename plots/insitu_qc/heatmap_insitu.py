#!/usr/bin/env python3
"""In-situ Rrs transect heatmaps, one per campaign/sensor/transect/plot-mode.

Typography matches plot_ramses_measurements.py's Ed/radiance figures
(DejaVu Sans, large absolute point sizes, bold main title + italic
location/date subtitle from campaign_locations.extract_location), at a
landscape (x-axis longer than y-axis) aspect. No axis grid, since it would
just sit under the pcolormesh's full-canvas color fill.

Y-axis is distance to shore [km] (performance_plot_common.compute_distance_to_shore_km,
the same per-transect raster distance-transform used for dist_to_shore_km in
the performance_metrics CSVs, and the one other scripts here rely on), not
distance along the transect: rows are plotted at uniform positions in
chronological/along-transect order, with tick labels showing the actual
distance-to-shore value at each row (performance_plot_common.set_distance_ticks,
shared with heatmap_error.py), since distance to shore is not generally
monotonic along a transect.

Runs the hard_filter_retained plot mode by default (matching what the
visualization website's galleries consume, at the same output location);
the other modes (all_measurements, hard_filter_strict_retained,
isolated_rrs_filter_retained) are available via --plot-mode for ad-hoc QC
review.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

INSITU_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
_CAMPAIGN_TRANSECTS_DIR = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\quality_control")
_CAMPAIGN_LOCATIONS_DIR = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\results\stats")
_PERFORMANCE_PLOT_COMMON_DIR = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\results\plots\processor_comparison")
for _extra_dir in (_CAMPAIGN_TRANSECTS_DIR, _CAMPAIGN_LOCATIONS_DIR, _PERFORMANCE_PLOT_COMMON_DIR):
    if str(_extra_dir) not in sys.path:
        sys.path.insert(0, str(_extra_dir))

from campaign_transects import get_campaign_transects  # noqa: E402
from campaign_locations import extract_location  # noqa: E402
from performance_plot_common import (  # noqa: E402
    CAMPAIGN_DEPTH_RASTER,
    compute_distance_to_shore_km,
    set_distance_ticks,
)


TIMESTAMP_COLUMN = "timestamp (UTC)"
X_COLUMN = "x-coordinate"
Y_COLUMN = "y-coordinate"
TRANSECT_COLUMN = "transect_nr"
QC_RETAINED_COLUMN = "qc_retained"
HARD_QC_RETAINED_COLUMN = "hard_filter_retain"
HARD_QC_STRICT_RETAINED_COLUMN = "hard_filter_retain_strict"
RRS_FILTER_COLUMNS = (
    "rrs_required_nonfinite",
    "rrs_required_nonpositive",
    "rrs_negative_baseline_shift",
    "rrs_positive_baseline_shift",
)

DEFAULT_QC_SUFFIX = "final_qc"
DEFAULT_SAVE_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu\all\transect_heatmaps")
DEFAULT_PLOT_MODES = ("hard_filter_retained",)
VALID_QC_SUFFIXES = (
    "final_qc",
    "current_final_qc",
    "current_new_final_qc",
    "current_qc",
)
VALID_PLOT_MODES = (
    "all_measurements",
    "hard_filter_retained",
    "hard_filter_strict_retained",
    "isolated_rrs_filter_retained",
)

DEFAULT_DIST_CACHE_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu\all\dist_to_shore_cache")

WAVELENGTH_MIN_NM = 415.0
LANDSAT_WAVELENGTH_MAX_NM = 680.0
SENTINEL_WAVELENGTH_MAX_NM = 720.0
TICK_STEP_NM = 50

# Matches plot_ramses_measurements.py's apply_plot_style()/style_axis() exactly,
# so this heatmap reads as part of the same figure family as the Ed/radiance
# plots: same font, same absolute point sizes, same tick/spine weights.
TITLE_FONTSIZE = 40
SUBTITLE_FONTSIZE = 34
AXIS_LABEL_FONTSIZE = 36
TICK_FONTSIZE = 30
CM_TO_IN = 1 / 2.54
FIGURE_WIDTH_CM = 35.56
FIGURE_HEIGHT_CM = 20.32
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM * CM_TO_IN
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_CM * CM_TO_IN
SAVE_DPI = 600

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
    }
)


def get_sensor_name(csv_path: Path) -> str:
    match = re.search(
        r"_metadata_with_Rrs_(?P<sensor>.+?)_"
        r"(?:hard_filter_qc|current_new_final_qc|current_final_qc|"
        r"final_qc|current_qc)\.csv$",
        csv_path.name,
    )
    if match is None:
        return "unknown_sensor"
    return match.group("sensor")


def sensor_wavelength_max(csv_path: Path) -> float:
    filename = csv_path.name.upper()
    if "S2A" in filename or "S2B" in filename:
        return SENTINEL_WAVELENGTH_MAX_NM
    if "L8" in filename or "L9" in filename:
        return LANDSAT_WAVELENGTH_MAX_NM
    raise ValueError(f"Unrecognized sensor in filename: {csv_path.name}")


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def get_qc_retained_column(df: pd.DataFrame) -> str:
    if HARD_QC_RETAINED_COLUMN in df.columns:
        return HARD_QC_RETAINED_COLUMN
    if QC_RETAINED_COLUMN in df.columns:
        return QC_RETAINED_COLUMN
    raise KeyError(
        f"Missing retained QC column. Expected '{HARD_QC_RETAINED_COLUMN}'"
        f" or '{QC_RETAINED_COLUMN}'."
    )


def get_retained_column_for_plot_mode(df: pd.DataFrame, plot_mode: str) -> str | None:
    if plot_mode == "all_measurements":
        return None
    if plot_mode == "hard_filter_retained":
        return get_qc_retained_column(df)
    if plot_mode == "hard_filter_strict_retained":
        if HARD_QC_STRICT_RETAINED_COLUMN not in df.columns:
            raise KeyError(
                f"Missing retained QC column. Expected "
                f"'{HARD_QC_STRICT_RETAINED_COLUMN}'."
            )
        return HARD_QC_STRICT_RETAINED_COLUMN
    if plot_mode == "isolated_rrs_filter_retained":
        missing = [col for col in RRS_FILTER_COLUMNS if col not in df.columns]
        if missing:
            raise KeyError(
                "Missing Rrs filter column(s): "
                + ", ".join(missing)
                + ". Expected current final_qc metadata with Rrs hard-filter flags."
            )
        return None
    raise ValueError(f"Unknown plot mode: {plot_mode}")


def retained_mask_for_plot_mode(df: pd.DataFrame, plot_mode: str) -> pd.Series:
    retained_column = get_retained_column_for_plot_mode(df, plot_mode)
    if retained_column is not None:
        return parse_bool_series(df[retained_column])

    if plot_mode == "isolated_rrs_filter_retained":
        exclude = pd.Series(False, index=df.index)
        for column in RRS_FILTER_COLUMNS:
            exclude = exclude | parse_bool_series(df[column])
        return ~exclude

    return pd.Series(True, index=df.index)


def compute_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("Cannot compute edges for an empty array.")

    if len(values) == 1:
        delta = 0.5
        return np.array([values[0] - delta, values[0] + delta], dtype=float)

    mid = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (values[1] - values[0]) / 2.0
    last = values[-1] + (values[-1] - values[-2]) / 2.0
    return np.concatenate([[first], mid, [last]])


def find_metadata_files(
    campaign: str,
    insitu_root: Path,
    qc_suffix: str,
    sensor: str | None = None,
    qc_subdir: Path | None = None,
) -> list[Path]:
    qc_dir = insitu_root / campaign / "visualization_analysis" / "quality_control"
    if qc_subdir is not None:
        qc_dir = qc_dir / qc_subdir
    pattern = f"{campaign}_metadata_with_Rrs_*_{qc_suffix}.csv"
    files = sorted(qc_dir.glob(pattern))
    if qc_suffix == "final_qc":
        files = [path for path in files if "_current_" not in path.name]
    if sensor is None:
        return files

    sensor_upper = sensor.upper()
    return [path for path in files if get_sensor_name(path).upper() == sensor_upper]


def find_scale_metadata_files(
    campaign: str,
    insitu_root: Path,
    sensor: str | None = None,
    qc_subdir: Path | None = None,
) -> list[Path]:
    files: list[Path] = []
    for qc_suffix in VALID_QC_SUFFIXES:
        files.extend(
            find_metadata_files(
                campaign,
                insitu_root,
                qc_suffix=qc_suffix,
                sensor=sensor,
                qc_subdir=qc_subdir,
            )
        )
    return sorted(set(files))


def find_color_scale_metadata_files(
    campaign: str,
    insitu_root: Path,
    qc_suffix: str,
    sensor: str | None = None,
    qc_subdir: Path | None = None,
) -> list[Path]:
    if qc_suffix in {"current_qc", "current_final_qc", "current_new_final_qc"}:
        return find_metadata_files(
            campaign,
            insitu_root,
            qc_suffix="final_qc",
            sensor=sensor,
        )
    return find_metadata_files(
        campaign,
        insitu_root,
        qc_suffix=qc_suffix,
        sensor=sensor,
        qc_subdir=qc_subdir,
    )


def get_campaigns(
    campaigns: list[str] | None,
    insitu_root: Path,
    qc_suffix: str,
    sensor: str | None = None,
    qc_subdir: Path | None = None,
) -> dict[str, list[int]]:
    configured = get_campaign_transects(require_metadata=False)

    if campaigns:
        unknown = sorted(set(campaigns) - set(configured))
        if unknown:
            raise KeyError(
                "No transect selection configured for campaign(s): "
                + ", ".join(unknown)
            )
        return {campaign: configured[campaign] for campaign in campaigns}

    return {
        campaign: []
        for campaign in configured
        if find_metadata_files(
            campaign,
            insitu_root,
            qc_suffix=qc_suffix,
            sensor=sensor,
            qc_subdir=qc_subdir,
        )
    }


def find_rrs_columns_windowed(df: pd.DataFrame, wl_min: float, wl_max: float) -> tuple[list[str], np.ndarray]:
    rrs_info: list[tuple[float, str]] = []
    for col in df.columns:
        match = re.match(r"Rrs_(\d+\.?\d*)$", str(col))
        if match:
            wavelength = float(match.group(1))
            if wl_min <= wavelength <= wl_max:
                rrs_info.append((wavelength, col))
    if not rrs_info:
        raise ValueError(f"No Rrs columns found within {wl_min:g}-{wl_max:g} nm window.")
    rrs_info.sort(key=lambda item: item[0])
    return [col for _, col in rrs_info], np.array([wl for wl, _ in rrs_info], dtype=float)


def read_metadata(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=None, engine="python").rename(
        columns=lambda c: c.replace(".0", "") if isinstance(c, str) else c
    )


def discover_transects(metadata_files: list[Path]) -> list[int]:
    transects: set[int] = set()

    for metadata_file in metadata_files:
        df = read_metadata(metadata_file)
        if TRANSECT_COLUMN not in df.columns:
            raise KeyError(
                f"Missing required column '{TRANSECT_COLUMN}' in {metadata_file}"
            )

        values = pd.to_numeric(df[TRANSECT_COLUMN], errors="coerce")
        for value in sorted(values[np.isfinite(values)].unique()):
            if value > 0 and float(value).is_integer():
                transects.add(int(value))

    if not transects:
        raise ValueError("No positive integer transect numbers found.")

    return sorted(transects)


def load_transect_data(
    csv_path: Path,
    transect: int,
    plot_mode: str,
    wl_min: float,
    wl_max: float,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    if plot_mode not in VALID_PLOT_MODES:
        raise ValueError(f"Unknown plot mode: {plot_mode}")

    insitu_df = read_metadata(csv_path)
    required_columns = [
        X_COLUMN,
        Y_COLUMN,
        TRANSECT_COLUMN,
        TIMESTAMP_COLUMN,
    ]
    retained_column = get_retained_column_for_plot_mode(insitu_df, plot_mode)
    if retained_column is not None:
        required_columns.append(retained_column)
    if plot_mode == "isolated_rrs_filter_retained":
        required_columns.extend(RRS_FILTER_COLUMNS)

    missing_columns = [col for col in required_columns if col not in insitu_df.columns]
    if missing_columns:
        raise KeyError(
            "Missing required column(s): "
            + ", ".join(missing_columns)
            + "\nAvailable columns: "
            + ", ".join(map(str, insitu_df.columns))
        )

    for col in [X_COLUMN, Y_COLUMN, TRANSECT_COLUMN]:
        insitu_df[col] = pd.to_numeric(insitu_df[col], errors="coerce")

    keep_mask = (
        np.isfinite(insitu_df[X_COLUMN])
        & np.isfinite(insitu_df[Y_COLUMN])
        & (insitu_df[TRANSECT_COLUMN] == transect)
    )
    if plot_mode != "all_measurements":
        keep_mask = keep_mask & retained_mask_for_plot_mode(insitu_df, plot_mode)

    insitu_df = insitu_df[keep_mask].copy()

    if insitu_df.empty:
        if plot_mode != "all_measurements":
            raise ValueError(
                f"No rows remain after transect and {plot_mode} filtering."
            )
        raise ValueError("No rows remain after transect filtering.")

    if verbose:
        print(f"  Keeping Rrs wavelengths {wl_min:g}-{wl_max:g} nm")
    rrs_cols, wavelengths = find_rrs_columns_windowed(insitu_df, wl_min, wl_max)
    insitu_df[rrs_cols] = insitu_df[rrs_cols].apply(pd.to_numeric, errors="coerce")

    insitu_df[TIMESTAMP_COLUMN] = pd.to_datetime(
        insitu_df[TIMESTAMP_COLUMN],
        utc=True,
        errors="coerce",
    )
    insitu_df = insitu_df.dropna(subset=[TIMESTAMP_COLUMN, X_COLUMN, Y_COLUMN]).copy()
    insitu_df = insitu_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    if insitu_df.empty:
        raise ValueError("No valid rows remain after timestamp/coordinate cleaning.")

    return insitu_df, rrs_cols, wavelengths


def compute_campaign_color_scale(
    metadata_files: list[Path],
    transects: list[int],
    plot_modes: tuple[str, ...] = ("hard_filter_retained",),
) -> tuple[float, float]:
    finite_chunks: list[np.ndarray] = []

    for metadata_file in metadata_files:
        wl_max = sensor_wavelength_max(metadata_file)
        for transect in transects:
            for plot_mode in plot_modes:
                try:
                    insitu_df, rrs_cols, _ = load_transect_data(
                        metadata_file,
                        transect,
                        plot_mode=plot_mode,
                        wl_min=WAVELENGTH_MIN_NM,
                        wl_max=wl_max,
                        verbose=False,
                    )
                except (KeyError, ValueError):
                    continue

                rrs_matrix = insitu_df[rrs_cols].to_numpy(dtype=float)
                finite_vals = rrs_matrix[np.isfinite(rrs_matrix)]
                if finite_vals.size == 0:
                    continue
                finite_chunks.append(finite_vals)

    if not finite_chunks:
        raise ValueError(
            "No finite retained Rrs values found for campaign color scaling."
        )

    finite = np.concatenate(finite_chunks)
    vmin = 0.0
    vmax = float(np.nanpercentile(finite, 95))

    if vmin == vmax:
        vmax = vmin + 1e-12

    return vmin, vmax


def build_heatmap_figure():
    return plt.subplots(1, 1, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))


def set_two_row_title(ax, top_text: str, bottom_text: str) -> None:
    ax.text(
        0.5, 1.14, top_text,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=TITLE_FONTSIZE, fontweight="bold",
    )
    ax.text(
        0.5, 1.03, bottom_text,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=SUBTITLE_FONTSIZE, fontstyle="italic", fontweight="normal",
    )


def style_heatmap_axis(ax, wl_min: float, wl_max: float) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.6, length=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_xlim(wl_min, wl_max)
    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP_NM))


def _dist_cache_path(cache_root: Path, campaign: str, sensor: str) -> Path:
    return cache_root / f"{campaign}_{sensor}_dist_to_shore_cache.csv"


def distance_to_shore_km_cached(
    insitu_df: pd.DataFrame,
    campaign: str,
    sensor: str,
    raster_path: Path,
    cache_root: Path,
) -> np.ndarray:
    """Per-measurement distance-to-shore, cached by timestamp across runs.

    Distance only depends on x/y, not QC status, transect, or plot mode, so
    one cache file per campaign/sensor is shared and grows to cover the full
    transect regardless of which plot mode first triggered a computation for
    a given point - later runs (any plot mode, any qc-suffix) reuse it
    instead of re-running the raster distance transform.
    """
    cache_path = _dist_cache_path(cache_root, campaign, sensor)
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame(columns=["timestamp_ns", "dist_to_shore_km"])

    cache_map = pd.Series(
        cache_df["dist_to_shore_km"].to_numpy(dtype=float),
        index=cache_df["timestamp_ns"].astype("int64"),
    )

    timestamp_ns = insitu_df[TIMESTAMP_COLUMN].astype("int64")
    result = timestamp_ns.map(cache_map).to_numpy(dtype=float, copy=True)
    missing_mask = ~np.isfinite(result)

    if missing_mask.any():
        print(f"  Computing distance-to-shore for {int(missing_mask.sum())} uncached measurement(s)")
        missing_df = insitu_df.loc[missing_mask]
        computed = compute_distance_to_shore_km(
            missing_df, raster_path, x_col=X_COLUMN, y_col=Y_COLUMN, transect_col=TRANSECT_COLUMN
        )
        result[missing_mask] = computed

        new_entries = pd.DataFrame(
            {
                "timestamp_ns": timestamp_ns[missing_mask].to_numpy(),
                "dist_to_shore_km": computed,
            }
        )
        cache_df = (
            pd.concat([cache_df, new_entries], ignore_index=True)
            .drop_duplicates(subset="timestamp_ns", keep="last")
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(cache_path, index=False)

    return result


def mode_title_for(plot_mode: str) -> str:
    if plot_mode == "all_measurements":
        return "all measurements"
    if plot_mode == "isolated_rrs_filter_retained":
        return "Rrs-filter retained measurements"
    if plot_mode == "hard_filter_strict_retained":
        return "strict hard-filter retained measurements"
    return "hard-filter retained measurements"


def plot_heatmap(
    csv_path: Path,
    output_path: Path,
    campaign: str,
    transect: int,
    plot_mode: str,
    color_scale: tuple[float, float],
    dist_cache_root: Path,
    title_note: str | None = None,
) -> None:
    print(f"Processing {csv_path.name}, transect {transect}, {plot_mode}")
    wl_max = sensor_wavelength_max(csv_path)
    insitu_df, rrs_cols, wavelengths = load_transect_data(
        csv_path, transect, plot_mode, WAVELENGTH_MIN_NM, wl_max
    )

    raster_path = CAMPAIGN_DEPTH_RASTER.get(campaign)
    if raster_path is None:
        raise KeyError(f"No distance-to-shore raster configured for campaign {campaign!r}")
    distance_to_shore_km = distance_to_shore_km_cached(
        insitu_df, campaign, get_sensor_name(csv_path), raster_path, dist_cache_root
    )

    rrs_matrix = (
        insitu_df[rrs_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .copy()
    )
    finite_vals = rrs_matrix[np.isfinite(rrs_matrix)]
    row_label = "Retained rows" if plot_mode != "all_measurements" else "Rows"
    print(f"  {row_label}: {len(insitu_df)}")
    print(f"  Rrs finite count: {finite_vals.size}")
    if finite_vals.size == 0:
        raise ValueError(f"No finite Rrs values left for {plot_mode}.")

    vmin, vmax = color_scale
    x_edges = compute_edges(wavelengths)
    # Rows are ordered chronologically along the transect (as loaded), but
    # plotted at uniform positions rather than literal distance-to-shore
    # values: distance to shore is not generally monotonic along a transect
    # (e.g. round-trip shore-to-shore transects), so it can't serve as a
    # pcolormesh axis directly. Uniform row spacing, with tick labels showing
    # the actual distance-to-shore value at each row (see set_distance_ticks).
    y_edges = compute_edges(np.arange(len(insitu_df), dtype=float))

    main_title = rf"$R_{{rs}}$ along transect {transect}"
    title_note_part = f", {title_note}" if title_note else ""
    # Keep the subtitle exactly "location, date" for the default case (no
    # qualifier needed since hard_filter_retained is what every other
    # plot/the website implicitly means by "the" in-situ heatmap); only
    # non-default modes/qc-suffixes get an explicit qualifier appended.
    qualifier = (
        "" if (plot_mode == "hard_filter_retained" and not title_note)
        else f" ({mode_title_for(plot_mode)}{title_note_part})"
    )
    subtitle = f"{extract_location(csv_path)}{qualifier}"

    fig, ax = build_heatmap_figure()
    cmap = plt.cm.turbo.copy()
    cmap.set_bad(color="white")
    mesh = ax.pcolormesh(x_edges, y_edges, rrs_matrix, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel(r"$\lambda\ [nm]$", labelpad=18, fontstyle="italic")
    ax.set_ylabel("distance to shore [km]", labelpad=22, fontstyle="italic")
    set_distance_ticks(ax, distance_to_shore_km)
    set_two_row_title(ax, main_title, subtitle)
    style_heatmap_axis(ax, WAVELENGTH_MIN_NM, wl_max)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$R_{rs}$ [sr$^{-1}$]", fontsize=AXIS_LABEL_FONTSIZE, fontstyle="italic")
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(left=0.11, right=0.97, top=0.76, bottom=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    print(f"  Saved plot to: {output_path}")


def output_path_for(
    csv_path: Path,
    campaign: str,
    transect: int,
    plot_mode: str,
    qc_suffix: str,
    save_root: Path,
) -> Path:
    sensor = get_sensor_name(csv_path)
    qc_part = f"{qc_suffix}_" if qc_suffix != "final_qc" else ""
    return save_root / f"{campaign}_{sensor}_t{transect}_{qc_part}{plot_mode}_insitu_heatmap.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create in-situ Rrs transect heatmaps (distance-to-shore y-axis) "
            "for QC metadata. Runs all campaigns by default, or only the "
            "campaign IDs supplied."
        )
    )
    parser.add_argument(
        "campaigns",
        nargs="*",
        help="Campaign IDs to process, e.g. 20230610_CST 20231008_ZRH. Default: all.",
    )
    parser.add_argument(
        "--insitu-root",
        type=Path,
        default=INSITU_ROOT,
        help=f"Root folder containing campaign directories. Default: {INSITU_ROOT}",
    )
    parser.add_argument(
        "--sensor",
        help="Optional sensor filter, e.g. S2B or L9. Default: all matching sensors.",
    )
    parser.add_argument(
        "--qc-suffix",
        choices=VALID_QC_SUFFIXES,
        default=DEFAULT_QC_SUFFIX,
        help=f"QC metadata suffix to process. Default: {DEFAULT_QC_SUFFIX}",
    )
    parser.add_argument(
        "--qc-subdir",
        type=Path,
        help=(
            "Optional subdirectory below each campaign's visualization_analysis/"
            "quality_control folder to read metadata from, e.g. 3C_O25."
        ),
    )
    parser.add_argument(
        "--plot-mode",
        choices=VALID_PLOT_MODES,
        action="append",
        dest="plot_modes",
        help=(
            "Plot mode to generate. Repeat to generate multiple modes. "
            "Default: hard_filter_retained."
        ),
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=DEFAULT_SAVE_ROOT,
        help=f"Output directory for the generated PNGs. Default: {DEFAULT_SAVE_ROOT}",
    )
    parser.add_argument(
        "--dist-cache-root",
        type=Path,
        default=DEFAULT_DIST_CACHE_ROOT,
        help=(
            "Directory for per-campaign/sensor distance-to-shore caches, "
            f"reused across runs. Default: {DEFAULT_DIST_CACHE_ROOT}"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot_modes = tuple(args.plot_modes) if args.plot_modes else DEFAULT_PLOT_MODES
    campaigns = get_campaigns(
        args.campaigns,
        args.insitu_root,
        qc_suffix=args.qc_suffix,
        sensor=args.sensor,
        qc_subdir=args.qc_subdir,
    )

    if not campaigns:
        print("No campaigns with QC metadata files found.")
        return 1

    failures: list[tuple[Path, int, Exception]] = []
    created = 0

    for campaign in campaigns:
        metadata_files = find_metadata_files(
            campaign,
            args.insitu_root,
            qc_suffix=args.qc_suffix,
            sensor=args.sensor,
            qc_subdir=args.qc_subdir,
        )
        if not metadata_files:
            print(f"Skipping {campaign}: no QC metadata files found")
            continue

        if campaign not in CAMPAIGN_DEPTH_RASTER:
            print(f"Skipping {campaign}: no distance-to-shore raster configured")
            continue

        try:
            transects = discover_transects(metadata_files)
        except Exception as exc:
            for metadata_file in metadata_files:
                failures.append((metadata_file, -1, exc))
            print(f"Skipping {campaign}: failed to discover transects: {exc}")
            continue

        try:
            scale_plot_modes = tuple(
                plot_mode for plot_mode in plot_modes if plot_mode != "all_measurements"
            )
            if not scale_plot_modes:
                scale_plot_modes = (
                    "hard_filter_retained",
                    "hard_filter_strict_retained",
                    "isolated_rrs_filter_retained",
                )
            scale_metadata_files = find_color_scale_metadata_files(
                campaign,
                args.insitu_root,
                qc_suffix=args.qc_suffix,
                sensor=args.sensor,
                qc_subdir=args.qc_subdir,
            )
            scale_transects = discover_transects(scale_metadata_files)
            color_scale = compute_campaign_color_scale(
                scale_metadata_files,
                scale_transects,
                plot_modes=scale_plot_modes,
            )
        except Exception as exc:
            for metadata_file in metadata_files:
                failures.append((metadata_file, -1, exc))
            print(f"Skipping {campaign}: failed to compute campaign color scale: {exc}")
            continue

        print(
            f"{campaign} shared campaign color scale: "
            f"vmin={color_scale[0]}, vmax={color_scale[1]}"
        )
        print(f"{campaign} transects: {', '.join(map(str, transects))}")

        for metadata_file in metadata_files:
            for transect in transects:
                for plot_mode in plot_modes:
                    output_path = output_path_for(
                        metadata_file,
                        campaign=campaign,
                        transect=transect,
                        plot_mode=plot_mode,
                        qc_suffix=args.qc_suffix,
                        save_root=args.save_root,
                    )
                    try:
                        title_note = (
                            "3C O25"
                            if args.qc_suffix
                            in {"current_qc", "current_final_qc", "current_new_final_qc"}
                            else None
                        )
                        plot_heatmap(
                            metadata_file,
                            output_path,
                            campaign,
                            transect,
                            plot_mode,
                            color_scale,
                            args.dist_cache_root,
                            title_note=title_note,
                        )
                    except ValueError as exc:
                        if (
                            plot_mode != "all_measurements"
                            and "No rows remain after transect and " in str(exc)
                        ):
                            print(f"  SKIPPED: {exc}")
                            continue
                        failures.append((metadata_file, transect, exc))
                        print(f"  FAILED: {exc}")
                    except Exception as exc:
                        failures.append((metadata_file, transect, exc))
                        print(f"  FAILED: {exc}")
                    else:
                        created += 1

    print(f"Created {created} heatmap(s).")
    if failures:
        print("Failures:")
        for metadata_file, transect, exc in failures:
            print(f"  {metadata_file.name}, transect {transect}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
