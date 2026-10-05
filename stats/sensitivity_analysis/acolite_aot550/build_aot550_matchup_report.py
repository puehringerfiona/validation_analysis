"""
Build a point-matchup-weighted AOT550 report for the ACOLITE/RAdCor processor
family, as an alternative to the scene-wide statistic produced by
update_acolite_aot550_report.py.

update_acolite_aot550_report.py reports min/mean/median/max of the *entire*
aot_550 NetCDF variable array (every pixel in the tile grid, including areas
far from any in-situ transect). This script instead:

  1. For each of the 11 campaign/sensor combinations used by the project's
     main Rrs matchup pipeline (processor_comparison/run_performance_metrics_batch.py),
     loads and QC-filters the retained in-situ points the same way
     performance_metrics.py does (mc.filter_insitu_rows), and reprojects them
     to EPSG:2056.
  2. For the three spatially-tiled ACOLITE-family processors (acolite,
     acolite_tmart, radcor_acolite), builds a KDTree over each processor's
     aot_550 NetCDF grid and matches every retained in-situ point to its
     nearest pixel (rejecting matches farther than 1.5 pixel-diagonals away,
     same guard as performance_metrics.py), then takes the 3x3-footprint
     median of aot_550 at that point (mc.footprint_3x3 + mc.summarize_footprint_values
     + mc.select_matchup_value(..., "median")) -- the same convention already
     used project-wide for Rrs matchups.
  3. Averages those per-point matched AOT550 values to get one
     matchup-weighted mean AOT550 per campaign/sensor/processor.
  4. For the native ACOLITE+RAdCor (acolite_radcor) processor, there is no
     spatial aot_550 array -- it is a single fixed scene-wide `ac_aot_550`
     NetCDF attribute -- so that value is simply recorded as-is (with the
     ZRH20260227 exception: that campaign was run as two separate per-transect
     scenes, t1 and t3, each with its own fixed ac_aot_550, so in-situ points
     get the value of whichever transect they belong to).

File discovery for the three spatial processors is NOT reimplemented from
scratch: it reuses the already-validated per-campaign nc_file paths in
aot550_report_best_only_with_log_aot.csv (built by update_acolite_aot550_report.py
against the real eo_data/output tree), disambiguated by sensor token and by
preferring a NetCDF that actually contains readable aot_550 data over a
log-only fallback / "_winter" duplicate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Transformer
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
STATS_DIR = SCRIPT_DIR.parents[1]
RESULTS_ROOT = SCRIPT_DIR.parents[2]
PROJECT_ROOT = RESULTS_ROOT.parent
DOCUMENTS_ROOT = PROJECT_ROOT.parent

REPORT_PATH = SCRIPT_DIR / "aot550_report_best_only_with_log_aot.csv"
OUT_LONG = SCRIPT_DIR / "aot550_matchup_report.csv"
OUT_WIDE = SCRIPT_DIR / "aot550_matchup_summary.csv"

RUN_BATCH_PATH = STATS_DIR / "processor_comparison" / "run_performance_metrics_batch.py"
MATCHUP_COMMON_PATH = STATS_DIR / "matchup_common.py"

EPSG_CH = 2056
X_COL = "x-coordinate"
Y_COL = "y-coordinate"
HARD_FILTER_COLUMN = "hard_filter_retain"

# Same footprint/rejection convention as performance_metrics.py's Rrs matchup.
CONTEXT_RADIUS_PIXELS = 1.5
MATCHUP_VALUE_METHOD = "median"

VISIBLE_SENSOR_BANDS = {
    "L8":  [443, 483, 561, 655],
    "L9":  [443, 482, 561, 654],
    "S2A": [443, 492, 560, 665, 704],
    "S2B": [442, 492, 559, 665, 704],
}

# Tokens used to disambiguate report rows (which cover a whole campaign,
# potentially with both a Landsat and a Sentinel product) by sensor.
SENSOR_NC_TOKENS = {
    "L8": ["LC08"],
    "L9": ["LC09"],
    "S2A": ["S2A_MSIL1C"],
    "S2B": ["S2B_MSIL1C"],
}

SPATIAL_PROCESSORS = {
    "acolite": "acolite_dsf_matchup",
    "acolite_tmart": "tmart_acolite_dsf_matchup",
    "radcor_acolite": "radcor_acolite_dsf_matchup",
}
FIXED_PROCESSOR = "acolite_radcor"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mc = load_module("matchup_common", MATCHUP_COMMON_PATH)
rpm_batch = load_module("run_performance_metrics_batch", RUN_BATCH_PATH)

RUNS = rpm_batch.RUNS
EO_OUTPUT_ROOT = rpm_batch.EO_OUTPUT_ROOT
print(f"[setup] Resolved EO_OUTPUT_ROOT = {EO_OUTPUT_ROOT} (exists={EO_OUTPUT_ROOT.exists()})")
print(f"[setup] Resolved INSITU_ROOT   = {rpm_batch.INSITU_ROOT} (exists={rpm_batch.INSITU_ROOT.exists()})")


def long_path(path) -> str:
    text = str(path)
    return text if text.startswith(r"\\?\\") else r"\\?\\" + text


transformer_lonlat_to_2056 = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG_CH}", always_xy=True)


def read_nc_var(ds: Dataset, name: str) -> np.ndarray:
    arr = ds.variables[name][:]
    if np.ma.isMaskedArray(arr):
        return np.ma.filled(arr.astype(float), np.nan)
    return np.asarray(arr, dtype=float)


def safe_transform(ds: Dataset):
    lon_var = "lon" if "lon" in ds.variables else "longitude"
    lat_var = "lat" if "lat" in ds.variables else "latitude"
    lon = read_nc_var(ds, lon_var)
    lat = read_nc_var(ds, lat_var)
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    x, y = transformer_lonlat_to_2056.transform(lon.ravel(), lat.ravel())
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    coords = np.column_stack((x[mask], y[mask]))
    valid_idx = np.argwhere(mask).ravel()
    return coords, valid_idx, lon.shape


def estimate_pixel_sizes_m(ds: Dataset, transformer: Transformer):
    lon_var = "lon" if "lon" in ds.variables else "longitude"
    lat_var = "lat" if "lat" in ds.variables else "latitude"
    lon = read_nc_var(ds, lon_var)
    lat = read_nc_var(ds, lat_var)
    if lon.ndim == 1 and lat.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
    else:
        lon2, lat2 = lon, lat
    i = lon2.shape[0] // 2
    j = lon2.shape[1] // 2
    x0, y0 = transformer.transform(lon2[i, j], lat2[i, j])
    x1, y1 = transformer.transform(lon2[i, j + 1], lat2[i, j + 1])
    x2, y2 = transformer.transform(lon2[i + 1, j], lat2[i + 1, j])
    dx = float(np.hypot(x1 - x0, y1 - y0))
    dy = float(np.hypot(x2 - x0, y2 - y0))
    return dx, dy


def load_report() -> pd.DataFrame:
    df = pd.read_csv(REPORT_PATH, sep=";")
    df["nc_file"] = df["nc_file"].astype(str)
    return df


def rows_for(report_df: pd.DataFrame, compact_campaign: str, sensor: str, processor: str) -> pd.DataFrame:
    sub = report_df[(report_df["campaign"] == compact_campaign) & (report_df["processor"] == processor)]
    tokens = SENSOR_NC_TOKENS[sensor]
    mask = sub["nc_file"].apply(lambda p: any(t in p for t in tokens))
    return sub[mask]


def pick_spatial_row(candidates: pd.DataFrame, notes: list):
    """Pick the row to use for a spatially-matchable processor: must have a
    readable aot_550 variable (aot_550_var_mean not null); prefer a
    non-"_winter" duplicate when more than one qualifies."""
    valid = candidates[candidates["aot_550_var_mean"].notna()]
    if valid.empty:
        return None
    if len(valid) > 1:
        non_winter = valid[~valid["nc_file"].str.contains("winter", case=False, na=False)]
        if not non_winter.empty:
            if len(non_winter) < len(valid):
                notes.append(f"multiple valid report rows; dropped {len(valid) - len(non_winter)} '_winter' duplicate(s)")
            valid = non_winter
    valid = valid.sort_values("nc_file")
    if len(valid) > 1:
        notes.append(f"multiple valid report rows remained ({len(valid)}); used the first by path sort order")
    return valid.iloc[0]


def transect_token_for_row(nc_file: str):
    low = str(nc_file).lower()
    if "kr5km_t1" in low:
        return 1
    if "kr5km_t3" in low:
        return 3
    return None


def match_points_to_aot(nc_path: str, insitu_x: np.ndarray, insitu_y: np.ndarray):
    """Nearest-pixel + 3x3-footprint-median AOT550 matchup for one NetCDF
    grid, mirroring performance_metrics.py's Rrs matchup loop."""
    n = len(insitu_x)
    values = np.full(n, np.nan)
    with Dataset(long_path(nc_path), "r") as ds:
        if "aot_550" not in ds.variables:
            return values
        aot = read_nc_var(ds, "aot_550").ravel()
        coords, valid_idx, grid_shape = safe_transform(ds)
        dx, dy = estimate_pixel_sizes_m(ds, transformer_lonlat_to_2056)

    if coords.shape[0] == 0:
        return values

    context_radius_m = CONTEXT_RADIUS_PIXELS * max(abs(dx), abs(dy))
    tree = cKDTree(coords)
    dists, idxs = tree.query(np.column_stack([insitu_x, insitu_y]))
    dists = np.atleast_1d(dists)
    idxs = np.atleast_1d(idxs)

    for i in range(n):
        dist_m = dists[i]
        if not np.isfinite(dist_m) or not np.isfinite(context_radius_m) or dist_m > context_radius_m:
            continue
        flat_idx = valid_idx[idxs[i]]
        footprint_idx = mc.footprint_3x3(flat_idx, grid_shape)
        footprint_vals = aot[footprint_idx]
        summary = mc.summarize_footprint_values(footprint_vals)
        nearest_val = aot[flat_idx]
        summary["nearest"] = float(nearest_val) if np.isfinite(nearest_val) else np.nan
        values[i] = mc.select_matchup_value(summary, MATCHUP_VALUE_METHOD)

    return values


def load_insitu_points(cfg) -> pd.DataFrame:
    csv_conv_file = cfg.csv_conv_file
    transects = cfg.transects
    sensor = cfg.sensor
    df = pd.read_csv(csv_conv_file, sep=None, engine="python")
    df = df.rename(columns=lambda c: c.replace(".0", "") if isinstance(c, str) else c)
    result = mc.filter_insitu_rows(
        df,
        transects=transects,
        metric_waves=VISIBLE_SENSOR_BANDS[sensor],
        x_col=X_COL,
        y_col=Y_COL,
        hard_filter_column=HARD_FILTER_COLUMN,
        require_hard_filter_column=False,
    )
    frame = result["frame"].reset_index(drop=True)
    frame["insitu_row_id"] = frame.index
    gdf = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame[X_COL], frame[Y_COL]),
        crs="EPSG:4326",
    ).to_crs(epsg=EPSG_CH)
    gdf["insitu_x_2056"] = gdf.geometry.x
    gdf["insitu_y_2056"] = gdf.geometry.y
    return gdf, result


def summarize_values(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n_matched_points": 0, "mean_aot_550": np.nan, "median_aot_550": np.nan,
                "min_aot_550": np.nan, "max_aot_550": np.nan}
    return {
        "n_matched_points": int(finite.size),
        "mean_aot_550": float(np.mean(finite)),
        "median_aot_550": float(np.median(finite)),
        "min_aot_550": float(np.min(finite)),
        "max_aot_550": float(np.max(finite)),
    }


def main() -> int:
    report_df = load_report()
    long_rows = []
    wide_rows = []

    for cfg in RUNS:
        campaign = cfg.campaign
        sensor = cfg.sensor
        compact_campaign = cfg.compact_campaign
        print(f"\n##### {campaign} {sensor} (report key {compact_campaign}) #####")

        gdf, filter_result = load_insitu_points(cfg)
        n_retained = len(gdf)
        print(f"  retained in-situ points after QC/transect filtering: {n_retained} "
              f"(input_rows={filter_result['input_rows']}, "
              f"after_transect_coord_selection={filter_result['rows_after_transect_coordinate_selection']})")

        insitu_x = gdf["insitu_x_2056"].to_numpy(dtype=float)
        insitu_y = gdf["insitu_y_2056"].to_numpy(dtype=float)
        transect_nr = pd.to_numeric(gdf["transect_nr"], errors="coerce").to_numpy()

        wide_row = {"campaign": compact_campaign, "sensor": sensor}

        # --- Spatially-matchable processors ---
        for processor, wide_prefix in SPATIAL_PROCESSORS.items():
            notes = []
            candidates = rows_for(report_df, compact_campaign, sensor, processor)
            if candidates.empty:
                notes.append("no report row found for this campaign/sensor/processor")
                row = None
            else:
                row = pick_spatial_row(candidates, notes)
                if row is None:
                    notes.append(f"{len(candidates)} report row(s) found but none had a readable aot_550 variable "
                                 f"(log-only fallback) -- skipped")

            if row is None:
                summary = summarize_values(np.full(n_retained, np.nan))
                long_rows.append({
                    "campaign": compact_campaign, "campaign_full": campaign, "sensor": sensor,
                    "processor": processor, "nc_file": "", "n_retained_insitu_points": n_retained,
                    **summary, "notes": "; ".join(notes),
                })
                wide_row[f"{wide_prefix}_mean"] = np.nan
                print(f"  {processor}: SKIPPED ({'; '.join(notes)})")
                continue

            nc_file = row["nc_file"]
            values = match_points_to_aot(nc_file, insitu_x, insitu_y)
            summary = summarize_values(values)
            long_rows.append({
                "campaign": compact_campaign, "campaign_full": campaign, "sensor": sensor,
                "processor": processor, "nc_file": nc_file, "n_retained_insitu_points": n_retained,
                **summary, "notes": "; ".join(notes),
            })
            wide_row[f"{wide_prefix}_mean"] = summary["mean_aot_550"]
            match_frac = summary["n_matched_points"] / n_retained if n_retained else np.nan
            print(f"  {processor}: n_matched_points={summary['n_matched_points']}/{n_retained} "
                  f"({match_frac:.1%}) mean={summary['mean_aot_550']:.5f} "
                  f"median={summary['median_aot_550']:.5f}"
                  if summary["n_matched_points"] else
                  f"  {processor}: n_matched_points=0/{n_retained} -- INVESTIGATE")

        # --- Fixed / native ACOLITE+RAdCor processor ---
        notes = []
        candidates = rows_for(report_df, compact_campaign, sensor, FIXED_PROCESSOR)
        valid = candidates[candidates["ac_aot_550"].notna()] if not candidates.empty else candidates
        if valid.empty:
            notes.append("no report row with ac_aot_550 found")
            values = np.full(n_retained, np.nan)
        elif len(valid) == 1:
            fixed_value = float(valid.iloc[0]["ac_aot_550"])
            values = np.full(n_retained, fixed_value)
        else:
            # ZRH20260227-style per-transect split: map transect_nr -> fixed value.
            transect_value_map = {}
            for _, r in valid.iterrows():
                token = transect_token_for_row(r["nc_file"])
                if token is not None:
                    transect_value_map[token] = float(r["ac_aot_550"])
            if not transect_value_map:
                notes.append(f"{len(valid)} report rows with ac_aot_550 but none matched a t1/t3 transect token")
                values = np.full(n_retained, np.nan)
            else:
                notes.append(f"per-transect fixed values: {transect_value_map}")
                values = np.array([transect_value_map.get(int(t), np.nan) if np.isfinite(t) else np.nan
                                    for t in transect_nr], dtype=float)
                unmatched = np.sum(~np.isfinite(values))
                if unmatched:
                    notes.append(f"{unmatched} in-situ point(s) had a transect_nr with no matching fixed-value file")

        summary = summarize_values(values)
        nc_files_used = "; ".join(str(v) for v in valid["nc_file"].tolist()) if not valid.empty else ""
        long_rows.append({
            "campaign": compact_campaign, "campaign_full": campaign, "sensor": sensor,
            "processor": FIXED_PROCESSOR, "nc_file": nc_files_used, "n_retained_insitu_points": n_retained,
            **summary, "notes": "; ".join(notes),
        })
        wide_row["acolite_tsdsf_radcor_fixed"] = summary["mean_aot_550"]
        print(f"  {FIXED_PROCESSOR}: n_matched_points={summary['n_matched_points']}/{n_retained} "
              f"mean={summary['mean_aot_550']}" + (f" ({'; '.join(notes)})" if notes else ""))

        wide_rows.append(wide_row)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)

    long_df.to_csv(OUT_LONG, index=False)
    wide_df.to_csv(OUT_WIDE, index=False)
    print(f"\nWrote {OUT_LONG}")
    print(f"Wrote {OUT_WIDE}")

    zero_match = long_df[(long_df["processor"] != FIXED_PROCESSOR) & (long_df["n_matched_points"] == 0)]
    if not zero_match.empty:
        print("\nWARNING: rows with 0 matched points:")
        print(zero_match[["campaign", "sensor", "processor", "notes"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
