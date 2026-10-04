#!/usr/bin/env python3
"""Backfill morphology AEC-risk indicators into per-datapoint metric CSVs.

Adds:
- land_water_edge_length_5km_m
- land_water_edge_density_5km_m_per_km2
- land_water_edge_valid_area_5km_km2
- lake_water_area_km2
- lake_shoreline_length_km
- shoreline_development_index

The local edge-density metric is computed from binary land/water masks inside a
5 km circular neighbourhood around each in-situ point. The shoreline development
index is computed once per lake/campaign mask:

    shoreline_length / (2 * sqrt(pi * lake_area))
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


RESULTS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RESULTS_ROOT.parent
DOCUMENTS_ROOT = PROJECT_ROOT.parent
RASTER_ROOT = DOCUMENTS_ROOT / "insitu" / "all" / "qgis_transect_bb"
DEFAULT_TARGET_ROOTS = [
    RESULTS_ROOT / "stats" / "performance_metrics",
    RESULTS_ROOT / "stats" / "performance_metrics_strict",
]

LANDWATER_MASKS = {
    "BIE": RASTER_ROOT / "binary_masks" / "BIE_20240618_binary.tif",
    "CST": RASTER_ROOT / "binary_masks" / "CST_20230610_binary.tif",
    "WAL": RASTER_ROOT / "binary_masks" / "WAL_20240619_binary.tif",
    "ZRH": RASTER_ROOT / "binary_masks" / "ZRH_20231008_binary.tif",
}

CAMPAIGN_LANDWATER_MASKS = {
    "20250303_CST": RASTER_ROOT / "binary_masks" / "CST_20250303_binary.tif",
}

OUTPUT_COLUMNS = [
    "land_water_edge_length_5km_m",
    "land_water_edge_density_5km_m_per_km2",
    "land_water_edge_valid_area_5km_km2",
    "lake_water_area_km2",
    "lake_shoreline_length_km",
    "shoreline_development_index",
]


def mask_path_for_campaign(campaign: str) -> Path | None:
    if campaign in CAMPAIGN_LANDWATER_MASKS:
        return CAMPAIGN_LANDWATER_MASKS[campaign]
    lake = campaign.split("_", 1)[-1]
    return LANDWATER_MASKS.get(lake)


def pixel_dimensions(src) -> tuple[float, float, float]:
    width = abs(float(src.transform.a))
    height = abs(float(src.transform.e))
    return width, height, width * height


def edge_length_m(water_mask: np.ndarray, valid_mask: np.ndarray, pixel_width_m: float, pixel_height_m: float) -> float:
    water = np.asarray(water_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    horizontal_edges = horizontal_valid & (water[:, :-1] != water[:, 1:])
    vertical_edges = vertical_valid & (water[:-1, :] != water[1:, :])
    return float(horizontal_edges.sum() * pixel_height_m + vertical_edges.sum() * pixel_width_m)


def shoreline_development(mask_path: Path) -> dict[str, float]:
    with rasterio.open(mask_path) as src:
        data = src.read(1, masked=False).astype(float, copy=False)
        valid = np.isfinite(data)
        water = data >= 0.5
        pixel_width, pixel_height, pixel_area = pixel_dimensions(src)
        water_area_km2 = float((water & valid).sum() * pixel_area / 1_000_000.0)
        shoreline_length_km = edge_length_m(water, valid, pixel_width, pixel_height) / 1000.0

    if water_area_km2 <= 0 or not np.isfinite(water_area_km2):
        sdi = np.nan
    else:
        water_area_m2 = water_area_km2 * 1_000_000.0
        sdi = float((shoreline_length_km * 1000.0) / (2.0 * np.sqrt(np.pi * water_area_m2)))
    return {
        "lake_water_area_km2": water_area_km2,
        "lake_shoreline_length_km": shoreline_length_km,
        "shoreline_development_index": sdi,
    }


def point_edge_density(
    x_2056: float,
    y_2056: float,
    radius_m: float,
    src,
    to_mask_crs: Transformer,
) -> dict[str, float]:
    from rasterio.windows import Window, from_bounds

    if not np.isfinite(x_2056) or not np.isfinite(y_2056):
        return {
            "land_water_edge_length_5km_m": np.nan,
            "land_water_edge_density_5km_m_per_km2": np.nan,
            "land_water_edge_valid_area_5km_km2": np.nan,
        }

    x_mask, y_mask = to_mask_crs.transform(float(x_2056), float(y_2056))
    raw_window = from_bounds(
        x_mask - radius_m,
        y_mask - radius_m,
        x_mask + radius_m,
        y_mask + radius_m,
        src.transform,
    ).round_offsets().round_lengths()

    col_off = max(0, int(raw_window.col_off))
    row_off = max(0, int(raw_window.row_off))
    col_max = min(src.width, int(raw_window.col_off + raw_window.width))
    row_max = min(src.height, int(raw_window.row_off + raw_window.height))
    if col_max <= col_off or row_max <= row_off:
        return {
            "land_water_edge_length_5km_m": np.nan,
            "land_water_edge_density_5km_m_per_km2": np.nan,
            "land_water_edge_valid_area_5km_km2": 0.0,
        }

    window = Window(col_off, row_off, col_max - col_off, row_max - row_off)
    data = src.read(1, window=window, masked=False).astype(float, copy=False)
    transform = src.window_transform(window)
    rows, cols = np.ogrid[0:data.shape[0], 0:data.shape[1]]
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    circle = ((xs - x_mask) ** 2 + (ys - y_mask) ** 2) <= radius_m ** 2
    valid = np.isfinite(data) & circle
    water = data >= 0.5
    pixel_width, pixel_height, pixel_area = pixel_dimensions(src)
    valid_area_km2 = float(valid.sum() * pixel_area / 1_000_000.0)
    length_m = edge_length_m(water, valid, pixel_width, pixel_height)
    density = float(length_m / valid_area_km2) if valid_area_km2 > 0 else np.nan
    return {
        "land_water_edge_length_5km_m": length_m,
        "land_water_edge_density_5km_m_per_km2": density,
        "land_water_edge_valid_area_5km_km2": valid_area_km2,
    }


def update_metrics_csv(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    if "campaign" not in df.columns:
        campaign = path.parent.name
        df["campaign"] = campaign
    else:
        campaign = str(df["campaign"].dropna().iloc[0]) if df["campaign"].notna().any() else path.parent.name

    mask_path = mask_path_for_campaign(campaign)
    if mask_path is None or not mask_path.exists():
        return {"file": str(path), "status": "missing_mask", "mask": str(mask_path) if mask_path else ""}

    if "insitu_row_id" not in df.columns or "insitu_x_2056" not in df.columns or "insitu_y_2056" not in df.columns:
        return {"file": str(path), "status": "missing_required_columns", "mask": str(mask_path)}

    point_df = (
        df[["campaign", "insitu_row_id", "insitu_x_2056", "insitu_y_2056"]]
        .drop_duplicates(subset=["campaign", "insitu_row_id"])
        .copy()
    )
    point_df["insitu_x_2056"] = pd.to_numeric(point_df["insitu_x_2056"], errors="coerce")
    point_df["insitu_y_2056"] = pd.to_numeric(point_df["insitu_y_2056"], errors="coerce")

    lake_metrics = shoreline_development(mask_path)
    point_rows = []
    with rasterio.open(mask_path) as src:
        to_mask_crs = Transformer.from_crs("EPSG:2056", src.crs, always_xy=True)
        for row in point_df.itertuples(index=False):
            edge = point_edge_density(row.insitu_x_2056, row.insitu_y_2056, 5000.0, src, to_mask_crs)
            point_rows.append({
                "campaign": row.campaign,
                "insitu_row_id": row.insitu_row_id,
                **edge,
            })
    point_metrics = pd.DataFrame(point_rows)

    for col in OUTPUT_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(point_metrics, on=["campaign", "insitu_row_id"], how="left")
    for col, value in lake_metrics.items():
        df[col] = value
    df.to_csv(path, index=False)
    return {
        "file": str(path),
        "status": "updated",
        "mask": str(mask_path),
        "n_rows": int(len(df)),
        "n_points": int(len(point_metrics)),
        **lake_metrics,
    }


def discover_metrics_files(root: Path) -> list[Path]:
    return sorted(root.glob("*/*_per_datapoint_metrics_with_per_wavelength_and_overall.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add morphology indicators to existing performance metric CSVs.")
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_TARGET_ROOTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    for root in args.roots:
        files = discover_metrics_files(root)
        print(f"{root}: found {len(files)} metrics files")
        for path in files:
            report = update_metrics_csv(path)
            reports.append(report)
            print(f"  {report['status']}: {path}")

    report_path = RESULTS_ROOT / "stats" / "processor_comparison" / "morphology_indicator_backfill_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
