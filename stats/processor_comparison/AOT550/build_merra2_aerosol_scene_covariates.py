"""
Build scene-level aerosol covariates for the performance metric export.

This script writes one row per configured campaign/sensor to:
    stats/processor_comparison/aerosol_scene_covariates.csv

Canonical AOT550 columns:
    merra2_totexttau_550 = MERRA-2 M2T1NXAER TOTEXTTAU, total aerosol
        extinction optical thickness at 550 nm.
    cams_aot550 = CAMS total aerosol optical depth at 550 nm.
    aeronet_aot550 = AERONET direct-sun AOT at 550 nm, or Angstrom-adjusted
        AOT550 when 550 nm is not reported directly.

The legacy MERRA-2 scattering column is retained for backward compatibility,
but downstream AOT550 fusion should use extinction AOT consistently:
MERRA-2 TOTEXTTAU, CAMS total AOD550, and AERONET AOT550.

NASA Earthdata authentication is required for MERRA-2 downloads. Use either a
configured .netrc entry for urs.earthdata.nasa.gov or the environment variables
EARTHDATA_USERNAME and EARTHDATA_PASSWORD.

CAMS extraction is cache-first. Supply --cams-files with local NetCDF/GRIB files,
or use --download-cams after installing/configuring cdsapi. AERONET direct-sun
data are downloaded from the public AERONET V3 web service unless cached files
are already present.
"""

from __future__ import annotations

import argparse
import configparser
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import netCDF4
import numpy as np
import pandas as pd
import requests
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
STATS_DIR = SCRIPT_DIR.parent
for import_path in [SCRIPT_DIR, STATS_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import matchup_common as mc
from run_performance_metrics_batch import ACTIVE_RUNS


RESULTS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RESULTS_ROOT.parent
MERRA2_COLLECTION = "M2T1NXAER.5.12.4"
MERRA2_AEROSOL_VARIABLES = {
    "TOTSCATAU": "merra2_totscatau_550",
    "TOTEXTTAU": "merra2_totexttau_550",
}
CANONICAL_MERRA2_VARIABLE = "TOTEXTTAU"
OUTPUT_CSV = SCRIPT_DIR / "aerosol_scene_covariates.csv"
MERRA2_CACHE_DIR = SCRIPT_DIR / "merra2_cache"
CAMS_CACHE_DIR = SCRIPT_DIR / "cams_cache"
AERONET_CACHE_DIR = SCRIPT_DIR / "aeronet_cache"
DEFAULT_EARTHDATA_CONFIG = (
    PROJECT_ROOT
    / "eo_data"
    / "processors"
    / "sentinel-hindcast"
    / "environments"
    / "docker_env.ini"
)
DEFAULT_CDS_CONFIG = DEFAULT_EARTHDATA_CONFIG
EARTHDATA_AUTH_HOSTS = {
    "goldsmr4.gesdisc.eosdis.nasa.gov",
    "urs.earthdata.nasa.gov",
}
CAMS_DATASET = "cams-global-reanalysis-eac4"
CAMS_FORECAST_DATASET = "cams-global-atmospheric-composition-forecasts"
CAMS_VARIABLE = "total_aerosol_optical_depth_550nm"
CAMS_API_URL = "https://ads.atmosphere.copernicus.eu/api"
CAMS_VARIABLE_CANDIDATES = [
    "aod550",
    "aod550nm",
    "aod550_total",
    "total_aerosol_optical_depth_550nm",
    "aerosol_optical_depth_550nm",
    "aerosol_optical_depth_at_550_nm",
]
CAMS_LAT_NAMES = ["latitude", "lat", "y"]
CAMS_LON_NAMES = ["longitude", "lon", "x"]
CAMS_TIME_NAMES = ["time", "valid_time"]
SWISS_AREA_NWSE = [48.3, 5.5, 45.4, 10.8]
AERONET_SEARCH_WINDOW_MIN = 180
AERONET_LEVELS = ["AOD20", "AOD15", "AOD10"]
AERONET_SITES = [
    {"site": "Laegeren", "lat": 47.478333, "lon": 8.364389, "elevation_m": 689.0},
    {"site": "Payerne", "lat": 46.812408, "lon": 6.942019, "elevation_m": 491.0},
    {"site": "JUNGFRAU", "lat": 46.547480, "lon": 7.984993, "elevation_m": 3564.0},
    {"site": "Davos", "lat": 46.812810, "lon": 9.843690, "elevation_m": 1589.0},
    {"site": "Ispra", "lat": 45.803050, "lon": 8.626700, "elevation_m": 235.0},
    {"site": "KEMPTEN_UAS", "lat": 47.715833, "lon": 10.314133, "elevation_m": 718.0},
    {"site": "Lindenberg", "lat": 52.209, "lon": 14.122, "elevation_m": 125.0},
]


class EarthdataSession(requests.Session):
    """Requests session that keeps Earthdata auth across NASA host redirects."""

    def rebuild_auth(self, prepared_request, response):
        super().rebuild_auth(prepared_request, response)
        host = urlparse(prepared_request.url).hostname
        if self.auth and host in EARTHDATA_AUTH_HOSTS:
            prepared_request.prepare_auth(self.auth)


@dataclass(frozen=True)
class Scene:
    campaign: str
    sensor: str
    scene_datetime: pd.Timestamp
    matchup_lon: float
    matchup_lat: float
    n_matchup_measurements: int
    scene_time_fallback_no_input_overpass: bool

    @property
    def key(self) -> tuple[str, str]:
        return self.campaign, self.sensor


def merra2_stream_for_year(year: int) -> int:
    if year < 1992:
        return 100
    if year < 2001:
        return 200
    if year < 2011:
        return 300
    return 400


def merra2_daily_url(day: pd.Timestamp) -> str:
    stream = merra2_stream_for_year(int(day.year))
    ymd = day.strftime("%Y%m%d")
    return (
        "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/"
        f"{MERRA2_COLLECTION}/{day:%Y}/{day:%m}/"
        f"MERRA2_{stream}.tavg1_2d_aer_Nx.{ymd}.nc4"
    )


def parse_scene_times(frame: pd.DataFrame) -> pd.Series:
    for column in ["satellite_overpass_utc", "satellite_overpass_datetime_utc", "scene_datetime"]:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
            parsed = parsed[pd.notna(parsed)]
            if len(parsed):
                return parsed
    return pd.Series(dtype="datetime64[ns, UTC]")


def read_filtered_insitu(cfg) -> pd.DataFrame:
    frame = pd.read_csv(cfg.csv_conv_file, sep=";")
    frame = frame.rename(columns=lambda c: c.replace(".0", "") if isinstance(c, str) else c)
    wavelengths = {
        "L8": [443, 483, 561, 655],
        "L9": [443, 482, 561, 654],
        "S2A": [443, 492, 560, 665, 704],
        "S2B": [442, 492, 559, 665, 704],
    }[cfg.sensor]
    filtered = mc.filter_insitu_rows(
        frame,
        transects=cfg.transects,
        metric_waves=wavelengths,
        x_col="x-coordinate",
        y_col="y-coordinate",
        hard_filter_column="hard_filter_retain",
        require_hard_filter_column=False,
    )["frame"].copy()
    if filtered.empty:
        raise ValueError(f"No retained in-situ rows for {cfg.campaign} {cfg.sensor}.")
    return filtered


def campaign_scene_definition(cfg) -> Scene:
    frame = read_filtered_insitu(cfg)
    lon = pd.to_numeric(frame["x-coordinate"], errors="coerce")
    lat = pd.to_numeric(frame["y-coordinate"], errors="coerce")
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        raise ValueError(f"No valid measurement coordinates for {cfg.campaign} {cfg.sensor}.")

    scene_times = parse_scene_times(frame)
    if len(scene_times):
        scene_time = scene_times.sort_values().iloc[len(scene_times) // 2]
    else:
        scene_time = pd.Timestamp(cfg.date, tz="UTC") + pd.Timedelta(hours=12)

    return Scene(
        campaign=cfg.campaign,
        sensor=cfg.sensor,
        scene_datetime=scene_time,
        matchup_lon=float(lon[valid].mean()),
        matchup_lat=float(lat[valid].mean()),
        n_matchup_measurements=int(valid.sum()),
        scene_time_fallback_no_input_overpass=not bool(len(scene_times)),
    )


def earthdata_credentials(config_path: Path | None = DEFAULT_EARTHDATA_CONFIG) -> tuple[str | None, str | None]:
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if username and password:
        return username, password

    if config_path is None or not config_path.exists():
        return None, None

    parser = configparser.RawConfigParser()
    parser.read(config_path, encoding="utf-8")
    if not parser.has_section("EARTHDATA"):
        return None, None

    username = parser.get("EARTHDATA", "username", fallback="").strip()
    password = parser.get("EARTHDATA", "password", fallback="").strip()
    if not username or not password or username.startswith("<") or password.startswith("<"):
        return None, None
    return username, password


def download_file(
    url: str,
    target: Path,
    force: bool = False,
    earthdata_config: Path | None = DEFAULT_EARTHDATA_CONFIG,
) -> None:
    if target.exists() and not force:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    for stale_tmp in target.parent.glob(f"{target.name}*.tmp"):
        try:
            stale_tmp.unlink()
        except OSError:
            pass
    session = EarthdataSession()
    username, password = earthdata_credentials(earthdata_config)
    if username and password:
        session.auth = (username, password)

    with session.get(url, stream=True, timeout=120) as response:
        if response.status_code == 401:
            source = (
                f"Earthdata config {earthdata_config}"
                if username and password and earthdata_config and earthdata_config.exists()
                else "EARTHDATA_USERNAME/EARTHDATA_PASSWORD or .netrc"
            )
            raise RuntimeError(
                f"Unauthorized MERRA-2 download from NASA Earthdata using {source}. "
                "Check that the credentials are valid and that the Earthdata account has accepted "
                "GES DISC data access terms."
            ) from None
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            raise RuntimeError(
                "MERRA-2 download returned HTML instead of NetCDF. Configure NASA Earthdata "
                "authentication via .netrc or EARTHDATA_USERNAME/EARTHDATA_PASSWORD."
            )

        fd, tmp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=str(target.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            tmp_path.replace(target)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(values.astype(float) - float(target))))


def coordinate_interpolation_plan(values: np.ndarray, target: float) -> dict:
    indexed_values = [
        (index, float(value))
        for index, value in enumerate(np.asarray(values, dtype=float))
        if np.isfinite(value)
    ]
    if not indexed_values:
        raise ValueError("No valid source coordinates are available for spatial interpolation.")

    indexed_values = sorted(indexed_values, key=lambda item: item[1])
    offsets = [abs(value - target) for _, value in indexed_values]
    nearest_position = int(np.argmin(offsets))
    nearest_index_value, nearest_value = indexed_values[nearest_position]

    lower = [item for item in indexed_values if item[1] <= target]
    upper = [item for item in indexed_values if item[1] >= target]
    if lower and upper:
        lower_index, lower_value = lower[-1]
        upper_index, upper_value = upper[0]
        span = upper_value - lower_value
        if span > 0:
            weight_upper = (target - lower_value) / span
            method = "linear"
        else:
            weight_upper = 0.0
            method = "exact"
        return {
            "method": method,
            "nearest_index": nearest_index_value,
            "nearest_value": nearest_value,
            "lower_index": lower_index,
            "lower_value": lower_value,
            "upper_index": upper_index,
            "upper_value": upper_value,
            "weight_upper": float(weight_upper),
        }

    return {
        "method": "nearest_no_bracket",
        "nearest_index": nearest_index_value,
        "nearest_value": nearest_value,
        "lower_index": nearest_index_value,
        "lower_value": nearest_value,
        "upper_index": nearest_index_value,
        "upper_value": nearest_value,
        "weight_upper": 0.0,
    }


def bilinear_grid_value(variable, time_index: int | None, lat_plan: dict, lon_plan: dict) -> float:
    def sample(lat_index: int, lon_index: int):
        if time_index is None:
            return variable[lat_index, lon_index]
        return variable[time_index, lat_index, lon_index]

    v_ll = interpolate_pair(
        sample(lat_plan["lower_index"], lon_plan["lower_index"]),
        sample(lat_plan["lower_index"], lon_plan["upper_index"]),
        lon_plan["weight_upper"],
    )
    v_ul = interpolate_pair(
        sample(lat_plan["upper_index"], lon_plan["lower_index"]),
        sample(lat_plan["upper_index"], lon_plan["upper_index"]),
        lon_plan["weight_upper"],
    )
    return interpolate_pair(v_ll, v_ul, lat_plan["weight_upper"])


def spatial_method(lat_plan: dict, lon_plan: dict) -> str:
    methods = {lat_plan["method"], lon_plan["method"]}
    if methods == {"exact"}:
        return "exact_grid"
    if "nearest_no_bracket" in methods:
        return "nearest_no_bracket"
    if lat_plan["method"] == "linear" and lon_plan["method"] == "linear":
        return "bilinear"
    if lat_plan["method"] == "linear":
        return "linear_lat"
    if lon_plan["method"] == "linear":
        return "linear_lon"
    return "exact_grid"


def spatial_metadata(prefix: str, lat_plan: dict, lon_plan: dict, target_lon: float, target_lat: float) -> dict:
    return {
        f"{prefix}_spatial_method": spatial_method(lat_plan, lon_plan),
        f"{prefix}_interpolation_target_lon": target_lon,
        f"{prefix}_interpolation_target_lat": target_lat,
        f"{prefix}_grid_lon": lon_plan["nearest_value"],
        f"{prefix}_grid_lat": lat_plan["nearest_value"],
        f"{prefix}_interpolation_lon_lower": lon_plan["lower_value"],
        f"{prefix}_interpolation_lon_upper": lon_plan["upper_value"],
        f"{prefix}_interpolation_lon_weight_upper": lon_plan["weight_upper"],
        f"{prefix}_interpolation_lat_lower": lat_plan["lower_value"],
        f"{prefix}_interpolation_lat_upper": lat_plan["upper_value"],
        f"{prefix}_interpolation_lat_weight_upper": lat_plan["weight_upper"],
    }


def utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def time_interpolation_plan(times, target: pd.Timestamp) -> dict:
    target = target.tz_convert("UTC")
    indexed_times = []
    for index, value in enumerate(times):
        timestamp = utc_timestamp(value)
        if pd.notna(timestamp):
            indexed_times.append((index, timestamp))
    if not indexed_times:
        raise ValueError("No valid source times are available for temporal interpolation.")

    indexed_times = sorted(indexed_times, key=lambda item: item[1])
    offsets = [abs((timestamp - target).total_seconds()) for _, timestamp in indexed_times]
    nearest_position = int(np.argmin(offsets))
    nearest_index_value, nearest_time = indexed_times[nearest_position]

    before = [item for item in indexed_times if item[1] <= target]
    after = [item for item in indexed_times if item[1] >= target]
    if before and after:
        before_index, before_time = before[-1]
        after_index, after_time = after[0]
        window_seconds = (after_time - before_time).total_seconds()
        if window_seconds > 0:
            weight_after = (target - before_time).total_seconds() / window_seconds
            method = "linear_time"
        else:
            weight_after = 0.0
            method = "exact_time"
        return {
            "method": method,
            "nearest_index": nearest_index_value,
            "nearest_time": nearest_time,
            "before_index": before_index,
            "before_time": before_time,
            "after_index": after_index,
            "after_time": after_time,
            "weight_after": float(weight_after),
            "window_min": abs(window_seconds) / 60.0,
        }

    return {
        "method": "nearest_no_bracket",
        "nearest_index": nearest_index_value,
        "nearest_time": nearest_time,
        "before_index": nearest_index_value,
        "before_time": nearest_time,
        "after_index": nearest_index_value,
        "after_time": nearest_time,
        "weight_after": 0.0,
        "window_min": 0.0,
    }


def interpolate_pair(before_value, after_value, weight_after: float) -> float:
    before = float(np.asarray(before_value).squeeze())
    after = float(np.asarray(after_value).squeeze())
    if not (np.isfinite(before) and np.isfinite(after)):
        return np.nan
    return before + (after - before) * weight_after


def interpolation_metadata(prefix: str, plan: dict, scene_time: pd.Timestamp) -> dict:
    nearest_time = plan["nearest_time"]
    return {
        f"{prefix}_source_time_utc": nearest_time.isoformat(),
        f"{prefix}_time_offset_min": abs((nearest_time - scene_time).total_seconds()) / 60.0,
        f"{prefix}_interpolation_method": plan["method"],
        f"{prefix}_interpolation_before_time_utc": plan["before_time"].isoformat(),
        f"{prefix}_interpolation_after_time_utc": plan["after_time"].isoformat(),
        f"{prefix}_interpolation_weight_after": plan["weight_after"],
        f"{prefix}_interpolation_window_min": plan["window_min"],
    }


def extract_merra2_variables(file_path: Path, scene: Scene, source_url: str) -> dict:
    scene_time = scene.scene_datetime.tz_convert("UTC")
    with netCDF4.Dataset(file_path) as dataset:
        time_var = dataset.variables["time"]
        time_values = netCDF4.num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        plan = time_interpolation_plan(time_values, scene_time)
        lat_values = np.asarray(dataset.variables["lat"][:], dtype=float)
        lat_plan = coordinate_interpolation_plan(lat_values, scene.matchup_lat)
        lon_values = np.asarray(dataset.variables["lon"][:], dtype=float)
        matchup_lon = scene.matchup_lon % 360 if np.nanmax(lon_values) > 180 else scene.matchup_lon
        lon_plan = coordinate_interpolation_plan(lon_values, matchup_lon)

        aerosol_values = {}
        for variable, output_column in MERRA2_AEROSOL_VARIABLES.items():
            if variable not in dataset.variables:
                raise KeyError(f"MERRA-2 file {file_path} does not contain variable {variable}.")
            raw_before = bilinear_grid_value(dataset.variables[variable], plan["before_index"], lat_plan, lon_plan)
            raw_after = bilinear_grid_value(dataset.variables[variable], plan["after_index"], lat_plan, lon_plan)
            value = interpolate_pair(raw_before, raw_after, plan["weight_after"])
            aerosol_values[output_column] = value if np.isfinite(value) else np.nan

        row = {
            "campaign": scene.campaign,
            "sensor": scene.sensor,
            "scene_datetime": scene_time.isoformat(),
            "merra2_matchup_lon": scene.matchup_lon,
            "merra2_matchup_lat": scene.matchup_lat,
            "merra2_variables": ",".join(MERRA2_AEROSOL_VARIABLES),
            "merra2_collection": MERRA2_COLLECTION,
            "merra2_source_url": source_url,
            "n_matchup_measurements": scene.n_matchup_measurements,
            "scene_time_fallback_no_input_overpass": scene.scene_time_fallback_no_input_overpass,
            "merra2_variable": CANONICAL_MERRA2_VARIABLE,
        }
        row.update(interpolation_metadata("merra2", plan, scene_time))
        row.update(spatial_metadata("merra2", lat_plan, lon_plan, matchup_lon, scene.matchup_lat))
        row.update(aerosol_values)
        return row


def normalize_longitudes(values: xr.DataArray, target_lon: float) -> tuple[xr.DataArray, float]:
    lon_values = np.asarray(values.values, dtype=float)
    if np.nanmax(lon_values) > 180 and target_lon < 0:
        return values, target_lon % 360
    if np.nanmax(lon_values) > 180:
        return values, target_lon % 360
    return values, target_lon


def select_name(candidates: list[str], names) -> str | None:
    lower_lookup = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def select_cams_variable(dataset: xr.Dataset) -> str:
    by_name = select_name(CAMS_VARIABLE_CANDIDATES, dataset.data_vars)
    if by_name:
        return by_name

    for name, var in dataset.data_vars.items():
        text = " ".join(
            str(var.attrs.get(key, ""))
            for key in ["long_name", "standard_name", "GRIB_name", "description"]
        ).lower()
        if "550" in text and ("aerosol optical depth" in text or "aod" in text):
            return str(name)

    raise KeyError(
        "Could not find a CAMS AOT550 variable. Tried "
        f"{', '.join(CAMS_VARIABLE_CANDIDATES)} and variable metadata containing 550/AOD."
    )


def absolute_time_offsets_seconds(times: pd.DatetimeIndex, target: pd.Timestamp) -> np.ndarray:
    return np.abs((times - target) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def xarray_select_flat_time(variable: xr.DataArray, time_coord: xr.DataArray, flat_index: int) -> xr.DataArray:
    unraveled = np.unravel_index(flat_index, time_coord.shape)
    time_indexers = {
        dim: int(index)
        for dim, index in zip(time_coord.dims, unraveled)
        if dim in variable.dims
    }
    if not time_indexers:
        return variable
    return variable.isel(time_indexers)


def xarray_bilinear_grid_value(
    variable: xr.DataArray,
    time_coord: xr.DataArray | None,
    time_index: int | None,
    lat_name: str,
    lon_name: str,
    lat_plan: dict,
    lon_plan: dict,
) -> float:
    if time_coord is not None and time_index is not None:
        variable = xarray_select_flat_time(variable, time_coord, time_index)

    def sample(lat_index: int, lon_index: int):
        return variable.isel({lat_name: lat_index, lon_name: lon_index}).values

    v_ll = interpolate_pair(
        sample(lat_plan["lower_index"], lon_plan["lower_index"]),
        sample(lat_plan["lower_index"], lon_plan["upper_index"]),
        lon_plan["weight_upper"],
    )
    v_ul = interpolate_pair(
        sample(lat_plan["upper_index"], lon_plan["lower_index"]),
        sample(lat_plan["upper_index"], lon_plan["upper_index"]),
        lon_plan["weight_upper"],
    )
    return interpolate_pair(v_ll, v_ul, lat_plan["weight_upper"])


def extract_cams_from_file(file_path: Path, scene: Scene) -> dict:
    scene_time = scene.scene_datetime.tz_convert("UTC")
    engine_kwargs = [{}]
    if file_path.suffix.lower() in {".grib", ".grb", ".grb2"}:
        engine_kwargs = [{"engine": "cfgrib"}, {}]

    last_error: Exception | None = None
    for kwargs in engine_kwargs:
        try:
            with xr.open_dataset(file_path, **kwargs) as ds:
                var_name = select_cams_variable(ds)
                lat_name = select_name(CAMS_LAT_NAMES, list(ds.coords) + list(ds.dims))
                lon_name = select_name(CAMS_LON_NAMES, list(ds.coords) + list(ds.dims))
                time_name = select_name(CAMS_TIME_NAMES, list(ds.coords) + list(ds.dims))
                if not lat_name or not lon_name:
                    raise KeyError(f"CAMS file {file_path} does not expose latitude/longitude coordinates.")

                variable = ds[var_name]
                lon_coord, target_lon = normalize_longitudes(ds[lon_name], scene.matchup_lon)
                lon_plan = coordinate_interpolation_plan(lon_coord.values, target_lon)
                lat_plan = coordinate_interpolation_plan(ds[lat_name].values, scene.matchup_lat)

                plan = None
                if time_name and time_name in ds.coords and np.ndim(ds[time_name].values) > 0:
                    time_coord = ds[time_name]
                    flat_times = pd.to_datetime(np.ravel(time_coord.values), utc=True, errors="coerce")
                    plan = time_interpolation_plan(flat_times, scene_time)
                    before = xarray_bilinear_grid_value(
                        variable, time_coord, plan["before_index"], lat_name, lon_name, lat_plan, lon_plan
                    )
                    after = xarray_bilinear_grid_value(
                        variable, time_coord, plan["after_index"], lat_name, lon_name, lat_plan, lon_plan
                    )
                    value = interpolate_pair(before, after, plan["weight_after"])
                else:
                    value = xarray_bilinear_grid_value(
                        variable, None, None, lat_name, lon_name, lat_plan, lon_plan
                    )
                if not np.isfinite(value):
                    value = np.nan

                row = {
                    "cams_aot550": value,
                    "cams_variable": var_name,
                    "cams_dataset": CAMS_FORECAST_DATASET if "forecast" in file_path.name else CAMS_DATASET,
                    "cams_source_file": str(file_path),
                }
                if plan:
                    row.update(interpolation_metadata("cams", plan, scene_time))
                else:
                    row.update({
                        "cams_source_time_utc": "",
                        "cams_time_offset_min": np.nan,
                        "cams_interpolation_method": "no_time_coordinate",
                        "cams_interpolation_before_time_utc": "",
                        "cams_interpolation_after_time_utc": "",
                        "cams_interpolation_weight_after": np.nan,
                        "cams_interpolation_window_min": np.nan,
                    })
                row.update(spatial_metadata("cams", lat_plan, lon_plan, target_lon, scene.matchup_lat))
                return row
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read CAMS AOT550 from {file_path}: {last_error}") from last_error


def cams_local_files(paths: list[Path] | None = None) -> list[Path]:
    files: list[Path] = []
    for path in paths or []:
        if path.is_dir():
            files.extend(path.rglob("*"))
        elif path.exists():
            files.append(path)
    if CAMS_CACHE_DIR.exists():
        files.extend(CAMS_CACHE_DIR.rglob("*"))

    allowed = {".nc", ".nc4", ".cdf", ".grib", ".grb", ".grb2"}
    return sorted({path.resolve() for path in files if path.is_file() and path.suffix.lower() in allowed})


def cds_credentials(config_path: Path | None = DEFAULT_CDS_CONFIG) -> tuple[str | None, str | None]:
    url = os.environ.get("CDSAPI_URL") or CAMS_API_URL
    key = os.environ.get("CDSAPI_KEY")
    if key:
        return url, key

    if config_path is None or not config_path.exists():
        return url, None

    parser = configparser.RawConfigParser()
    parser.read(config_path, encoding="utf-8")
    if not parser.has_section("CDS"):
        return url, None

    api_key = parser.get("CDS", "api_key", fallback="").strip()
    uid = parser.get("CDS", "uid", fallback="").strip()
    username = parser.get("CDS", "username", fallback="").strip()
    password = parser.get("CDS", "password", fallback="").strip()
    configured_url = parser.get("CDS", "url", fallback="").strip()
    if configured_url:
        url = configured_url
    if api_key:
        return url, api_key
    if username and password:
        return url, f"{username}:{password}"
    return url, None


def cams_request_for_day(day: pd.Timestamp) -> tuple[str, dict, str]:
    if int(day.year) <= 2025:
        return (
            CAMS_DATASET,
            {
                "date": day.strftime("%Y-%m-%d"),
                "time": [f"{hour:02d}:00" for hour in range(0, 24, 3)],
                "variable": [CAMS_VARIABLE],
                "area": SWISS_AREA_NWSE,
                "format": "netcdf",
            },
            "eac4_reanalysis",
        )

    return (
        CAMS_FORECAST_DATASET,
        {
            "date": day.strftime("%Y-%m-%d"),
            "type": "forecast",
            "time": "00:00",
            "leadtime_hour": [str(hour) for hour in range(0, 24)],
            "variable": CAMS_VARIABLE,
            "area": SWISS_AREA_NWSE,
            "format": "netcdf",
        },
        "forecast_00utc",
    )


def download_cams_day(
    day: pd.Timestamp,
    target: Path,
    force: bool = False,
    cds_config: Path | None = DEFAULT_CDS_CONFIG,
) -> Path:
    if target.exists() and not force:
        return target
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is not installed. Install/configure cdsapi or provide local CAMS files via --cams-files."
        ) from exc

    url, key = cds_credentials(cds_config)
    if not key:
        raise RuntimeError(
            "No CDS/ADS API credentials found. Set CDSAPI_URL/CDSAPI_KEY, create .cdsapirc, "
            f"or provide a config with a [CDS] section. Checked: {cds_config}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url=url, key=key)
    dataset, request, _ = cams_request_for_day(day)
    client.retrieve(dataset, request, str(target))
    return target


def extract_cams(
    scene: Scene,
    cams_files: list[Path],
    download: bool,
    force_download: bool,
    cds_config: Path | None = DEFAULT_CDS_CONFIG,
) -> dict:
    candidate_files = list(cams_files)
    scene_day_token = scene.scene_datetime.tz_convert("UTC").strftime("%Y%m%d")
    matching_day_files = [path for path in candidate_files if scene_day_token in path.name]
    if matching_day_files:
        candidate_files = matching_day_files
    if download:
        day = scene.scene_datetime.tz_convert("UTC")
        _, _, product_kind = cams_request_for_day(day)
        target = CAMS_CACHE_DIR / f"cams_{product_kind}_aot550_{day:%Y%m%d}.nc"
        candidate_files.insert(0, download_cams_day(day, target, force=force_download, cds_config=cds_config))

    if not candidate_files:
        return {
            "cams_aot550": np.nan,
            "cams_grid_lon": np.nan,
            "cams_grid_lat": np.nan,
            "cams_spatial_method": "missing_local_file",
            "cams_interpolation_target_lon": scene.matchup_lon,
            "cams_interpolation_target_lat": scene.matchup_lat,
            "cams_interpolation_lon_lower": np.nan,
            "cams_interpolation_lon_upper": np.nan,
            "cams_interpolation_lon_weight_upper": np.nan,
            "cams_interpolation_lat_lower": np.nan,
            "cams_interpolation_lat_upper": np.nan,
            "cams_interpolation_lat_weight_upper": np.nan,
            "cams_source_time_utc": "",
            "cams_time_offset_min": np.nan,
            "cams_variable": "",
            "cams_dataset": CAMS_DATASET,
            "cams_source_file": "",
            "cams_status": "missing_local_file",
        }

    errors = []
    for file_path in candidate_files:
        try:
            row = extract_cams_from_file(file_path, scene)
            row["cams_status"] = "ok"
            return row
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    return {
        "cams_aot550": np.nan,
        "cams_grid_lon": np.nan,
        "cams_grid_lat": np.nan,
        "cams_spatial_method": "error",
        "cams_interpolation_target_lon": scene.matchup_lon,
        "cams_interpolation_target_lat": scene.matchup_lat,
        "cams_interpolation_lon_lower": np.nan,
        "cams_interpolation_lon_upper": np.nan,
        "cams_interpolation_lon_weight_upper": np.nan,
        "cams_interpolation_lat_lower": np.nan,
        "cams_interpolation_lat_upper": np.nan,
        "cams_interpolation_lat_weight_upper": np.nan,
        "cams_source_time_utc": "",
        "cams_time_offset_min": np.nan,
        "cams_variable": "",
        "cams_dataset": CAMS_DATASET,
        "cams_source_file": "",
        "cams_status": "error: " + " | ".join(errors[:3]),
    }


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def aeronet_url(site: str, day: pd.Timestamp, level: str) -> str:
    return (
        "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
        f"?site={site}"
        f"&year={day:%Y}&month={int(day.month)}&day={int(day.day)}"
        f"&year2={day:%Y}&month2={int(day.month)}&day2={int(day.day)}"
        f"&{level}=1&AVG=10&if_no_html=1"
    )


def download_aeronet(site: str, day: pd.Timestamp, level: str, force: bool = False) -> tuple[Path, str]:
    url = aeronet_url(site, day, level)
    cache_file = AERONET_CACHE_DIR / f"{site}_{level}_{day:%Y%m%d}.csv"
    if cache_file.exists() and not force:
        return cache_file, url

    AERONET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    cache_file.write_text(response.text, encoding="utf-8")
    return cache_file, url


def read_aeronet_table(file_path: Path) -> pd.DataFrame:
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if "Date(dd:mm:yyyy)" in line and "Time(hh:mm:ss)" in line:
            header_index = index
            break
    if header_index is None:
        return pd.DataFrame()
    return pd.read_csv(file_path, skiprows=header_index)


def parse_wavelength_nm(column: str) -> int | None:
    match = re.search(r"(?:AOD|Optical_Depth)[_\s-]*(\d{3,4})", column, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def find_angstrom_column(frame: pd.DataFrame) -> str | None:
    candidates = [
        "440-870_Angstrom_Exponent",
        "440-675_Angstrom_Exponent",
        "500-870_Angstrom_Exponent",
    ]
    by_name = select_name(candidates, frame.columns)
    if by_name:
        return by_name
    for column in frame.columns:
        if "angstrom" in str(column).lower() and "440" in str(column) and "870" in str(column):
            return str(column)
    for column in frame.columns:
        if "angstrom" in str(column).lower():
            return str(column)
    return None


def compute_aeronet_aot550(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["datetime_utc"] = pd.to_datetime(
        frame["Date(dd:mm:yyyy)"].astype(str) + " " + frame["Time(hh:mm:ss)"].astype(str),
        format="%d:%m:%Y %H:%M:%S",
        utc=True,
        errors="coerce",
    )

    wavelength_columns = []
    for column in frame.columns:
        wavelength = parse_wavelength_nm(str(column))
        if wavelength:
            wavelength_columns.append((wavelength, str(column)))

    direct_550 = [column for wavelength, column in wavelength_columns if wavelength == 550]
    if direct_550:
        values = pd.to_numeric(frame[direct_550[0]], errors="coerce")
        frame["aot550"] = values.where(values >= 0)
        frame["aot550_method"] = "direct_550"
        return frame

    angstrom_column = find_angstrom_column(frame)
    if not angstrom_column:
        return pd.DataFrame()
    available = [
        (abs(wavelength - 550), wavelength, column)
        for wavelength, column in wavelength_columns
        if 300 <= wavelength <= 1100
    ]
    if not available:
        return pd.DataFrame()
    alpha = pd.to_numeric(frame[angstrom_column], errors="coerce")
    frame["aot550"] = np.nan
    frame["aot550_method"] = ""
    for _, wavelength, column in sorted(available):
        base_aot = pd.to_numeric(frame[column], errors="coerce")
        converted = base_aot * ((550.0 / float(wavelength)) ** (-alpha))
        valid = (base_aot >= 0) & (alpha >= 0) & (converted >= 0) & pd.notna(converted)
        missing = pd.isna(frame["aot550"])
        use = missing & valid
        frame.loc[use, "aot550"] = converted[use]
        frame.loc[use, "aot550_method"] = f"angstrom_from_{wavelength}nm"
    return frame


def extract_aeronet(scene: Scene, force_download: bool = False) -> dict:
    scene_time = scene.scene_datetime.tz_convert("UTC")
    ranked_sites = sorted(
        AERONET_SITES,
        key=lambda site: haversine_km(scene.matchup_lon, scene.matchup_lat, site["lon"], site["lat"]),
    )

    best: dict | None = None
    errors = []
    for site in ranked_sites:
        distance_km = haversine_km(scene.matchup_lon, scene.matchup_lat, site["lon"], site["lat"])
        for level in AERONET_LEVELS:
            try:
                file_path, url = download_aeronet(site["site"], scene_time, level, force=force_download)
                table = compute_aeronet_aot550(read_aeronet_table(file_path))
                if "AERONET_Site" in table.columns:
                    table = table[table["AERONET_Site"].astype(str) == site["site"]]
                if table.empty or "aot550" not in table:
                    continue
                table["time_offset_min"] = np.abs((table["datetime_utc"] - scene_time).dt.total_seconds()) / 60.0
                table = table[pd.notna(table["aot550"]) & pd.notna(table["time_offset_min"])]
                table = table[table["time_offset_min"] <= AERONET_SEARCH_WINDOW_MIN]
                if table.empty:
                    continue
                table = table.sort_values("datetime_utc").reset_index(drop=True)
                plan = time_interpolation_plan(table["datetime_utc"], scene_time)
                selected = table.iloc[plan["nearest_index"]]
                before = table.iloc[plan["before_index"]]
                after = table.iloc[plan["after_index"]]
                value = interpolate_pair(before["aot550"], after["aot550"], plan["weight_after"])
                if not np.isfinite(value):
                    continue
                candidate = {
                    "aeronet_aot550": value,
                    "aeronet_site": site["site"],
                    "aeronet_site_lon": site["lon"],
                    "aeronet_site_lat": site["lat"],
                    "aeronet_site_elevation_m": site["elevation_m"],
                    "aeronet_distance_km": distance_km,
                    "aeronet_spatial_method": "nearest_station",
                    "aeronet_level": level,
                    "aeronet_aot550_method": selected["aot550_method"],
                    "aeronet_source_url": url,
                    "aeronet_status": "ok",
                }
                candidate.update(interpolation_metadata("aeronet", plan, scene_time))
                candidate.update({
                    "aeronet_interpolation_before_aot550_method": before["aot550_method"],
                    "aeronet_interpolation_after_aot550_method": after["aot550_method"],
                })
                if best is None or candidate["aeronet_time_offset_min"] < best["aeronet_time_offset_min"]:
                    best = candidate
            except Exception as exc:
                errors.append(f"{site['site']} {level}: {exc}")

    if best:
        return best
    return {
        "aeronet_aot550": np.nan,
        "aeronet_site": "",
        "aeronet_site_lon": np.nan,
        "aeronet_site_lat": np.nan,
        "aeronet_site_elevation_m": np.nan,
        "aeronet_distance_km": np.nan,
        "aeronet_spatial_method": "missing_match",
        "aeronet_level": "",
        "aeronet_source_time_utc": "",
        "aeronet_time_offset_min": np.nan,
        "aeronet_aot550_method": "",
        "aeronet_source_url": "",
        "aeronet_status": "missing_match" if not errors else "error: " + " | ".join(errors[:3]),
    }


def build_scene_base(scene: Scene) -> dict:
    return {
        "campaign": scene.campaign,
        "sensor": scene.sensor,
        "scene_datetime": scene.scene_datetime.tz_convert("UTC").isoformat(),
        "n_matchup_measurements": scene.n_matchup_measurements,
        "scene_time_fallback_no_input_overpass": scene.scene_time_fallback_no_input_overpass,
    }


def build_covariates(
    force_download: bool = False,
    earthdata_config: Path | None = DEFAULT_EARTHDATA_CONFIG,
    include_cams: bool = True,
    cams_files: list[Path] | None = None,
    download_cams: bool = False,
    cds_config: Path | None = DEFAULT_CDS_CONFIG,
    include_aeronet: bool = True,
    skip_merra2_download: bool = False,
) -> pd.DataFrame:
    local_cams_files = cams_local_files(cams_files)
    rows = []

    for cfg in ACTIVE_RUNS:
        scene = campaign_scene_definition(cfg)
        day = scene.scene_datetime.tz_convert("UTC")
        print(
            f"[aot550] {scene.campaign} {scene.sensor}: "
            f"centroid=({scene.matchup_lon:.5f}, {scene.matchup_lat:.5f}) time={day.isoformat()}"
        )

        row = build_scene_base(scene)
        url = merra2_daily_url(day)
        local_file = MERRA2_CACHE_DIR / f"{Path(url).name}"
        if not local_file.exists() and skip_merra2_download:
            raise FileNotFoundError(f"Missing cached MERRA-2 file and --skip-merra2-download was used: {local_file}")
        download_file(url, local_file, force=force_download, earthdata_config=earthdata_config)
        row.update(extract_merra2_variables(local_file, scene, url))

        if include_cams:
            row.update(
                extract_cams(
                    scene,
                    local_cams_files,
                    download=download_cams,
                    force_download=force_download,
                    cds_config=cds_config,
                )
            )
        if include_aeronet:
            row.update(extract_aeronet(scene, force_download=force_download))

        rows.append(row)
        frame = pd.DataFrame(rows).sort_values(["campaign", "sensor"], kind="mergesort")
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUTPUT_CSV, index=False)
        print(f"[aot550] wrote incremental covariates to {OUTPUT_CSV}")

    frame = pd.DataFrame(rows).sort_values(["campaign", "sensor"], kind="mergesort")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true", help="Download source files even if cached.")
    parser.add_argument(
        "--earthdata-config",
        type=Path,
        default=DEFAULT_EARTHDATA_CONFIG,
        help="INI file with an [EARTHDATA] username/password section. Defaults to sentinel-hindcast docker_env.ini.",
    )
    parser.add_argument(
        "--cams-files",
        nargs="*",
        type=Path,
        default=[],
        help="Local CAMS NetCDF/GRIB files or directories. Cache directory is searched automatically.",
    )
    parser.add_argument("--download-cams", action="store_true", help="Download daily CAMS AOT550 NetCDF files with cdsapi.")
    parser.add_argument(
        "--cds-config",
        type=Path,
        default=DEFAULT_CDS_CONFIG,
        help="INI file with a [CDS] section. Defaults to sentinel-hindcast docker_env.ini.",
    )
    parser.add_argument("--no-cams", action="store_true", help="Skip CAMS extraction.")
    parser.add_argument("--no-aeronet", action="store_true", help="Skip AERONET extraction.")
    parser.add_argument(
        "--skip-merra2-download",
        action="store_true",
        help="Use only cached MERRA-2 files. Fails if a required daily file is missing.",
    )
    args = parser.parse_args()

    frame = build_covariates(
        force_download=args.force_download,
        earthdata_config=args.earthdata_config,
        include_cams=not args.no_cams,
        cams_files=args.cams_files,
        download_cams=args.download_cams,
        cds_config=args.cds_config,
        include_aeronet=not args.no_aeronet,
        skip_merra2_download=args.skip_merra2_download,
    )
    print(f"Wrote {len(frame)} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
