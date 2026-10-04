"""
Per-datapoint EO match-up metrics export (single CSV, per-wavelength + overall, no plotting)

What this script does:
- Uses your loading + KDTree match-up logic (NetCDF + GAAC GeoTIFF support, processor discovery in PRODUCT_ROOT).
- Computes distance-to-shore (km) for every in situ point from a binary/depth raster.
- Matches every in situ datapoint to the nearest EO pixel per processor.
- For each in situ datapoint and processor:
    * extracts the EO spectrum at the matched pixel
    * computes per-wavelength metrics
    * computes overall metrics aggregated across all valid common wavelengths
- Writes one single CSV:
    * one row per input datapoint × processor
    * includes overall metrics
    * includes per-wavelength values and per-wavelength metrics as columns

Notes:
- No plotting is performed.
- SAM is computed only as an overall spectral metric across all valid common wavelengths.
"""

import os
import re
import sys
import importlib.util
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import rasterio
import rasterio.enums
import rasterio.features
import rasterio.vrt
from shapely.geometry import shape
from shapely.ops import unary_union
from scipy import ndimage
from scipy.spatial import cKDTree
from pyproj import Transformer

SCRIPT_DIR = Path(__file__).resolve().parent
STATS_DIR = SCRIPT_DIR.parent
for import_path in [SCRIPT_DIR, STATS_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import matchup_common as mc
from campaign_locations import extract_location


# =========================
# Configuration
# =========================
RESULTS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RESULTS_ROOT.parent
DOCUMENTS_ROOT = PROJECT_ROOT.parent


def first_existing_path(*candidates: Path, required_child: str | None = None) -> Path:
    for candidate in candidates:
        if candidate.exists() and (required_child is None or (candidate / required_child).exists()):
            return candidate
    return candidates[0]


INSITU_ROOT = first_existing_path(
    PROJECT_ROOT / "insitu",
    DOCUMENTS_ROOT / "insitu",
    required_child="20231008_ZRH",
)
RASTER_ROOT = first_existing_path(
    PROJECT_ROOT / "insitu" / "all" / "qgis_transect_bb",
    DOCUMENTS_ROOT / "insitu" / "all" / "qgis_transect_bb",
    required_child="binary_masks",
)

CSV_HYPER_FILE = str(INSITU_ROOT / "20231008_ZRH" / "visualization_analysis" / "quality_control" / "20231008_ZRH_metadata_with_Rrs_S2B_final_qc.csv")
CSV_CONV_FILE = str(INSITU_ROOT / "20231008_ZRH" / "visualization_analysis" / "quality_control" / "20231008_ZRH_metadata_with_Rrs_conv_S2B_qc.csv")

DEPTH_RASTER_FILE = str(RASTER_ROOT / "binary_masks" / "ZRH_mask.tif")
DEM_RASTER_FILE = str(RASTER_ROOT / "depth" / "DEM.tif")
LANDWATER_RASTER_FILE = str(RASTER_ROOT / "binary_masks" / "ZRH_20231008_binary.tif")
VIEWING_ANGLES_DIR = str(first_existing_path(PROJECT_ROOT / "eo_data" / "viewing_angles", DOCUMENTS_ROOT / "eo_data" / "viewing_angles", required_child="all_campaigns_transect_viewing_angles.csv"))
AEROSOL_COVARIATES_FILE = str(first_existing_path(RESULTS_ROOT / "stats" / "performance_metrics", SCRIPT_DIR / "AOT550", SCRIPT_DIR, required_child="aerosol_scene_covariates.csv") / "aerosol_scene_covariates.csv")
INPUT_L2SR_ROOT = str(first_existing_path(PROJECT_ROOT / "eo_data" / "input_l2sr", DOCUMENTS_ROOT / "eo_data" / "input_l2sr"))
TRANSECT_CONFIG_PATH = PROJECT_ROOT / "insitu" / "quality_control" / "campaign_transects.py"

output_dir = str(RESULTS_ROOT / "stats" / "performance_metrics" / "20231008_ZRH" / "test")
os.makedirs(output_dir, exist_ok=True)

PRODUCT_ROOT = str(first_existing_path(PROJECT_ROOT / "eo_data" / "output", DOCUMENTS_ROOT / "eo_data" / "output", required_child="acolite"))
PRODUCT_KEYWORDS = ["ZRH", "2023", "S2", "TMT"]

x_col = "x-coordinate"
y_col = "y-coordinate"
EPSG_CH = 2056

USE_MANUAL_EXTENTS = True
EXTENTS = ["3km"]

# Matchup value used for sat_rrs_<wavelength> and all derived metrics.
# Options: "nearest", "mean", "median".
MATCHUP_VALUE_METHOD = "median"
INSITU_HARD_FILTER_COLUMN = "hard_filter_retain"
GREEN_PEAK_MIN_NM = 540
GREEN_PEAK_MAX_NM = 590

# Local footprint/context used for footprint averaging, land/shore fraction, and
# invalid/negative EO fractions.
# A radius of 1.5 pixels accepts a nearest pixel whose 3x3 window overlaps the in situ point.
# on a regular grid.
FOOTPRINT_RADIUS_PIXELS = 1.5
CONTEXT_RADIUS_PIXELS = FOOTPRINT_RADIUS_PIXELS

PRECISE_EO_ANGLE_COLUMNS = [
    "eo_sun_zenith_deg",
    "eo_sun_azimuth_deg",
    "eo_view_zenith_deg",
    "eo_view_azimuth_deg",
    "eo_relative_azimuth_deg",
]

REDUNDANT_ANGLE_COLUMNS = [
    "sun_zenith_deg",
    "sun_azimuth_deg",
    "view_zenith_deg",
    "view_azimuth_deg",
    "relative_azimuth_deg",
    "eo_sun_azimuth_deg_min",
    "eo_sun_azimuth_deg_max",
    "eo_sun_azimuth_deg_mean",
    "eo_sun_azimuth_deg_count",
    "eo_sun_azimuth_deg_n_measurements_with_coordinates",
    "eo_sun_zenith_deg_min",
    "eo_sun_zenith_deg_max",
    "eo_sun_zenith_deg_mean",
    "eo_sun_zenith_deg_count",
    "eo_sun_zenith_deg_n_measurements_with_coordinates",
    "eo_view_azimuth_deg_min",
    "eo_view_azimuth_deg_max",
    "eo_view_azimuth_deg_mean",
    "eo_view_azimuth_deg_count",
    "eo_view_azimuth_deg_n_measurements_with_coordinates",
    "eo_view_zenith_deg_min",
    "eo_view_zenith_deg_max",
    "eo_view_zenith_deg_mean",
    "eo_view_zenith_deg_count",
    "eo_view_zenith_deg_n_measurements_with_coordinates",
    "insitu_sun_azimuth_deg_min",
    "insitu_sun_azimuth_deg_max",
    "insitu_sun_azimuth_deg_mean",
    "insitu_sun_azimuth_deg_count",
    "insitu_sun_azimuth_deg_n_measurements_with_coordinates",
    "insitu_sun_zenith_deg_min",
    "insitu_sun_zenith_deg_max",
    "insitu_sun_zenith_deg_mean",
    "insitu_sun_zenith_deg_count",
    "insitu_sun_zenith_deg_n_measurements_with_coordinates",
    "insitu_relative_azimuth_deg_min",
    "insitu_relative_azimuth_deg_max",
    "insitu_relative_azimuth_deg_mean",
    "insitu_relative_azimuth_deg_count",
    "insitu_relative_azimuth_deg_n_measurements_with_coordinates",
    "insitu_raw_relative_azimuth_deg_min",
    "insitu_raw_relative_azimuth_deg_max",
    "insitu_raw_relative_azimuth_deg_mean",
    "insitu_raw_relative_azimuth_deg_count",
    "insitu_raw_relative_azimuth_deg_n_measurements_with_coordinates",
    "insitu_heading_deg_min",
    "insitu_heading_deg_max",
    "insitu_heading_deg_mean",
    "insitu_heading_deg_count",
    "insitu_heading_deg_n_measurements_with_coordinates",
    "insitu_inclination_v_deg_min",
    "insitu_inclination_v_deg_max",
    "insitu_inclination_v_deg_mean",
    "insitu_inclination_v_deg_count",
    "insitu_inclination_v_deg_n_measurements_with_coordinates",
    "insitu_inclination_x_deg_min",
    "insitu_inclination_x_deg_max",
    "insitu_inclination_x_deg_mean",
    "insitu_inclination_x_deg_count",
    "insitu_inclination_x_deg_n_measurements_with_coordinates",
    "insitu_inclination_y_deg_min",
    "insitu_inclination_y_deg_max",
    "insitu_inclination_y_deg_mean",
    "insitu_inclination_y_deg_count",
    "insitu_inclination_y_deg_n_measurements_with_coordinates",
    "eo_relative_azimuth_deg_mean",
]


# =============================================================================
# SENSOR BANDS / INDICES
# =============================================================================
SENSOR_BANDS = {
    "L8":  [443, 483, 561, 655, 865],
    "L9":  [443, 482, 561, 654, 865],
    "S2A": [443, 492, 560, 665, 704, 740, 783, 842, 865],
    "S2B": [442, 492, 559, 665, 704, 739, 780, 842, 865]
}

VISIBLE_SENSOR_BANDS = {
    "L8":  [443, 483, 561, 655],
    "L9":  [443, 482, 561, 654],
    "S2A": [443, 492, 560, 665, 704],
    "S2B": [442, 492, 559, 665, 704],
}

L2SR_CONTRAST_BANDS = {
    "L8": [443, 482, 561, 655],
    "L9": [443, 482, 561, 655],
    "S2A": [443, 492, 560, 665, 704],
    "S2B": [442, 492, 559, 665, 704],
}

# Land-side contrast is computed at these radii (land fraction/reach of
# adjacency-effect contamination). The water side uses a separate, much
# smaller, fixed radius (see WATER_LOCAL_RADIUS_M) so it stays sensitive to
# local water optical gradients instead of being smeared over the land-side
# disc.
CONTRAST_RADII_KM = [1, 5]
WATER_LOCAL_RADIUS_M = 100.0


def insitu_rrs_column_map(sensor):
    """Map each L2SR_CONTRAST_BANDS wavelength to the in-situ Rrs_ column for
    the same physical band. The two band-label dicts disagree by ~1nm for a
    couple of Landsat bands (e.g. L8 blue is 482 here but 483 in
    VISIBLE_SENSOR_BANDS, which is what in-situ Rrs_ columns are keyed by),
    so match by nearest wavelength rather than assuming an exact label match.
    """
    available = VISIBLE_SENSOR_BANDS[sensor]
    return {
        wl: f"Rrs_{int(min(available, key=lambda a: abs(a - wl)))}"
        for wl in L2SR_CONTRAST_BANDS[sensor]
    }

L2SR_LANDSAT_BANDS = {
    "L8": {443: "B1", 482: "B2", 561: "B3", 655: "B4"},
    "L9": {443: "B1", 482: "B2", 561: "B3", 655: "B4"},
}

L2SR_SAFE_BANDS = {
    "S2A": {443: "B01", 492: "B02", 560: "B03", 665: "B04", 704: "B05"},
    "S2B": {442: "B01", 492: "B02", 559: "B03", 665: "B04", 704: "B05"},
}

S2_BAND_ID = {
    "B01": 0,
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}

LANDSAT_L2SR_SCALE = 0.0000275
LANDSAT_L2SR_ADD = -0.2
L2SR_REFLECTANCE_MIN = 0.0
L2SR_REFLECTANCE_MAX = 1.0

GAAC_BAND_INDEX = {
    "L8": {443: 1, 483: 2, 561: 3, 655: 4, 865: 5},
    "L9": {443: 1, 482: 2, 561: 3, 654: 4, 865: 5},
    "S2A": {443: 1, 492: 2, 560: 3, 665: 4, 704: 5, 740: 6, 783: 7, 842: 8, 865: 9},
    "S2B": {442: 1, 492: 2, 559: 3, 665: 4, 704: 5, 739: 6, 780: 7, 842: 8, 865: 9},
}

C2RCC_BAND_INDEX = {
    "L8": {443: 1, 483: 2, 561: 3, 655: 4, 865: 5},
    "L9": {443: 1, 482: 2, 561: 3, 654: 4, 865: 5},
    "S2A": {443: 1, 492: 2, 560: 3, 665: 4, 704: 5, 740: 6, 783: 7, 865: 8},
    "S2B": {442: 1, 492: 2, 559: 3, 665: 4, 704: 5, 739: 6, 780: 7,  865: 8},
}



C2RCC_S2_WL_TO_BAND = {
    "S2A": {443: "B1", 492: "B2", 560: "B3", 665: "B4", 704: "B5", 740: "B6", 783: "B7", 865: "B8A"},
    "S2B": {442: "B1", 492: "B2", 559: "B3", 665: "B4", 704: "B5", 739: "B6", 780: "B7", 865: "B8A"},
}

KM_TO_PX = {
    "L8": {"3km": "100x100", "5km": "167x167", "10km": "334x334", "15km": "500x500"},
    "L9": {"3km": "100x100", "5km": "167x167", "10km": "334x334", "15km": "500x500"},
    "S2A": {"3km": "150x150", "5km": "250x250", "10km": "500x500", "15km": "750x750"},
    "S2B": {"3km": "150x150", "5km": "250x250", "10km": "500x500", "15km": "750x750"},
}

# =============================================================================
# PROCESSORS
# =============================================================================
PROCESSORS = {
    "ACOLITE": {
        "enabled": True,
        "must_contain": ["acolite_dsf"],
        "band_prefixes": ["Rrs_"],
        "divide_by_pi": False,
        "extent_mode": "px"
    },
    "ACOLITE+RAdCor": {
        "enabled": True,
        "must_contain": ["acolite_radcor"],
        "band_prefixes": ["Rrs_"],
        "divide_by_pi": False,
        "extent_mode": "km"
    },
    "RAdCor+ACOLITE": {
        "enabled": True,
        "must_contain": ["radcor_acolite"],
        "band_prefixes": ["Rrs_"],
        "divide_by_pi": False,
        "extent_mode": "single"
    },
    "ACOLITE+T-Mart": {
        "enabled": True,
        "must_contain": ["acolite_tmart"],
        "band_prefixes": ["Rrs_"],
        "divide_by_pi": False,
        "extent_mode": "px"
    },
    "Polymer": {
        "enabled": True,
        "must_contain": ["polymer"],
        "band_prefixes": ["rhow", "Rw", "rhow_"],
        "divide_by_pi": True,
        "extent_mode": "single"
    },
    "Polymer+T-Mart": {
        "enabled": True,
        "must_contain": ["polymer_tmart"],
        "band_prefixes": ["rhow", "Rw", "rhow_"],
        "divide_by_pi": True,
        "extent_mode": "single"
    },
    "Polymer+Radcor": {
        "enabled": True,
        "must_contain": ["radcor_polymer"],
        "band_prefixes": ["rhow", "Rw", "rhow_"],
        "divide_by_pi": True,
        "extent_mode": "single"
    },
    "C2RCC": {
        "enabled": True,
        "must_contain": ["c2rcc"],
        "band_prefixes": ["Rw", "rhow", "rhow_B", "rhow_"],
        "divide_by_pi": True,
        "extent_mode": "single"
    },
    "C2RCC+T-Mart": {
        "enabled": True,
        "must_contain": ["c2rcc_tmart"],
        "band_prefixes": ["Rw", "rhow", "rhow_B", "rhow_"],
        "divide_by_pi": True,
        "extent_mode": "single"
    },
    "C2RCC+RAdCor": {
        "enabled": True,
        "must_contain": ["c2rcc_radcor"],
        "band_prefixes": ["Rw", "rhow", "rhow_B", "rhow_", "B"],
        "divide_by_pi": True,
        "extent_mode": "single"
    },
    "GAAC": {
        "enabled": True,
        "must_contain": ["gaac", "rhow"],
        "band_prefixes": ["band_"],
        "divide_by_pi": True,
        "extent_mode": "single"
    },
}


# =============================================================================
# HELPERS
# =============================================================================
def extract_campaign(csv_path):
    base = os.path.basename(csv_path)
    m = re.match(r"(\d{8}_[A-Z]+)", base)
    return m.group(1) if m else "UNKNOWN"

def extract_sensor_from_csv(csv_path):
    base = os.path.basename(csv_path)
    if "_L9" in base or "L9" in base: return "L9"
    if "_L8" in base or "L8" in base: return "L8"
    if "_S2A" in base: return "S2A"
    if "_S2B" in base: return "S2B"
    return "S2A"

def load_campaign_transects(campaign):
    spec = importlib.util.spec_from_file_location("campaign_transects", TRANSECT_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load campaign transects from {TRANSECT_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_campaign_transects(campaign, require_metadata=False)

def _has_all(path: str, tokens):
    low = path.lower()
    return all(t.lower() in low for t in tokens)

def extract_wavelength_from_name(name):
    m = re.search(r"(\d{3,4}(\.\d+)?)", str(name))
    return float(m.group(1)) if m else None

def find_closest_bandname(target_wl, bandnames):
    pairs = [(b, extract_wavelength_from_name(b)) for b in bandnames if extract_wavelength_from_name(b) is not None]
    if not pairs:
        raise ValueError("No numeric wavelengths found in bandnames")
    names, wls = zip(*pairs)
    idx = np.abs(np.array(wls) - float(target_wl)).argmin()
    return names[idx]

def get_band_candidates(ds: xr.Dataset, prefixes: list[str]) -> list[str]:
    prefixes_low = [p.lower() for p in prefixes]
    out = []
    for v in ds.data_vars:
        vl = str(v).lower()
        if any(vl.startswith(p) for p in prefixes_low):
            out.append(v)
    return out


def is_raster_stack(obj) -> bool:
    return hasattr(obj, "dims") and "band" in obj.dims and "x" in obj.coords and "y" in obj.coords


def raster_stack_band_map(obj, wavelengths, preferred_prefixes=("Rrs", "rhow", "rhos")):
    names = obj.attrs.get("long_name")
    if names is None:
        names = [str(value) for value in obj.coords["band"].values]
    if isinstance(names, str):
        names = [names]
    names = list(names)
    bands = list(obj.coords["band"].values)
    pairs = []
    for band, name in zip(bands, names):
        text = str(name)
        wl = extract_wavelength_from_name(text)
        if wl is None:
            continue
        prefix_rank = len(preferred_prefixes)
        lowered = text.lower()
        for rank, prefix in enumerate(preferred_prefixes):
            if lowered.startswith(prefix.lower()):
                prefix_rank = rank
                break
        pairs.append((int(band), text, wl, prefix_rank))
    mapping = {}
    for wl in wavelengths:
        candidates = [item for item in pairs if item[3] < len(preferred_prefixes)]
        if not candidates:
            candidates = pairs
        if not candidates:
            mapping[float(wl)] = None
            continue
        best = min(candidates, key=lambda item: (item[3], abs(item[2] - float(wl))))
        mapping[float(wl)] = best[0]
    return mapping

def proc_token_for_plot_extent(proc_name: str, proc_cfg: dict, plot_extent_km: str, sensor: str) -> str | None:
    mode = proc_cfg.get("extent_mode", "px")
    if mode == "km":
        return plot_extent_km
    if mode == "px":
        return KM_TO_PX.get(sensor, {}).get(plot_extent_km)
    if mode == "single":
        return None
    raise ValueError(f"Unknown extent_mode={mode} for processor {proc_name}")

def discover_files_for_processor(root_dir: str, proc_name: str, proc_cfg: dict,
                                 plot_extents_km: list[str], sensor: str, global_keywords: list[str]):
    out = {}

    keyword_text = " ".join(str(token).lower() for token in global_keywords)
    if proc_name == "ACOLITE+RAdCor" and sensor == "S2B" and "zrh" in keyword_text and "20260227" in keyword_text:
        transect_root_candidates = [
            PROJECT_ROOT / "eo_data" / "output" / "acolite_radcor" / "Sentinel" / "ZRH_20260227",
            DOCUMENTS_ROOT / "eo_data" / "output" / "acolite_radcor" / "Sentinel" / "ZRH_20260227",
        ]
        transect_files = {}
        for transect_id, token in [(1, "kr5km_t1"), (3, "kr5km_t3")]:
            matches = []
            for transect_root in transect_root_candidates:
                if not transect_root.is_dir():
                    continue
                for dp, _, files in os.walk(transect_root):
                    for f in files:
                        full = os.path.join(dp, f)
                        low = full.lower().replace(chr(92), "/")
                        product_name = os.path.basename(os.path.dirname(dp)).lower()
                        name_low = f.lower()
                        if (
                            name_low.endswith((".tif", ".tiff", ".nc"))
                            and token in product_name
                            and "l2acolite" in low
                            and "acolite_radcor_tsdsf" in low
                            and "20260227" in low
                        ):
                            matches.append(full)
            if matches:
                transect_files[transect_id] = sorted(
                    matches,
                    key=lambda p: (0 if p.lower().endswith((".tif", ".tiff")) else 1, len(p), p),
                )[0]

        if transect_files:
            out["__by_transect__"] = transect_files
            first_output = transect_files.get(1) or next(iter(transect_files.values()))
            out["__single__"] = first_output
            for pe in plot_extents_km:
                out[pe] = first_output
            return out

    if proc_name == "RAdCor+ACOLITE":
        roots = [
            PROJECT_ROOT / "eo_data" / "output" / "radcor_acolite",
            DOCUMENTS_ROOT / "eo_data" / "output" / "radcor_acolite",
        ]
        candidates = []
        for search_root in roots:
            if not search_root.is_dir():
                continue
            for dp, _, files in os.walk(search_root):
                for f in files:
                    name_low = f.lower()
                    if not name_low.endswith((".nc", ".tif", ".tiff")):
                        continue
                    full = os.path.join(dp, f)
                    low = full.lower().replace(chr(92), "/")
                    if not _has_all(full, global_keywords):
                        continue
                    if "_watermask" in name_low or "_rgb" in name_low or "_reproducibility" in low:
                        continue
                    if "l2acolite" not in low and "l2w" not in name_low:
                        continue
                    candidates.append(full)
        if candidates:
            first_output = sorted(
                set(candidates),
                key=lambda p: (0 if p.lower().endswith((".tif", ".tiff")) else 1, len(p), p),
            )[0]
            out["__single__"] = first_output
            for pe in plot_extents_km:
                out[pe] = first_output
            return out

    wanted_tokens = {}
    if proc_cfg.get("extent_mode") != "single":
        for pe in plot_extents_km:
            tok = proc_token_for_plot_extent(proc_name, proc_cfg, pe, sensor)
            if tok is not None:
                wanted_tokens[pe] = tok.lower()

    for dp, _, files in os.walk(root_dir):
        for f in files:
            if not f.lower().endswith((".nc", ".tif", ".tiff")):
                continue
            full = os.path.join(dp, f)
            if not _has_all(full, global_keywords):
                continue
            if proc_name == "GAAC":
                name_low = f.lower()
                full_low = full.lower()
                if not (
                    name_low.endswith((".tif", ".tiff"))
                    and "rhow" in name_low
                    and "gaac" in full_low
                    and "_watermask" not in name_low
                    and "_rgb" not in name_low
                ):
                    continue
            elif not _has_all(full, proc_cfg["must_contain"]):
                continue

            mode = proc_cfg.get("extent_mode", "px")
            if mode == "single":
                out.setdefault("__single__", full)
                continue

            low = full.lower()
            for pe, tok in wanted_tokens.items():
                if tok in low:
                    out[pe] = full
    return out

def nearest_distance_km(pt, line_geom):
    if pt.is_empty or not pt.is_valid or line_geom.is_empty:
        return float("nan")
    return pt.distance(line_geom) / 1000.0


def raster_pixel_sizes_m(transform, raster_crs, shape_hw):
    h, w = shape_hw
    center_col = max(0, min(w - 2, w // 2))
    center_row = max(0, min(h - 2, h // 2))
    to_2056 = Transformer.from_crs(raster_crs, f"EPSG:{EPSG_CH}", always_xy=True)
    x0, y0 = transform * (center_col, center_row)
    x1, y1 = transform * (center_col + 1, center_row)
    x2, y2 = transform * (center_col, center_row + 1)
    x0m, y0m = to_2056.transform(x0, y0)
    x1m, y1m = to_2056.transform(x1, y1)
    x2m, y2m = to_2056.transform(x2, y2)
    px_m = float(np.hypot(x1m - x0m, y1m - y0m))
    py_m = float(np.hypot(x2m - x0m, y2m - y0m))
    return px_m, py_m


def fill_small_internal_mask_holes(water_mask, pixel_area_m2, max_hole_area_m2=4.0):
    if not np.any(water_mask) or not np.any(~water_mask) or not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        return water_mask
    labels, n_labels = ndimage.label(~water_mask, structure=np.ones((3, 3), dtype=bool))
    if n_labels == 0:
        return water_mask

    border_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    candidate_labels = np.setdiff1d(np.arange(1, n_labels + 1), border_labels, assume_unique=False)
    if candidate_labels.size == 0:
        return water_mask

    counts = np.bincount(labels.ravel())
    max_pixels = max(1, int(np.floor(max_hole_area_m2 / pixel_area_m2)))
    small_labels = candidate_labels[counts[candidate_labels] <= max_pixels]
    if small_labels.size == 0:
        return water_mask

    cleaned = water_mask.copy()
    cleaned[np.isin(labels, small_labels)] = True
    return cleaned


def parse_scene_datetime_from_path(path):
    """Best-effort overpass datetime parser from common Landsat/Sentinel names."""
    if not path:
        return pd.NaT
    text = os.path.basename(str(path))

    m = re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", text)
    if m:
        y, mo, d, h, mi, s = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=s)

    m = re.search(r"(\d{8})T(\d{6})", text)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S", errors="coerce")

    return pd.NaT


def _angle_feature_name(angle_domain, metric, statistic):
    domain_prefix = {
        "eo_angles": "eo",
        "insitu_angles": "insitu",
    }.get(str(angle_domain), safe_output_name(angle_domain))
    metric_name = safe_output_name(metric)
    suffix = "_deg" if any(token in metric_name for token in ("azimuth", "zenith", "inclination", "heading")) else ""
    return f"{domain_prefix}_{metric_name}{suffix}_{statistic}"


_ANGLE_TABLE_CACHE: dict[str, pd.DataFrame] = {}

_EO_METRIC_TO_PRECISE_COLUMN = {
    "sun_zenith": "eo_sun_zenith_deg",
    "sun_azimuth": "eo_sun_azimuth_deg",
    "view_zenith": "eo_view_zenith_deg",
    "view_azimuth": "eo_view_azimuth_deg",
}


def _load_angle_table(angle_dir):
    """Cached read of the consolidated, per-transect eo/insitu viewing-angle
    table that covers all campaigns (all_campaigns_transect_viewing_angles.csv).
    Superseded the older per-campaign angle_stats.json/csv files, which are no
    longer read here."""
    key = str(angle_dir)
    if key not in _ANGLE_TABLE_CACHE:
        path = os.path.join(str(angle_dir), "all_campaigns_transect_viewing_angles.csv")
        _ANGLE_TABLE_CACHE[key] = pd.read_csv(path, sep=None, engine="python") if os.path.exists(path) else pd.DataFrame()
    return _ANGLE_TABLE_CACHE[key]


def _relative_azimuth_deg(sun_azimuth_deg, view_azimuth_deg):
    """Angular separation between sun and view azimuth, wrapped to [0, 180] deg."""
    if not (np.isfinite(sun_azimuth_deg) and np.isfinite(view_azimuth_deg)):
        return np.nan
    diff = abs(sun_azimuth_deg - view_azimuth_deg) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def _angle_features_from_eo_rows(eo_rows):
    """eo_rows: rows with angle_domain == 'eo_angles' for one campaign/sensor,
    optionally spanning several transects. Returns the precise single-value
    columns (PRECISE_EO_ANGLE_COLUMNS; count-weighted mean across transects
    where more than one is present) plus the diagnostic _min/_max/_mean/_count/
    _n_measurements_with_coordinates columns matching REDUNDANT_ANGLE_COLUMNS."""
    features = {}
    precise = {}
    for metric, precise_col in _EO_METRIC_TO_PRECISE_COLUMN.items():
        sub = eo_rows[eo_rows["metric"] == metric]
        if sub.empty:
            continue
        weights = sub["count"].astype(float).clip(lower=1.0)
        weighted_mean = float(np.average(sub["mean"].astype(float), weights=weights))
        features[_angle_feature_name("eo_angles", metric, "min")] = float(sub["min"].min())
        features[_angle_feature_name("eo_angles", metric, "max")] = float(sub["max"].max())
        features[_angle_feature_name("eo_angles", metric, "mean")] = weighted_mean
        features[_angle_feature_name("eo_angles", metric, "count")] = float(sub["count"].sum())
        if "n_measurements_with_coordinates" in sub.columns:
            features[_angle_feature_name("eo_angles", metric, "n_measurements_with_coordinates")] = float(
                sub["n_measurements_with_coordinates"].sum()
            )
        features[precise_col] = weighted_mean
        precise[metric] = weighted_mean
    features["eo_relative_azimuth_deg"] = _relative_azimuth_deg(
        precise.get("sun_azimuth", np.nan), precise.get("view_azimuth", np.nan)
    )
    return features


def _campaign_sensor_eo_rows(angle_dir, campaign, sensor):
    table = _load_angle_table(angle_dir)
    if table.empty:
        return table
    return table[
        (table["campaign"].astype(str) == str(campaign))
        & (table["sensor_token"].astype(str).str.upper() == str(sensor).upper())
        & (table["angle_domain"] == "eo_angles")
    ]


def load_viewing_angle_features(angle_dir, campaign, sensor):
    """Scene-level (campaign x sensor) eo viewing/illumination angles, aggregated
    (count-weighted) across all transects. Applied to every row as the base
    value; load_transect_angle_features overrides it with transect-specific
    values where a row's own transect has a more precise entry."""
    eo_rows = _campaign_sensor_eo_rows(angle_dir, campaign, sensor)
    if eo_rows.empty:
        return {}
    return _angle_features_from_eo_rows(eo_rows)


def load_transect_angle_features(angle_dir, campaign, sensor):
    """Per-transect eo viewing/illumination angles, keyed by transect_nr (int)."""
    eo_rows = _campaign_sensor_eo_rows(angle_dir, campaign, sensor)
    if eo_rows.empty:
        return {}
    out = {}
    for transect_nr, part in eo_rows.groupby("transect"):
        try:
            key = int(transect_nr)
        except (TypeError, ValueError):
            continue
        out[key] = _angle_features_from_eo_rows(part)
    return out


AEROSOL_FEATURE_DEFAULTS = {
    "merra2_totexttau_550": np.nan,
    "merra2_source_time_utc": "",
    "merra2_matchup_lon": np.nan,
    "merra2_matchup_lat": np.nan,
    "merra2_time_offset_min": np.nan,
    "merra2_source_url": "",
}


def load_scene_aerosol_features(path, campaign, sensor):
    """Load one cached MERRA-2 scene-level aerosol row for this campaign/sensor."""
    features = AEROSOL_FEATURE_DEFAULTS.copy()
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"MERRA-2 aerosol covariate file not found: {path}. "
            "Run stats/processor_comparison/AOT550/build_merra2_aerosol_scene_covariates.py "
            "before running performance metrics."
        )

    try:
        aerosol_df = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise RuntimeError(f"Could not read MERRA-2 aerosol covariate file: {path}") from exc

    required = {"campaign"}
    if not required.issubset(aerosol_df.columns):
        missing = ", ".join(sorted(required - set(aerosol_df.columns)))
        raise ValueError(f"MERRA-2 aerosol covariate file is missing required column(s): {missing}")

    subset = aerosol_df[aerosol_df["campaign"].astype(str).eq(str(campaign))].copy()
    if subset.empty:
        available = ", ".join(sorted(aerosol_df["campaign"].dropna().astype(str).unique()))
        raise ValueError(
            f"No MERRA-2 aerosol covariate row for campaign={campaign!r}. "
            f"Available campaigns: {available}"
        )

    if "sensor" in subset.columns:
        sensor_subset = subset[subset["sensor"].astype(str).str.upper().eq(str(sensor).upper())].copy()
        if not sensor_subset.empty:
            subset = sensor_subset
        else:
            available = ", ".join(sorted(subset["sensor"].dropna().astype(str).unique()))
            raise ValueError(
                f"No MERRA-2 aerosol covariate row for campaign={campaign!r}, sensor={sensor!r}. "
                f"Available sensors for campaign: {available}"
            )

    if "scene_datetime" in subset.columns:
        subset["_scene_datetime_sort"] = pd.to_datetime(subset["scene_datetime"], utc=True, errors="coerce")
        subset = subset.sort_values("_scene_datetime_sort", kind="mergesort")

    aerosol_row = subset.iloc[0]
    for key in features:
        if key in aerosol_row and pd.notna(aerosol_row[key]):
            features[key] = aerosol_row[key]

    for key in ["merra2_totexttau_550", "merra2_matchup_lon", "merra2_matchup_lat", "merra2_time_offset_min"]:
        features[key] = pd.to_numeric(pd.Series([features[key]]), errors="coerce").iloc[0]
    return features


def landwater_fraction_near_point(point_2056, radius_m, mask_src, to_mask_crs):
    if mask_src is None or not np.isfinite(radius_m) or radius_m <= 0:
        return {
            "land_fraction": np.nan,
            "water_fraction": np.nan,
            "valid_fraction": np.nan,
        }

    from rasterio.windows import Window, from_bounds

    x_mask, y_mask = to_mask_crs.transform(point_2056.x, point_2056.y)
    raw_window = from_bounds(
        x_mask - radius_m,
        y_mask - radius_m,
        x_mask + radius_m,
        y_mask + radius_m,
        mask_src.transform,
    ).round_offsets().round_lengths()

    col_off = max(0, int(raw_window.col_off))
    row_off = max(0, int(raw_window.row_off))
    col_max = min(mask_src.width, int(raw_window.col_off + raw_window.width))
    row_max = min(mask_src.height, int(raw_window.row_off + raw_window.height))
    if col_max <= col_off or row_max <= row_off:
        return {
            "land_fraction": np.nan,
            "water_fraction": np.nan,
            "valid_fraction": 0.0,
        }

    window = Window(col_off, row_off, col_max - col_off, row_max - row_off)
    # Read unmasked because these binary masks encode outside/land as 0 even
    # when the GeoTIFF nodata tag also says 0.
    data = mask_src.read(1, window=window, masked=False).astype(float, copy=False)
    transform = mask_src.window_transform(window)

    rows, cols = np.ogrid[0:data.shape[0], 0:data.shape[1]]
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    circle = ((xs - x_mask) ** 2 + (ys - y_mask) ** 2) <= radius_m ** 2
    values = data[circle]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "land_fraction": np.nan,
            "water_fraction": np.nan,
            "valid_fraction": 0.0,
        }

    water_fraction = float(np.mean(values >= 0.5))
    pixel_area = abs(float(mask_src.transform.a) * float(mask_src.transform.e))
    expected_pixels = np.pi * radius_m * radius_m / pixel_area if pixel_area > 0 else np.nan
    valid_fraction = float(min(values.size / expected_pixels, 1.0)) if np.isfinite(expected_pixels) and expected_pixels > 0 else np.nan
    return {
        "land_fraction": float(1.0 - water_fraction),
        "water_fraction": water_fraction,
        "valid_fraction": valid_fraction,
    }


def transect_order_and_distance_axis(group):
    sort_series = None
    for column in ["timestamp (UTC)", "datetime (UTC)", "insitu_datetime", "datetime (CET)"]:
        if column not in group.columns:
            continue
        if column == "timestamp (UTC)":
            parsed = pd.to_datetime(pd.to_numeric(group[column], errors="coerce"), unit="s", utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(group[column], utc=True, errors="coerce")
        if parsed.notna().any():
            sort_series = parsed
            break

    ordered = group.copy()
    if sort_series is not None:
        ordered["_transect_sort_time"] = sort_series
        ordered = ordered.sort_values(["_transect_sort_time", "insitu_row_id"], kind="mergesort")
        ordered = ordered.drop(columns=["_transect_sort_time"])
    else:
        ordered = ordered.sort_values("insitu_row_id", kind="mergesort")

    coords = np.array([(pt.x, pt.y) for pt in ordered.geometry], dtype=float)
    if len(coords) <= 1:
        return ordered, np.zeros(len(ordered), dtype=float)

    segment_lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    distance_axis = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if not np.isfinite(distance_axis).all() or float(np.nanmax(distance_axis)) <= 0:
        distance_axis = np.arange(len(ordered), dtype=float)
    return ordered, distance_axis


def interpolate_anchor_values(distance_axis, anchor_positions, anchor_values):
    values = np.asarray(anchor_values, dtype=float)
    anchor_axis = np.asarray(distance_axis, dtype=float)[np.asarray(anchor_positions, dtype=int)]
    valid = np.isfinite(anchor_axis) & np.isfinite(values)
    if not np.any(valid):
        return np.full(len(distance_axis), np.nan, dtype=float)
    if np.sum(valid) == 1:
        return np.full(len(distance_axis), float(values[valid][0]), dtype=float)

    anchor_frame = pd.DataFrame({"x": anchor_axis[valid], "value": values[valid]})
    anchor_frame = anchor_frame.groupby("x", as_index=False)["value"].mean().sort_values("x")
    if len(anchor_frame) == 1:
        return np.full(len(distance_axis), float(anchor_frame["value"].iloc[0]), dtype=float)

    return np.interp(
        np.asarray(distance_axis, dtype=float),
        anchor_frame["x"].to_numpy(dtype=float),
        anchor_frame["value"].to_numpy(dtype=float),
    )


def add_transect_landwater_fraction(gdf, mask_path):
    out = gdf.copy()
    if "transect_context_distance_m" not in out.columns:
        out["transect_context_distance_m"] = np.nan
    for radius_km in [1, 5, 10]:
        out[f"land_fraction_{radius_km}km"] = np.nan
        out[f"water_fraction_{radius_km}km"] = np.nan
        out[f"landwater_valid_fraction_{radius_km}km"] = np.nan

    if not mask_path or not os.path.exists(mask_path) or out.empty:
        return out

    with rasterio.open(mask_path) as mask_src:
        to_mask_crs = Transformer.from_crs(f"EPSG:{EPSG_CH}", mask_src.crs, always_xy=True)

        for _, group in out.groupby("transect_nr", dropna=False):
            ordered, distance_axis = transect_order_and_distance_axis(group)
            if ordered.empty:
                continue

            out.loc[ordered.index, "transect_context_distance_m"] = distance_axis
            context_by_radius = {radius_km: [] for radius_km in [1, 5, 10]}
            for point in ordered.geometry:
                for radius_km in [1, 5, 10]:
                    context_by_radius[radius_km].append(
                        landwater_fraction_near_point(point, radius_km * 1000.0, mask_src, to_mask_crs)
                    )

            for radius_km in [1, 5, 10]:
                for source_name, out_name in [
                    ("land_fraction", f"land_fraction_{radius_km}km"),
                    ("water_fraction", f"water_fraction_{radius_km}km"),
                    ("valid_fraction", f"landwater_valid_fraction_{radius_km}km"),
                ]:
                    out.loc[ordered.index, out_name] = [
                        context[source_name] for context in context_by_radius[radius_km]
                    ]

    return out


def dem_relief_p95_p5_near_point(point_2056, radius_m, dem_src, to_dem_crs):
    if dem_src is None or not np.isfinite(radius_m) or radius_m <= 0:
        return np.nan

    from rasterio.windows import Window, from_bounds

    x_dem, y_dem = to_dem_crs.transform(point_2056.x, point_2056.y)
    raw_window = from_bounds(
        x_dem - radius_m,
        y_dem - radius_m,
        x_dem + radius_m,
        y_dem + radius_m,
        dem_src.transform,
    ).round_offsets().round_lengths()

    col_off = max(0, int(raw_window.col_off))
    row_off = max(0, int(raw_window.row_off))
    col_max = min(dem_src.width, int(raw_window.col_off + raw_window.width))
    row_max = min(dem_src.height, int(raw_window.row_off + raw_window.height))
    if col_max <= col_off or row_max <= row_off:
        return np.nan

    window = Window(col_off, row_off, col_max - col_off, row_max - row_off)
    elev = dem_src.read(1, window=window, masked=True)
    transform = dem_src.window_transform(window)

    rows, cols = np.ogrid[0:elev.shape[0], 0:elev.shape[1]]
    xs, ys = transform * (cols, rows)
    circle = ((xs - x_dem) ** 2 + (ys - y_dem) ** 2) <= radius_m ** 2
    values = np.asarray(elev)[circle]
    if np.ma.isMaskedArray(elev):
        valid = circle & ~np.ma.getmaskarray(elev)
        values = np.asarray(elev.data)[valid]

    values = values.astype(float, copy=False)
    values = values[np.isfinite(values)]
    if dem_src.nodata is not None and np.isfinite(dem_src.nodata):
        values = values[values != float(dem_src.nodata)]
    if values.size < 2:
        return np.nan

    return float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))


def dem_relief_p95_p5_multi_radius_near_point(point_2056, radius_m_values, dem_src, to_dem_crs):
    radii = [float(radius_m) for radius_m in radius_m_values if np.isfinite(radius_m) and radius_m > 0]
    if dem_src is None or not radii:
        return {float(radius_m): np.nan for radius_m in radius_m_values}

    from rasterio.windows import Window, from_bounds

    max_radius = max(radii)
    x_dem, y_dem = to_dem_crs.transform(point_2056.x, point_2056.y)
    raw_window = from_bounds(
        x_dem - max_radius,
        y_dem - max_radius,
        x_dem + max_radius,
        y_dem + max_radius,
        dem_src.transform,
    ).round_offsets().round_lengths()

    col_off = max(0, int(raw_window.col_off))
    row_off = max(0, int(raw_window.row_off))
    col_max = min(dem_src.width, int(raw_window.col_off + raw_window.width))
    row_max = min(dem_src.height, int(raw_window.row_off + raw_window.height))
    if col_max <= col_off or row_max <= row_off:
        return {float(radius_m): np.nan for radius_m in radius_m_values}

    window = Window(col_off, row_off, col_max - col_off, row_max - row_off)
    elev = dem_src.read(1, window=window, masked=True)
    transform = dem_src.window_transform(window)
    rows, cols = np.ogrid[0:elev.shape[0], 0:elev.shape[1]]
    xs, ys = transform * (cols, rows)
    dist2 = (xs - x_dem) ** 2 + (ys - y_dem) ** 2
    mask = np.ma.getmaskarray(elev) if np.ma.isMaskedArray(elev) else np.zeros(elev.shape, dtype=bool)
    data = np.asarray(elev.data if np.ma.isMaskedArray(elev) else elev, dtype=float)

    results = {}
    for radius_m in radius_m_values:
        radius = float(radius_m)
        if not np.isfinite(radius) or radius <= 0:
            results[radius] = np.nan
            continue
        valid = (dist2 <= radius ** 2) & ~mask & np.isfinite(data)
        values = data[valid]
        if dem_src.nodata is not None and np.isfinite(dem_src.nodata):
            values = values[values != float(dem_src.nodata)]
        if values.size < 2:
            results[radius] = np.nan
        else:
            results[radius] = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))
    return results


def add_transect_topographic_relief(gdf, dem_path):
    out = gdf.copy()
    if "transect_context_distance_m" not in out.columns:
        out["transect_context_distance_m"] = np.nan
    for radius_km in [5]:
        out[f"topo_relief_p95_p5_m_{radius_km}km"] = np.nan

    if not dem_path or not os.path.exists(dem_path) or out.empty:
        return out

    with rasterio.open(dem_path) as dem_src:
        to_dem_crs = Transformer.from_crs(f"EPSG:{EPSG_CH}", dem_src.crs, always_xy=True)

        for _, group in out.groupby("transect_nr", dropna=False):
            ordered, distance_axis = transect_order_and_distance_axis(group)
            if ordered.empty:
                continue

            out.loc[ordered.index, "transect_context_distance_m"] = distance_axis
            relief_by_radius = {5: []}
            for point in ordered.geometry:
                relief = dem_relief_p95_p5_multi_radius_near_point(point, [5000.0], dem_src, to_dem_crs)
                relief_by_radius[5].append(relief.get(5000.0, np.nan))
            for radius_km in [5]:
                relief_values = np.asarray(relief_by_radius[radius_km], dtype=float)
                out.loc[ordered.index, f"topo_relief_p95_p5_m_{radius_km}km"] = relief_values

    return out


def round_output_metrics(frame):
    return mc.round_output_metrics(frame)


def first_present(row, names, default=np.nan):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


@dataclass(frozen=True)
class L2SRBandSource:
    wavelength: int
    path: Path
    band_index: int
    scale: float = 1.0
    offset: float = 0.0


def empty_landwater_contrast_values(sensor):
    values = {}
    for wl in L2SR_CONTRAST_BANDS[sensor]:
        values.update({
            f"insitu_water_median_100m_{wl}": np.nan,
            f"insitu_water_n_points_100m_{wl}": 0,
            f"insitu_water_used_fallback_100m_{wl}": np.nan,
        })
        for radius_km in CONTRAST_RADII_KM:
            values.update({
                f"l2sr_land_mean_{radius_km}km_{wl}": np.nan,
                f"land_water_ratio_{radius_km}km_{wl}": np.nan,
                f"land_water_log_ratio_{radius_km}km_{wl}": np.nan,
                f"land_water_adjacency_contrast_{radius_km}km_{wl}": np.nan,
            })
    return values


def find_landsat_l2sr_dir(campaign, sensor):
    directory = Path(INPUT_L2SR_ROOT) / campaign
    if not directory.exists() or sensor not in {"L8", "L9"}:
        return None
    prefixes = ["LC08"] if sensor == "L8" else ["LC09", "LC08"]
    for prefix in prefixes:
        if sorted(directory.glob(f"{prefix}_L2*_SR_B1.TIF")):
            return directory
    candidates = []
    for prefix in prefixes:
        candidates.extend(path for path in directory.glob(f"{prefix}_L2*") if path.is_dir())
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda path: (0 if sensor == "L9" and path.name.startswith("LC09") else 1, path.name),
    )[0]


def landsat_l2sr_band_sources(campaign, sensor):
    l2_dir = find_landsat_l2sr_dir(campaign, sensor)
    if l2_dir is None:
        raise FileNotFoundError(f"No Landsat L2 SR directory/files found for {campaign} {sensor}")
    sources = []
    for wavelength, band_name in L2SR_LANDSAT_BANDS[sensor].items():
        band_number = int(band_name[1:])
        matches = sorted(l2_dir.glob(f"*_SR_B{band_number}.TIF"))
        if not matches:
            raise FileNotFoundError(f"No SR_{band_name} TIFF found under {l2_dir}")
        sources.append(
            L2SRBandSource(
                wavelength=wavelength,
                path=matches[0],
                band_index=1,
                scale=LANDSAT_L2SR_SCALE,
                offset=LANDSAT_L2SR_ADD / LANDSAT_L2SR_SCALE,
            )
        )
    return sources


def find_l2a_safe(campaign, sensor):
    directory = Path(INPUT_L2SR_ROOT) / campaign
    if not directory.exists() or not sensor.startswith("S2"):
        return None
    candidates = sorted(path for path in directory.glob(f"{sensor}_MSIL2A*.SAFE") if path.is_dir())
    return candidates[0] if candidates else None


def parse_l2a_radiometry(safe_path):
    metadata = safe_path / "MTD_MSIL2A.xml"
    root = ET.parse(metadata).getroot()
    quantification = 10000.0
    offsets = {}
    for element in root.iter():
        tag = element.tag.split("}", 1)[-1]
        if tag == "BOA_QUANTIFICATION_VALUE" and element.text:
            quantification = float(element.text.strip())
        if tag == "BOA_ADD_OFFSET":
            band_id = element.attrib.get("band_id")
            if band_id is not None and element.text:
                for band, idx in S2_BAND_ID.items():
                    if str(idx) == str(band_id):
                        offsets[band] = float(element.text.strip())
    return quantification, offsets


def l2a_band_sources(campaign, sensor):
    safe_path = find_l2a_safe(campaign, sensor)
    if safe_path is None:
        raise FileNotFoundError(f"No Sentinel-2 L2A SAFE found for {campaign} {sensor}")
    quantification, offsets = parse_l2a_radiometry(safe_path)
    granule_dirs = sorted((safe_path / "GRANULE").glob("*"))
    if not granule_dirs:
        raise FileNotFoundError(f"No GRANULE directory under {safe_path}")
    img_dir = granule_dirs[0] / "IMG_DATA"
    sources = []
    for wavelength, band_name in L2SR_SAFE_BANDS[sensor].items():
        matches = []
        for resolution in ("R10m", "R20m", "R60m"):
            matches = sorted((img_dir / resolution).glob(f"*_{band_name}_*.jp2"))
            if matches:
                break
        if not matches:
            raise FileNotFoundError(f"No {band_name} L2A JP2 found under {img_dir}")
        sources.append(
            L2SRBandSource(
                wavelength=wavelength,
                path=matches[0],
                band_index=1,
                scale=1.0 / quantification,
                offset=offsets.get(band_name, 0.0),
            )
        )
    return sources


def discover_l2sr_contrast_sources(campaign, sensor):
    if sensor in {"L8", "L9"}:
        return landsat_l2sr_band_sources(campaign, sensor)
    if sensor.startswith("S2"):
        return l2a_band_sources(campaign, sensor)
    raise ValueError(f"Unsupported L2 SR contrast sensor: {sensor}")


def l2sr_bounded_window(src, x, y, radius_m):
    from rasterio.windows import Window, from_bounds

    raw = from_bounds(x - radius_m, y - radius_m, x + radius_m, y + radius_m, src.transform)
    raw = raw.round_offsets().round_lengths()
    col_off = max(0, int(raw.col_off))
    row_off = max(0, int(raw.row_off))
    col_max = min(src.width, int(raw.col_off + raw.width))
    row_max = min(src.height, int(raw.row_off + raw.height))
    if col_max <= col_off or row_max <= row_off:
        return None
    return Window(col_off, row_off, col_max - col_off, row_max - row_off)


def l2sr_circle_mask(src, window, x, y, radius_m):
    transform = src.window_transform(window)
    rows, cols = np.ogrid[0:int(window.height), 0:int(window.width)]
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    return ((xs - x) ** 2 + (ys - y) ** 2) <= radius_m ** 2


def read_l2sr_reflectance(src, source, window):
    values = src.read(source.band_index, window=window, masked=False).astype(float, copy=False)
    values = (values + source.offset) * source.scale
    nodata = src.nodata
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    return values


def summarize_l2sr_land_mean_band(source, mask_path, points, radius_km):
    radius_m = radius_km * 1000.0
    rows = []
    with rasterio.open(source.path) as src, rasterio.open(mask_path) as mask_src:
        to_src = Transformer.from_crs(f"EPSG:{EPSG_CH}", src.crs, always_xy=True)
        with rasterio.vrt.WarpedVRT(
            mask_src,
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            resampling=rasterio.enums.Resampling.nearest,
            src_nodata=None,
            nodata=None,
        ) as mask_vrt:
            for point in points.itertuples(index=False):
                wl = source.wavelength
                values = {
                    "insitu_row_id": point.insitu_row_id,
                    f"l2sr_land_mean_{radius_km}km_{wl}": np.nan,
                }
                x_src, y_src = to_src.transform(point.insitu_x_2056, point.insitu_y_2056)
                window = l2sr_bounded_window(src, x_src, y_src, radius_m)
                if window is None:
                    rows.append(values)
                    continue
                circle = l2sr_circle_mask(src, window, x_src, y_src, radius_m)
                mask = mask_vrt.read(1, window=window, out_shape=circle.shape, masked=False).astype(float, copy=False)
                reflectance = read_l2sr_reflectance(src, source, window)
                valid = (
                    circle
                    & np.isfinite(mask)
                    & np.isfinite(reflectance)
                    & (reflectance > L2SR_REFLECTANCE_MIN)
                    & (reflectance < L2SR_REFLECTANCE_MAX)
                )
                land_values = reflectance[valid & (mask < 0.5)]
                values[f"l2sr_land_mean_{radius_km}km_{wl}"] = (
                    float(np.mean(land_values)) if land_values.size else np.nan
                )
                rows.append(values)
    return pd.DataFrame(rows)


def nearby_insitu_water_reflectance(points_df, wavelength_col, radius_m=WATER_LOCAL_RADIUS_M):
    """Local in-situ water reflectance per point: median Rrs*pi of same-campaign
    points within radius_m (self included). Falls back to the nearest point
    with a finite value if none are found within the radius."""
    ids = points_df["insitu_row_id"].to_numpy()
    xs = points_df["insitu_x_2056"].to_numpy(dtype=float)
    ys = points_df["insitu_y_2056"].to_numpy(dtype=float)
    rrs = pd.to_numeric(points_df[wavelength_col], errors="coerce").to_numpy(dtype=float)

    n = len(points_df)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.hypot(dx, dy)

    finite = np.isfinite(rrs)
    rows = []
    for i in range(n):
        within = (dist[i] <= radius_m) & finite
        used_fallback = False
        if not np.any(within):
            if np.any(finite):
                nearest_idx = np.argmin(np.where(finite, dist[i], np.inf))
                within = np.zeros(n, dtype=bool)
                within[nearest_idx] = True
                used_fallback = True
            else:
                rows.append({
                    "insitu_row_id": ids[i],
                    "water_median": np.nan,
                    "n_points": 0,
                    "used_fallback": np.nan,
                })
                continue
        neighbor_values = rrs[within]
        rows.append({
            "insitu_row_id": ids[i],
            "water_median": float(np.median(neighbor_values)) * math.pi,
            "n_points": int(within.sum()),
            "used_fallback": used_fallback,
        })
    return pd.DataFrame(rows)


def build_landwater_contrast_by_point(campaign, sensor, points, mask_path):
    rrs_column_map = insitu_rrs_column_map(sensor)
    rrs_columns = sorted(set(rrs_column_map.values()))
    base = points[[
        "insitu_row_id", "insitu_x_2056", "insitu_y_2056",
        *[f"land_fraction_{radius_km}km" for radius_km in CONTRAST_RADII_KM],
        *rrs_columns,
    ]].drop_duplicates("insitu_row_id").copy()
    for radius_km in CONTRAST_RADII_KM:
        base[f"land_fraction_{radius_km}km"] = pd.to_numeric(base[f"land_fraction_{radius_km}km"], errors="coerce")

    covariates = base[["insitu_row_id"]].copy()

    land_sources = discover_l2sr_contrast_sources(campaign, sensor)
    for source in land_sources:
        for radius_km in CONTRAST_RADII_KM:
            covariates = covariates.merge(
                summarize_l2sr_land_mean_band(source, mask_path, base, radius_km),
                on="insitu_row_id",
                how="left",
                validate="one_to_one",
            )

    for wl in L2SR_CONTRAST_BANDS[sensor]:
        wavelength_col = rrs_column_map[wl]
        water = nearby_insitu_water_reflectance(base, wavelength_col)
        covariates = covariates.merge(
            water.rename(columns={
                "water_median": f"insitu_water_median_100m_{wl}",
                "n_points": f"insitu_water_n_points_100m_{wl}",
                "used_fallback": f"insitu_water_used_fallback_100m_{wl}",
            }),
            on="insitu_row_id",
            how="left",
            validate="one_to_one",
        )

    covariates = covariates.merge(
        base[["insitu_row_id", *[f"land_fraction_{radius_km}km" for radius_km in CONTRAST_RADII_KM]]],
        on="insitu_row_id",
        how="left",
        validate="one_to_one",
    )

    for wl in L2SR_CONTRAST_BANDS[sensor]:
        water_mean = covariates[f"insitu_water_median_100m_{wl}"]
        for radius_km in CONTRAST_RADII_KM:
            land_mean = covariates[f"l2sr_land_mean_{radius_km}km_{wl}"]
            land_fraction = covariates[f"land_fraction_{radius_km}km"]
            ratio = np.where(
                np.isfinite(land_mean) & np.isfinite(water_mean) & (water_mean > 0),
                land_mean / water_mean,
                np.nan,
            )
            positive_ratio = np.isfinite(ratio) & (ratio > 0)
            log_ratio = np.where(positive_ratio, np.log10(np.where(positive_ratio, ratio, 1.0)), np.nan)
            # land_fraction == 0 means no land pixels fell within the radius, so
            # log_ratio is undefined (no land reflectance to sample) even though
            # the contrast itself is well-defined: no land present -> no possible
            # land-driven contrast contribution, i.e. 0, not missing.
            no_land = np.isfinite(land_fraction) & (land_fraction == 0)
            has_signal = np.isfinite(land_fraction) & (land_fraction > 0) & np.isfinite(log_ratio)
            contrast = np.select([no_land, has_signal], [0.0, land_fraction * log_ratio], default=np.nan)
            covariates[f"land_water_ratio_{radius_km}km_{wl}"] = ratio
            covariates[f"land_water_log_ratio_{radius_km}km_{wl}"] = log_ratio
            covariates[f"land_water_adjacency_contrast_{radius_km}km_{wl}"] = contrast

    covariates = covariates.drop(columns=[f"land_fraction_{radius_km}km" for radius_km in CONTRAST_RADII_KM])
    return covariates.set_index("insitu_row_id").to_dict(orient="index")


def green_peak_rrs_columns(columns):
    green_columns = []
    wavelengths_by_column = {}
    for column in columns:
        text = str(column)
        if not text.startswith("Rrs_"):
            continue
        try:
            wavelength = float(text.split("_", 1)[1])
        except ValueError:
            continue
        if GREEN_PEAK_MIN_NM <= wavelength <= GREEN_PEAK_MAX_NM:
            green_columns.append(column)
            wavelengths_by_column[column] = wavelength
    return green_columns, wavelengths_by_column


def add_hyperspectral_green_peak_metrics(frame, hyper_file):
    frame = frame.copy()
    frame["insitu_green_peak_rrs"] = np.nan
    frame["insitu_green_peak_wavelength_nm"] = np.nan

    if not hyper_file or not os.path.exists(hyper_file):
        print(f"[warning] hyperspectral in-situ QC file not found for green peak metrics: {hyper_file}")
        return frame

    hyper_header = pd.read_csv(hyper_file, sep=None, engine="python", nrows=0).columns
    green_columns, wavelengths_by_column = green_peak_rrs_columns(hyper_header)
    if not green_columns:
        print(f"[warning] no hyperspectral Rrs columns in {GREEN_PEAK_MIN_NM}-{GREEN_PEAK_MAX_NM} nm: {hyper_file}")
        return frame

    key_candidates = ["fid_ramses", "timestamp (UTC)", "datetime (UTC)", "datetime (CET)"]
    key_columns = [column for column in key_candidates if column in frame.columns and column in hyper_header]
    if not key_columns:
        print(f"[warning] no common key columns for green peak lookup: {hyper_file}")
        return frame

    usecols = list(dict.fromkeys([*key_columns, *green_columns]))
    hyper = pd.read_csv(hyper_file, sep=None, engine="python", usecols=usecols)
    green = hyper[green_columns].apply(pd.to_numeric, errors="coerce")
    valid_rows = green.notna().any(axis=1)
    peak_column = pd.Series(index=green.index, dtype=object)
    peak_column.loc[valid_rows] = green.loc[valid_rows].idxmax(axis=1, skipna=True)

    lookup = hyper[key_columns].copy()
    lookup["insitu_green_peak_rrs_from_hyperspectral"] = green.max(axis=1, skipna=True)
    lookup["insitu_green_peak_wavelength_nm_from_hyperspectral"] = peak_column.map(wavelengths_by_column)
    lookup = lookup.dropna(subset=["insitu_green_peak_rrs_from_hyperspectral"], how="all")
    lookup = lookup.drop_duplicates(key_columns, keep="first")

    merged = frame.merge(lookup, on=key_columns, how="left")
    merged["insitu_green_peak_rrs"] = pd.to_numeric(
        merged["insitu_green_peak_rrs_from_hyperspectral"], errors="coerce"
    )
    merged["insitu_green_peak_wavelength_nm"] = pd.to_numeric(
        merged["insitu_green_peak_wavelength_nm_from_hyperspectral"], errors="coerce"
    )
    return merged.drop(
        columns=[
            "insitu_green_peak_rrs_from_hyperspectral",
            "insitu_green_peak_wavelength_nm_from_hyperspectral",
        ],
        errors="ignore",
    )


def parse_input_scene_datetime(row):
    value = first_present(row, ["satellite_overpass_utc", "satellite_overpass_datetime_utc", "scene_datetime"], pd.NaT)
    if pd.isna(value):
        return pd.NaT
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return parsed.tz_convert(None)


def safe_output_name(column):
    return str(column).strip().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")


def row_metadata(row):
    metadata = {}
    direct_columns = {
        "source_filename": "source_filename",
        "filename_ramses": "filename_ramses",
        "fid_ramses": "fid_ramses",
        "timestamp (UTC)": "timestamp_utc",
        "datetime (UTC)": "datetime_utc",
        "datetime (CET)": "datetime_cet",
        "heading": "insitu_heading_deg",
        "instantaneous solar azimuth angle": "insitu_sun_azimuth_deg",
        "instantaneous solar zenith angle": "insitu_sun_zenith_deg",
        "relative azimuth angle": "insitu_relative_azimuth_deg",
        "raw_relative_azimuth_angle": "insitu_raw_relative_azimuth_deg",
        "speed": "speed",
        "Incl_V": "insitu_inclination_v_deg",
        "Incl_X": "insitu_inclination_x_deg",
        "Incl_Y": "insitu_inclination_y_deg",
        "depth_m": "depth_m",
        "satellite_overpass_utc": "satellite_overpass_utc",
        "eo_product": "input_eo_product",
        "temporal_offset_min": "input_temporal_offset_min",
    }
    for source, target in direct_columns.items():
        if source in row:
            metadata[target] = row[source]

    bottom_reflectance_keep_columns = {
        "bottom_tau",
        "bottom_class",
        "bottom_spectral_z",
        "bottom_exclude",
    }
    tokens = (
        "eligibility",
        "hard_filter",
        "rrs_",
        "metric_",
        "flag_",
        "flags_",
        "qc_",
        "propagated_",
    )
    for source in row.index:
        lower = str(source).lower()
        if source in direct_columns or lower.startswith("rrs_"):
            continue
        if lower in bottom_reflectance_keep_columns:
            metadata[safe_output_name(source)] = row[source]
            continue
        if any(lower.startswith(token) for token in tokens):
            metadata[safe_output_name(source)] = row[source]
    return metadata


# =============================================================================
# LOAD IN-SITU + SHORELINE DISTANCE
# =============================================================================
campaign = extract_campaign(CSV_CONV_FILE)
sensor   = extract_sensor_from_csv(CSV_CONV_FILE)
location = extract_location(CSV_CONV_FILE)
wavelengths = VISIBLE_SENSOR_BANDS[sensor]
analysis_transects = load_campaign_transects(campaign)
angle_features = load_viewing_angle_features(VIEWING_ANGLES_DIR, campaign, sensor)
transect_angle_features = load_transect_angle_features(VIEWING_ANGLES_DIR, campaign, sensor)
aerosol_features = load_scene_aerosol_features(AEROSOL_COVARIATES_FILE, campaign, sensor)

insitu_conv_df = pd.read_csv(CSV_CONV_FILE, sep=None, engine="python")
insitu_conv_df = insitu_conv_df.rename(columns=lambda c: c.replace(".0", "") if isinstance(c, str) else c)
insitu_conv_df = add_hyperspectral_green_peak_metrics(insitu_conv_df, CSV_HYPER_FILE)
if "datetime (CET)" in insitu_conv_df.columns:
    insitu_conv_df["insitu_datetime"] = pd.to_datetime(insitu_conv_df["datetime (CET)"], errors="coerce")
else:
    insitu_conv_df["insitu_datetime"] = pd.NaT
insitu_filter_result = mc.filter_insitu_rows(
    insitu_conv_df,
    transects=analysis_transects,
    metric_waves=wavelengths,
    x_col=x_col,
    y_col=y_col,
    hard_filter_column=INSITU_HARD_FILTER_COLUMN,
    require_hard_filter_column=False,
)
insitu_conv_df = insitu_filter_result["frame"]
if "timestamp (UTC)" in insitu_conv_df.columns:
    sort_time = pd.to_datetime(pd.to_numeric(insitu_conv_df["timestamp (UTC)"], errors="coerce"), unit="s", utc=True, errors="coerce")
elif "datetime (UTC)" in insitu_conv_df.columns:
    sort_time = pd.to_datetime(insitu_conv_df["datetime (UTC)"], utc=True, errors="coerce")
else:
    sort_time = pd.to_datetime(insitu_conv_df["insitu_datetime"], utc=True, errors="coerce")
insitu_conv_df["_insitu_sort_time"] = sort_time
insitu_conv_df = insitu_conv_df.sort_values(["_insitu_sort_time", "transect_nr"], kind="mergesort").drop(columns=["_insitu_sort_time"])

insitu_conv_df = insitu_conv_df.reset_index(drop=True)
insitu_conv_df["insitu_row_id"] = insitu_conv_df.index

gdf_conv = gpd.GeoDataFrame(
    insitu_conv_df,
    geometry=gpd.points_from_xy(insitu_conv_df[x_col], insitu_conv_df[y_col]),
    crs="EPSG:4326"
).to_crs(epsg=EPSG_CH)
gdf_conv["insitu_x_2056"] = gdf_conv.geometry.x
gdf_conv["insitu_y_2056"] = gdf_conv.geometry.y

if DEPTH_RASTER_FILE:
    with rasterio.open(DEPTH_RASTER_FILE) as src:
        depth = src.read(1)
        transform = src.transform
        raster_crs = src.crs
        valid_mask = np.isfinite(depth)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid_mask &= depth != src.nodata
        water_mask = valid_mask & (depth > 0)
        if not np.any(water_mask) and np.any(valid_mask):
            water_mask = valid_mask
        px_m, py_m = raster_pixel_sizes_m(transform, raster_crs, water_mask.shape)
        water_mask = fill_small_internal_mask_holes(water_mask, abs(px_m * py_m), max_hole_area_m2=4.0)
        shapes_gen = rasterio.features.shapes(water_mask.astype("uint8"), mask=water_mask, transform=transform)
        water_polys = [shape(geom) for geom, val in shapes_gen if val == 1]

    water_union = unary_union(water_polys)
    shoreline = water_union.boundary
    shoreline_gdf = gpd.GeoDataFrame(geometry=[shoreline], crs=raster_crs).to_crs(epsg=EPSG_CH)
    shoreline_geom = shoreline_gdf.geometry.unary_union
    gdf_conv["dist_to_shore_km"] = gdf_conv.geometry.apply(lambda p: nearest_distance_km(p, shoreline_geom))
else:
    gdf_conv = gdf_conv.sort_values(["transect_nr", "datetime (CET)"] if "datetime (CET)" in gdf_conv.columns else ["transect_nr"])
    gdf_conv["transect_distance_m"] = 0.0
    for t_id, group in gdf_conv.groupby("transect_nr"):
        coords = np.array([(pt.x, pt.y) for pt in group.geometry])
        if len(coords) <= 1:
            gdf_conv.loc[group.index, "transect_distance_m"] = 0.0
            continue
        dists = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        gdf_conv.loc[group.index, "transect_distance_m"] = np.concatenate(([0.0], np.cumsum(dists)))
    gdf_conv["dist_to_shore_km"] = gdf_conv["transect_distance_m"] / 1000.0

gdf_conv = add_transect_topographic_relief(gdf_conv, DEM_RASTER_FILE)
gdf_conv = add_transect_landwater_fraction(gdf_conv, LANDWATER_RASTER_FILE)
_landwater_contrast_columns = [
    "insitu_row_id", "insitu_x_2056", "insitu_y_2056",
    *[f"land_fraction_{radius_km}km" for radius_km in CONTRAST_RADII_KM],
    *sorted(set(insitu_rrs_column_map(sensor).values())),
]
landwater_contrast_by_point = build_landwater_contrast_by_point(
    campaign,
    sensor,
    gdf_conv[_landwater_contrast_columns].drop_duplicates("insitu_row_id"),
    LANDWATER_RASTER_FILE,
)


# =============================================================================
# MATCH-UP HELPERS
# =============================================================================
transformer_lonlat_to_2056 = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG_CH}", always_xy=True)
transformer_2056_to_lonlat = Transformer.from_crs(f"EPSG:{EPSG_CH}", "EPSG:4326", always_xy=True)

def safe_transform(ds):
    lon_var = "lon" if "lon" in ds else "longitude"
    lat_var = "lat" if "lat" in ds else "latitude"
    lon, lat = ds[lon_var].values, ds[lat_var].values
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    x, y = transformer_lonlat_to_2056.transform(lon.ravel(), lat.ravel())
    mask = np.isfinite(x) & np.isfinite(y)
    coords = np.column_stack((x[mask], y[mask]))
    valid_idx = np.argwhere(mask).ravel()
    return coords, valid_idx, lon.shape


def footprint_3x3(flat_index, grid_shape):
    return mc.footprint_3x3(flat_index, grid_shape)

def estimate_pixel_sizes_m(ds, transformer):
    lon_var = "lon" if "lon" in ds else "longitude"
    lat_var = "lat" if "lat" in ds else "latitude"
    lon = ds[lon_var].values
    lat = ds[lat_var].values
    if lon.ndim == 1 and lat.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
    else:
        lon2, lat2 = lon, lat
    i = lon2.shape[0] // 2
    j = lon2.shape[1] // 2
    x0, y0 = transformer.transform(lon2[i, j], lat2[i, j])
    x1, y1 = transformer.transform(lon2[i, j+1], lat2[i, j+1])
    x2, y2 = transformer.transform(lon2[i+1, j], lat2[i+1, j])
    dx = np.hypot(x1 - x0, y1 - y0)
    dy = np.hypot(x2 - x0, y2 - y0)
    return dx, dy

def build_band_map(proc, obj, sensor, wavelengths):
    if is_raster_stack(obj):
        if proc == "GAAC":
            return raster_stack_band_map(obj, wavelengths, preferred_prefixes=("rhow", "Rrs", "Rw", "rhos", "band_"))
        return raster_stack_band_map(obj, wavelengths)

    if proc == "GAAC":
        idx_map = GAAC_BAND_INDEX.get(sensor)
        if idx_map is None:
            raise ValueError(f"No GAAC_BAND_INDEX mapping for sensor={sensor}")
        if "band" not in obj.dims:
            raise ValueError("GAAC DataArray missing 'band' dimension.")
        return {float(wl): idx_map.get(int(wl)) for wl in wavelengths}

    ds = obj
    prefixes = enabled_procs[proc]["band_prefixes"]
    band_candidates = get_band_candidates(ds, prefixes)
    if not band_candidates:
        raise ValueError(f"{proc}: No band candidates found for prefixes={prefixes}")

    if proc.startswith("C2RCC"):
        proc_map = {}
        rhow_vars_low = {str(v).lower(): v for v in ds.data_vars if str(v).lower().startswith("rhow")}
        has_rhow_b = any(k.startswith("rhow_b") for k in rhow_vars_low.keys())

        if has_rhow_b:
            if sensor not in ("S2A", "S2B"):
                raise ValueError(f"{proc}: file uses rhow_B* variables but sensor={sensor} is not S2A/S2B.")
            wl_to_band = C2RCC_S2_WL_TO_BAND.get(sensor, {})
            for wl in wavelengths:
                wl_int = int(wl)
                band_id = wl_to_band.get(wl_int)
                if band_id is None:
                    proc_map[wl] = None
                    continue

                cand1 = f"rhow_{band_id}".lower()
                cand2 = f"rhown_{band_id}".lower()

                if cand1 in rhow_vars_low:
                    proc_map[wl] = rhow_vars_low[cand1]
                elif cand2 in rhow_vars_low:
                    proc_map[wl] = rhow_vars_low[cand2]
                else:
                    proc_map[wl] = None
            return proc_map

        idx_map = C2RCC_BAND_INDEX.get(sensor)
        if idx_map is None:
            raise ValueError(f"{proc}: No C2RCC_BAND_INDEX defined for sensor={sensor}")

        for wl in wavelengths:
            wl_int = int(wl)
            band_idx = idx_map.get(wl_int)
            if band_idx is None:
                proc_map[wl] = None
                continue

            cand1 = f"rhow_{band_idx}"
            cand2 = f"rhown_{band_idx}"

            if cand1 in ds.data_vars:
                proc_map[wl] = cand1
            elif cand2 in ds.data_vars:
                proc_map[wl] = cand2
            else:
                proc_map[wl] = None
        return proc_map

    return {float(wl): find_closest_bandname(wl, band_candidates) for wl in wavelengths}

def get_sat_value(proc, obj, flat_idx, band_ref):
    if band_ref is None:
        return np.nan

    if is_raster_stack(obj):
        band_arr = obj.sel(band=int(band_ref)).values
        val = float(band_arr.ravel()[flat_idx])
    else:
        val = float(obj[band_ref].values.ravel()[flat_idx])

    if enabled_procs[proc]["divide_by_pi"]:
        val = val / np.pi
    return val


def get_sat_values(proc, obj, flat_indices, band_ref):
    if band_ref is None or len(flat_indices) == 0:
        return np.array([], dtype=float)

    if is_raster_stack(obj):
        arr = obj.sel(band=int(band_ref)).values.ravel()
    else:
        arr = obj[band_ref].values.ravel()

    vals = arr[np.asarray(flat_indices, dtype=int)].astype(float)
    if enabled_procs[proc]["divide_by_pi"]:
        vals = vals / np.pi
    return vals


def summarize_context_values(proc, obj, band_refs, flat_indices):
    """Summarize invalid/negative fractions and local Rrs variability around a matchup.

    Only the invalid fraction is tracked (not its complement, valid fraction):
    valid_fraction == 1 - invalid_fraction by construction, so keeping both
    would just duplicate the same information."""
    flat_indices = np.asarray(flat_indices, dtype=int)
    if flat_indices.size == 0:
        return {
            "eo_context_n_pixels": 0,
            "eo_context_invalid_fraction": np.nan,
            "eo_context_negative_fraction": np.nan,
            "eo_context_rrs_std_mean": np.nan,
            "eo_context_rrs_cv_mean": np.nan,
        }, {}

    invalid_counts = []
    negative_counts = []
    stds = []
    cvs = []
    per_wl = {}

    for wl, band_ref in band_refs.items():
        vals = get_sat_values(proc, obj, flat_indices, band_ref)
        n = vals.size
        finite = np.isfinite(vals)
        negative = finite & (vals < 0)
        invalid_fraction = float(np.mean(~finite)) if n else np.nan
        negative_fraction = float(np.mean(negative)) if n else np.nan

        valid_vals = vals[finite]
        local_std = float(np.nanstd(valid_vals, ddof=1)) if valid_vals.size > 1 else 0.0 if valid_vals.size == 1 else np.nan
        local_mean = float(np.nanmean(valid_vals)) if valid_vals.size else np.nan
        local_cv = float(local_std / abs(local_mean)) if np.isfinite(local_std) and np.isfinite(local_mean) and local_mean != 0 else np.nan

        wl_int = int(float(wl))
        per_wl[wl_int] = {
            "invalid_fraction": invalid_fraction,
            "negative_fraction": negative_fraction,
            "std": local_std,
            "cv": local_cv,
        }

        if n:
            invalid_counts.append(np.sum(~finite))
            negative_counts.append(np.sum(negative))
        if np.isfinite(local_std):
            stds.append(local_std)
        if np.isfinite(local_cv):
            cvs.append(local_cv)

    total = flat_indices.size * max(len(band_refs), 1)
    summary = {
        "eo_context_n_pixels": int(flat_indices.size),
        "eo_context_invalid_fraction": float(np.sum(invalid_counts) / total) if total else np.nan,
        "eo_context_negative_fraction": float(np.sum(negative_counts) / total) if total else np.nan,
        "eo_context_rrs_std_mean": float(np.mean(stds)) if stds else np.nan,
        "eo_context_rrs_cv_mean": float(np.mean(cvs)) if cvs else np.nan,
    }
    return summary, per_wl


# =============================================================================
# DISCOVER EO FILES
# =============================================================================
SPATIAL_SUBSETS = EXTENTS if USE_MANUAL_EXTENTS else EXTENTS
enabled_procs = {p: cfg for p, cfg in PROCESSORS.items() if cfg.get("enabled", False)}
processor_files = {}
if MATCHUP_VALUE_METHOD not in {"nearest", "mean", "median"}:
    raise ValueError("MATCHUP_VALUE_METHOD must be one of: 'nearest', 'mean', 'median'")

for proc, cfg in enabled_procs.items():
    processor_files[proc] = discover_files_for_processor(
        PRODUCT_ROOT, proc, cfg, SPATIAL_SUBSETS, sensor, PRODUCT_KEYWORDS
    )
    print(f"{proc}: found {len(processor_files[proc])} files")


# =============================================================================
# MAIN
# =============================================================================
rows_out = []

for subset in SPATIAL_SUBSETS:
    print(f"\n=== EXTENT {subset} ===")

    datasets = {}
    scene_datetime_map = {}
    for proc, cfg in enabled_procs.items():
        by_transect_files = processor_files[proc].get("__by_transect__")
        if by_transect_files:
            datasets[proc] = {"__by_transect__": {}}
            scene_datetime_map[(proc, "__default__")] = pd.NaT
            for transect_id, fp in sorted(by_transect_files.items()):
                if fp.lower().endswith((".tif", ".tiff")):
                    datasets[proc]["__by_transect__"][int(transect_id)] = rioxarray.open_rasterio(fp)
                else:
                    datasets[proc]["__by_transect__"][int(transect_id)] = xr.open_dataset(fp)
                scene_datetime_map[(proc, int(transect_id))] = parse_scene_datetime_from_path(fp)
            continue

        fp = processor_files[proc].get("__single__") if cfg.get("extent_mode") == "single" else processor_files[proc].get(subset)
        if fp is None:
            datasets[proc] = None
            scene_datetime_map[(proc, "__default__")] = pd.NaT
            continue
        if fp.lower().endswith((".tif", ".tiff")):
            datasets[proc] = rioxarray.open_rasterio(fp)
        else:
            datasets[proc] = xr.open_dataset(fp)
        scene_datetime_map[(proc, "__default__")] = parse_scene_datetime_from_path(fp)

    if all(v is None for v in datasets.values()):
        print(f"  -> Skipping {subset}: no EO products found.")
        continue

    obj_map = {}
    tree_map = {}
    valid_idx_map = {}
    context_radius_map = {}
    coords_map = {}
    grid_shape_map = {}
    band_map = {}

    for proc, proc_datasets in datasets.items():
        if proc_datasets is None:
            continue

        if isinstance(proc_datasets, dict) and "__by_transect__" in proc_datasets:
            proc_items = [((proc, int(transect_id)), obj) for transect_id, obj in proc_datasets["__by_transect__"].items()]
        else:
            proc_items = [((proc, "__default__"), proc_datasets)]

        for proc_key, obj in proc_items:
            obj_map[proc_key] = obj

            if is_raster_stack(obj):
                da = obj
                xs = da["x"].values
                ys = da["y"].values
                X, Y = np.meshgrid(xs, ys)
                coords = np.column_stack([X.ravel(), Y.ravel()])

                raster_crs = da.rio.crs
                if raster_crs is None:
                    raise ValueError(f"{proc} GeoTIFF has no CRS.")
                if str(raster_crs.to_epsg()) != str(EPSG_CH):
                    to_ch = Transformer.from_crs(raster_crs, f"EPSG:{EPSG_CH}", always_xy=True)
                    x_ch, y_ch = to_ch.transform(coords[:, 0], coords[:, 1])
                    coords_ch = np.column_stack([x_ch, y_ch])
                else:
                    coords_ch = coords

                coords_map[proc_key] = coords_ch
                valid_idx_map[proc_key] = np.arange(coords_ch.shape[0])
                grid_shape_map[proc_key] = (len(ys), len(xs))
                tree_map[proc_key] = cKDTree(coords_ch)

                dx = float(np.median(np.diff(xs)))
                dy = float(np.median(np.diff(ys)))
                context_radius_map[proc_key] = CONTEXT_RADIUS_PIXELS * max(abs(dx), abs(dy))
            else:
                ds = obj
                proc_coords, proc_valid_idx, proc_grid_shape = safe_transform(ds)
                coords_map[proc_key] = proc_coords
                valid_idx_map[proc_key] = proc_valid_idx
                grid_shape_map[proc_key] = proc_grid_shape
                tree_map[proc_key] = cKDTree(proc_coords)

                dx, dy = estimate_pixel_sizes_m(ds, transformer_lonlat_to_2056)
                context_radius_map[proc_key] = CONTEXT_RADIUS_PIXELS * max(abs(dx), abs(dy))

            band_map[proc_key] = build_band_map(proc, obj, sensor, wavelengths)

    for _, row in gdf_conv.iterrows():
        insitu_x_2056 = float(row.geometry.x)
        insitu_y_2056 = float(row.geometry.y)
        insitu_lon, insitu_lat = transformer_2056_to_lonlat.transform(insitu_x_2056, insitu_y_2056)

        insitu_spec_dict = {}
        for wl in wavelengths:
            col = f"Rrs_{int(wl)}"
            insitu_spec_dict[float(wl)] = float(row[col]) if col in row and np.isfinite(row[col]) else np.nan

        for proc in enabled_procs.keys():
            transect_id = int(row["transect_nr"])
            proc_key = (proc, transect_id) if (proc, transect_id) in obj_map else (proc, "__default__")
            input_scene_dt = parse_input_scene_datetime(row)
            scene_dt = input_scene_dt if pd.notna(input_scene_dt) else scene_datetime_map.get(proc_key, scene_datetime_map.get((proc, "__default__"), pd.NaT))
            insitu_dt = row.get("insitu_datetime", pd.NaT)
            input_temporal_offset = first_present(row, ["temporal_offset_min", "metric_temporal_offset_min"], np.nan)
            if pd.notna(input_temporal_offset):
                temporal_offset_min = float(input_temporal_offset)
            elif pd.notna(scene_dt) and pd.notna(insitu_dt):
                temporal_offset_min = abs((pd.Timestamp(insitu_dt) - pd.Timestamp(scene_dt)).total_seconds()) / 60.0
            else:
                temporal_offset_min = np.nan

            base_row = {
                "campaign": campaign,
                "location": location,
                "sensor": sensor,
                "insitu_row_id": int(row["insitu_row_id"]),
                "transect_nr": transect_id,
                "insitu_x_2056": insitu_x_2056,
                "insitu_y_2056": insitu_y_2056,
                "insitu_lon": insitu_lon,
                "insitu_lat": insitu_lat,
                "insitu_datetime": insitu_dt,
                "insitu_green_peak_rrs": first_present(row, ["insitu_green_peak_rrs"], np.nan),
                "insitu_green_peak_wavelength_nm": first_present(row, ["insitu_green_peak_wavelength_nm"], np.nan),
                "scene_datetime": scene_dt,
                "temporal_offset_min": temporal_offset_min,
                "dist_to_shore_km": float(row["dist_to_shore_km"]),
                "transect_context_distance_m": first_present(row, ["transect_context_distance_m"], np.nan),
                "topo_relief_p95_p5_m_5km": first_present(row, ["topo_relief_p95_p5_m_5km"], np.nan),
                "land_fraction_1km": first_present(row, ["land_fraction_1km"], np.nan),
                "landwater_valid_fraction_1km": first_present(row, ["landwater_valid_fraction_1km"], np.nan),
                "land_fraction_5km": first_present(row, ["land_fraction_5km"], np.nan),
                "landwater_valid_fraction_5km": first_present(row, ["landwater_valid_fraction_5km"], np.nan),
                "land_fraction_10km": first_present(row, ["land_fraction_10km"], np.nan),
                "landwater_valid_fraction_10km": first_present(row, ["landwater_valid_fraction_10km"], np.nan),
                "processor": proc,
            }
            base_row.update(angle_features)
            base_row.update(transect_angle_features.get(int(row["transect_nr"]), {}))
            base_row["merra2_totexttau_550"] = aerosol_features.get("merra2_totexttau_550", np.nan)
            base_row.update(row_metadata(row))
            base_row.update(landwater_contrast_by_point.get(int(row["insitu_row_id"]), empty_landwater_contrast_values(sensor)))

            obj = obj_map.get(proc_key)
            if obj is None or proc_key not in tree_map:
                out_row = base_row.copy()
                out_row.update({
                    "eo_match_found": False,
                    "eo_pixel_distance_m": np.nan,
                    "eo_x_2056": np.nan,
                    "eo_y_2056": np.nan,
                    "eo_lon": np.nan,
                    "eo_lat": np.nan,
                    "eo_context_n_pixels": 0,
                    "eo_context_invalid_fraction": np.nan,
                    "eo_context_negative_fraction": np.nan,
                    "eo_context_rrs_std_mean": np.nan,
                    "eo_context_rrs_cv_mean": np.nan,
                    "n_common_wavelengths": 0,
                    "positive_pair_count_overall": 0,
                    "MSA_percent_overall": np.nan,
                    "SB_percent_overall": np.nan,
                    "MAE_overall": np.nan,
                    "RMSE_overall": np.nan,
                    "SAM_deg_overall": np.nan,
                })
                for wl in wavelengths:
                    wl_int = int(wl)
                    out_row[f"insitu_rrs_{wl_int}"] = insitu_spec_dict.get(float(wl), np.nan)
                    out_row[f"sat_rrs_nearest_{wl_int}"] = np.nan
                    out_row[f"sat_rrs_mean_{wl_int}"] = np.nan
                    out_row[f"sat_rrs_median_{wl_int}"] = np.nan
                    out_row[f"sat_rrs_std_{wl_int}"] = np.nan
                    out_row[f"sat_rrs_n_{wl_int}"] = 0
                    out_row[f"error_{wl_int}"] = np.nan
                    out_row[f"abs_error_{wl_int}"] = np.nan
                    out_row[f"rel_error_percent_{wl_int}"] = np.nan
                    out_row[f"MSA_percent_{wl_int}"] = np.nan
                    out_row[f"SB_percent_{wl_int}"] = np.nan
                    out_row[f"eo_context_invalid_fraction_{wl_int}"] = np.nan
                    out_row[f"eo_context_negative_fraction_{wl_int}"] = np.nan
                    out_row[f"eo_context_rrs_cv_{wl_int}"] = np.nan
                rows_out.append(out_row)
                continue

            dist_m, idx_near = tree_map[proc_key].query([insitu_x_2056, insitu_y_2056])

            context_radius_m = context_radius_map.get(proc_key, np.nan)

            if not np.isfinite(dist_m) or not np.isfinite(context_radius_m) or dist_m > context_radius_m:
                out_row = base_row.copy()
                out_row.update({
                    "eo_match_found": False,
                    "eo_pixel_distance_m": float(dist_m) if np.isfinite(dist_m) else np.nan,
                    "eo_x_2056": np.nan,
                    "eo_y_2056": np.nan,
                    "eo_lon": np.nan,
                    "eo_lat": np.nan,
                    "eo_context_n_pixels": 0,
                    "eo_context_invalid_fraction": np.nan,
                    "eo_context_negative_fraction": np.nan,
                    "eo_context_rrs_std_mean": np.nan,
                    "eo_context_rrs_cv_mean": np.nan,
                    "n_common_wavelengths": 0,
                    "positive_pair_count_overall": 0,
                    "MSA_percent_overall": np.nan,
                    "SB_percent_overall": np.nan,
                    "MAE_overall": np.nan,
                    "RMSE_overall": np.nan,
                    "SAM_deg_overall": np.nan,
                })
                for wl in wavelengths:
                    wl_int = int(wl)
                    out_row[f"insitu_rrs_{wl_int}"] = insitu_spec_dict.get(float(wl), np.nan)
                    out_row[f"sat_rrs_nearest_{wl_int}"] = np.nan
                    out_row[f"sat_rrs_mean_{wl_int}"] = np.nan
                    out_row[f"sat_rrs_median_{wl_int}"] = np.nan
                    out_row[f"sat_rrs_std_{wl_int}"] = np.nan
                    out_row[f"sat_rrs_n_{wl_int}"] = 0
                    out_row[f"error_{wl_int}"] = np.nan
                    out_row[f"abs_error_{wl_int}"] = np.nan
                    out_row[f"rel_error_percent_{wl_int}"] = np.nan
                    out_row[f"MSA_percent_{wl_int}"] = np.nan
                    out_row[f"SB_percent_{wl_int}"] = np.nan
                    out_row[f"eo_context_invalid_fraction_{wl_int}"] = np.nan
                    out_row[f"eo_context_negative_fraction_{wl_int}"] = np.nan
                    out_row[f"eo_context_rrs_cv_{wl_int}"] = np.nan
                rows_out.append(out_row)
                continue

            flat_idx = valid_idx_map[proc_key][idx_near]
            eo_x = float(coords_map[proc_key][idx_near, 0])
            eo_y = float(coords_map[proc_key][idx_near, 1])
            eo_lon, eo_lat = transformer_2056_to_lonlat.transform(eo_x, eo_y)
            context_flat_indices = footprint_3x3(flat_idx, grid_shape_map[proc_key])
            context_summary, context_per_wl = summarize_context_values(
                proc,
                obj,
                band_map[proc_key],
                context_flat_indices,
            )

            sat_spec_dict = {}
            footprint_stats_dict = {}
            for wl in wavelengths:
                b = band_map[proc_key].get(float(wl))
                nearest_val = get_sat_value(proc, obj, flat_idx, b)
                footprint_vals = get_sat_values(proc, obj, context_flat_indices, b)
                footprint_summary = mc.summarize_footprint_values(footprint_vals)
                footprint_summary["nearest"] = nearest_val
                chosen_val = mc.select_matchup_value(footprint_summary, MATCHUP_VALUE_METHOD)

                sat_spec_dict[float(wl)] = chosen_val
                footprint_stats_dict[int(wl)] = {
                    "nearest": footprint_summary["nearest"],
                    "mean": footprint_summary["mean"],
                    "median": footprint_summary["median"],
                    "std": footprint_summary["std"],
                    "n": footprint_summary["n"],
                }

            common_wls = np.array([
                wl for wl in wavelengths
                if np.isfinite(insitu_spec_dict.get(float(wl), np.nan)) and np.isfinite(sat_spec_dict.get(float(wl), np.nan))
            ], dtype=float)
            y_true = np.array([insitu_spec_dict.get(float(wl), np.nan) for wl in wavelengths], dtype=float)
            y_pred = np.array([sat_spec_dict.get(float(wl), np.nan) for wl in wavelengths], dtype=float)
            overall_metrics = mc.per_spectrum_metrics(y_true, y_pred)

            out_row = base_row.copy()
            out_row.update({
                "eo_match_found": True,
                "eo_pixel_distance_m": float(dist_m),
                "eo_x_2056": eo_x,
                "eo_y_2056": eo_y,
                "eo_lon": eo_lon,
                "eo_lat": eo_lat,
                "n_common_wavelengths": int(common_wls.size),
                "positive_pair_count_overall": overall_metrics["positive_pair_count"],
                "MSA_percent_overall": float(overall_metrics["msa_percent"]) if np.isfinite(overall_metrics["msa_percent"]) else np.nan,
                "SB_percent_overall": float(overall_metrics["sb_percent"]) if np.isfinite(overall_metrics["sb_percent"]) else np.nan,
                "MAE_overall": float(overall_metrics["mae"]) if np.isfinite(overall_metrics["mae"]) else np.nan,
                "RMSE_overall": float(overall_metrics["rmse"]) if np.isfinite(overall_metrics["rmse"]) else np.nan,
                "SAM_deg_overall": float(overall_metrics["spectral_angle_deg"]) if np.isfinite(overall_metrics["spectral_angle_deg"]) else np.nan,
            })
            out_row.update(context_summary)

            for wl in wavelengths:
                wl_f = float(wl)
                wl_int = int(wl)
                ins_val = insitu_spec_dict.get(wl_f, np.nan)
                sat_val = sat_spec_dict.get(wl_f, np.nan)
                wl_context = context_per_wl.get(wl_int, {})
                wl_footprint = footprint_stats_dict.get(wl_int, {})

                out_row[f"insitu_rrs_{wl_int}"] = ins_val
                out_row[f"sat_rrs_nearest_{wl_int}"] = wl_footprint.get("nearest", np.nan)
                out_row[f"sat_rrs_mean_{wl_int}"] = wl_footprint.get("mean", np.nan)
                out_row[f"sat_rrs_median_{wl_int}"] = wl_footprint.get("median", np.nan)
                out_row[f"sat_rrs_std_{wl_int}"] = wl_footprint.get("std", np.nan)
                out_row[f"sat_rrs_n_{wl_int}"] = wl_footprint.get("n", 0)
                out_row[f"eo_context_invalid_fraction_{wl_int}"] = wl_context.get("invalid_fraction", np.nan)
                out_row[f"eo_context_negative_fraction_{wl_int}"] = wl_context.get("negative_fraction", np.nan)
                out_row[f"eo_context_rrs_cv_{wl_int}"] = wl_context.get("cv", np.nan)

                if np.isfinite(ins_val) and np.isfinite(sat_val):
                    err = sat_val - ins_val
                    out_row[f"error_{wl_int}"] = err
                    out_row[f"abs_error_{wl_int}"] = abs(err)
                    out_row[f"rel_error_percent_{wl_int}"] = (err / ins_val) * 100 if ins_val != 0 else np.nan

                    wl_metrics = mc.per_wavelength_scalar_metrics(ins_val, sat_val)
                    out_row[f"MSA_percent_{wl_int}"] = wl_metrics["MSA_percent"]
                    out_row[f"SB_percent_{wl_int}"] = wl_metrics["SB_percent"]
                else:
                    out_row[f"error_{wl_int}"] = np.nan
                    out_row[f"abs_error_{wl_int}"] = np.nan
                    out_row[f"rel_error_percent_{wl_int}"] = np.nan
                    out_row[f"MSA_percent_{wl_int}"] = np.nan
                    out_row[f"SB_percent_{wl_int}"] = np.nan

            rows_out.append(out_row)


# =============================================================================
# WRITE SINGLE CSV
# =============================================================================
metrics_csv = os.path.join(output_dir, f"{campaign}_{sensor}_per_datapoint_metrics_with_per_wavelength_and_overall.csv")
output_frame = pd.DataFrame(rows_out)
if "extent" in output_frame.columns:
    output_frame = output_frame.drop(columns=["extent"])
for column in PRECISE_EO_ANGLE_COLUMNS:
    if column not in output_frame.columns:
        output_frame[column] = np.nan
output_frame = output_frame.drop(columns=REDUNDANT_ANGLE_COLUMNS, errors="ignore")
round_output_metrics(output_frame).to_csv(metrics_csv, index=False)

print("\nCSV output written:")
print(f"  - {metrics_csv}")
