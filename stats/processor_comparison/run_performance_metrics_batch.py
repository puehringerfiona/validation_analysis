"""
Batch runner for performance_metrics.py using the post-migration data layout.

This keeps the original metric/matchup implementation intact and injects a
campaign-specific configuration before executing it for each campaign/sensor.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
import importlib.util
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


SOURCE_SCRIPT = Path(__file__).with_name("performance_metrics.py")
RESULTS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RESULTS_ROOT.parent
DOCUMENTS_ROOT = PROJECT_ROOT.parent


def first_existing_path(*candidates: Path, required_child: str | None = None) -> Path:
    for candidate in candidates:
        if candidate.exists() and (required_child is None or (candidate / required_child).exists()):
            return candidate
    return candidates[0]


OUTPUT_ROOT = RESULTS_ROOT / "stats" / "performance_metrics"
HARD_FILTER_COLUMN = "hard_filter_retain"
AEROSOL_COVARIATES_FILE = first_existing_path(
    OUTPUT_ROOT,
    SOURCE_SCRIPT.parent / "AOT550",
    SOURCE_SCRIPT.parent,
    required_child="aerosol_scene_covariates.csv",
) / "aerosol_scene_covariates.csv"
INSITU_ROOT = first_existing_path(
    PROJECT_ROOT / "insitu",
    DOCUMENTS_ROOT / "insitu",
    required_child="20230610_CST",
)
EO_OUTPUT_ROOT = first_existing_path(
    PROJECT_ROOT / "eo_data" / "output",
    DOCUMENTS_ROOT / "eo_data" / "output",
    required_child="acolite",
)
VIEWING_ANGLES_DIR = first_existing_path(
    PROJECT_ROOT / "eo_data" / "viewing_angles",
    DOCUMENTS_ROOT / "eo_data" / "viewing_angles",
    required_child="all_campaigns_transect_viewing_angles.csv",
)
RASTER_ROOT = first_existing_path(
    PROJECT_ROOT / "insitu" / "all" / "qgis_transect_bb",
    DOCUMENTS_ROOT / "insitu" / "all" / "qgis_transect_bb",
    required_child="depth",
)
DEM_RASTER_FILE = RASTER_ROOT / "depth" / "DEM.tif"
ZRH_SHORE_MASK = RASTER_ROOT / "binary_masks" / "ZRH_mask.tif"
LANDWATER_MASKS = {
    "BIE": RASTER_ROOT / "binary_masks" / "BIE_20240618_binary.tif",
    "CST": RASTER_ROOT / "binary_masks" / "CST_20230610_binary.tif",
    "WAL": RASTER_ROOT / "binary_masks" / "WAL_20240619_binary.tif",
    "ZRH": RASTER_ROOT / "binary_masks" / "ZRH_20231008_binary.tif",
}
TRANSECT_CONFIG_PATH = Path(
    PROJECT_ROOT / "insitu" / "quality_control" / "campaign_transects.py"
)

# Product discovery is no longer extent/subset-driven; keep a single output label for CSV compatibility.
EXTENTS = ["selected_output"]
LOG_FILE = OUTPUT_ROOT / "run.log"
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


@dataclass(frozen=True)
class RunConfig:
    campaign: str
    sensor: str
    depth_raster: Path

    @property
    def lake(self) -> str:
        return self.campaign.split("_", 1)[1]

    @property
    def date(self) -> str:
        return self.campaign.split("_", 1)[0]

    @property
    def compact_campaign(self) -> str:
        return f"{self.lake}{self.date}"

    @property
    def csv_conv_file(self) -> Path:
        return (
            INSITU_ROOT
            / self.campaign
            / "visualization_analysis"
            / "quality_control"
            / f"{self.campaign}_metadata_with_Rrs_conv_{self.sensor}_qc.csv"
        )

    @property
    def csv_hyper_file(self) -> Path:
        return (
            INSITU_ROOT
            / self.campaign
            / "visualization_analysis"
            / "quality_control"
            / f"{self.campaign}_metadata_with_Rrs_{self.sensor}_final_qc.csv"
        )

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.campaign

    @property
    def metrics_csv(self) -> Path:
        return self.output_dir / f"{self.campaign}_{self.sensor}_per_datapoint_metrics_with_per_wavelength_and_overall.csv"

    @property
    def landwater_raster(self) -> Path:
        return LANDWATER_MASKS[self.lake]

    @property
    def product_keywords(self) -> list[str]:
        # Include both naming styles seen under eo_data/output, for example
        # CST_20230610 and CST20230610. The runner patches discovery to accept
        # either campaign token while still requiring the sensor token.
        return [self.lake, self.date, self.compact_campaign, self.sensor]

    @property
    def transects(self) -> list[int]:
        return load_campaign_transects(self.campaign)


def load_campaign_transects(campaign: str) -> list[int]:
    spec = importlib.util.spec_from_file_location("campaign_transects", TRANSECT_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load campaign transects from {TRANSECT_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_campaign_transects(campaign, require_metadata=False)


RUNS = [
    RunConfig("20230610_CST", "L8", RASTER_ROOT / "depth" / "20230610_CST_depth.tif"),
    RunConfig("20230610_CST", "S2B", RASTER_ROOT / "depth" / "20230610_CST_depth.tif"),
    RunConfig("20231008_ZRH", "L9", ZRH_SHORE_MASK),
    RunConfig("20231008_ZRH", "S2B", ZRH_SHORE_MASK),
    RunConfig("20240618_BIE", "L9", RASTER_ROOT / "depth" / "20240618_BIE_depth.tif"),
    RunConfig("20240619_WAL", "S2A", RASTER_ROOT / "depth" / "20240619_WAL_depth.tif"),
    RunConfig("20250303_CST", "L9", RASTER_ROOT / "binary_masks" / "CST_20250303_binary.tif"),
    RunConfig("20260226_ZRH", "L8", ZRH_SHORE_MASK),
    RunConfig("20260227_ZRH", "S2B", ZRH_SHORE_MASK),
    RunConfig("20260423_ZRH", "L9", ZRH_SHORE_MASK),
    RunConfig("20260430_ZRH", "L9", ZRH_SHORE_MASK),
]

RERUN_CAMPAIGNS = {cfg.campaign for cfg in RUNS}
ACTIVE_RUNS = RUNS


def quote_path(path: Path) -> str:
    return repr(str(path))


def replace_assignment(source: str, name: str, value: str) -> str:
    pattern = rf"^{name}\s*=.*$"
    replacement = f"{name} = {value}"
    updated, count = re.subn(pattern, lambda _match: replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not patch assignment for {name!r}.")
    return updated


def patch_processor_discovery(source: str) -> str:
    """Make processor matching less ambiguous in the new eo_data/output tree."""
    source = source.replace('"must_contain": ["polymer_outputs"]', '"must_contain": [r"\\\\polymer\\\\"]')
    source = source.replace('"must_contain": ["c2rcc"]', '"must_contain": [r"\\\\c2rcc\\\\"]', 1)
    source = source.replace('"must_contain": ["c2rcc_tmart"]', '"must_contain": [r"\\\\c2rcc_tmart\\\\"]')
    source = source.replace('"must_contain": ["c2rcc_radcor"]', '"must_contain": [r"\\\\c2rcc_radcor\\\\"]')
    return source


def patch_keyword_logic(source: str) -> str:
    old = """def _has_all(path: str, tokens):
    low = path.lower()
    return all(t.lower() in low for t in tokens)
"""
    new = """def _has_all(path: str, tokens):
    low = path.lower()
    tokens_low = [str(t).lower() for t in tokens]
    if len(tokens_low) == 4:
        lake, date, compact_campaign, sensor = tokens_low
        campaign_matches = (
            f"{lake}_{date}" in low
            or compact_campaign in low
            or date in low
            or date.replace("-", "") in low
        )
        sensor_aliases = {
            "l8": ["l8", "lc08", "oli"],
            "l9": ["l9", "lc09", "oli"],
            "s2a": ["s2a"],
            "s2b": ["s2b"],
        }.get(sensor, [sensor])
        sensor_matches = any(alias in low for alias in sensor_aliases)
        return campaign_matches and sensor_matches
    return all(t in low for t in tokens_low)
"""
    if old not in source:
        raise RuntimeError("Could not patch _has_all keyword logic.")
    return source.replace(old, new, 1)


def patch_file_discovery(source: str) -> str:
    old = """def discover_files_for_processor(root_dir: str, proc_name: str, proc_cfg: dict,
                                 plot_extents_km: list[str], sensor: str, global_keywords: list[str]):
    out = {}
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
            if not _has_all(full, proc_cfg["must_contain"]):
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
"""
    new = """def discover_files_for_processor(root_dir: str, proc_name: str, proc_cfg: dict,
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

    mode = proc_cfg.get("extent_mode", "px")

    multi_output_roots = {
        "ACOLITE": ["acolite_dsf"],
        "ACOLITE+RAdCor": ["acolite_radcor_tsdsf"],
        "ACOLITE+T-Mart": ["acolite_tmart"],
        "Polymer+Radcor": ["polymer_radcor"],
        "C2RCC+RAdCor": ["c2rcc_radcor"],
    }
    single_output_roots = {
        "ACOLITE": ["acolite"],
        "ACOLITE+RAdCor": ["acolite_radcor"],
        "RAdCor+ACOLITE": ["radcor_acolite"],
        "ACOLITE+T-Mart": ["acolite_tmart", "tmart"],
        "Polymer": ["polymer"],
        "Polymer+T-Mart": ["polymer_tmart"],
        "Polymer+Radcor": ["radcor_polymer", "polymer_radcor"],
        "C2RCC": ["c2rcc"],
        "C2RCC+T-Mart": ["c2rcc_tmart"],
        "C2RCC+RAdCor": ["c2rcc_radcor"],
        "GAAC": ["gaac"],
    }

    def sensor_family():
        sensor_token = str(sensor).lower()
        if sensor_token.startswith("s2"):
            return "sentinel_products"
        if sensor_token.startswith("l"):
            return "landsat_products"
        return ""

    def search_roots():
        roots = []
        family = sensor_family()
        for segment in multi_output_roots.get(proc_name, []):
            if family:
                roots.append(os.path.join(root_dir, "sencast", family, segment))
            roots.append(os.path.join(root_dir, "sencast", segment))
        for segment in single_output_roots.get(proc_name, []):
            if family:
                roots.append(os.path.join(root_dir, "sencast", family, segment))
            roots.append(os.path.join(root_dir, segment))
        seen = set()
        existing = []
        for candidate in roots:
            key = os.path.normcase(os.path.abspath(candidate))
            if key in seen:
                continue
            seen.add(key)
            if os.path.isdir(candidate):
                existing.append(candidate)
        return existing

    def looks_like_output(path):
        low = path.lower()
        norm_low = low.replace(chr(92), "/")
        name = os.path.basename(low)
        if (
            name.endswith(("_rgb.tif", "_watermask.tif"))
            or "_watermask" in name
            or "mosaic" in low
            or "/non_best/" in norm_low
            or "/_non_best/" in norm_low
            or "/winter/" in norm_low
            or "_winter/" in norm_low
            or "/winter_" in norm_low
            or "_winter_" in norm_low
            or "_non_best_products" in low
            or "_reproducibility" in low
        ):
            return False
        if proc_name in {"ACOLITE", "ACOLITE+RAdCor", "RAdCor+ACOLITE", "ACOLITE+T-Mart"}:
            return ("l2w" in name) or ("l2acolite" in low)
        if proc_name.startswith("Polymer"):
            return name.endswith(".nc")
        if proc_name.startswith("C2RCC"):
            if not name.endswith(".nc"):
                return False
            if "/l1p/" in norm_low or "l1p_" in name or "_l1p" in name:
                return False
            return "l2c2rcc" in low or "c2rcc" in name
        if proc_name == "GAAC":
            return name.endswith((".tif", ".tiff")) and ("rhow" in name)
        return True

    def candidate_rank(path):
        low = path.lower()
        norm_low = low.replace(chr(92), "/")
        score = 0
        if "/sencast/" in norm_low:
            score -= 100
        if "/sencast/" not in norm_low and proc_name in {"ACOLITE", "ACOLITE+RAdCor", "RAdCor+ACOLITE", "ACOLITE+T-Mart", "Polymer+Radcor", "C2RCC+RAdCor"}:
            score += 100
        if proc_name.startswith("C2RCC"):
            if "l2c2rcc" in low:
                score -= 40
            if "/l1p/" in norm_low or "l1p_" in os.path.basename(low):
                score += 1000
        if (
            proc_name in {"ACOLITE", "ACOLITE+RAdCor", "RAdCor+ACOLITE", "ACOLITE+T-Mart"}
            and low.endswith((".tif", ".tiff"))
            and "l2w_stack" in low
        ):
            score -= 70
        if proc_name == "RAdCor+ACOLITE" and low.endswith((".tif", ".tiff")):
            score -= 50
        if "reproj" in low:
            score -= 5
        if "unproj" in low:
            score += 10
        return (score, len(path), path)

    candidates = []
    roots_to_search = search_roots()
    if not roots_to_search:
        return out

    for search_root in roots_to_search:
        for dp, dirs, files in os.walk(search_root):
            dirs[:] = [
                d for d in dirs
                if d not in {"non_best", "_non_best", "_cache", "_run_control", "__pycache__"}
                and "winter" not in d.lower()
            ]
            for f in files:
                if not f.lower().endswith((".nc", ".tif", ".tiff")):
                    continue
                full = os.path.join(dp, f)
                if not _has_all(full, global_keywords):
                    continue
                if not looks_like_output(full):
                    continue
                candidates.append(full)

    candidates = sorted(set(candidates), key=candidate_rank)
    if not candidates:
        return out

    if proc_name == "ACOLITE+RAdCor" and sensor == "S2B" and "zrh" in keyword_text and "20260227" in keyword_text:
        transect_files = {}
        for candidate in candidates:
            low = candidate.lower()
            transect_match = re.search(r"(?:^|[_\\/])t([13])(?:[_\\/])", low)
            if transect_match and int(transect_match.group(1)) not in transect_files:
                transect_files[int(transect_match.group(1))] = candidate
        if transect_files:
            out["__by_transect__"] = transect_files
            first_output = transect_files.get(1) or next(iter(transect_files.values()))
            out["__single__"] = first_output
            for pe in plot_extents_km:
                out[pe] = first_output
            return out

    first_output = candidates[0]
    if mode == "single":
        out["__single__"] = first_output
        return out

    for pe in plot_extents_km:
        out[pe] = first_output
    return out
"""
    if old not in source:
        pattern = (
            r"def discover_files_for_processor\(root_dir: str, proc_name: str, proc_cfg: dict,"
            r"\n\s+plot_extents_km: list\[str\], sensor: str, global_keywords: list\[str\]\):"
            r"\n[\s\S]*?\n(?=def nearest_distance_km)"
        )
        updated, count = re.subn(pattern, new + "\n", source, count=1)
        if count != 1:
            raise RuntimeError("Could not patch EO file discovery.")
        return updated
    return source.replace(old, new, 1)


def patch_progress_logging(source: str) -> str:
    replacements = [
        (
            """angle_features = load_viewing_angle_features(VIEWING_ANGLES_DIR, campaign, sensor)
""",
            """angle_features = load_viewing_angle_features(VIEWING_ANGLES_DIR, campaign, sensor)
print(f"[progress] configured run campaign={campaign} sensor={sensor}")
print(f"[progress] CSV_CONV_FILE={CSV_CONV_FILE}")
print(f"[progress] PRODUCT_ROOT={PRODUCT_ROOT}")
""",
        ),
        (
            """insitu_conv_df = insitu_conv_df.reset_index(drop=True)
insitu_conv_df["insitu_row_id"] = insitu_conv_df.index
""",
            """insitu_conv_df = insitu_conv_df.reset_index(drop=True)
insitu_conv_df["insitu_row_id"] = insitu_conv_df.index
print(f"[progress] retained in-situ rows selected for requested transects: {len(insitu_conv_df)}")
""",
        ),
        (
            """if DEPTH_RASTER_FILE:
    with rasterio.open(DEPTH_RASTER_FILE) as src:
""",
            """if DEPTH_RASTER_FILE:
    print(f"[progress] computing distance-to-shore from raster: {DEPTH_RASTER_FILE}")
    with rasterio.open(DEPTH_RASTER_FILE) as src:
""",
        ),
        (
            """for proc, cfg in enabled_procs.items():
    processor_files[proc] = discover_files_for_processor(
""",
            """print(f"[progress] discovering EO files for processors and extents: {SPATIAL_SUBSETS}")
for proc, cfg in enabled_procs.items():
    processor_files[proc] = discover_files_for_processor(
""",
        ),
        (
            """        if fp.lower().endswith((".tif", ".tiff")):
            datasets[proc] = rioxarray.open_rasterio(fp)
        else:
            datasets[proc] = xr.open_dataset(fp)
""",
            """        print(f"[progress] opening {proc} file for subset {subset}: {fp}")
        if fp.lower().endswith((".tif", ".tiff")):
            datasets[proc] = rioxarray.open_rasterio(fp)
        else:
            datasets[proc] = xr.open_dataset(fp)
""",
        ),
        (
            """    for _, row in gdf_conv.iterrows():
""",
            """    print(f"[progress] matching {len(gdf_conv)} in-situ rows for subset {subset}")
    for _, row in gdf_conv.iterrows():
""",
        ),
        (
            """metrics_csv = os.path.join(output_dir, f"{campaign}_{sensor}_per_datapoint_metrics_with_per_wavelength_and_overall.csv")
output_frame = pd.DataFrame(rows_out)
if "extent" in output_frame.columns:
    output_frame = output_frame.drop(columns=["extent"])
for column in PRECISE_EO_ANGLE_COLUMNS:
    if column not in output_frame.columns:
        output_frame[column] = np.nan
output_frame = output_frame.drop(columns=REDUNDANT_ANGLE_COLUMNS, errors="ignore")
round_output_metrics(output_frame).to_csv(metrics_csv, index=False)
""",
            """metrics_csv = os.path.join(output_dir, f"{campaign}_{sensor}_per_datapoint_metrics_with_per_wavelength_and_overall.csv")
print(f"[progress] writing {len(rows_out)} rows to {metrics_csv}")
output_frame = pd.DataFrame(rows_out)
if "extent" in output_frame.columns:
    output_frame = output_frame.drop(columns=["extent"])
for column in PRECISE_EO_ANGLE_COLUMNS:
    if column not in output_frame.columns:
        output_frame[column] = np.nan
output_frame = output_frame.drop(columns=REDUNDANT_ANGLE_COLUMNS, errors="ignore")
round_output_metrics(output_frame).to_csv(metrics_csv, index=False)
""",
        ),
    ]
    for old, new in replacements:
        if old not in source:
            if "[progress] opening {proc} file for subset {subset}" in new:
                continue
            raise RuntimeError("Could not patch progress logging.")
        source = source.replace(old, new, 1)
    return source


def patch_satellite_band_cache(source: str) -> str:
    old_getters = """def get_sat_value(proc, obj, flat_idx, band_ref):
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
"""
    new_getters = """SATELLITE_BAND_CACHE = {}


def get_sat_value(proc, obj, flat_idx, band_ref):
    if band_ref is None:
        return np.nan

    cached = SATELLITE_BAND_CACHE.get((proc, id(obj), band_ref))
    if cached is not None:
        return float(cached[int(flat_idx)])

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

    cached = SATELLITE_BAND_CACHE.get((proc, id(obj), band_ref))
    if cached is not None:
        return cached[np.asarray(flat_indices, dtype=int)].astype(float)

    if is_raster_stack(obj):
        arr = obj.sel(band=int(band_ref)).values.ravel()
    else:
        arr = obj[band_ref].values.ravel()

    vals = arr[np.asarray(flat_indices, dtype=int)].astype(float)
    if enabled_procs[proc]["divide_by_pi"]:
        vals = vals / np.pi
    return vals
"""
    if old_getters not in source:
        raise RuntimeError("Could not patch satellite value cache getters.")
    source = source.replace(old_getters, new_getters, 1)

    old_band_map = """            band_map[proc_key] = build_band_map(proc, obj, sensor, wavelengths)
"""
    new_band_map = """            band_map[proc_key] = build_band_map(proc, obj, sensor, wavelengths)
            unique_band_refs = []
            for band_ref in band_map[proc_key].values():
                if band_ref is not None and band_ref not in unique_band_refs:
                    unique_band_refs.append(band_ref)
            print(f"[progress] caching {len(unique_band_refs)} satellite bands for {proc} {proc_key[1]} in subset {subset}")
            for band_ref in unique_band_refs:
                if is_raster_stack(obj):
                    cached_arr = obj.sel(band=int(band_ref)).values.ravel().astype(float)
                else:
                    cached_arr = obj[band_ref].values.ravel().astype(float)
                if enabled_procs[proc]["divide_by_pi"]:
                    cached_arr = cached_arr / np.pi
                SATELLITE_BAND_CACHE[(proc, id(obj), band_ref)] = cached_arr
            print(f"[progress] cached {len(unique_band_refs)} satellite bands for {proc} {proc_key[1]} in subset {subset}")
"""
    if old_band_map not in source:
        raise RuntimeError("Could not patch satellite band cache builder.")
    return source.replace(old_band_map, new_band_map, 1)


def patch_processor_match_progress(source: str) -> str:
    old_loop = """    print(f"[progress] matching {len(gdf_conv)} in-situ rows for subset {subset}")
    for _, row in gdf_conv.iterrows():
"""
    new_loop = """    match_total = len(gdf_conv)
    print(f"[progress] matching {match_total} in-situ rows for subset {subset}")
    for _, row in gdf_conv.iterrows():
"""
    if old_loop not in source:
        raise RuntimeError("Could not patch match total progress.")
    source = source.replace(old_loop, new_loop, 1)

    old_proc_loop = """        for proc in enabled_procs.keys():
"""
    new_proc_loop = """        for proc in enabled_procs.keys():
            row_pos = int(row["insitu_row_id"]) + 1
            if row_pos == 1 or row_pos % 25 == 0 or row_pos == match_total:
                print(f"[progress] matching processor {proc}: row {row_pos}/{match_total} for subset {subset}")
"""
    if old_proc_loop not in source:
        raise RuntimeError("Could not patch per-processor match progress.")
    return source.replace(old_proc_loop, new_proc_loop, 1)


def patch_distance_transform(source: str) -> str:
    new = """if DEPTH_RASTER_FILE:
    print(f"[progress] computing distance-to-shore with per-transect raster distance transforms: {DEPTH_RASTER_FILE}")
    from scipy.ndimage import distance_transform_edt
    from rasterio.windows import Window, from_bounds

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

    def pixel_sizes_m(transform, water_mask, to_2056):
        center_col = max(0, min(water_mask.shape[1] - 2, water_mask.shape[1] // 2))
        center_row = max(0, min(water_mask.shape[0] - 2, water_mask.shape[0] // 2))
        x0, y0 = transform * (center_col, center_row)
        x1, y1 = transform * (center_col + 1, center_row)
        x2, y2 = transform * (center_col, center_row + 1)
        x0m, y0m = to_2056.transform(x0, y0)
        x1m, y1m = to_2056.transform(x1, y1)
        x2m, y2m = to_2056.transform(x2, y2)
        px_m = float(np.hypot(x1m - x0m, y1m - y0m))
        py_m = float(np.hypot(x2m - x0m, y2m - y0m))
        return px_m, py_m

    dist_cache_file = os.path.join(output_dir, f"{campaign}_dist_to_shore_{'_'.join(map(str, analysis_transects))}.csv")
    distance_cache_loaded = os.path.exists(dist_cache_file)
    if distance_cache_loaded:
        print(f"[progress] loading cached distance-to-shore values: {dist_cache_file}")
        dist_cache_df = pd.read_csv(dist_cache_file)
        dist_map = dict(zip(dist_cache_df["insitu_row_id"].astype(int), dist_cache_df["dist_to_shore_km"].astype(float)))
        gdf_conv["dist_to_shore_km"] = gdf_conv["insitu_row_id"].map(dist_map)
    else:
        gdf_conv["dist_to_shore_km"] = np.nan
    with rasterio.open(DEPTH_RASTER_FILE) as src:
        raster_crs = src.crs
        to_2056 = Transformer.from_crs(raster_crs, f"EPSG:{EPSG_CH}", always_xy=True)
        to_raster_crs = Transformer.from_crs(f"EPSG:{EPSG_CH}", raster_crs, always_xy=True)

        for transect_nr, transect_group in gdf_conv.groupby("transect_nr"):
            ordered_group = transect_group.sort_values("insitu_row_id")
            chunk_size = 60
            chunk_total = int(np.ceil(len(ordered_group) / chunk_size))
            for chunk_idx, chunk_start in enumerate(range(0, len(ordered_group), chunk_size), start=1):
                group_2056 = ordered_group.iloc[chunk_start:chunk_start + chunk_size]
                gdf_raster = group_2056.to_crs(raster_crs)
                minx, miny, maxx, maxy = gdf_raster.total_bounds
                pad = 0.03 if raster_crs.is_geographic else 3000.0
                raw_window = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, src.transform)
                raw_window = raw_window.round_offsets().round_lengths()
                col_off = max(0, int(raw_window.col_off))
                row_off = max(0, int(raw_window.row_off))
                col_max = min(src.width, int(raw_window.col_off + raw_window.width))
                row_max = min(src.height, int(raw_window.row_off + raw_window.height))
                if col_max <= col_off or row_max <= row_off:
                    print(f"[progress] warning: empty depth raster crop for transect {transect_nr} chunk {chunk_idx}/{chunk_total}")
                    continue
                window = Window(col_off, row_off, col_max - col_off, row_max - row_off)
                print(f"[progress] depth raster transect {transect_nr} chunk {chunk_idx}/{chunk_total} crop rows={int(window.height)} cols={int(window.width)}")

                depth = src.read(1, window=window)
                transform = src.window_transform(window)
                valid_mask = np.isfinite(depth)
                if src.nodata is not None and np.isfinite(src.nodata):
                    valid_mask &= depth != src.nodata
                water_mask = valid_mask & (depth > 0)
                if not np.any(water_mask) and np.any(valid_mask):
                    water_mask = valid_mask
                px_m, py_m = pixel_sizes_m(transform, water_mask, to_2056)
                water_mask = fill_small_internal_mask_holes(water_mask, abs(px_m * py_m), max_hole_area_m2=4.0)
                if not np.any(~water_mask):
                    print(f"[progress] warning: transect {transect_nr} chunk {chunk_idx}/{chunk_total} crop contains no land pixels; expanding crop once.")
                    pad2 = 0.08 if raster_crs.is_geographic else 8000.0
                    raw_window = from_bounds(minx - pad2, miny - pad2, maxx + pad2, maxy + pad2, src.transform).round_offsets().round_lengths()
                    col_off = max(0, int(raw_window.col_off))
                    row_off = max(0, int(raw_window.row_off))
                    col_max = min(src.width, int(raw_window.col_off + raw_window.width))
                    row_max = min(src.height, int(raw_window.row_off + raw_window.height))
                    window = Window(col_off, row_off, col_max - col_off, row_max - row_off)
                    print(f"[progress] depth raster transect {transect_nr} chunk {chunk_idx}/{chunk_total} expanded crop rows={int(window.height)} cols={int(window.width)}")
                    depth = src.read(1, window=window)
                    transform = src.window_transform(window)
                    valid_mask = np.isfinite(depth)
                    if src.nodata is not None and np.isfinite(src.nodata):
                        valid_mask &= depth != src.nodata
                    water_mask = valid_mask & (depth > 0)
                    if not np.any(water_mask) and np.any(valid_mask):
                        water_mask = valid_mask
                    px_m, py_m = pixel_sizes_m(transform, water_mask, to_2056)
                    water_mask = fill_small_internal_mask_holes(water_mask, abs(px_m * py_m), max_hole_area_m2=4.0)

                if distance_cache_loaded:
                    continue

                center_col = max(0, min(water_mask.shape[1] - 2, water_mask.shape[1] // 2))
                center_row = max(0, min(water_mask.shape[0] - 2, water_mask.shape[0] // 2))
                x0, y0 = transform * (center_col, center_row)
                x1, y1 = transform * (center_col + 1, center_row)
                x2, y2 = transform * (center_col, center_row + 1)
                x0m, y0m = to_2056.transform(x0, y0)
                x1m, y1m = to_2056.transform(x1, y1)
                x2m, y2m = to_2056.transform(x2, y2)
                px_m = float(np.hypot(x1m - x0m, y1m - y0m))
                py_m = float(np.hypot(x2m - x0m, y2m - y0m))
                dist_to_land_m = distance_transform_edt(water_mask, sampling=(py_m, px_m))

                for idx, point_2056 in group_2056.geometry.items():
                    x_r, y_r = to_raster_crs.transform(point_2056.x, point_2056.y)
                    col_f, row_f = (~transform) * (x_r, y_r)
                    row_i = int(round(row_f))
                    col_i = int(round(col_f))
                    if 0 <= row_i < dist_to_land_m.shape[0] and 0 <= col_i < dist_to_land_m.shape[1]:
                        gdf_conv.loc[idx, "dist_to_shore_km"] = float(dist_to_land_m[row_i, col_i] / 1000.0)
    if not distance_cache_loaded:
        cache_cols = ["insitu_row_id", "transect_nr", "dist_to_shore_km"]
        gdf_conv[cache_cols].to_csv(dist_cache_file, index=False)
        print(f"[progress] cached distance-to-shore values: {dist_cache_file}")
else:
"""
    pattern = r"if DEPTH_RASTER_FILE:\n[\s\S]*?\nelse:\n"
    updated, count = re.subn(pattern, new, source, count=1)
    if count != 1:
        raise RuntimeError("Could not patch distance-to-shore computation.")
    return updated


def patch_temporal_offset_timezone(source: str) -> str:
    if "input_scene_dt = parse_input_scene_datetime(row)" in source:
        return source
    old = """            scene_dt = scene_datetime_map.get(proc, pd.NaT)
            insitu_dt = row.get("insitu_datetime", pd.NaT)
            if pd.notna(scene_dt) and pd.notna(insitu_dt):
                temporal_offset_min = abs((pd.Timestamp(insitu_dt) - pd.Timestamp(scene_dt)).total_seconds()) / 60.0
            else:
                temporal_offset_min = np.nan
"""
    new = """            scene_dt = scene_datetime_map.get(proc, pd.NaT)
            insitu_dt = row.get("insitu_datetime", pd.NaT)
            if pd.notna(scene_dt) and pd.notna(insitu_dt):
                insitu_ts = pd.Timestamp(insitu_dt)
                scene_ts = pd.Timestamp(scene_dt)
                if insitu_ts.tzinfo is not None:
                    insitu_ts = insitu_ts.tz_convert(None)
                if scene_ts.tzinfo is not None:
                    scene_ts = scene_ts.tz_convert(None)
                temporal_offset_min = abs((insitu_ts - scene_ts).total_seconds()) / 60.0
            else:
                temporal_offset_min = np.nan
"""
    if old not in source:
        raise RuntimeError("Could not patch temporal offset timezone handling.")
    return source.replace(old, new, 1)


def clean_metrics_angle_columns(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    frame = pd.read_csv(path, sep=sep, low_memory=False)
    for column in PRECISE_EO_ANGLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.drop(columns=REDUNDANT_ANGLE_COLUMNS, errors="ignore")
    frame.to_csv(path, sep=sep, index=False)


def build_source_for_run(base_source: str, cfg: RunConfig) -> str:
    source = base_source
    source = replace_assignment(source, "CSV_HYPER_FILE", quote_path(cfg.csv_hyper_file))
    source = replace_assignment(source, "CSV_CONV_FILE", quote_path(cfg.csv_conv_file))
    source = replace_assignment(source, "DEPTH_RASTER_FILE", quote_path(cfg.depth_raster))
    source = replace_assignment(source, "DEM_RASTER_FILE", quote_path(DEM_RASTER_FILE))
    source = replace_assignment(source, "LANDWATER_RASTER_FILE", quote_path(cfg.landwater_raster))
    source = replace_assignment(source, "VIEWING_ANGLES_DIR", quote_path(VIEWING_ANGLES_DIR))
    source = replace_assignment(source, "AEROSOL_COVARIATES_FILE", quote_path(AEROSOL_COVARIATES_FILE))
    source = replace_assignment(source, "output_dir", quote_path(cfg.output_dir))
    source = replace_assignment(source, "PRODUCT_ROOT", quote_path(EO_OUTPUT_ROOT))
    source = replace_assignment(source, "PRODUCT_KEYWORDS", repr(cfg.product_keywords))
    source = replace_assignment(source, "EXTENTS", repr(EXTENTS))
    source = replace_assignment(source, "INSITU_HARD_FILTER_COLUMN", repr(HARD_FILTER_COLUMN))
    source = patch_processor_discovery(source)
    source = patch_keyword_logic(source)
    source = patch_file_discovery(source)
    source = patch_progress_logging(source)
    source = patch_satellite_band_cache(source)
    source = patch_processor_match_progress(source)
    source = patch_distance_transform(source)
    source = patch_temporal_offset_timezone(source)
    return source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run performance metric exports for all configured campaign/sensor pairs.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--hard-filter-column",
        default=HARD_FILTER_COLUMN,
        choices=["hard_filter_retain", "hard_filter_retain_strict"],
    )
    return parser.parse_args(argv)


def configure_run(args: argparse.Namespace) -> None:
    global OUTPUT_ROOT, HARD_FILTER_COLUMN, LOG_FILE
    OUTPUT_ROOT = args.output_root
    HARD_FILTER_COLUMN = args.hard_filter_column
    LOG_FILE = OUTPUT_ROOT / "run.log"


class Tee:
    def __init__(self, *targets):
        self.targets = targets

    def write(self, text: str) -> int:
        for target in self.targets:
            target.write(text)
            target.flush()
        return len(text)

    def flush(self) -> None:
        for target in self.targets:
            target.flush()


def validate_inputs() -> None:
    missing = []
    for cfg in ACTIVE_RUNS:
        for path in (cfg.csv_conv_file, cfg.csv_hyper_file, cfg.depth_raster):
            if not path.exists():
                missing.append(path)
    if not DEM_RASTER_FILE.exists():
        missing.append(DEM_RASTER_FILE)
    for path in LANDWATER_MASKS.values():
        if not path.exists():
            missing.append(path)
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required inputs:\n{missing_text}")


def main(argv: list[str] | None = None) -> int:
    configure_run(parse_args(argv))
    validate_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_dir = str(SOURCE_SCRIPT.parent)
    stats_dir = str(SOURCE_SCRIPT.parent.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    if stats_dir not in sys.path:
        sys.path.insert(0, stats_dir)
    base_source = SOURCE_SCRIPT.read_text(encoding="utf-8")
    log_handle = LOG_FILE.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    print(f"\n\n##### Batch launch {datetime.now().isoformat(timespec='seconds')} #####")
    print(f"Log file: {LOG_FILE}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Hard-filter column: {HARD_FILTER_COLUMN}")

    failures = []
    try:
        print(f"Active rerun campaigns: {sorted(RERUN_CAMPAIGNS)}")
        print("Other existing campaign outputs are not touched by this launch.")
        for cfg in ACTIVE_RUNS:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            if cfg.metrics_csv.exists():
                print(f"\n\n##### Skipping {cfg.campaign} {cfg.sensor}: output already exists #####")
                print(f"Existing output: {cfg.metrics_csv}")
                continue
            print(f"\n\n##### Running {cfg.campaign} {cfg.sensor} #####")
            print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Transects: {cfg.transects}")
            print(f"Depth raster: {cfg.depth_raster}")
            print(f"Output dir: {cfg.output_dir}")

            module_name = f"performance_metrics_{cfg.campaign}_{cfg.sensor}"
            module = types.ModuleType(module_name)
            module.__file__ = str(SOURCE_SCRIPT)
            namespace = module.__dict__
            sys.modules[module_name] = module
            try:
                exec(compile(build_source_for_run(base_source, cfg), str(SOURCE_SCRIPT), "exec"), namespace)
                clean_metrics_angle_columns(cfg.metrics_csv)
                print(f"Finished: {datetime.now().isoformat(timespec='seconds')}")
            except Exception:
                failures.append(f"{cfg.campaign} {cfg.sensor}")
                traceback.print_exc()
            finally:
                sys.modules.pop(module_name, None)

        print("\n\n##### Batch complete #####")
        if failures:
            print("Failed runs:")
            for run_id in failures:
                print(f"  - {run_id}")
            return 1

        print(f"All outputs written under: {OUTPUT_ROOT}")
        return 0
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
