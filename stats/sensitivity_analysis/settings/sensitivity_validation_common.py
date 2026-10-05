"""Shared validation helpers for sensitivity-analysis entry points."""

import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matchup_common as mc


FOOTPRINT_RADIUS_PIXELS = 1.5
TRANSECT_CONFIG_PATH = Path(
    r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\visualization_analysis\quality_control\campaign_transects.py"
)
COMMON_EXTENTS = {
    "0.5 km": {"MSI": "25x25", "OLI_TIRS": "17x17"},
    "1 km": {"MSI": "50x50", "OLI_TIRS": "34x34"},
    "3 km": {"MSI": "150x150", "OLI_TIRS": "100x100"},
    "5 km": {"MSI": "250x250", "OLI_TIRS": "167x167"},
    "7 km": {"MSI": "350x350", "OLI_TIRS": "234x234"},
    "10 km": {"MSI": "500x500", "OLI_TIRS": "334x334"},
    "15 km": {"MSI": "750x750", "OLI_TIRS": "500x500"},
    "30 km": {"MSI": "1500x1500", "OLI_TIRS": "1000x1000"},
}
ALLOWED_TILES_BY_SENSOR = {
    sensor: {tiles[sensor] for tiles in COMMON_EXTENTS.values()}
    for sensor in ("MSI", "OLI_TIRS")
}
C2RCC_BAND_INDEX = {
    "L8": {443: 1, 483: 2, 561: 3, 655: 4, 865: 5},
    "L9": {443: 1, 482: 2, 561: 3, 654: 4, 865: 5},
    "S2A": {443: 1, 492: 2, 560: 3, 665: 4, 704: 5, 740: 6, 783: 7, 865: 8},
    "S2B": {442: 1, 492: 2, 559: 3, 665: 4, 704: 5, 739: 6, 780: 7, 865: 8},
}
C2RCC_S2_WL_TO_BAND = {
    "S2A": {443: "B1", 492: "B2", 560: "B3", 665: "B4", 704: "B5", 740: "B6", 783: "B7", 865: "B8A"},
    "S2B": {442: "B1", 492: "B2", 559: "B3", 665: "B4", 704: "B5", 739: "B6", 780: "B7", 865: "B8A"},
}


def netcdf_path(path):
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(path)
    return path


def round_output_metrics(frame):
    return mc.round_output_metrics(frame)


def insitu_campaign_name(campaign):
    match = re.fullmatch(r"([A-Z]+)(\d{8})", campaign)
    if not match:
        raise ValueError(f"Cannot convert campaign ID to in-situ folder name: {campaign}")
    site, date = match.groups()
    return f"{date}_{site}"


def load_campaign_transects():
    spec = importlib.util.spec_from_file_location("campaign_transects", TRANSECT_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load campaign transects from {TRANSECT_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_campaign_transects(require_metadata=False)


def sensor_code_from_csv(path):
    match = re.search(r"_metadata_with_Rrs_conv_([^_]+)_qc\.csv$", str(path))
    if not match:
        raise ValueError(f"Could not parse sensor code from input CSV path: {path}")
    return match.group(1)


def qc_csv(documents, campaign, sensor_config):
    campaign_folder = insitu_campaign_name(campaign)
    sensor_code = sensor_code_from_csv(sensor_config["csv"])
    return (
        documents
        / "insitu"
        / campaign_folder
        / "visualization_analysis"
        / "quality_control"
        / f"{campaign_folder}_metadata_with_Rrs_conv_{sensor_code}_qc.csv"
    )


def configure_convolved_qc_inputs(campaign_config, documents, hard_filter_column="hard_filter_retain"):
    transects = load_campaign_transects()
    configured = {}
    for campaign, definition in campaign_config.items():
        campaign_folder = insitu_campaign_name(campaign)
        if campaign_folder not in transects:
            raise KeyError(f"No predefined transects configured for {campaign_folder}")
        sensors = {}
        for sensor, sensor_config in definition["sensors"].items():
            csv = qc_csv(documents, campaign, sensor_config)
            sensor_code = sensor_code_from_csv(csv)
            if not csv.exists():
                raise FileNotFoundError(f"Convolved QC input not found: {csv}")
            sensors[sensor] = {
                **sensor_config,
                "csv": csv,
                "sensor_code": sensor_code,
                "hard_filter_column": hard_filter_column,
            }
        configured[campaign] = {
            **definition,
            "scenarios": ["predefined_transects"],
            "scenario_transects": {"predefined_transects": transects[campaign_folder]},
            "sensors": sensors,
        }
    return configured


def get_scenario_transects(definition, scenario):
    if "scenario_transects" in definition and scenario in definition["scenario_transects"]:
        return definition["scenario_transects"][scenario]
    raise KeyError(f"Scenario {scenario!r} has no transects from campaign_transects.py")


def boolean_mask(series):
    return mc.boolean_mask(series)


def read_and_filter_insitu(config, transects):
    frame = pd.read_csv(config["csv"], sep=";")
    result = mc.filter_insitu_rows(
        frame,
        transects=transects,
        metric_waves=config["metric_waves"],
        hard_filter_column=config.get("hard_filter_column"),
        require_hard_filter_column=bool(config.get("hard_filter_column")),
    )
    return (
        result["frame"],
        result["input_rows"],
        result["rows_after_transect_coordinate_selection"],
        result["rows_after_insitu_filtering"],
    )


def product_label(path, campaign, sensor, resolution):
    text = str(path)
    c2rcc_radcor_match = re.search(
        rf"{campaign}_{sensor}_c2rcc_radcor_tsdsf_(autoLUT|MOD1|MOD2)(?:_[^\\/]+)?_lb([^_\\/]+)_kr(\d+km)(?:_|\\|/)",
        text,
    )
    if c2rcc_radcor_match:
        lut, _land_buffer, kernel_radius = c2rcc_radcor_match.groups()
        return {
            "lut": lut,
            "tile_dimensions": "campaign",
            "physical_tile_extent": "campaign",
            "kernel_radius": kernel_radius,
            "land_buffer": "",
            "product_transect": np.nan,
        }

    polymer_radcor_match = re.search(
        rf"{campaign}_{sensor}_polymer_radcor_tsdsf_(autoLUT|MOD1|MOD2)(?:_[^\\/]+)?_lb([^_\\/]+)_kr(\d+km)(?:_|\\|/)",
        text,
    )
    if polymer_radcor_match:
        lut, _land_buffer, kernel_radius = polymer_radcor_match.groups()
        return {
            "lut": lut,
            "tile_dimensions": "campaign",
            "physical_tile_extent": "campaign",
            "kernel_radius": kernel_radius,
            "land_buffer": "",
            "product_transect": np.nan,
        }

    radcor_match = re.search(
        rf"{campaign}_{sensor}_acolite_radcor_tsdsf_(autoLUT|MOD1|MOD2)_t(\d+)_(\d+km)_lb([^_]+)_kr(\d+km)(?:_|\\|/)",
        text,
    )
    if radcor_match:
        lut, transect, subset, _land_buffer, kernel_radius = radcor_match.groups()
        return {
            "lut": lut,
            "tile_dimensions": subset,
            "physical_tile_extent": subset.replace("km", " km"),
            "kernel_radius": kernel_radius,
            "land_buffer": "",
            "product_transect": int(transect),
        }

    campaign_radcor_match = re.search(
        rf"{campaign}_{sensor}_acolite_radcor_tsdsf_(autoLUT|MOD1|MOD2)_campaign_(\d+km)_lb([^_]+)_kr(\d+km)(?:_|\\|/)",
        text,
    )
    if campaign_radcor_match:
        lut, subset, _land_buffer, kernel_radius = campaign_radcor_match.groups()
        return {
            "lut": lut,
            "tile_dimensions": subset,
            "physical_tile_extent": subset.replace("km", " km"),
            "kernel_radius": kernel_radius,
            "land_buffer": "",
            "product_transect": np.nan,
        }

    short_radcor_match = re.search(
        rf"{campaign}_{sensor}_acolite_radcor_tsdsf_(autoLUT|MOD1|MOD2)_(\d+km)_lb([^_]+)_kr(\d+km)(?:_|\\|/)",
        text,
    )
    if short_radcor_match:
        lut, subset, _land_buffer, kernel_radius = short_radcor_match.groups()
        return {
            "lut": lut,
            "tile_dimensions": subset,
            "physical_tile_extent": subset.replace("km", " km"),
            "kernel_radius": kernel_radius,
            "land_buffer": "",
            "product_transect": np.nan,
        }

    acolite_match = re.search(rf"{campaign}_{sensor}_acolite_dsf_(autoLUT|MOD1|MOD2)_([^_\\/]+)(?:_|\\|/)", text)
    if not acolite_match:
        raise ValueError(f"Could not parse product path: {path}")
    lut, tile = acolite_match.groups()
    px_x, px_y = [int(value) for value in tile.split("x")]
    return {
        "lut": lut,
        "tile_dimensions": tile,
        "physical_tile_extent": f"{px_x * resolution / 1000:g}x{px_y * resolution / 1000:g} km",
        "kernel_radius": "",
        "land_buffer": "",
        "product_transect": np.nan,
    }


def grid_index(reference_product, resolution):
    with netCDF4.Dataset(netcdf_path(reference_product)) as dataset:
        lon_name, lat_name = coordinate_variable_names(dataset)
        lon = np.ma.filled(dataset.variables[lon_name][:], np.nan)
        lat = np.ma.filled(dataset.variables[lat_name][:], np.nan)
        if lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        grid_shape = lon.shape
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x, y = transformer.transform(lon.ravel(), lat.ravel())
    valid = np.isfinite(x) & np.isfinite(y)
    valid_indices = np.flatnonzero(valid)
    tree = cKDTree(np.column_stack((x[valid], y[valid])))
    return transformer, tree, valid_indices, grid_shape, FOOTPRINT_RADIUS_PIXELS * resolution


def coordinate_variable_names(dataset):
    if "longitude" in dataset.variables and "latitude" in dataset.variables:
        longitude = dataset.variables["longitude"]
        latitude = dataset.variables["latitude"]
        if len(longitude.shape) == len(latitude.shape) == 2:
            return "longitude", "latitude"
    if "lon" in dataset.variables and "lat" in dataset.variables:
        return "lon", "lat"
    if "longitude" in dataset.variables and "latitude" in dataset.variables:
        return "longitude", "latitude"
    raise KeyError("No lon/lat or longitude/latitude coordinate variables found")


def wavelength_from_variable(name):
    match = re.search(r"(\d{3,4})", str(name))
    return int(match.group(1)) if match else None


def resolve_satellite_variables(dataset, config):
    if config.get("c2rcc_band_mapping"):
        return resolve_c2rcc_variables(dataset, config)

    resolved = {}
    configured = config["variables"]
    available = set(dataset.variables)
    candidates = [
        variable
        for variable in dataset.variables
        if re.match(r"^(Rrs_|Rw|rhow)", str(variable), flags=re.IGNORECASE)
    ]
    for wave, variable in configured.items():
        if variable in available:
            resolved[wave] = variable
            continue
        exact_candidates = [
            candidate
            for candidate in candidates
            if wavelength_from_variable(candidate) == int(wave)
        ]
        if exact_candidates:
            resolved[wave] = sorted(exact_candidates, key=lambda item: (not str(item).startswith("Rrs_"), str(item)))[0]
            continue
        numeric_candidates = [
            (abs(wavelength_from_variable(candidate) - int(wave)), candidate)
            for candidate in candidates
            if wavelength_from_variable(candidate) is not None
        ]
        numeric_candidates = [item for item in numeric_candidates if item[0] <= 2]
        if numeric_candidates:
            resolved[wave] = sorted(numeric_candidates, key=lambda item: (item[0], str(item[1])))[0][1]
            continue
        raise KeyError(f"Could not find satellite variable for wavelength {wave} nm")
    return resolved


def resolve_c2rcc_variables(dataset, config):
    resolved = {}
    sensor_code = config.get("sensor_code")
    variables_lower = {str(variable).lower(): variable for variable in dataset.variables}
    has_s2_band_names = any(name.startswith("rhow_b") or name.startswith("rhown_b") for name in variables_lower)

    if has_s2_band_names:
        band_map = C2RCC_S2_WL_TO_BAND.get(sensor_code, {})
        for wave in config["metric_waves"]:
            band_id = band_map.get(int(wave))
            candidates = []
            if band_id:
                candidates = [f"rhow_{band_id}".lower(), f"rhown_{band_id}".lower()]
            for candidate in candidates:
                if candidate in variables_lower:
                    resolved[wave] = variables_lower[candidate]
                    break
            if wave not in resolved:
                raise KeyError(f"Could not find C2RCC variable for {sensor_code} wavelength {wave} nm")
        return resolved

    band_map = C2RCC_BAND_INDEX.get(sensor_code, {})
    for wave in config["metric_waves"]:
        band_index = band_map.get(int(wave))
        candidates = []
        if band_index:
            candidates = [f"rhow_{band_index}".lower(), f"rhown_{band_index}".lower()]
        for candidate in candidates:
            if candidate in variables_lower:
                resolved[wave] = variables_lower[candidate]
                break
        if wave not in resolved:
            raise KeyError(f"Could not find C2RCC variable for {sensor_code} wavelength {wave} nm")
    return resolved


def satellite_values(dataset, variable, divide_by_pi=False):
    values = np.ma.filled(dataset.variables[variable][:], np.nan).astype(float).ravel()
    if divide_by_pi:
        values = values / np.pi
    return values


def footprint_3x3(flat_index, grid_shape):
    return mc.footprint_3x3(flat_index, grid_shape)


def extract_metrics(product, config, points, tree, valid_indices, grid_shape, transformer, radius):
    point_x, point_y = transformer.transform(points["x-coordinate"].to_numpy(), points["y-coordinate"].to_numpy())
    footprints = []
    for x, y in zip(point_x, point_y):
        distance, tree_index = tree.query([x, y])
        if not np.isfinite(distance) or distance > radius:
            footprints.append(np.array([], dtype=int))
        else:
            footprints.append(footprint_3x3(valid_indices[tree_index], grid_shape))
    with netCDF4.Dataset(netcdf_path(product)) as dataset:
        variables = resolve_satellite_variables(dataset, config)
        values = {
            wave: satellite_values(dataset, variable, bool(config.get("divide_by_pi")))
            for wave, variable in variables.items()
        }
    reference_vectors = []
    estimate_vectors = []
    matchup_rows = []
    for row_index, row in points.iterrows():
        reference = []
        estimate = []
        flat_indices = footprints[row_index]
        matchup_start = len(matchup_rows)
        for wave in config["metric_waves"]:
            insitu = float(row[f"Rrs_{wave}"])
            reference.append(insitu)
            selected = values[wave][flat_indices]
            finite = np.isfinite(selected)
            invalid_count = int((~finite).sum())
            negative_count = int((finite & (selected < 0)).sum())
            finite_count = int(finite.sum())
            satellite_median = float(np.nanmedian(selected)) if finite_count else np.nan
            estimate.append(satellite_median)
            matchup_rows.append({
                "row_index": row_index,
                "fid_ramses": row["fid_ramses"] if "fid_ramses" in row else np.nan,
                "transect_nr": row["transect_nr"] if "transect_nr" in row else np.nan,
                "x_coordinate": row["x-coordinate"],
                "y_coordinate": row["y-coordinate"],
                "wavelength_nm": wave,
                "satellite_variable": variables[wave],
                "insitu_rrs": insitu,
                "satellite_median_rrs": satellite_median,
                "footprint_pixel_count": int(selected.size),
                "satellite_finite_pixel_count": finite_count,
                "satellite_invalid_pixel_count": invalid_count,
                "satellite_negative_pixel_count": negative_count,
                "satellite_has_invalid_pixels": invalid_count > 0,
                "satellite_has_negative_pixels": negative_count > 0,
                "satellite_median_is_invalid": not np.isfinite(satellite_median),
                "satellite_median_is_nonpositive": bool(np.isfinite(satellite_median) and satellite_median <= 0),
                "insitu_is_invalid": not np.isfinite(insitu),
                "insitu_is_nonpositive": bool(np.isfinite(insitu) and insitu <= 0),
            })
        angle = mc.spectral_angle(np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float))
        for matchup_row in matchup_rows[matchup_start:]:
            matchup_row["spectral_angle_deg"] = angle
        reference_vectors.append(reference)
        estimate_vectors.append(estimate)
    reference_array = np.asarray(reference_vectors, dtype=float)
    estimate_array = np.asarray(estimate_vectors, dtype=float)
    if reference_array.size == 0:
        return empty_overall(), [], matchup_rows
    overall = mc.acolite_overall_metrics(reference_array, estimate_array, matchup_rows)
    wavelength_rows = mc.acolite_wavelength_metrics(
        reference_array,
        estimate_array,
        config["metric_waves"],
        variables,
        matchup_rows,
    )
    return overall, wavelength_rows, matchup_rows


def empty_overall():
    return mc.empty_acolite_overall_metrics()


def selected_lut(log_path):
    if not log_path or not Path(log_path).exists():
        return "Selection log unavailable", np.nan, np.nan
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    selected = re.findall(r"Selected (?:model )?ACOLITE-LUT-202110-(MOD[12])", text)
    mod1 = re.findall(r"ACOLITE-LUT-202110-MOD1:\s*([0-9.]+)%", text)
    mod2 = re.findall(r"ACOLITE-LUT-202110-MOD2:\s*([0-9.]+)%", text)
    return (
        selected[-1] if selected else "Selection log unavailable",
        float(mod1[-1]) if mod1 else np.nan,
        float(mod2[-1]) if mod2 else np.nan,
    )


def setting_group_columns(frame):
    columns = ["lut", "tile_dimensions", "physical_tile_extent"]
    for optional in ["kernel_radius", "land_buffer", "product_transect"]:
        if optional in frame.columns and frame[optional].notna().any() and not frame[optional].fillna("").eq("").all():
            columns.append(optional)
    return columns


def setting_label(frame):
    values = frame["lut"].astype(str) + "_" + frame["tile_dimensions"].astype(str) + "_" + frame["physical_tile_extent"].astype(str)
    if "kernel_radius" in frame.columns and not frame["kernel_radius"].fillna("").eq("").all():
        values = values + "_kr" + frame["kernel_radius"].fillna("").astype(str)
    if "product_transect" in frame.columns and not frame["product_transect"].fillna("").eq("").all():
        values = values + "_t" + frame["product_transect"].fillna("").astype(str)
    return values


def write_autolut_comparison(metrics, campaign, output_dir, prefix, log_roots):
    rows = []
    group_columns = ["sensor", "tile_dimensions"]
    for optional in ["kernel_radius", "land_buffer", "product_transect"]:
        if optional in metrics.columns and metrics[optional].notna().any() and not metrics[optional].fillna("").eq("").all():
            group_columns.append(optional)
    for scenario in metrics["scenario"].unique():
        subset_scenario = metrics[metrics.scenario == scenario]
        for group_key, subset in subset_scenario.groupby(group_columns, sort=False, dropna=False):
            group_values = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
            sensor = group_values["sensor"]
            tile = group_values["tile_dimensions"]
            settings = subset.set_index("lut")
            if not {"autoLUT", "MOD1", "MOD2"}.issubset(settings.index):
                continue
            log_path = ""
            for root in log_roots:
                candidate = Path(root) / f"{campaign}_{sensor}_acolite_dsf_autoLUT_{tile}.log"
                if candidate.exists():
                    log_path = str(candidate)
                    break
            chosen, mod1_vote, mod2_vote = selected_lut(log_path)
            auto = settings.loc["autoLUT"]
            mod1 = settings.loc["MOD1"]
            mod2 = settings.loc["MOD2"]
            valid_fixed = [row for row in [mod1, mod2] if row.n_matchups > 0 and np.isfinite(row.msa_percent)]
            valid_fixed.sort(key=lambda row: row.msa_percent)
            best = valid_fixed[0] if valid_fixed else None
            auto_valid = auto.n_matchups > 0 and np.isfinite(auto.msa_percent)
            best_lut = best.name if best is not None else "No valid fixed LUT"
            best_msa = best.msa_percent if best is not None else np.nan
            rows.append({
                "scenario": scenario,
                "sensor": sensor,
                "extent": auto.physical_tile_extent,
                "tile_dimensions": tile,
            "kernel_radius": group_values.get("kernel_radius", ""),
            "product_transect": group_values.get("product_transect", ""),
                "autoLUT_selected": chosen,
                "mod1_tile_win_percent": mod1_vote,
                "mod2_tile_win_percent": mod2_vote,
                "autoLUT_n_matchups": int(auto.n_matchups),
                "autoLUT_MSA_percent": auto.msa_percent if auto_valid else np.nan,
                "MOD1_n_matchups": int(mod1.n_matchups),
                "MOD1_MSA_percent": mod1.msa_percent if mod1.n_matchups > 0 else np.nan,
                "MOD2_n_matchups": int(mod2.n_matchups),
                "MOD2_MSA_percent": mod2.msa_percent if mod2.n_matchups > 0 else np.nan,
                "best_fixed_LUT": best_lut,
                "best_fixed_MSA_percent": best_msa,
                "autoLUT_best_at_extent": "Yes" if auto_valid and np.isclose(auto.msa_percent, best_msa) else "No",
                "autoLUT_selection_matches_best_fixed": "Yes" if chosen == best_lut else "No",
                "selection_log": log_path,
            })
    round_output_metrics(pd.DataFrame(rows)).to_csv(output_dir / f"{prefix}_autolut_selected_vs_fixed_at_same_extent.csv", index=False)


def discover_products(pattern):
    patterns = [str(pattern)]
    pattern_text = str(pattern)
    campaign_level_non_best = pattern_text.replace(f"{os.sep}*{os.sep}", f"{os.sep}*{os.sep}non_best{os.sep}", 1)
    if campaign_level_non_best != pattern_text:
        patterns.append(campaign_level_non_best)
    for product_group in ["sentinel_products", "landsat_products"]:
        for processor_dir in ["acolite_dsf", "acolite_radcor_tsdsf", "acolite_tmart"]:
            marker = f"{os.sep}{product_group}{os.sep}{processor_dir}{os.sep}"
            if marker in pattern_text:
                patterns.append(pattern_text.replace(marker, f"{marker}_non_best_products{os.sep}", 1))
        # c2rcc_radcor / polymer_radcor archive non-primary-best runs into a nested
        # "non_best" subfolder (e.g. .../c2rcc_radcor/non_best/<run>/...), unlike the
        # ACOLITE-family processors above which use a sibling "<processor>_non_best_products"
        # folder -- without this, sensitivity runs for these two processors silently miss
        # every setting except whichever one currently sits in the non-archived slot.
        for processor_dir in ["c2rcc_radcor", "polymer_radcor"]:
            marker = f"{os.sep}{product_group}{os.sep}{processor_dir}{os.sep}"
            if marker in pattern_text:
                patterns.append(pattern_text.replace(marker, f"{marker}non_best{os.sep}", 1))
    products = []
    for candidate in patterns:
        products.extend(glob.glob(candidate))
    return sorted(set(products))


def run_campaign(campaign, definition, output_root):
    output_dir = Path(output_root) / campaign
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    wavelength_rows = []
    matchup_rows = []
    filter_rows = []
    skipped_product_rows = []
    for sensor, config in definition["sensors"].items():
        products = discover_products(config["products"])
        if config.get("allowed_tiles") and "acolite_radcor_tsdsf" not in str(config["products"]):
            products = [
                product for product in products
                if product_label(product, campaign, sensor, config["resolution"])["tile_dimensions"] in config["allowed_tiles"]
            ]
        if not products:
            if definition.get("skip_missing_products"):
                continue
            raise ValueError(f"No products found for {campaign} {sensor}: {config['products']}")
        for scenario in definition["scenarios"]:
            selected_transects = get_scenario_transects(definition, scenario)
            points, input_rows, input_rows_after_selection, retained_rows = read_and_filter_insitu(config, selected_transects)
            filter_rows.append({
                "sensor": sensor,
                "scenario": scenario,
                "input_rows": input_rows,
                "transects": ",".join(str(value) for value in selected_transects),
                "rows_after_transect_coordinate_selection": input_rows_after_selection,
                "rows_after_insitu_filtering": retained_rows,
                "insitu_filtering_applied": bool(config.get("hard_filter_column")),
                "insitu_filter_column": config.get("hard_filter_column", ""),
            })
            for product in products:
                label = product_label(product, campaign, sensor, config["resolution"])
                scenario_transects = selected_transects
                if np.isfinite(label["product_transect"]) and int(label["product_transect"]) not in scenario_transects:
                    continue
                try:
                    transformer, tree, valid_indices, grid_shape, radius = grid_index(product, config["resolution"])
                    overall, per_wave, per_matchup = extract_metrics(product, config, points, tree, valid_indices, grid_shape, transformer, radius)
                except Exception as exc:
                    if not definition.get("skip_unreadable_products"):
                        raise
                    skipped_product_rows.append({
                        "campaign": campaign,
                        "sensor": sensor,
                        "scenario": scenario,
                        "selected_transects": ",".join(str(value) for value in selected_transects),
                        "product": product,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    continue
                common = {
                    "sensor": sensor,
                    "scenario": scenario,
                    "selected_transects": ",".join(str(value) for value in selected_transects),
                    "lut": label["lut"],
                    "tile_dimensions": label["tile_dimensions"],
                    "physical_tile_extent": label["physical_tile_extent"],
                    "kernel_radius": label["kernel_radius"],
                    "product_transect": label["product_transect"],
                    "product": product,
                }
                metric_rows.append({**common, **overall})
                for row in per_wave:
                    wavelength_rows.append({**common, **row})
                for row in per_matchup:
                    matchup_rows.append({**common, **row})
    prefix = campaign.lower()
    if not metric_rows:
        return pd.DataFrame(), pd.DataFrame(filter_rows)
    metrics = pd.DataFrame(metric_rows)
    metrics["rank_msa"] = metrics.groupby(["sensor", "scenario"])["msa_percent"].rank(method="min")
    metrics = metrics.sort_values(["sensor", "scenario", "rank_msa", "sam_deg_median", "rmse"])
    filters = pd.DataFrame(filter_rows)
    wavelengths = pd.DataFrame(wavelength_rows)
    matchups = pd.DataFrame(matchup_rows)
    skipped_products = pd.DataFrame(skipped_product_rows)
    if not wavelengths.empty:
        wavelengths["rank_msa_within_transect_wavelength"] = wavelengths.groupby(
            ["sensor", "scenario", "wavelength_nm"]
        )["msa_percent"].rank(method="min")
        wavelengths = wavelengths.sort_values(["sensor", "scenario", "wavelength_nm", "rank_msa_within_transect_wavelength"])
    round_output_metrics(metrics).to_csv(output_dir / f"{prefix}_acolite_insitu_parameter_ranking.csv", index=False)
    filters.to_csv(output_dir / f"{prefix}_acolite_insitu_filter_counts.csv", index=False)
    round_output_metrics(wavelengths).to_csv(output_dir / f"{prefix}_acolite_all_settings_by_transect_visible_wavelength.csv", index=False)
    matchups.to_csv(output_dir / f"{prefix}_acolite_matchup_flags_by_wavelength.csv", index=False)
    if not skipped_products.empty:
        skipped_products.to_csv(output_dir / f"{prefix}_skipped_unreadable_products.csv", index=False)
    best = metrics.groupby(["sensor", "scenario"], as_index=False).first()
    write_autolut_comparison(metrics, campaign, output_dir, prefix, definition["log_roots"])
    return best, filters


def config(documents, processor="acolite"):
    insitu = documents / "insitu"
    sencast = documents / "eo_data" / "output" / "sencast"
    if processor == "radcor":
        sent = sencast / "sentinel_products" / "acolite_radcor_tsdsf"
        land = sencast / "landsat_products" / "acolite_radcor_tsdsf"
        processor_token = "acolite_radcor_tsdsf"
    elif processor == "tmart":
        sent = sencast / "sentinel_products" / "acolite_tmart"
        land = sencast / "landsat_products" / "acolite_tmart"
        processor_token = "acolite_dsf"
    else:
        sent = sencast / "sentinel_products" / "acolite_dsf"
        land = sencast / "landsat_products" / "acolite_dsf"
        processor_token = "acolite_dsf"

    def product_pattern(root, campaign, sensor, suffix):
        if processor == "radcor":
            return str(root / "*" / f"{campaign}_{sensor}_{processor_token}_*" / "L2ACOLITE" / "L2ACOLITE_*.nc")
        return str(root / "*" / f"{campaign}_{sensor}_{processor_token}_*" / "L2ACOLITE" / "L2ACOLITE_*.nc")

    campaign_config = {
        "CST20230610": {
            "scenarios": ["transect2_only"],
            "log_roots": [sent / "_run_control" / "logs"],
            "sensors": {
                "MSI": {
                    "csv": insitu / "20230610_CST" / "visualization_analysis" / "quality_control" / "20230610_CST_metadata_with_Rrs_conv_S2B_qc.csv",
                    "products": product_pattern(sent, "CST20230610", "MSI", "constance_2023-06-10_2023-06-10"),
                    "resolution": 20.0,
                    "filter_waves": [442, 492, 559, 665, 704, 739, 780, 842, 865],
                    "metric_waves": [442, 492, 559, 665, 704],
                    "variables": {442: "Rrs_442", 492: "Rrs_492", 559: "Rrs_559", 665: "Rrs_665", 704: "Rrs_704"},
                },
                "OLI_TIRS": {
                    "csv": insitu / "20230610_CST" / "visualization_analysis" / "quality_control" / "20230610_CST_metadata_with_Rrs_conv_L8_qc.csv",
                    "products": product_pattern(land, "CST20230610", "OLI_TIRS", "constance_2023-06-10_2023-06-10"),
                    "resolution": 30.0,
                    "filter_waves": [443, 483, 561, 655, 865],
                    "metric_waves": [443, 483, 561, 655],
                    "variables": {443: "Rrs_443", 483: "Rrs_483", 561: "Rrs_561", 655: "Rrs_655"},
                },
            },
        },
        "ZRH20231008": {
            "scenarios": ["transect1_only", "transect2_only", "transect3_only", "transect4_only", "all_suitable_transects"],
            "log_roots": [sent / "_run_control" / "ZRH20231008" / "logs"],
            "sensors": {
                "MSI": {
                    "csv": insitu / "20231008_ZRH" / "visualization_analysis" / "quality_control" / "20231008_ZRH_metadata_with_Rrs_conv_S2B_qc.csv",
                    "products": product_pattern(sent, "ZRH20231008", "MSI", "zurich_2023-10-08_2023-10-08"),
                    "resolution": 20.0,
                    "filter_waves": [442, 492, 559, 665, 704, 739, 780, 842, 865],
                    "metric_waves": [442, 492, 559, 665, 704],
                    "variables": {442: "Rrs_442", 492: "Rrs_492", 559: "Rrs_559", 665: "Rrs_665", 704: "Rrs_704"},
                },
                "OLI_TIRS": {
                    "csv": insitu / "20231008_ZRH" / "visualization_analysis" / "quality_control" / "20231008_ZRH_metadata_with_Rrs_conv_L9_qc.csv",
                    "products": product_pattern(land, "ZRH20231008", "OLI_TIRS", "zurich_2023-10-08_2023-10-08"),
                    "resolution": 30.0,
                    "filter_waves": [443, 482, 561, 654, 865],
                    "metric_waves": [443, 482, 561, 654],
                    "variables": {443: "Rrs_443", 482: "Rrs_482", 561: "Rrs_561", 654: "Rrs_654"},
                },
            },
        },
        "BIE20240618": {
            "scenarios": ["transect1_only"],
            "log_roots": [land / "_run_control" / "BIE20240618" / "logs"],
            "sensors": {
                "OLI_TIRS": {
                    "csv": insitu / "20240618_BIE" / "visualization_analysis" / "quality_control" / "20240618_BIE_metadata_with_Rrs_conv_L9_qc.csv",
                    "products": product_pattern(land, "BIE20240618", "OLI_TIRS", "biel_20240618_entire_2024-06-18_2024-06-18"),
                    "resolution": 30.0,
                    "filter_waves": [443, 482, 561, 654, 865],
                    "metric_waves": [443, 482, 561, 654],
                    "variables": {443: "Rrs_443", 482: "Rrs_482", 561: "Rrs_561", 654: "Rrs_654"},
                }
            },
        },
        "WAL20240619": {
            "scenarios": ["transect2_only", "transect4_only"],
            "log_roots": [sent / "_run_control" / "WAL20240619" / "logs"],
            "sensors": {
                "MSI": {
                    "csv": insitu / "20240619_WAL" / "visualization_analysis" / "quality_control" / "20240619_WAL_metadata_with_Rrs_conv_S2A_qc.csv",
                    "products": product_pattern(sent, "WAL20240619", "MSI", "walen_20240619_entire_2024-06-19_2024-06-19"),
                    "resolution": 20.0,
                    "filter_waves": [443, 492, 560, 665, 704, 740, 783, 842, 865],
                    "metric_waves": [443, 492, 560, 665, 704],
                    "variables": {443: "Rrs_443", 492: "Rrs_492", 560: "Rrs_560", 665: "Rrs_665", 704: "Rrs_704"},
                }
            },
        },
        "CST20250303": {
            "scenarios": ["transect1_only"],
            "log_roots": [land / "_run_control" / "CST20250303" / "logs"],
            "sensors": {
                "OLI_TIRS": {
                    "csv": insitu / "20250303_CST" / "visualization_analysis" / "quality_control" / "20250303_CST_metadata_with_Rrs_conv_L9_qc.csv",
                    "products": product_pattern(land, "CST20250303", "OLI_TIRS", "constance_2025-03-03_2025-03-03"),
                    "resolution": 30.0,
                    "filter_waves": [443, 482, 561, 654, 865],
                    "metric_waves": [443, 482, 561, 654],
                    "variables": {443: "Rrs_443", 482: "Rrs_482", 561: "Rrs_561", 654: "Rrs_654"},
                }
            },
        },
        "ZRH20260226": {
            "scenarios": ["transect1_only", "transect2_only", "transect3_only", "transect4_only", "all_suitable_transects"],
            "log_roots": [land / "_run_control" / "ZRH20260226" / "logs"],
            "sensors": {
                "OLI_TIRS": {
                    "csv": insitu / "20260226_ZRH" / "visualization_analysis" / "quality_control" / "20260226_ZRH_metadata_with_Rrs_conv_L8_qc.csv",
                    "products": product_pattern(land, "ZRH20260226", "OLI_TIRS", "zurich_2026-02-26_2026-02-26"),
                    "resolution": 30.0,
                    "filter_waves": [443, 483, 561, 655, 865],
                    "metric_waves": [443, 483, 561, 655],
                    "variables": {443: "Rrs_443", 483: "Rrs_483", 561: "Rrs_561", 655: "Rrs_655"},
                }
            },
        },
        "ZRH20260227": {
            "scenarios": ["transect1_only", "transect2_only", "transect3_only", "transect4_only", "all_suitable_transects"],
            "log_roots": [sent / "_run_control" / "ZRH20260227" / "logs"],
            "sensors": {
                "MSI": {
                    "csv": insitu / "20260227_ZRH" / "visualization_analysis" / "quality_control" / "20260227_ZRH_metadata_with_Rrs_conv_S2B_qc.csv",
                    "products": product_pattern(sent, "ZRH20260227", "MSI", "zurich_2026-02-27_2026-02-27"),
                    "resolution": 20.0,
                    "filter_waves": [442, 492, 559, 665, 704, 739, 780, 842, 865],
                    "metric_waves": [442, 492, 559, 665, 704],
                    "variables": {442: "Rrs_442", 492: "Rrs_492", 559: "Rrs_559", 665: "Rrs_665", 704: "Rrs_704"},
                }
            },
        },
        "ZRH20260423": {
            "scenarios": ["transect1_only", "transect2_only", "transect3_only", "transect4_only", "all_suitable_transects"],
            "log_roots": [land / "_run_control" / "ZRH20260423" / "logs"],
            "sensors": {
                "OLI_TIRS": {
                    "csv": insitu / "20260423_ZRH" / "visualization_analysis" / "quality_control" / "20260423_ZRH_metadata_with_Rrs_conv_L9_qc.csv",
                    "products": product_pattern(land, "ZRH20260423", "OLI_TIRS", "zurich_2026-04-23_2026-04-23"),
                    "resolution": 30.0,
                    "filter_waves": [443, 482, 561, 654, 865],
                    "metric_waves": [443, 482, 561, 654],
                    "variables": {443: "Rrs_443", 482: "Rrs_482", 561: "Rrs_561", 654: "Rrs_654"},
                }
            },
        },
        "ZRH20260430": {
            "scenarios": ["transect1_only", "transect2_only", "transect3_only", "transect4_only", "transect5_only", "all_suitable_transects"],
            "log_roots": [land / "_run_control" / "ZRH20260430" / "logs"],
            "sensors": {
                "OLI_TIRS": {
                    "csv": insitu / "20260430_ZRH" / "visualization_analysis" / "quality_control" / "20260430_ZRH_metadata_with_Rrs_conv_L9_qc.csv",
                    "products": product_pattern(land, "ZRH20260430", "OLI_TIRS", "zurich_2026-04-30_2026-04-30"),
                    "resolution": 30.0,
                    "filter_waves": [443, 482, 561, 654, 865],
                    "metric_waves": [443, 482, 561, 654],
                    "variables": {443: "Rrs_443", 482: "Rrs_482", 561: "Rrs_561", 654: "Rrs_654"},
                }
            },
        },
    }

    for definition in campaign_config.values():
        for sensor, sensor_config in definition["sensors"].items():
            sensor_config["allowed_tiles"] = ALLOWED_TILES_BY_SENSOR[sensor].copy()

    return campaign_config
