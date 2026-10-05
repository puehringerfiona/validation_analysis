from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt


RESULTS_ROOT = Path(r"C:\Users\puehrifi\Documents\results\stats\sensitivity_analysis")
DEFAULT_PLOT_ROOT = Path(r"C:\Users\puehrifi\Documents\results\plots\sensitivity_analysis")
BINARY_MASK_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu\all\qgis_transect_bb\binary_masks")
DOCUMENTS = Path(r"C:\Users\puehrifi\Documents")
FILTER_VARIANTS = ("hard_filter_retain", "hard_filter_retain_strict")
PROCESSOR_FOLDERS = {
    "ACOLITE": "acolite",
    "ACOLITE_TMart": "acolite_tmart",
    "ACOLITE_RAdCor": "acolite_radcor",
    "C2RCC_RAdCor": "c2rcc_radcor",
    "Polymer_RAdCor": "polymer_radcor",
}
PROCESSOR_LABELS = {
    "acolite": "ACOLITE",
    "acolite_tmart": "ACOLITE+T-Mart",
    "acolite_radcor": "ACOLITE+RAdCor",
    "c2rcc_radcor": "C2RCC+RAdCor",
    "polymer_radcor": "Polymer+RAdCor",
}
KERNEL_PROCESSORS = {"acolite_radcor", "c2rcc_radcor", "polymer_radcor"}
WAVELENGTH_COLORS = {
    442: "#5bc0eb",
    443: "#5bc0eb",
    482: "#2066b2",
    483: "#2066b2",
    490: "#123f8c",
    492: "#123f8c",
    559: "#2ca25f",
    560: "#2ca25f",
    561: "#2ca25f",
    654: "#d7301f",
    655: "#d7301f",
    665: "#b2182b",
    704: "#7a0177",
}


STATS_DIR = Path(__file__).resolve().parents[2] / "stats"
if str(STATS_DIR) not in sys.path:
    sys.path.insert(0, str(STATS_DIR))

import sensitivity_analysis.sensitivity_validation_common as validation


_INSITU_CACHE = {}
_SHORE_DISTANCE_CACHE = {}


def read_csv_auto(path: Path) -> pd.DataFrame:
    header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    separator = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=separator)


def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def lake_from_campaign(campaign: object) -> str:
    match = re.match(r"([A-Za-z]+)", str(campaign))
    return match.group(1).upper() if match else str(campaign)[:3].upper()


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def setting_key(row: pd.Series) -> str:
    lut = str(row.get("lut", "")).strip()
    processor = str(row.get("processor", "")).strip()
    if processor in KERNEL_PROCESSORS:
        return f"{lut} | {str(row.get('kernel_radius', '')).strip()}"
    extent = row.get("extent_km", "")
    if not pd.isna(extent) and str(extent).strip() != "":
        return f"{lut} | {float(extent):g} km"
    extent_text = str(row.get("physical_tile_extent", "")).strip()
    return f"{lut} | {extent_text}"


def add_setting_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "processor" not in frame.columns:
        frame["processor"] = ""
    frame["setting_key"] = frame.apply(setting_key, axis=1)
    return frame


def processor_root(results_root: Path, variant: str, processor_folder: str) -> Path:
    return results_root / variant / processor_folder


def load_best_settings(results_root: Path, variant: str) -> pd.DataFrame:
    frames = []
    for folder, processor in PROCESSOR_FOLDERS.items():
        path = processor_root(results_root, variant, folder) / "best_sensitivity_settings_by_lake_processor.csv"
        if not path.exists():
            continue
        frame = read_csv_auto(path)
        if frame.empty:
            continue
        frame["filter_variant"] = variant
        frame["processor_folder"] = folder
        frame["processor"] = frame.get("processor", processor).fillna(processor)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = numeric(
        data,
        [
            "median_msa",
            "p90_msa",
            "median_spectral_angle_deg",
            "setting_rank",
            "delta_median_msa_from_best",
        ],
    )
    if "is_primary_best" in data.columns:
        data["is_primary_best"] = data["is_primary_best"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        data["is_primary_best"] = data["setting_rank"].eq(1)
    return add_setting_key(data)


def load_setting_scores(results_root: Path, variant: str) -> pd.DataFrame:
    frames = []
    for folder, processor in PROCESSOR_FOLDERS.items():
        path = processor_root(results_root, variant, folder) / "best_sensitivity_settings_by_lake_processor_all_setting_scores.csv"
        if not path.exists():
            continue
        frame = read_csv_auto(path)
        if frame.empty:
            continue
        frame["filter_variant"] = variant
        frame["processor_folder"] = folder
        frame["processor"] = frame.get("processor", processor).fillna(processor)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    return add_setting_key(
        numeric(data, ["median_msa", "p90_msa", "median_spectral_angle_deg", "p90_spectral_angle_deg"])
    )


def load_parameter_rankings(metrics_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(metrics_root.glob("*/*_acolite_insitu_parameter_ranking.csv")):
        frame = read_csv_auto(path)
        if frame.empty:
            continue
        frame["campaign"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_matchups(metrics_root: Path, processor: str, campaigns: list[str] | None = None, with_context: bool = True) -> pd.DataFrame:
    frames = []
    for path in sorted(metrics_root.glob("*/*_acolite_matchup_flags_by_wavelength.csv")):
        campaign = path.parent.name
        if campaigns and campaign not in campaigns:
            continue
        frame = read_csv_auto(path)
        if frame.empty:
            continue
        frame["campaign"] = campaign
        frame["processor"] = processor
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return prepare_matchups(pd.concat(frames, ignore_index=True), with_context=with_context)


def campaign_mask_path(campaign: str) -> Path | None:
    lake = str(campaign)[:3]
    date = str(campaign)[3:]
    exact = BINARY_MASK_ROOT / f"{lake}_{date}_binary.tif"
    if exact.exists():
        return exact
    fallback = BINARY_MASK_ROOT / f"{lake}_mask.tif"
    return fallback if fallback.exists() else None


def shore_distance_sampler(campaign: str):
    if campaign in _SHORE_DISTANCE_CACHE:
        return _SHORE_DISTANCE_CACHE[campaign]

    mask_path = campaign_mask_path(campaign)
    if mask_path is None:
        _SHORE_DISTANCE_CACHE[campaign] = None
        return None

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
    water = np.isfinite(mask) & (mask > 0)
    pixel_size = (abs(transform.a), abs(transform.e))
    distance_m = distance_transform_edt(water, sampling=(pixel_size[1], pixel_size[0]))
    _SHORE_DISTANCE_CACHE[campaign] = {
        "distance_m": distance_m,
        "transform": transform,
        "transformer": Transformer.from_crs("EPSG:4326", crs, always_xy=True),
    }
    return _SHORE_DISTANCE_CACHE[campaign]


def sample_distance_to_shore_km(campaign: str, lon_values, lat_values):
    sampler = shore_distance_sampler(campaign)
    if sampler is None:
        return np.full(len(lon_values), np.nan, dtype=float)
    x, y = sampler["transformer"].transform(np.asarray(lon_values, dtype=float), np.asarray(lat_values, dtype=float))
    rows, cols = rasterio.transform.rowcol(sampler["transform"], x, y)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    data = sampler["distance_m"]
    valid = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    out = np.full(len(rows), np.nan, dtype=float)
    out[valid] = data[rows[valid], cols[valid]] / 1000.0
    return out


def parse_selected_transects(value):
    if pd.isna(value):
        return None
    transects = []
    for part in str(value).split(","):
        text = part.strip()
        if text:
            transects.append(int(float(text)))
    return transects or None


def insitu_metadata_for(campaign: str, sensor: str, selected_transects=None):
    selected_key = "" if selected_transects is None else ",".join(str(value) for value in selected_transects)
    key = (campaign, sensor, selected_key)
    if key in _INSITU_CACHE:
        return _INSITU_CACHE[key]

    campaign_config = validation.configure_convolved_qc_inputs(
        validation.config(DOCUMENTS, processor="acolite"),
        DOCUMENTS,
    )
    if campaign not in campaign_config or sensor not in campaign_config[campaign]["sensors"]:
        _INSITU_CACHE[key] = pd.DataFrame()
        return _INSITU_CACHE[key]

    sensor_config = campaign_config[campaign]["sensors"][sensor]
    transects = selected_transects or campaign_config[campaign].get("scenario_transects", {}).get("predefined_transects")
    if not transects:
        _INSITU_CACHE[key] = pd.DataFrame()
        return _INSITU_CACHE[key]

    frame, *_ = validation.read_and_filter_insitu(sensor_config, transects)
    frame["row_index"] = frame.index
    timestamp_column = "timestamp (UTC)" if "timestamp (UTC)" in frame.columns else "datetime (UTC)"
    frame["insitu_timestamp_utc"] = pd.to_datetime(frame.get(timestamp_column), errors="coerce", utc=True)
    frame["distance_to_shore_km"] = sample_distance_to_shore_km(
        campaign,
        frame["x-coordinate"].to_numpy(),
        frame["y-coordinate"].to_numpy(),
    )
    _INSITU_CACHE[key] = frame[["row_index", "insitu_timestamp_utc", "distance_to_shore_km"]]
    return _INSITU_CACHE[key]


def add_insitu_context(matchups: pd.DataFrame) -> pd.DataFrame:
    context_frames = []
    group_columns = ["campaign", "sensor", "selected_transects"]
    for keys, group in matchups.groupby(group_columns, dropna=False):
        values = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        metadata = insitu_metadata_for(
            values["campaign"],
            values["sensor"],
            selected_transects=parse_selected_transects(values.get("selected_transects")),
        )
        if not metadata.empty:
            context_frames.append(group.merge(metadata, on="row_index", how="left"))
    return pd.concat(context_frames, ignore_index=True) if context_frames else matchups


def prepare_matchups(matchups: pd.DataFrame, with_context: bool = True) -> pd.DataFrame:
    matchups = matchups.copy()
    matchups = matchups[matchups["lut"].isin(["MOD1", "MOD2"])].copy()
    for column in ["kernel_radius", "product_transect", "tile_dimensions", "physical_tile_extent", "selected_transects"]:
        if column not in matchups.columns:
            matchups[column] = ""
        matchups[column] = matchups[column].fillna("").astype(str)
    matchups["lake"] = matchups["campaign"].map(lake_from_campaign)
    matchups["setting_key"] = matchups.apply(setting_key, axis=1)
    for column in ["insitu_rrs", "satellite_median_rrs", "x_coordinate", "y_coordinate", "wavelength_nm", "transect_nr"]:
        if column in matchups.columns:
            matchups[column] = pd.to_numeric(matchups[column], errors="coerce")
    positive = (
        np.isfinite(matchups["insitu_rrs"])
        & np.isfinite(matchups["satellite_median_rrs"])
        & (matchups["insitu_rrs"] > 0)
        & (matchups["satellite_median_rrs"] > 0)
    )
    matchups["msa_pair_percent"] = np.nan
    matchups.loc[positive, "msa_pair_percent"] = (
        100.0
        * (
            np.exp(
                np.abs(
                    np.log(matchups.loc[positive, "satellite_median_rrs"] / matchups.loc[positive, "insitu_rrs"])
                )
            )
            - 1.0
        )
    )
    return add_insitu_context(matchups) if with_context else matchups


def write_best_settings_matrix(best: pd.DataFrame, output_root: Path):
    primary = best[best["is_primary_best"]].copy()
    matrix = primary.pivot_table(index="lake", columns="processor", values="setting_key", aggfunc="first")
    matrix = matrix.rename(columns=PROCESSOR_LABELS)
    matrix.to_csv(output_root / "best_settings_matrix_by_lake_processor.csv")
    return matrix


def plot_best_settings_matrix(matrix: pd.DataFrame, output_root: Path):
    if matrix.empty:
        return []
    fig_width = max(10, 2.2 * len(matrix.columns))
    fig_height = max(2.8, 0.55 * len(matrix.index) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=matrix.fillna("").values,
        rowLabels=matrix.index,
        colLabels=matrix.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    ax.set_title("Primary Best Settings by Lake and Processor", pad=16)
    out = output_root / "best_settings_matrix_by_lake_processor.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out]


def write_processor_family_consistency(best: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    primary = best[best["is_primary_best"]].copy()
    wide = primary.pivot_table(index="lake", columns="processor", values="setting_key", aggfunc="first")
    rows = []
    for lake, row in wide.iterrows():
        acolite_pair = [row.get("acolite"), row.get("acolite_tmart")]
        radcor_family = [row.get("acolite_radcor"), row.get("c2rcc_radcor"), row.get("polymer_radcor")]
        rows.append(
            {
                "lake": lake,
                "acolite": row.get("acolite", ""),
                "acolite_tmart": row.get("acolite_tmart", ""),
                "acolite_family_same_setting": len(set(v for v in acolite_pair if pd.notna(v))) == 1,
                "acolite_radcor": row.get("acolite_radcor", ""),
                "c2rcc_radcor": row.get("c2rcc_radcor", ""),
                "polymer_radcor": row.get("polymer_radcor", ""),
                "radcor_family_same_setting": len(set(v for v in radcor_family if pd.notna(v))) == 1,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "processor_family_best_setting_consistency.csv", index=False)
    return result


def write_setting_sensitivity(scores: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    for (processor, lake), group in scores.groupby(["processor", "lake"], dropna=False):
        msa = pd.to_numeric(group["median_msa"], errors="coerce").dropna()
        p90 = pd.to_numeric(group["p90_msa"], errors="coerce").dropna()
        if msa.empty:
            continue
        rows.append(
            {
                "processor": processor,
                "processor_label": PROCESSOR_LABELS.get(processor, processor),
                "lake": lake,
                "n_settings": int(len(group)),
                "best_median_msa": float(msa.min()),
                "worst_median_msa": float(msa.max()),
                "median_setting_msa": float(msa.median()),
                "msa_range": float(msa.max() - msa.min()),
                "msa_iqr": float(msa.quantile(0.75) - msa.quantile(0.25)),
                "best_p90_msa": float(p90.min()) if not p90.empty else np.nan,
                "worst_p90_msa": float(p90.max()) if not p90.empty else np.nan,
                "p90_msa_range": float(p90.max() - p90.min()) if not p90.empty else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "setting_sensitivity_tuning_value_by_lake_processor.csv", index=False)
    return result


def plot_setting_sensitivity(sensitivity: pd.DataFrame, output_root: Path):
    if sensitivity.empty:
        return []
    outputs = []
    for metric, ylabel in [("msa_range", "Range of median MSA across settings"), ("msa_iqr", "IQR of median MSA across settings")]:
        pivot = sensitivity.pivot(index="lake", columns="processor_label", values=metric)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("Lake")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Setting Sensitivity by Processor ({metric})")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Processor", fontsize=8)
        fig.tight_layout()
        out = output_root / f"setting_sensitivity_{metric}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_aec_stabilization(sensitivity: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    pairs = [("acolite", "acolite_tmart"), ("acolite", "acolite_radcor")]
    indexed = sensitivity.set_index(["processor", "lake"])
    for base_processor, aec_processor in pairs:
        for lake in sorted(sensitivity["lake"].dropna().unique()):
            if (base_processor, lake) not in indexed.index or (aec_processor, lake) not in indexed.index:
                continue
            base = indexed.loc[(base_processor, lake)]
            aec = indexed.loc[(aec_processor, lake)]
            rows.append(
                {
                    "lake": lake,
                    "base_processor": base_processor,
                    "aec_processor": aec_processor,
                    "base_msa_range": base["msa_range"],
                    "aec_msa_range": aec["msa_range"],
                    "delta_msa_range": aec["msa_range"] - base["msa_range"],
                    "base_msa_iqr": base["msa_iqr"],
                    "aec_msa_iqr": aec["msa_iqr"],
                    "delta_msa_iqr": aec["msa_iqr"] - base["msa_iqr"],
                    "aec_reduces_range": aec["msa_range"] < base["msa_range"],
                    "aec_reduces_iqr": aec["msa_iqr"] < base["msa_iqr"],
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "aec_stabilization_vs_acolite.csv", index=False)
    return result


def plot_aec_stabilization(aec: pd.DataFrame, output_root: Path):
    if aec.empty:
        return []
    outputs = []
    for metric in ["delta_msa_range", "delta_msa_iqr"]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, group in aec.groupby("aec_processor"):
            ax.plot(group["lake"], group[metric], marker="o", label=PROCESSOR_LABELS.get(label, label))
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Lake")
        ax.set_ylabel(f"{metric} versus ACOLITE")
        ax.set_title("AEC Effect on Setting Sensitivity")
        ax.grid(alpha=0.25)
        ax.legend(title="AEC processor")
        fig.tight_layout()
        out = output_root / f"aec_stabilization_{metric}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outputs.append(out)
    return outputs


def wavelength_color(wavelength: object) -> str:
    try:
        wave = int(round(float(wavelength)))
    except (TypeError, ValueError):
        return "#666666"
    nearest = min(WAVELENGTH_COLORS, key=lambda item: abs(item - wave))
    return WAVELENGTH_COLORS[nearest]


def filter_campaigns(frame: pd.DataFrame, campaigns: list[str] | None) -> pd.DataFrame:
    if not campaigns or frame.empty or "campaign" not in frame.columns:
        return frame
    return frame[frame["campaign"].isin(campaigns)].copy()


def save_msa_transect_heatmaps(metrics_root: Path, plot_root: Path, processor: str, campaigns: list[str] | None = None):
    matchups = load_matchups(metrics_root, processor, campaigns=campaigns, with_context=True)
    if matchups.empty or "distance_to_shore_km" not in matchups.columns:
        return []
    heatmap_root = plot_root / "msa_transect_heatmaps" / PROCESSOR_LABELS.get(processor, processor)
    heatmap_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    group_columns = ["campaign", "sensor", "selected_transects", "transect_nr", "setting_key"]
    for keys, subset in matchups.groupby(group_columns, dropna=False):
        campaign, sensor, selected_transects, transect_nr, setting = keys
        subset = subset.dropna(subset=["msa_pair_percent", "insitu_timestamp_utc", "distance_to_shore_km", "wavelength_nm"])
        if subset.empty:
            continue
        ordered_points = (
            subset[["row_index", "insitu_timestamp_utc", "distance_to_shore_km"]]
            .drop_duplicates()
            .sort_values(["insitu_timestamp_utc", "row_index"])
            .reset_index(drop=True)
        )
        ordered_points["measurement_order"] = ordered_points.index
        subset = subset.merge(ordered_points[["row_index", "measurement_order"]], on="row_index", how="left")
        matrix = subset.pivot_table(index="measurement_order", columns="wavelength_nm", values="msa_pair_percent", aggfunc="median")
        if matrix.empty:
            continue
        finite_values = matrix.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            continue
        vmax = min(100.0, float(np.nanpercentile(finite_values, 98)))
        fig_height = max(4.8, 0.12 * len(matrix.index))
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="Reds", vmin=0, vmax=max(vmax, 1.0))
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels([int(wl) for wl in matrix.columns], rotation=90)
        y_labels = ordered_points.set_index("measurement_order").reindex(matrix.index)["distance_to_shore_km"].to_numpy(float)
        y_positions = np.linspace(0, len(matrix.index) - 1, min(8, len(matrix.index)), dtype=int)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{y_labels[pos]:.2f}" for pos in y_positions])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Distance to shore (km)")
        ax.set_title(f"{campaign} {sensor} T{transect_nr:g} {setting}\nMSA (%)")
        fig.colorbar(image, ax=ax, label="MSA (%)")
        fig.tight_layout()
        out = heatmap_root / (
            f"{safe_filename(campaign)}_{safe_filename(sensor)}_t{safe_filename(transect_nr)}_"
            f"{safe_filename(setting)}_msa_heatmap.png"
        )
        fig.savefig(out, dpi=260, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def save_matchup_scatter(metrics_root: Path, plot_root: Path, processor: str, campaigns: list[str] | None = None):
    matchups = load_matchups(metrics_root, processor, campaigns=campaigns, with_context=False)
    if matchups.empty:
        return []
    scatter_root = plot_root / "scatter_by_wavelength" / PROCESSOR_LABELS.get(processor, processor)
    scatter_root.mkdir(parents=True, exist_ok=True)
    matchups = matchups.dropna(subset=["insitu_rrs", "satellite_median_rrs", "wavelength_nm"]).copy()
    matchups = matchups[np.isfinite(matchups["insitu_rrs"]) & np.isfinite(matchups["satellite_median_rrs"])].copy()
    if matchups.empty:
        return []
    outputs = []
    for keys, subset in matchups.groupby(["campaign", "sensor", "setting_key"], dropna=False):
        campaign, sensor, setting = keys
        subset = subset.copy()
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.6, 6.2))
        for wavelength, group in subset.groupby("wavelength_nm"):
            ax.scatter(
                group["insitu_rrs"],
                group["satellite_median_rrs"],
                s=18,
                alpha=0.68,
                color=wavelength_color(wavelength),
                label=f"{int(float(wavelength))} nm",
                edgecolors="none",
            )
        values = pd.concat([subset["insitu_rrs"], subset["satellite_median_rrs"]])
        values = values[np.isfinite(values)]
        if not values.empty:
            low = float(values.min())
            high = float(values.max())
            pad = (high - low) * 0.05 if high > low else max(abs(high) * 0.05, 1e-5)
            ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", linewidth=1, linestyle="--")
            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(low - pad, high + pad)
        transects = ",".join(sorted({str(int(float(v))) for v in subset["transect_nr"].dropna().unique()}))
        ax.set_xlabel("In situ Rrs")
        ax.set_ylabel("EO median Rrs")
        ax.set_title(f"{campaign} {sensor} {setting}\nTransects: {transects or 'all available'}")
        ax.grid(alpha=0.25)
        ax.legend(title="Wavelength", fontsize=8, ncol=2)
        fig.tight_layout()
        out = scatter_root / f"{safe_filename(campaign)}_{safe_filename(sensor)}_{safe_filename(setting)}_scatter.png"
        fig.savefig(out, dpi=240, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def build_summary_outputs(results_root: Path, output_root: Path):
    saved = []
    for variant in FILTER_VARIANTS:
        variant_output = output_root / variant
        variant_output.mkdir(parents=True, exist_ok=True)
        best = load_best_settings(results_root, variant)
        scores = load_setting_scores(results_root, variant)
        if best.empty or scores.empty:
            continue
        best.to_csv(variant_output / "best_settings_all_processors_by_lake.csv", index=False)
        scores.to_csv(variant_output / "setting_scores_all_processors_by_lake.csv", index=False)
        matrix = write_best_settings_matrix(best, variant_output)
        consistency = write_processor_family_consistency(best, variant_output)
        sensitivity = write_setting_sensitivity(scores, variant_output)
        aec = write_aec_stabilization(sensitivity, variant_output)
        saved.extend(
            [
                variant_output / "best_settings_all_processors_by_lake.csv",
                variant_output / "setting_scores_all_processors_by_lake.csv",
                variant_output / "best_settings_matrix_by_lake_processor.csv",
                variant_output / "processor_family_best_setting_consistency.csv",
                variant_output / "setting_sensitivity_tuning_value_by_lake_processor.csv",
                variant_output / "aec_stabilization_vs_acolite.csv",
            ]
        )
        saved.extend(plot_best_settings_matrix(matrix, variant_output))
        saved.extend(plot_setting_sensitivity(sensitivity, variant_output))
        saved.extend(plot_aec_stabilization(aec, variant_output))
        consistency.to_csv(variant_output / "processor_family_best_setting_consistency.csv", index=False)
    return saved


def build_diagnostic_plots(
    results_root: Path,
    output_root: Path,
    plot_names: list[str],
    variants: list[str],
    processors: list[str] | None,
    campaigns: list[str] | None,
):
    saved = []
    selected_processors = set(processors or PROCESSOR_FOLDERS)
    for variant in variants:
        variant_output = output_root / variant
        variant_output.mkdir(parents=True, exist_ok=True)
        for folder, processor in PROCESSOR_FOLDERS.items():
            if folder not in selected_processors and processor not in selected_processors:
                continue
            root = processor_root(results_root, variant, folder)
            if not root.exists():
                continue
            if "msa_transect_heatmaps" in plot_names:
                saved.extend(save_msa_transect_heatmaps(root, variant_output, processor, campaigns=campaigns))
            if "matchup_scatter" in plot_names:
                saved.extend(save_matchup_scatter(root, variant_output, processor, campaigns=campaigns))
    return saved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create sensitivity-analysis summary tables and diagnostic plots from existing validation outputs."
    )
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--plot-root", default=str(DEFAULT_PLOT_ROOT))
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["summaries", "msa_transect_heatmaps", "matchup_scatter"],
        default=["summaries"],
        help="Outputs to generate. Summaries are generated by default.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(FILTER_VARIANTS),
        default=list(FILTER_VARIANTS),
        help="Filter variants to process. Defaults to both.",
    )
    parser.add_argument(
        "--processors",
        nargs="+",
        choices=[*PROCESSOR_FOLDERS.keys(), *PROCESSOR_FOLDERS.values()],
        help="Optional processor folders or processor ids for diagnostic plots.",
    )
    parser.add_argument(
        "--campaigns",
        nargs="+",
        help="Optional campaign IDs for diagnostic plots.",
    )
    parser.add_argument("--list-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_plots:
        print("Available outputs:")
        print("  - summaries")
        print("  - msa_transect_heatmaps")
        print("  - matchup_scatter")
        return

    results_root = Path(args.results_root)
    plot_root = Path(args.plot_root)
    plot_root.mkdir(parents=True, exist_ok=True)
    saved = []
    if "summaries" in args.plots:
        saved.extend(build_summary_outputs(results_root, plot_root))
    diagnostic_plots = [plot for plot in args.plots if plot in {"msa_transect_heatmaps", "matchup_scatter"}]
    if diagnostic_plots:
        saved.extend(
            build_diagnostic_plots(
                results_root,
                plot_root,
                diagnostic_plots,
                variants=args.variants,
                processors=args.processors,
                campaigns=args.campaigns,
            )
        )
    print(f"Wrote {len(saved)} outputs under {plot_root}")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
