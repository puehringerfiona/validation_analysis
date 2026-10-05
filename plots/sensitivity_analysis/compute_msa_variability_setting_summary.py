"""Build the per-setting MSA variability tables used downstream by
plot_settings_variability.py.

This used to also render its own box+swarm figures (plot_variability,
plot_variability_by_campaign), but those were superseded by the consolidated,
processor-faceted figures in plot_settings_variability.py (see that script's
docstring and build_visualization_website.py's infer_sensitivity_asset(),
which no longer registers the per-campaign msa_variability_by_processor_sensor_
<campaign>.png files this script used to produce). Only the table-building
half survives here, since plot_settings_variability.py still reads
msa_variability_setting_summary.csv as its input.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path(r"C:\Users\puehrifi\Documents\results\stats\sensitivity_analysis")
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\puehrifi\Documents\results\plots\sensitivity_analysis")
FILTER_VARIANTS = ("hard_filter_retain", "hard_filter_retain_strict")
PROCESSOR_FOLDERS = {
    "ACOLITE": "acolite",
    "ACOLITE_TMart": "acolite_tmart",
    "ACOLITE_RAdCor": "acolite_radcor",
    "C2RCC_RAdCor": "c2rcc_radcor",
    "Polymer_RAdCor": "polymer_radcor",
}
PROCESSOR_LABELS = {
    "acolite": "ACOLITE DSF",
    "acolite_tmart": "T-Mart+ACOLITE DSF",
    "acolite_radcor": "ACOLITE TSDSF+RAdCor",
    "c2rcc_radcor": "RAdCor+C2RCC",
    "polymer_radcor": "RAdCor+POLYMER",
}
KERNEL_PROCESSORS = {"acolite_radcor", "c2rcc_radcor", "polymer_radcor"}
LUTS = ("MOD1", "MOD2")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def read_csv_auto(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.readline()
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=delimiter)


def lake_from_campaign(campaign: str) -> str:
    match = re.match(r"([A-Za-z]+)", campaign)
    return match.group(1).upper() if match else campaign[:3].upper()


def parse_extent_km(value: object) -> float:
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not numbers:
        return math.nan
    return float(numbers[0])


def setting_label(row: pd.Series) -> str:
    lut = str(row["lut"]).strip()
    if row["processor"] in KERNEL_PROCESSORS:
        return f"{lut} | {str(row.get('kernel_radius', '')).strip()}"
    extent = row.get("extent_km")
    if pd.notna(extent):
        return f"{lut} | {float(extent):g} km"
    return f"{lut} | {str(row.get('physical_tile_extent', '')).strip()}"


def load_all_settings(results_root: Path, variant: str) -> pd.DataFrame:
    frames = []
    variant_root = results_root / variant
    for processor_folder, processor in PROCESSOR_FOLDERS.items():
        processor_root = variant_root / processor_folder
        for path in sorted(processor_root.glob("*/*_all_settings_by_transect_visible_wavelength.csv")):
            frame = read_csv_auto(path)
            if frame.empty:
                continue
            campaign = path.parent.name
            frame["filter_variant"] = variant
            frame["processor_folder"] = processor_folder
            frame["processor"] = processor
            frame["processor_label"] = PROCESSOR_LABELS[processor]
            frame["campaign"] = campaign
            frame["lake"] = lake_from_campaign(campaign)
            frame["source_table"] = str(path)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No all-settings CSVs found under {variant_root}")

    data = pd.concat(frames, ignore_index=True)
    for column in ["msa_percent", "wavelength_nm", "n_matchups"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["kernel_radius", "tile_dimensions", "physical_tile_extent", "land_buffer", "product_transect"]:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    data = data[data["lut"].isin(LUTS) & data["msa_percent"].notna()].copy()
    data["extent_km"] = data["physical_tile_extent"].map(parse_extent_km)
    data.loc[data["processor"].isin(KERNEL_PROCESSORS), "extent_km"] = np.nan
    data["setting_label"] = data.apply(setting_label, axis=1)
    return data


def summarize_settings(data: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "filter_variant",
        "lake",
        "campaign",
        "sensor",
        "processor",
        "processor_label",
        "lut",
        "extent_km",
        "kernel_radius",
        "setting_label",
    ]
    summary = (
        data.groupby(group_columns, dropna=False)
        .agg(
            setting_median_msa_percent=("msa_percent", "median"),
            setting_mean_msa_percent=("msa_percent", "mean"),
            setting_p90_msa_percent=("msa_percent", lambda values: values.quantile(0.9)),
            n_wavelength_rows=("msa_percent", "size"),
            n_matchups_total=("n_matchups", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values(["sensor", "processor", "lake", "campaign", "setting_median_msa_percent"])


def summarize_variability(setting_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, processor, sensor), group in setting_summary.groupby(["filter_variant", "processor", "sensor"]):
        values = group["setting_median_msa_percent"].dropna().to_numpy(float)
        if values.size == 0:
            continue
        rows.append(
            {
                "filter_variant": variant,
                "processor": processor,
                "processor_label": PROCESSOR_LABELS.get(processor, processor),
                "sensor": sensor,
                "n_campaign_settings": int(values.size),
                "best_setting_median_msa_percent": float(np.min(values)),
                "median_setting_msa_percent": float(np.median(values)),
                "worst_setting_median_msa_percent": float(np.max(values)),
                "iqr_setting_msa_percent": float(np.percentile(values, 75) - np.percentile(values, 25)),
                "p90_minus_p10_setting_msa_percent": float(np.percentile(values, 90) - np.percentile(values, 10)),
            }
        )
    return pd.DataFrame(rows).sort_values(["sensor", "processor"])


def write_outputs(results_root: Path, output_root: Path, variant: str) -> list[Path]:
    output_dir = output_root / variant
    data = load_all_settings(results_root, variant)
    setting_summary = summarize_settings(data)
    variability = summarize_variability(setting_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    setting_summary_path = output_dir / "msa_variability_setting_summary.csv"
    variability_path = output_dir / "msa_variability_summary_by_processor_sensor.csv"
    setting_summary.to_csv(setting_summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    variability.to_csv(variability_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return [setting_summary_path, variability_path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MSA variability tables across atmospheric-processor sensitivity settings."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="+", default=list(FILTER_VARIANTS), choices=FILTER_VARIANTS)
    args = parser.parse_args()

    outputs = []
    for variant in args.variants:
        outputs.extend(write_outputs(args.results_root, args.output_root, variant))

    print("Wrote:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
