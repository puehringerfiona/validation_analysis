from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sensitivity_validation_common as validation


DOCUMENTS = Path(r"C:\Users\puehrifi\Documents")
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\puehrifi\Documents\results\sensitivity_analysis\RAdCor")
DEFAULT_SENTINEL_ROOT = Path(
    r"C:\Users\puehrifi\Documents\eo_data\output\sencast\sentinel_products\acolite_radcor_tsdsf"
)
DEFAULT_LANDSAT_ROOT = Path(
    r"C:\Users\puehrifi\Documents\eo_data\output\sencast\landsat_products\acolite_radcor_tsdsf"
)
DEFAULT_POLYMER_SENTINEL_ROOT = Path(
    r"C:\Users\puehrifi\Documents\eo_data\output\sencast\sentinel_products\polymer_radcor"
)
DEFAULT_POLYMER_LANDSAT_ROOT = Path(
    r"C:\Users\puehrifi\Documents\eo_data\output\sencast\landsat_products\polymer_radcor"
)
DEFAULT_C2RCC_SENTINEL_ROOT = Path(
    r"C:\Users\puehrifi\Documents\eo_data\output\sencast\sentinel_products\c2rcc_radcor"
)
DEFAULT_C2RCC_LANDSAT_ROOT = Path(
    r"C:\Users\puehrifi\Documents\eo_data\output\sencast\landsat_products\c2rcc_radcor"
)

def insitu_campaign_name(campaign: str) -> str:
    site = campaign[:3]
    date = campaign[3:]
    return f"{date}_{site}"

def product_glob(sentinel_root: Path, landsat_root: Path, campaign: str, sensor: str, processor: str) -> str:
    product_root = sentinel_root if sensor == "MSI" else landsat_root
    if processor == "polymer_radcor":
        return str(
            product_root
            / f"{campaign}_{sensor}_polymer_radcor_tsdsf_*"
            / "L2POLY"
            / "L2POLY_reproj_*.nc"
        )
    if processor == "c2rcc_radcor":
        return str(
            product_root
            / f"{campaign}_{sensor}_c2rcc_radcor_tsdsf_*"
            / "L2C2RCC"
            / "L2C2RCC_*.nc"
        )
    folder = insitu_campaign_name(campaign)
    return str(
        product_root
        / folder
        / f"{campaign}_{sensor}_acolite_radcor_tsdsf_*"
        / "L2ACOLITE"
        / "L2ACOLITE_*.nc"
    )


def configured_campaigns(sentinel_root: Path, landsat_root: Path, hard_filter_column: str, processor: str) -> dict:
    campaign_config = validation.configure_convolved_qc_inputs(
        validation.config(DOCUMENTS, processor="acolite"),
        DOCUMENTS,
        hard_filter_column=hard_filter_column,
    )
    configured = {}
    for campaign, definition in campaign_config.items():
        sensors = {}
        for sensor, sensor_config in definition["sensors"].items():
            sensors[sensor] = {
                **sensor_config,
                "products": product_glob(sentinel_root, landsat_root, campaign, sensor, processor),
                "allowed_tiles": set(),
                "divide_by_pi": processor in {"polymer_radcor", "c2rcc_radcor"},
                "c2rcc_band_mapping": processor == "c2rcc_radcor",
            }
        configured[campaign] = {
            **definition,
            "sensors": sensors,
            "log_roots": [],
            "skip_missing_products": True,
        }
    return configured


def filter_campaigns(campaign_config: dict, campaigns: list[str] | None) -> dict:
    if not campaigns:
        return campaign_config
    missing = [campaign for campaign in campaigns if campaign not in campaign_config]
    if missing:
        raise ValueError(f"Campaigns not configured: {', '.join(missing)}")
    return {campaign: campaign_config[campaign] for campaign in campaigns}


def filter_sensor_source(campaign_config: dict, sensor_source: str) -> dict:
    if sensor_source == "all":
        return campaign_config
    root_token = "sentinel_products" if sensor_source == "sentinel" else "landsat_products"
    filtered_config = {}
    for campaign, definition in campaign_config.items():
        sensors = {
            sensor: sensor_config
            for sensor, sensor_config in definition["sensors"].items()
            if root_token in str(sensor_config["products"]).lower()
        }
        if sensors:
            filtered_config[campaign] = {**definition, "sensors": sensors}
    return filtered_config


def expand_predefined_transects(campaign_config: dict) -> dict:
    expanded = {}
    for campaign, definition in campaign_config.items():
        transects = definition.get("scenario_transects", {}).get("predefined_transects")
        if not transects:
            expanded[campaign] = definition
            continue
        scenarios = [str(transect) for transect in transects]
        expanded[campaign] = {
            **definition,
            "scenarios": scenarios,
            "scenario_transects": {
                str(transect): [transect]
                for transect in transects
            },
        }
    return expanded


def drop_land_buffer_columns(output_root: Path) -> None:
    for path in sorted(output_root.glob("*/*.csv")):
        try:
            header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            separator = ";" if header.count(";") > header.count(",") else ","
            frame = pd.read_csv(path, sep=separator)
        except pd.errors.EmptyDataError:
            continue
        except IndexError:
            continue
        if "land_buffer" not in frame.columns:
            continue
        frame = frame.drop(columns=["land_buffer"])
        frame.to_csv(path, index=False, sep=separator)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run kernel-based RAdCor sensitivity validation. The current product reader "
            "supports ACOLITE+RAdCor, Polymer+RAdCor, and C2RCC+RAdCor NetCDF products "
            "and ranks MOD1/MOD2 plus kernel radius. "
            "Land buffer is not used as a ranking dimension."
        )
    )
    parser.add_argument(
        "--processor",
        choices=["acolite_radcor", "polymer_radcor", "c2rcc_radcor"],
        default="acolite_radcor",
        help="RAdCor-family product layout to validate. Defaults to ACOLITE+RAdCor.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sentinel-root", default=str(DEFAULT_SENTINEL_ROOT))
    parser.add_argument("--landsat-root", default=str(DEFAULT_LANDSAT_ROOT))
    parser.add_argument("--campaigns", nargs="+", help="Campaign IDs to process. Defaults to all configured campaigns.")
    parser.add_argument(
        "--sensor-source",
        choices=["all", "sentinel", "landsat"],
        default="all",
        help="Filter configured sensors by product source. Defaults to all configured sensors.",
    )
    parser.add_argument(
        "--hard-filter-column",
        default="hard_filter_retain",
        help="In-situ QC boolean column to retain measurements. Use hard_filter_retain_strict for strict filtering.",
    )
    parser.add_argument(
        "--skip-unreadable-products",
        action="store_true",
        help="Skip NetCDF products that cannot be opened and write *_skipped_unreadable_products.csv per campaign.",
    )
    args = parser.parse_args()
    if args.processor == "polymer_radcor":
        if args.sentinel_root == str(DEFAULT_SENTINEL_ROOT):
            args.sentinel_root = str(DEFAULT_POLYMER_SENTINEL_ROOT)
        if args.landsat_root == str(DEFAULT_LANDSAT_ROOT):
            args.landsat_root = str(DEFAULT_POLYMER_LANDSAT_ROOT)
    elif args.processor == "c2rcc_radcor":
        if args.sentinel_root == str(DEFAULT_SENTINEL_ROOT):
            args.sentinel_root = str(DEFAULT_C2RCC_SENTINEL_ROOT)
        if args.landsat_root == str(DEFAULT_LANDSAT_ROOT):
            args.landsat_root = str(DEFAULT_C2RCC_LANDSAT_ROOT)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    campaign_config = configured_campaigns(
        Path(args.sentinel_root),
        Path(args.landsat_root),
        args.hard_filter_column,
        args.processor,
    )
    campaign_config = filter_campaigns(campaign_config, args.campaigns)
    campaign_config = filter_sensor_source(campaign_config, args.sensor_source)
    campaign_config = expand_predefined_transects(campaign_config)
    if args.skip_unreadable_products:
        campaign_config = {
            campaign: {**definition, "skip_unreadable_products": True}
            for campaign, definition in campaign_config.items()
        }

    skipped = []
    for campaign, definition in campaign_config.items():
        has_products = any(glob.glob(sensor_config["products"]) for sensor_config in definition["sensors"].values())
        if not has_products:
            skipped.append(campaign)
            continue
        validation.run_campaign(campaign, definition, output_root)
    if skipped:
        pd.DataFrame({"campaign": skipped}).to_csv(output_root / "skipped_campaigns_no_products.csv", index=False)

    drop_land_buffer_columns(output_root)


if __name__ == "__main__":
    main()
