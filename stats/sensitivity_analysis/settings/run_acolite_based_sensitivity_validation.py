from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sensitivity_validation_common as validation


DOCUMENTS = Path(r"C:\Users\puehrifi\Documents")
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\puehrifi\Documents\results\sensitivity_analysis\acolite_based")
PROCESSOR_OUTPUT_DIRS = {
    "acolite": "ACOLITE",
    "tmart": "ACOLITE_TMart",
}

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
    token = f"{sensor_source}_products"
    filtered_config = {}
    for campaign, definition in campaign_config.items():
        sensors = {
            sensor: sensor_config
            for sensor, sensor_config in definition["sensors"].items()
            if token in str(sensor_config["products"])
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

def run_processor(processor: str, output_root: Path, args: argparse.Namespace) -> None:
    processor_output_root = output_root / PROCESSOR_OUTPUT_DIRS[processor]
    processor_output_root.mkdir(parents=True, exist_ok=True)

    campaign_config = validation.configure_convolved_qc_inputs(
        validation.config(DOCUMENTS, processor=processor),
        DOCUMENTS,
        hard_filter_column=args.hard_filter_column,
    )
    campaign_config = filter_campaigns(campaign_config, args.campaigns)
    campaign_config = filter_sensor_source(campaign_config, args.sensor_source)
    campaign_config = expand_predefined_transects(campaign_config)
    if args.skip_missing_products:
        campaign_config = {
            campaign: {**definition, "skip_missing_products": True}
            for campaign, definition in campaign_config.items()
        }
    if args.skip_unreadable_products:
        campaign_config = {
            campaign: {**definition, "skip_unreadable_products": True}
            for campaign, definition in campaign_config.items()
        }

    for campaign, definition in campaign_config.items():
        validation.run_campaign(campaign, definition, processor_output_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ACOLITE-family sensitivity validation. This compares MOD1/MOD2 "
            "and DSF extent for plain ACOLITE and ACOLITE+T-Mart products."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--processors",
        nargs="+",
        choices=sorted(PROCESSOR_OUTPUT_DIRS),
        default=["acolite", "tmart"],
    )
    parser.add_argument("--campaigns", nargs="+", help="Campaign IDs to process. Defaults to all configured campaigns.")
    parser.add_argument(
        "--sensor-source",
        choices=["all", "sentinel", "landsat"],
        default="all",
        help="Filter configured sensors by product source. Defaults to all configured sensors.",
    )
    parser.add_argument(
        "--skip-missing-products",
        action="store_true",
        help="Skip configured sensors/campaigns whose product glob has no matching files.",
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

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for processor in args.processors:
        run_processor(processor, output_root, args)


if __name__ == "__main__":
    main()
