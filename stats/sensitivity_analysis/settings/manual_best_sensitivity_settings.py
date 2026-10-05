from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


CSV_PATH = Path(__file__).with_name("manual_best_sensitivity_settings.csv")

PROCESSOR_ALIASES = {
    "acolite": "acolite",
    "acolite_dsf": "acolite",
    "tmart": "acolite_tmart",
    "acolite_tmart": "acolite_tmart",
    "acolite_t_mart": "acolite_tmart",
    "radcor": "acolite_radcor",
    "acolite_radcor": "acolite_radcor",
    "c2rcc_radcor": "c2rcc_radcor",
    "polymer_radcor": "polymer_radcor",
}

MANUAL_BEST_SETTINGS: tuple[dict[str, Any], ...] = (
    {"lake": "BIE", "processor": "acolite", "processor_label": "ACOLITE", "parameter_family": "lut_dsf_extent", "lut": "MOD2", "dsf_extent_km": 5.0, "kernel_radius": "", "msi_tile_dimensions": "250x250", "oli_tirs_tile_dimensions": "167x167", "setting_label": "MOD2 | 5.0 km"},
    {"lake": "BIE", "processor": "acolite_tmart", "processor_label": "ACOLITE+T-Mart", "parameter_family": "lut_dsf_extent", "lut": "MOD2", "dsf_extent_km": 5.0, "kernel_radius": "", "msi_tile_dimensions": "250x250", "oli_tirs_tile_dimensions": "167x167", "setting_label": "MOD2 | 5.0 km"},
    {"lake": "BIE", "processor": "acolite_radcor", "processor_label": "ACOLITE+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD2", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD2 | 5km"},
    {"lake": "BIE", "processor": "c2rcc_radcor", "processor_label": "C2RCC+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD2", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD2 | 5km"},
    {"lake": "BIE", "processor": "polymer_radcor", "processor_label": "Polymer+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD2", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD2 | 5km"},
    {"lake": "CST", "processor": "acolite", "processor_label": "ACOLITE", "parameter_family": "lut_dsf_extent", "lut": "MOD1", "dsf_extent_km": 0.5, "kernel_radius": "", "msi_tile_dimensions": "25x25", "oli_tirs_tile_dimensions": "17x17", "setting_label": "MOD1 | 0.5 km"},
    {"lake": "CST", "processor": "acolite_tmart", "processor_label": "ACOLITE+T-Mart", "parameter_family": "lut_dsf_extent", "lut": "MOD1", "dsf_extent_km": 0.5, "kernel_radius": "", "msi_tile_dimensions": "25x25", "oli_tirs_tile_dimensions": "17x17", "setting_label": "MOD1 | 0.5 km"},
    {"lake": "CST", "processor": "acolite_radcor", "processor_label": "ACOLITE+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
    {"lake": "CST", "processor": "c2rcc_radcor", "processor_label": "C2RCC+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
    {"lake": "CST", "processor": "polymer_radcor", "processor_label": "Polymer+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
    {"lake": "WAL", "processor": "acolite", "processor_label": "ACOLITE", "parameter_family": "lut_dsf_extent", "lut": "MOD1", "dsf_extent_km": 1.0, "kernel_radius": "", "msi_tile_dimensions": "50x50", "oli_tirs_tile_dimensions": "33x33", "setting_label": "MOD1 | 1.0 km"},
    {"lake": "WAL", "processor": "acolite_tmart", "processor_label": "ACOLITE+T-Mart", "parameter_family": "lut_dsf_extent", "lut": "MOD1", "dsf_extent_km": 1.0, "kernel_radius": "", "msi_tile_dimensions": "50x50", "oli_tirs_tile_dimensions": "33x33", "setting_label": "MOD1 | 1.0 km"},
    {"lake": "WAL", "processor": "acolite_radcor", "processor_label": "ACOLITE+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
    {"lake": "WAL", "processor": "c2rcc_radcor", "processor_label": "C2RCC+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
    {"lake": "WAL", "processor": "polymer_radcor", "processor_label": "Polymer+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
    {"lake": "ZRH", "processor": "acolite", "processor_label": "ACOLITE", "parameter_family": "lut_dsf_extent", "lut": "MOD1", "dsf_extent_km": 5.0, "kernel_radius": "", "msi_tile_dimensions": "250x250", "oli_tirs_tile_dimensions": "167x167", "setting_label": "MOD1 | 5.0 km"},
    {"lake": "ZRH", "processor": "acolite_tmart", "processor_label": "ACOLITE+T-Mart", "parameter_family": "lut_dsf_extent", "lut": "MOD1", "dsf_extent_km": 5.0, "kernel_radius": "", "msi_tile_dimensions": "250x250", "oli_tirs_tile_dimensions": "167x167", "setting_label": "MOD1 | 5.0 km"},
    {"lake": "ZRH", "processor": "acolite_radcor", "processor_label": "ACOLITE+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
    {"lake": "ZRH", "processor": "c2rcc_radcor", "processor_label": "C2RCC+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
    {"lake": "ZRH", "processor": "polymer_radcor", "processor_label": "Polymer+RAdCor", "parameter_family": "lut_kernel_radius", "lut": "MOD1", "dsf_extent_km": "", "kernel_radius": "5km", "msi_tile_dimensions": "", "oli_tirs_tile_dimensions": "", "setting_label": "MOD1 | 5km"},
)


def canonical_processor(processor: str) -> str:
    key = processor.strip().lower().replace("+", "_").replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return PROCESSOR_ALIASES.get(key, key)


def iter_best_settings() -> tuple[dict[str, Any], ...]:
    return MANUAL_BEST_SETTINGS


def settings_by_lake_processor() -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["lake"], row["processor"]): row for row in MANUAL_BEST_SETTINGS}


def get_best_setting(lake: str, processor: str) -> dict[str, Any]:
    key = (lake.strip().upper(), canonical_processor(processor))
    try:
        return settings_by_lake_processor()[key]
    except KeyError as exc:
        valid = ", ".join(f"{lake_id}/{proc}" for lake_id, proc in sorted(settings_by_lake_processor()))
        raise KeyError(f"No manual best setting for {key[0]}/{key[1]}. Valid pairs: {valid}") from exc


def write_csv(path: Path = CSV_PATH) -> None:
    fieldnames = [
        "lake",
        "processor",
        "processor_label",
        "parameter_family",
        "lut",
        "dsf_extent_km",
        "kernel_radius",
        "msi_tile_dimensions",
        "oli_tirs_tile_dimensions",
        "setting_label",
        "source",
    ]
    rows = [dict(row, source="manual_selection_from_sensitivity_analysis") for row in MANUAL_BEST_SETTINGS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the manually selected best sensitivity settings table.")
    parser.add_argument("--output", type=Path, default=CSV_PATH, help="Output CSV path.")
    args = parser.parse_args()
    write_csv(args.output)
    print(f"Wrote {len(MANUAL_BEST_SETTINGS)} manual best settings to {args.output}")


if __name__ == "__main__":
    main()
