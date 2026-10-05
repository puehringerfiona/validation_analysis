"""Quantify how much median MSA varies across the tested settings grid (LUT x
extent/kernel-radius) per campaign, for all five processor combinations, to support
a results section on whether tuning is worthwhile and which processors are most
robust to settings choice.

For each processor x campaign: best = min median MSA across all tested settings,
worst = max median MSA across all tested settings, ratio = worst / best,
range_pts = worst - best (percentage points, since MSA is already a percent).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(r"C:\Users\puehrifi\Documents\results\stats\sensitivity_analysis\hard_filter_retain")

CAMPAIGNS = [
    "CST20230610",
    "ZRH20231008",
    "BIE20240618",
    "WAL20240619",
    "CST20250303",
    "ZRH20260226",
    "ZRH20260227",
    "ZRH20260423",
    "ZRH20260430",
]

PROCESSORS = {
    "ACOLITE": "extent",
    "ACOLITE_TMart": "extent",
    "ACOLITE_RAdCor": "kernel_radius",
    "C2RCC_RAdCor": "kernel_radius",
    "Polymer_RAdCor": "kernel_radius",
}

LUTS = ["MOD1", "MOD2"]


def read_csv_auto(path: Path) -> pd.DataFrame:
    header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    separator = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=separator)


def load_campaign(processor_folder: str, campaign: str, setting_kind: str) -> pd.DataFrame | None:
    campaign_dir = RESULTS_ROOT / processor_folder / campaign
    matches = list(campaign_dir.glob("*_acolite_all_settings_by_transect_visible_wavelength.csv"))
    if not matches:
        return None
    frame = read_csv_auto(matches[0])
    if "lut" not in frame.columns:
        return None
    frame = frame[frame["lut"].isin(LUTS)].copy()
    if frame.empty:
        return None
    frame["msa_percent"] = pd.to_numeric(frame["msa_percent"], errors="coerce")
    if setting_kind == "extent":
        frame["setting"] = frame["physical_tile_extent"].astype(str).str.extract(r"([\d.]+)").astype(float)
    else:
        frame["setting"] = frame["kernel_radius"].astype(str).str.extract(r"([\d.]+)").astype(float)
    return frame


def main():
    rows = []
    for processor_folder, setting_kind in PROCESSORS.items():
        for campaign in CAMPAIGNS:
            frame = load_campaign(processor_folder, campaign, setting_kind)
            if frame is None:
                continue
            per_setting = frame.groupby(["lut", "setting"])["msa_percent"].median().dropna()
            if per_setting.empty:
                continue
            best = per_setting.min()
            worst = per_setting.max()
            best_key = per_setting.idxmin()
            worst_key = per_setting.idxmax()
            rows.append(
                {
                    "processor": processor_folder,
                    "campaign": campaign,
                    "n_settings_tested": len(per_setting),
                    "best_median_msa": round(best, 2),
                    "best_setting": best_key,
                    "worst_median_msa": round(worst, 2),
                    "worst_setting": worst_key,
                    "ratio_worst_best": round(worst / best, 2) if best > 0 else float("nan"),
                    "range_pts": round(worst - best, 2),
                }
            )
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(table.to_string(index=False))

    print("\n=== Per-processor summary across campaigns (median MSA ratio worst/best) ===")
    summary = table.groupby("processor").agg(
        n_campaigns=("campaign", "count"),
        median_ratio=("ratio_worst_best", "median"),
        min_ratio=("ratio_worst_best", "min"),
        max_ratio=("ratio_worst_best", "max"),
        median_range_pts=("range_pts", "median"),
        min_range_pts=("range_pts", "min"),
        max_range_pts=("range_pts", "max"),
    )
    print(summary.to_string())

    out_path = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\results\plots\sensitivity_analysis\settings_variation_summary.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
