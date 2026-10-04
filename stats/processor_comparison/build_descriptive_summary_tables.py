"""Aggregate per-datapoint descriptive performance metrics into summary tables.

Reads the QC-filtered per-datapoint metric tables (MSA, SB, MAE, RMSE per
wavelength and overall, plus overall SAM) and writes tidy summary tables
(median, IQR, min-max, n) per processor at several grouping levels: overall,
by sensor, by lake, by campaign, and by transect within campaign.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT_ROOT = Path(
    r"C:\Users\puehrifi\Documents\AE_personal_migration\results\stats\performance_metrics"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\puehrifi\Documents\AE_personal_migration\results\stats\processor_comparison"
    r"\descriptive_analysis"
)

# Common spectral band groups shared across MSI and OLI/OLI-TIRS wavelengths,
# matching common_band_group() in plot_adjacency_risk_aec_improvement.py.
BAND_GROUPS: dict[tuple[float, float], str] = {
    (435, 445): "443",
    (475, 495): "490",
    (555, 565): "560",
    (650, 670): "665",
    (698, 710): "704",
}

METRICS = ["MSA_percent", "SB_percent", "MAE", "RMSE"]
ALL_METRICS = METRICS + ["SAM_deg"]

GROUP_LEVELS: dict[str, list[str]] = {
    "overall": ["processor"],
    "by_sensor": ["processor", "sensor_group"],
    "by_lake": ["processor", "lake"],
    "by_campaign": ["processor", "campaign"],
    "by_campaign_transect": ["processor", "campaign", "transect_nr"],
}


def common_band_group(wavelength: float) -> str | float:
    if wavelength is None or not np.isfinite(wavelength):
        return np.nan
    for (lo, hi), label in BAND_GROUPS.items():
        if lo <= wavelength <= hi:
            return label
    return np.nan


def sensor_group(sensor: str) -> str:
    sensor = str(sensor).upper()
    if sensor in {"S2A", "S2B", "MSI"}:
        return "MSI"
    if sensor in {"L8", "L9", "OLI", "OLI_TIRS"}:
        return "OLI_TIRS"
    return "unknown"


def lake_name(row: pd.Series) -> str:
    location = str(row.get("location", "")).strip()
    campaign = str(row.get("campaign", "")).strip()
    label = location if location and location.lower() != "nan" else campaign
    if "," in label:
        label = label.split(",", 1)[0]
    return label.strip()


def load_long_table(input_root: Path) -> pd.DataFrame:
    files = sorted(input_root.glob("*/*_per_datapoint_metrics_with_per_wavelength_and_overall.csv"))
    if not files:
        raise FileNotFoundError(f"No per-datapoint metric CSVs found under {input_root}")

    keep_cols = ["campaign", "location", "sensor", "processor", "transect_nr"]
    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        base = df[keep_cols].copy()
        base["lake"] = df.apply(lake_name, axis=1)
        base["sensor_group"] = base["sensor"].map(sensor_group)

        overall = base.copy()
        overall["band"] = "overall"
        for metric in METRICS:
            overall[metric] = df.get(f"{metric}_overall")
        overall["SAM_deg"] = df.get("SAM_deg_overall")
        frames.append(overall)

        wavelength_cols = [c for c in df.columns if c.startswith("MSA_percent_") and c != "MSA_percent_overall"]
        wavelengths = sorted({c.replace("MSA_percent_", "") for c in wavelength_cols}, key=float)
        for wl in wavelengths:
            band = common_band_group(float(wl))
            if pd.isna(band):
                continue
            sub = base.copy()
            sub["band"] = band
            for metric in METRICS:
                if metric in ("MAE", "RMSE"):
                    # Per-wavelength MAE_λ/RMSE_λ columns are no longer written to
                    # the per-datapoint CSVs: at a single matchup-wavelength pair
                    # both are identical to abs_error_λ, so abs_error_λ supplies
                    # the same population of values (MAE(λ) and RMSE(λ) below are
                    # therefore numerically identical at the per-band grain, same
                    # as before this column was dropped; only the overall/spectrum
                    # band's MAE/RMSE are computed from >1 wavelength and diverge).
                    sub[metric] = df.get(f"abs_error_{wl}")
                else:
                    sub[metric] = df.get(f"{metric}_{wl}")
            sub["SAM_deg"] = np.nan
            frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def campaign_weighted_source(long_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Collapse each campaign to one median value per metric/band first, so a
    heavily-sampled campaign doesn't dominate the cross-campaign summary below
    (mirrors the campaign-weighted approach in overall_processor_performance_summary.py)."""
    per_campaign_cols = group_cols + ["campaign", "band"]
    records = []
    for keys, sub in long_df.groupby(per_campaign_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(per_campaign_cols, keys))
        for metric in ALL_METRICS:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[metric] = values.median() if not values.empty else np.nan
        records.append(row)
    return pd.DataFrame(records)


def summarize(long_df: pd.DataFrame, group_cols: list[str], weighting: str = "campaign") -> pd.DataFrame:
    # "by_campaign"/"by_campaign_transect" are already single-campaign per
    # group, so row-pooling vs. campaign-weighting is moot there. The other
    # levels ("overall", "by_sensor", "by_lake") pool rows from multiple
    # campaigns: by default they're computed from per-campaign medians first
    # (campaign-weighted), so a heavily-sampled campaign doesn't dominate.
    # --weighting row overrides this and pools matchup rows directly instead.
    if "campaign" in group_cols or weighting == "row":
        source_df = long_df
    else:
        source_df = campaign_weighted_source(long_df, group_cols)

    records = []
    for keys, sub in source_df.groupby(group_cols + ["band"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + ["band"], keys))
        for metric in ALL_METRICS:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            prefix = metric
            if values.empty:
                row[f"{prefix}_n"] = 0
                for stat in ("median", "q25", "q75", "min", "max"):
                    row[f"{prefix}_{stat}"] = np.nan
                continue
            row[f"{prefix}_n"] = int(values.size)
            row[f"{prefix}_median"] = values.median()
            row[f"{prefix}_q25"] = values.quantile(0.25)
            row[f"{prefix}_q75"] = values.quantile(0.75)
            row[f"{prefix}_min"] = values.min()
            row[f"{prefix}_max"] = values.max()
        records.append(row)
    summary = pd.DataFrame(records)

    rank_cols = [c for c in group_cols if c != "processor"] + ["band"]
    summary["rank_msa"] = (
        summary.groupby(rank_cols)["MSA_percent_median"].rank(method="min", ascending=True).astype("Int64")
    )
    ordered_cols = group_cols + ["rank_msa", "band"] + [
        c for c in summary.columns if c not in group_cols + ["rank_msa", "band"]
    ]
    summary = summary[ordered_cols]
    return summary.sort_values(rank_cols + ["rank_msa"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--weighting",
        choices=["campaign", "row"],
        default="campaign",
        help=(
            "How 'overall'/'by_sensor'/'by_lake' aggregate across campaigns. "
            "'campaign' (default) takes the median/IQR of each campaign's own median, "
            "so a heavily-sampled campaign doesn't dominate. "
            "'row' pools every matchup row directly instead."
        ),
    )
    args = parser.parse_args()

    long_df = load_long_table(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, group_cols in GROUP_LEVELS.items():
        # Weighting only affects levels that pool rows across multiple campaigns;
        # "by_campaign"/"by_campaign_transect" are already single-campaign per
        # group, so skip writing a redundant duplicate for those under --weighting row.
        weighting_applies = "campaign" not in group_cols
        if args.weighting == "row" and not weighting_applies:
            continue
        suffix = "" if args.weighting == "campaign" else f"_{args.weighting}_weighted"
        summary = summarize(long_df, group_cols, weighting=args.weighting)
        out_path = args.output_dir / f"summary_{name}{suffix}.csv"
        summary.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
