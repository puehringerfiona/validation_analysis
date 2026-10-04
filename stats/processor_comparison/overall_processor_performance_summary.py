"""
Overall processor performance summary (Descriptives, "Overall Processor
Performance" section): median MSA/SB/SAM per processor under two
aggregation schemes.

  - row-weighted: median across every matchup row for that processor
    (pools all campaigns/lakes; heavily-sampled campaigns dominate).
  - campaign-weighted: median of the 9 per-campaign medians (each campaign
    counted once, regardless of how many matchups it contributed).

Lake-weighting was dropped: it reduces to the median of only 4 values (a
noisy statistic on its own), and per-lake performance is already examined
in full in the "Across Lakes" section rather than needing a rough summary
figure here too.

Row-weighted IQR (25th/75th percentile) is also reported, to support the
"widest IQR" claims in the text.
"""
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = RESULTS_ROOT / "stats" / "performance_metrics"
OUT_DIR = Path(__file__).resolve().parent / "overall_performance"

METRICS = ["MSA_percent_overall", "SB_percent_overall", "SAM_deg_overall"]


def load_data() -> pd.DataFrame:
    files = sorted(PERF_DIR.glob("*/*_per_datapoint_metrics_with_per_wavelength_and_overall.csv"))
    frames = [pd.read_csv(f, low_memory=False, usecols=["campaign", "processor", *METRICS]) for f in files]
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    row_w = df.groupby("processor")[metric].median()
    campaign_medians = df.groupby(["processor", "campaign"])[metric].median().reset_index()
    campaign_w = campaign_medians.groupby("processor")[metric].median()
    q25 = df.groupby("processor")[metric].quantile(0.25)
    q75 = df.groupby("processor")[metric].quantile(0.75)
    out = pd.DataFrame({
        "row_weighted": row_w,
        "campaign_weighted": campaign_w,
        "row_weighted_p25": q25,
        "row_weighted_p75": q75,
    })
    out["row_weighted_iqr_width"] = out["row_weighted_p75"] - out["row_weighted_p25"]
    out["row_vs_campaign_divergence"] = (out["campaign_weighted"] - out["row_weighted"]).abs()
    return out.sort_values("row_weighted")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    for metric in METRICS:
        table = summarize(df, metric)
        out_path = OUT_DIR / f"{metric.lower()}_by_weighting.csv"
        table.round(3).to_csv(out_path)
        print(f"=== {metric} ===")
        print(table.round(3).to_string())
        print(f"saved {out_path}\n")


if __name__ == "__main__":
    main()
