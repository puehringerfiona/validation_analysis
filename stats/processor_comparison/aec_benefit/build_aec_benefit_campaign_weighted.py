"""Campaign-weighted AEC-benefit tables (Section 3.3.1.6): does adding an AEC
to a given AC base help or hurt, by how much, for which metric -- using the
full dataset, no near-shore restriction (that's a separate, narrower question
already answered by transect_proxy_association.py's near-shore-restricted
delta_MSA/delta_SB/delta_SAM, which asks whether AEC *benefit itself* tracks
land influence, not what the benefit is overall).

Campaign-weighting: median of each campaign's own median, so each campaign
counts once regardless of how many matchups it contributed -- the same
convention used throughout Section 3.3.1 (see plot_lake_wavelength_heatmap.py's
campaign_weighted_median()). Delta convention: AC - AEC, positive =
improvement, matching plot_tiered_delta_slope.py and
plot_delta_msa_per_lake_band.py (see Section 3.3.1.6 Methods note on sign
convention).

Bias is tested via |SB| (magnitude), not signed SB: adjacency contamination
can surface as growing overestimation or deepening underestimation depending
on a processor's baseline direction, so "AEC reduces bias" means it reduces
the magnitude, not that it pushes SB toward a fixed sign (same reasoning as
transect_proxy_association.py). The magnitude is taken as abs() of the
campaign-weighted signed median, not a campaign-weighted median of per-row
abs(SB): the latter would answer "how much does bias swing in either
direction campaign to campaign", a different and noisier quantity than "how
big is the typical (signed) bias magnitude" -- the two coincide only when a
processor's bias never crosses zero across campaigns.

Outputs (written to build_aec_benefit_campaign_weighted/):
  - pairing_summary.csv: one row per AEC pairing with campaign-weighted AC
    value, AEC value, and delta for MSA / |SB| / SAM. This is Table X.
  - pairing_summary_by_band.csv: same delta, MSA only, broken out by band
    group (CA/blue/green/red/red_edge, respecting the near-zero-signal QC
    rejection already applied per matchup). This is Table Y.
  - processor_summary.csv: row-weighted and campaign-weighted MSA_percent_
    overall (median, row-weighted IQR) for all 11 processors, plus their
    row-to-campaign divergence -- supports the GAAC / ACOLITE TSDSF+RAdCor
    architectural-instability discussion later in the section.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PERF_DIR = ROOT / "stats" / "performance_metrics"
OUTPUT_DIR = Path(__file__).resolve().parent / "build_aec_benefit_campaign_weighted"

BAND_GROUP = {442: "CA", 443: "CA", 482: "blue", 483: "blue", 492: "blue",
              559: "green", 560: "green", 561: "green",
              654: "red", 655: "red", 665: "red", 704: "red_edge"}
BAND_ORDER = ["CA", "blue", "green", "red", "red_edge"]

ALL_PROCESSORS = [
    "ACOLITE", "ACOLITE+T-Mart", "RAdCor+ACOLITE", "ACOLITE+RAdCor",
    "Polymer", "Polymer+T-Mart", "Polymer+Radcor",
    "C2RCC", "C2RCC+T-Mart", "C2RCC+RAdCor", "GAAC",
]
# (AC base, AEC pairing) -- T-Mart/RAdCor applied before AC in every pairing
# except ACOLITE+RAdCor (native ACOLITE TSDSF+RAdCor, where ACOLITE's own
# aerosol estimation precedes RAdCor).
AEC_PAIRS = [
    ("ACOLITE", "ACOLITE+T-Mart"),
    ("ACOLITE", "RAdCor+ACOLITE"),
    ("ACOLITE", "ACOLITE+RAdCor"),
    ("Polymer", "Polymer+T-Mart"),
    ("Polymer", "Polymer+Radcor"),
    ("C2RCC", "C2RCC+T-Mart"),
    ("C2RCC", "C2RCC+RAdCor"),
]
DISPLAY_NAME = {
    "ACOLITE": "ACOLITE DSF",
    "ACOLITE+T-Mart": "T-Mart + ACOLITE DSF",
    "RAdCor+ACOLITE": "RAdCor + ACOLITE DSF",
    "ACOLITE+RAdCor": "ACOLITE TSDSF + RAdCor",
    "GAAC": "GAAC",
    "Polymer": "POLYMER",
    "Polymer+T-Mart": "T-Mart + POLYMER",
    "Polymer+Radcor": "RAdCor + POLYMER",
    "C2RCC": "C2RCC",
    "C2RCC+T-Mart": "T-Mart + C2RCC",
    "C2RCC+RAdCor": "RAdCor + C2RCC",
}


def load_full() -> pd.DataFrame:
    files = sorted(PERF_DIR.glob("*/*_per_datapoint_metrics_with_per_wavelength_and_overall.csv"))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(frames, ignore_index=True)


def campaign_weighted_median(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    """Median of each campaign's own median: each campaign counts once
    regardless of matchup count. `group_cols` must include "campaign"."""
    per_campaign = df.groupby(group_cols)[value_col].median()
    other_cols = [c for c in group_cols if c != "campaign"]
    return per_campaign.groupby(other_cols).median() if other_cols else per_campaign.median()


def build_pairing_summary(full: pd.DataFrame) -> pd.DataFrame:
    metrics = {"MSA": "MSA_percent_overall", "SAM": "SAM_deg_overall"}
    cw = {}
    for mname, col in metrics.items():
        cw[mname] = campaign_weighted_median(full, ["processor", "campaign"], col)
    # abs() applied AFTER campaign-weighting the signed value, not before (see
    # module docstring): a processor whose bias crosses zero across campaigns
    # would otherwise get an inflated "magnitude" from abs-ing each campaign
    # first, even though its typical signed bias is small.
    cw_signed_sb = campaign_weighted_median(full, ["processor", "campaign"], "SB_percent_overall")
    cw["abs_SB"] = cw_signed_sb.abs()

    rows = []
    for ac, aec in AEC_PAIRS:
        row = {"ac_processor": ac, "aec_processor": aec, "pairing": DISPLAY_NAME[aec]}
        for mname in metrics:
            ac_val = cw[mname][ac]
            aec_val = cw[mname][aec]
            row[f"{mname}_AC"] = round(ac_val, 4)
            row[f"{mname}_AEC"] = round(aec_val, 4)
            row[f"delta_{mname}"] = round(ac_val - aec_val, 4)
        ac_sb, aec_sb = cw["abs_SB"][ac], cw["abs_SB"][aec]
        row["abs_SB_AC"] = round(ac_sb, 4)
        row["abs_SB_AEC"] = round(aec_sb, 4)
        row["delta_abs_SB"] = round(ac_sb - aec_sb, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairing_summary_by_band(full: pd.DataFrame) -> pd.DataFrame:
    wl_cols = [c for c in full.columns if c.startswith("MSA_percent_") and c != "MSA_percent_overall"]
    rejected_default = pd.Series([""] * len(full), index=full.index)
    rejected = full.get("qc_near_zero_signal_rejected_wavelengths", rejected_default).fillna("").astype(str)

    long_frames = []
    for col in wl_cols:
        wl = int(col.replace("MSA_percent_", ""))
        band = BAND_GROUP.get(wl)
        if band is None:
            continue
        keep = ~rejected.str.split(",").apply(lambda parts, wl=wl: str(wl) in parts)
        piece = full.loc[keep, ["processor", "campaign", col]].rename(columns={col: "MSA"})
        piece["band"] = band
        long_frames.append(piece)
    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["MSA"] = pd.to_numeric(long_df["MSA"], errors="coerce")
    long_df = long_df.dropna(subset=["MSA"])

    cw = campaign_weighted_median(long_df, ["processor", "band", "campaign"], "MSA").unstack()
    cw = cw.reindex(columns=BAND_ORDER)

    rows = []
    for ac, aec in AEC_PAIRS:
        row = {"ac_processor": ac, "aec_processor": aec, "pairing": DISPLAY_NAME[aec]}
        for band in BAND_ORDER:
            ac_val = cw.loc[ac, band]
            aec_val = cw.loc[aec, band]
            row[f"delta_MSA_{band}"] = round(ac_val - aec_val, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_processor_summary(full: pd.DataFrame) -> pd.DataFrame:
    col = "MSA_percent_overall"
    row_weighted = full.groupby("processor")[col].median()
    row_q1 = full.groupby("processor")[col].quantile(0.25)
    row_q3 = full.groupby("processor")[col].quantile(0.75)
    campaign_weighted = campaign_weighted_median(full, ["processor", "campaign"], col)

    rows = []
    for proc in ALL_PROCESSORS:
        rows.append({
            "processor": proc,
            "display_name": DISPLAY_NAME.get(proc, proc),
            "row_weighted_MSA": round(row_weighted[proc], 4),
            "row_weighted_IQR": round(row_q3[proc] - row_q1[proc], 4),
            "campaign_weighted_MSA": round(campaign_weighted[proc], 4),
            "row_to_campaign_divergence": round(abs(campaign_weighted[proc] - row_weighted[proc]), 4),
        })
    return pd.DataFrame(rows).sort_values("campaign_weighted_MSA").reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full = load_full()

    pairing_summary = build_pairing_summary(full)
    pairing_path = OUTPUT_DIR / "pairing_summary.csv"
    pairing_summary.to_csv(pairing_path, index=False)
    print(f"saved {pairing_path}")
    print(pairing_summary.to_string(index=False))

    by_band = build_pairing_summary_by_band(full)
    band_path = OUTPUT_DIR / "pairing_summary_by_band.csv"
    by_band.to_csv(band_path, index=False)
    print(f"\nsaved {band_path}")
    print(by_band.to_string(index=False))

    processor_summary = build_processor_summary(full)
    proc_path = OUTPUT_DIR / "processor_summary.csv"
    processor_summary.to_csv(proc_path, index=False)
    print(f"\nsaved {proc_path}")
    print(processor_summary.to_string(index=False))


if __name__ == "__main__":
    main()
