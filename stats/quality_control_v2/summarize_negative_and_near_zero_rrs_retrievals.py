import glob
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

PERF_DIR = Path("C:/Users/puehrifi/Documents/AE_personal_migration/results/stats/performance_metrics")

GAAC_RRS_LOWER_BOUND = 1e-5
GAAC_RHOW_LOWER_BOUND = 0.0000314
RRS_TOLERANCE = 1e-10
THRESHOLD = max(GAAC_RRS_LOWER_BOUND, GAAC_RHOW_LOWER_BOUND / math.pi) + RRS_TOLERANCE

files = sorted(PERF_DIR.glob("*/*_per_datapoint_metrics_with_per_wavelength_and_overall.csv"))
print(f"found {len(files)} files")

all_rows = []
matchup_keys = set()
total_pxw = 0
neg_records = []
lowbound_records = []

for f in files:
    df = pd.read_csv(f, low_memory=False)
    wls = sorted({int(m.group(1)) for c in df.columns for m in [re.match(r"sat_rrs_median_(\d+)$", c)] if m})
    # matchup key excludes processor
    for _, row in df.iterrows():
        key = (row.get("campaign"), row.get("sensor"), row.get("insitu_row_id"), row.get("scene_datetime"))
        matchup_keys.add(key)

    for wl in wls:
        col = f"sat_rrs_median_{wl}"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        finite = vals.notna()
        total_pxw += int(finite.sum())
        neg = finite & (vals < 0)
        low = finite & (vals <= THRESHOLD) & (vals >= 0)  # non-negative near-zero lower bound hits
        low_any = finite & (vals <= THRESHOLD)  # includes negatives too, if any <=threshold but also negative

        if neg.any():
            sub = df.loc[neg, ["processor"]].copy()
            sub["wavelength"] = wl
            sub["file"] = f.name
            sub["value"] = vals[neg].values
            neg_records.append(sub)

        if low_any.any():
            sub2 = df.loc[low_any, ["processor"]].copy()
            sub2["wavelength"] = wl
            sub2["file"] = f.name
            sub2["value"] = vals[low_any].values
            lowbound_records.append(sub2)

print(f"\nTotal finite processor x wavelength values: {total_pxw}")
print(f"Total unique matchups (campaign,sensor,insitu_row_id,scene_datetime): {len(matchup_keys)}")

neg_df = pd.concat(neg_records, ignore_index=True) if neg_records else pd.DataFrame(columns=["processor","wavelength","file","value"])
low_df = pd.concat(lowbound_records, ignore_index=True) if lowbound_records else pd.DataFrame(columns=["processor","wavelength","file","value"])

print(f"\n=== NEGATIVE RETRIEVALS ===")
print(f"Total negative values: {len(neg_df)}  ({100*len(neg_df)/total_pxw:.3f}% of all processor x wavelength values)")
print("\nBy processor:")
print(neg_df.groupby("processor").size().sort_values(ascending=False))
print("\nBy wavelength:")
print(neg_df.groupby("wavelength").size().sort_values(ascending=False))
print("\nBy processor x wavelength:")
print(neg_df.groupby(["processor","wavelength"]).size().sort_values(ascending=False))
if len(neg_df):
    wl704_frac = (neg_df["wavelength"] == 704).sum() / len(neg_df)
    print(f"\nFraction of all negatives at 704nm: {100*wl704_frac:.1f}%")

print(f"\n\n=== LOWER-BOUND / NEAR-ZERO-SIGNAL (<= {THRESHOLD}) INCLUDING NEGATIVES ===")
print(f"Total hits: {len(low_df)}  ({100*len(low_df)/total_pxw:.3f}% of all processor x wavelength values)")
print("\nBy processor:")
print(low_df.groupby("processor").size().sort_values(ascending=False))
print("\nBy processor x wavelength (top 20):")
print(low_df.groupby(["processor","wavelength"]).size().sort_values(ascending=False).head(20))

# POLYMER at 704 specifically - fraction of matchups at that band
print("\n\n=== POLYMER @ 704nm lower-bound rate ===")
for f in files:
    df = pd.read_csv(f, low_memory=False)
    if "sat_rrs_median_704" not in df.columns:
        continue
    if "processor" not in df.columns:
        continue
    sub = df[df["processor"].astype(str).str.fullmatch("POLYMER", case=False, na=False)]
    if len(sub) == 0:
        # try contains bare polymer without RAdCor/T-Mart prefix
        continue
    vals = pd.to_numeric(sub["sat_rrs_median_704"], errors="coerce")
    finite = vals.notna()
    if finite.sum() == 0:
        continue
    hits = finite & (vals <= THRESHOLD)
    print(f"{f.name}: POLYMER n={finite.sum()} lowbound_hits={hits.sum()} frac={100*hits.sum()/finite.sum():.1f}%")
