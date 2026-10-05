"""
Generalizes the previous GAAC-only lower-bound QC rule into a processor-agnostic,
band-wise hard rejection applied directly to the base performance_metrics/ CSVs
(pre-analysis), replacing the old row-wise GAAC-only filter that lived in
`stats/quality control/apply_quality_control.py`.

Rule: for any processor x wavelength, if sat_rrs_<wl> <= threshold (~1e-5 sr^-1,
same physical bound as the original GAAC rule), the retrieval at that band is
noise-floor/non-physical. We null out only that band's derived metric columns
for that row (error_/abs_error_/rel_error_percent_/MSA_percent_/SB_percent_/
MAE_/RMSE_<wl>), leaving other bands of the same matchup intact, then recompute
the row's *_overall columns from the surviving (non-rejected) wavelengths using
the exact same formulas as performance_metrics.py (matchup_common.py).

Modifies files in place under stats/performance_metrics/, after copying the
current tree to a timestamped backup.
"""
import glob
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Users/puehrifi/Documents/AE_personal_migration/results/stats")
import matchup_common as mc

PERF_DIR = Path("C:/Users/puehrifi/Documents/AE_personal_migration/results/stats/performance_metrics")

GAAC_RRS_LOWER_BOUND = 1e-5
GAAC_RHOW_LOWER_BOUND = 0.0000314
RRS_TOLERANCE = 1e-10
THRESHOLD = max(GAAC_RRS_LOWER_BOUND, GAAC_RHOW_LOWER_BOUND / math.pi) + RRS_TOLERANCE

WL_METRIC_COLS = ["error", "abs_error", "rel_error_percent", "MSA_percent", "SB_percent"]


def wavelengths_in(df: pd.DataFrame) -> list[int]:
    import re
    wls = sorted({int(m.group(1)) for c in df.columns for m in [re.match(r"MSA_percent_(\d+)$", c)] if m})
    return wls


def recompute_overall(row: pd.Series, wls: list[int], rejected: set[int]) -> dict:
    y_true, y_pred = [], []
    for wl in wls:
        if wl in rejected:
            continue
        ins = row.get(f"insitu_rrs_{wl}", np.nan)
        sat = row.get(f"sat_rrs_median_{wl}", np.nan)
        if np.isfinite(ins) and np.isfinite(sat):
            y_true.append(ins)
            y_pred.append(sat)
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    m = mc.per_spectrum_metrics(y_true, y_pred)
    return {
        "n_common_wavelengths": int(len(y_true)),
        "positive_pair_count_overall": m["positive_pair_count"],
        "MSA_percent_overall": float(m["msa_percent"]) if np.isfinite(m["msa_percent"]) else np.nan,
        "SB_percent_overall": float(m["sb_percent"]) if np.isfinite(m["sb_percent"]) else np.nan,
        "MAE_overall": float(m["mae"]) if np.isfinite(m["mae"]) else np.nan,
        "RMSE_overall": float(m["rmse"]) if np.isfinite(m["rmse"]) else np.nan,
        "SAM_deg_overall": float(m["spectral_angle_deg"]) if np.isfinite(m["spectral_angle_deg"]) else np.nan,
    }


def process_file(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    wls = wavelengths_in(df)
    n_band_rejections = 0

    rejected_per_row: list[set[int]] = [set() for _ in range(len(df))]
    pos_by_label = {label: i for i, label in enumerate(df.index)}

    for wl in wls:
        sat_col = f"sat_rrs_median_{wl}"
        if sat_col not in df.columns:
            continue
        hit = df[sat_col].notna() & (df[sat_col] <= THRESHOLD)
        if not hit.any():
            continue
        n_band_rejections += int(hit.sum())
        for label in df.index[hit]:
            rejected_per_row[pos_by_label[label]].add(wl)
        for base in WL_METRIC_COLS:
            col = f"{base}_{wl}"
            if col in df.columns:
                df.loc[hit, col] = np.nan

    df["qc_near_zero_signal_rejected_wavelengths"] = [
        ",".join(str(w) for w in sorted(s)) for s in rejected_per_row
    ]

    rows_with_rejection = df.index[df["qc_near_zero_signal_rejected_wavelengths"] != ""]
    n_rows_touched = len(rows_with_rejection)

    for label in rows_with_rejection:
        rejected = rejected_per_row[pos_by_label[label]]
        overall = recompute_overall(df.loc[label], wls, rejected)
        for k, v in overall.items():
            df.at[label, k] = v

    df.to_csv(path, index=False)
    return {
        "file": str(path),
        "n_wavelength_band_rejections": n_band_rejections,
        "n_rows_touched": n_rows_touched,
        "n_rows_total": len(df),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PERF_DIR / f"backup_before_near_zero_signal_hard_filter_{ts}"
    campaign_dirs = [
        p for p in PERF_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("backup_")
    ]
    backup_dir.mkdir()
    for d in campaign_dirs:
        shutil.copytree(d, backup_dir / d.name)
    print(f"backed up {len(campaign_dirs)} campaign dirs to {backup_dir}")

    files = sorted(PERF_DIR.glob("*/*_per_datapoint_metrics_with_per_wavelength_and_overall.csv"))
    print(f"processing {len(files)} files, threshold={THRESHOLD}")

    reports = [process_file(f) for f in files]
    report_df = pd.DataFrame(reports)
    report_path = Path("C:/Users/puehrifi/Documents/AE_personal_migration/results/stats/quality_control_v2/near_zero_signal_hard_filter_report.csv")
    report_path.parent.mkdir(exist_ok=True)
    report_df.to_csv(report_path, index=False)
    print(report_df.to_string(index=False))
    print()
    print("total band-level rejections:", report_df.n_wavelength_band_rejections.sum())
    print("total rows touched (>=1 band rejected):", report_df.n_rows_touched.sum())


if __name__ == "__main__":
    main()
