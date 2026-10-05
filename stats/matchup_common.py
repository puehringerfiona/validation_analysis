import re

import numpy as np
import pandas as pd


def boolean_mask(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def filter_insitu_rows(
    frame,
    transects,
    metric_waves,
    x_col="x-coordinate",
    y_col="y-coordinate",
    transect_col="transect_nr",
    hard_filter_column=None,
    require_hard_filter_column=True,
):
    frame = frame.copy()
    input_rows = len(frame)
    metric_columns = [f"Rrs_{int(wave)}" for wave in metric_waves]
    required = {x_col, y_col, transect_col, *metric_columns}
    if hard_filter_column and (require_hard_filter_column or hard_filter_column in frame.columns):
        required.add(hard_filter_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Input table is missing required columns: {', '.join(missing)}")

    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    frame[transect_col] = pd.to_numeric(frame[transect_col], errors="coerce")
    mask = (
        np.isfinite(frame[x_col])
        & np.isfinite(frame[y_col])
        & frame[transect_col].isin(transects)
    )
    rows_after_transect_coordinate_selection = int(mask.sum())

    hard_filter_applied = False
    if hard_filter_column and hard_filter_column in frame.columns:
        mask = mask & boolean_mask(frame[hard_filter_column])
        hard_filter_applied = True

    frame = frame[mask].copy()
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return {
        "frame": frame.reset_index(drop=True),
        "input_rows": input_rows,
        "rows_after_transect_coordinate_selection": rows_after_transect_coordinate_selection,
        "rows_after_insitu_filtering": len(frame),
        "hard_filter_applied": hard_filter_applied,
    }


def footprint_3x3(flat_index, grid_shape):
    row, col = np.unravel_index(int(flat_index), grid_shape)
    rows = range(max(0, row - 1), min(grid_shape[0], row + 2))
    cols = range(max(0, col - 1), min(grid_shape[1], col + 2))
    return np.asarray([np.ravel_multi_index((r, c), grid_shape) for r in rows for c in cols], dtype=int)


def summarize_footprint_values(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    negative = finite & (values < 0)
    return {
        "nearest": np.nan,
        "mean": float(np.nanmean(finite_values)) if finite_values.size else np.nan,
        "median": float(np.nanmedian(finite_values)) if finite_values.size else np.nan,
        "std": float(np.nanstd(finite_values, ddof=1)) if finite_values.size > 1 else 0.0 if finite_values.size == 1 else np.nan,
        "n": int(finite_values.size),
        "footprint_pixel_count": int(values.size),
        "finite_pixel_count": int(finite.sum()),
        "invalid_pixel_count": int((~finite).sum()),
        "negative_pixel_count": int(negative.sum()),
        "has_invalid_pixels": bool((~finite).any()),
        "has_negative_pixels": bool(negative.any()),
    }


def select_matchup_value(summary, method="median"):
    if method not in {"nearest", "mean", "median"}:
        raise ValueError("method must be one of: 'nearest', 'mean', 'median'")
    return summary[method]


def median_symmetric_accuracy(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(estimate) & (reference > 0) & (estimate > 0)
    if not mask.any():
        return np.nan
    ratios = np.abs(np.log(estimate[mask] / reference[mask]))
    return float(100.0 * (np.exp(np.nanmedian(ratios)) - 1.0))


def symmetric_bias(reference, estimate):
    """Symmetric signed percentage bias (SSPB), Morley et al. (2018).

    The sign is applied outside the exponential so that a systematic 2x
    overestimate and a systematic 2x underestimate score as +-100%, not
    +100%/-50%: exp(median(X))-1 alone is not sign-symmetric."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(estimate) & (reference > 0) & (estimate > 0)
    if not mask.any():
        return np.nan
    median_log_ratio = np.nanmedian(np.log(estimate[mask] / reference[mask]))
    return float(100.0 * np.sign(median_log_ratio) * (np.exp(np.abs(median_log_ratio)) - 1.0))


def spectral_angle(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(estimate) & (reference > 0) & (estimate > 0)
    if mask.sum() < 2:
        return np.nan
    reference = reference[mask]
    estimate = estimate[mask]
    denominator = np.linalg.norm(reference) * np.linalg.norm(estimate)
    if denominator == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(reference, estimate) / denominator, -1.0, 1.0))))


def positive_pair_count(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return int((np.isfinite(reference) & np.isfinite(estimate) & (reference > 0) & (estimate > 0)).sum())


def paired_mae(reference, estimate):
    """Mean absolute error over finite pairs only.

    Unlike MSA/SB/SAM, MAE tolerates any sign (no positivity requirement),
    but like them it must mask NaNs per-pair rather than propagate a single
    missing band into an all-NaN result via a plain np.mean."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(estimate)
    if not mask.any():
        return np.nan
    differences = estimate[mask] - reference[mask]
    return float(np.mean(np.abs(differences)))


def paired_rmse(reference, estimate):
    """Root-mean-square error over finite pairs only (see paired_mae)."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(estimate)
    if not mask.any():
        return np.nan
    differences = estimate[mask] - reference[mask]
    return float(np.sqrt(np.mean(np.square(differences))))


def per_spectrum_metrics(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return {
        "positive_pair_count": positive_pair_count(reference, estimate),
        "msa_percent": median_symmetric_accuracy(reference, estimate),
        "sb_percent": symmetric_bias(reference, estimate),
        "mae": paired_mae(reference, estimate),
        "rmse": paired_rmse(reference, estimate),
        "spectral_angle_deg": spectral_angle(reference, estimate),
    }


def per_wavelength_scalar_metrics(reference, estimate):
    """MAE/RMSE are spectrum-level metrics only (see per_spectrum_metrics):
    at a single wavelength they reduce identically to the absolute error,
    so they are not reported here."""
    metrics = per_spectrum_metrics([reference], [estimate])
    return {
        "MSA_percent": metrics["msa_percent"],
        "SB_percent": metrics["sb_percent"],
    }


def empty_acolite_overall_metrics():
    return {
        "n_matchups": 0,
        "positive_pair_count": 0,
        "msa_percent": np.nan,
        "sb_percent": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "sam_deg_median": np.nan,
        "spectral_angle_deg_median": np.nan,
        "spectral_angle_deg_mean": np.nan,
        "spectral_angle_n": 0,
        "satellite_total_pixel_count": 0,
        "satellite_invalid_pixel_count": 0,
        "satellite_negative_pixel_count": 0,
        "satellite_matchups_with_invalid_pixels": 0,
        "satellite_matchups_with_negative_pixels": 0,
        "satellite_matchups_with_invalid_median": 0,
        "satellite_matchups_with_nonpositive_median": 0,
    }


def _satellite_flag_totals(matchup_rows):
    return {
        "satellite_total_pixel_count": int(sum(row["footprint_pixel_count"] for row in matchup_rows)),
        "satellite_invalid_pixel_count": int(sum(row["satellite_invalid_pixel_count"] for row in matchup_rows)),
        "satellite_negative_pixel_count": int(sum(row["satellite_negative_pixel_count"] for row in matchup_rows)),
        "satellite_matchups_with_invalid_pixels": int(sum(row["satellite_has_invalid_pixels"] for row in matchup_rows)),
        "satellite_matchups_with_negative_pixels": int(sum(row["satellite_has_negative_pixels"] for row in matchup_rows)),
        "satellite_matchups_with_invalid_median": int(sum(row["satellite_median_is_invalid"] for row in matchup_rows)),
        "satellite_matchups_with_nonpositive_median": int(sum(row["satellite_median_is_nonpositive"] for row in matchup_rows)),
    }


def acolite_overall_metrics(reference_array, estimate_array, matchup_rows):
    reference_array = np.asarray(reference_array, dtype=float)
    estimate_array = np.asarray(estimate_array, dtype=float)
    if reference_array.size == 0:
        return empty_acolite_overall_metrics()

    angles = np.asarray([spectral_angle(ref, est) for ref, est in zip(reference_array, estimate_array)])
    metrics = {
        "n_matchups": int(reference_array.shape[0]),
        "positive_pair_count": positive_pair_count(reference_array.ravel(), estimate_array.ravel()),
        "msa_percent": median_symmetric_accuracy(reference_array.ravel(), estimate_array.ravel()),
        "sb_percent": symmetric_bias(reference_array.ravel(), estimate_array.ravel()),
        "mae": paired_mae(reference_array, estimate_array),
        "rmse": paired_rmse(reference_array, estimate_array),
        "sam_deg_median": float(np.nanmedian(angles)),
        "spectral_angle_deg_median": float(np.nanmedian(angles)),
        "spectral_angle_deg_mean": float(np.nanmean(angles)),
        "spectral_angle_n": int(np.isfinite(angles).sum()),
    }
    metrics.update(_satellite_flag_totals(matchup_rows))
    return metrics


def acolite_wavelength_metrics(reference_array, estimate_array, metric_waves, variables, matchup_rows):
    reference_array = np.asarray(reference_array, dtype=float)
    estimate_array = np.asarray(estimate_array, dtype=float)
    if reference_array.size == 0:
        return []

    angles = np.asarray([spectral_angle(ref, est) for ref, est in zip(reference_array, estimate_array)])
    rows = []
    for index, wave in enumerate(metric_waves):
        reference = reference_array[:, index]
        estimate = estimate_array[:, index]
        wave_matchups = [row for row in matchup_rows if row["wavelength_nm"] == wave]
        row = {
            "wavelength_nm": wave,
            "satellite_variable": variables[wave],
            "n_matchups": int(reference.size),
            "positive_pair_count": positive_pair_count(reference, estimate),
            "msa_percent": median_symmetric_accuracy(reference, estimate),
            "sb_percent": symmetric_bias(reference, estimate),
            "mae": paired_mae(reference, estimate),
            "rmse": paired_rmse(reference, estimate),
            "spectral_angle_deg_median": float(np.nanmedian(angles)),
            "spectral_angle_deg_mean": float(np.nanmean(angles)),
            "spectral_angle_n": int(np.isfinite(angles).sum()),
            "mean_insitu_rrs": float(np.mean(reference)),
            "mean_satellite_rrs": float(np.mean(estimate)),
        }
        row.update(_satellite_flag_totals(wave_matchups))
        rows.append(row)
    return rows


def round_output_metrics(frame):
    frame = frame.copy()

    def format_metric(value, digits):
        if pd.isna(value):
            return ""
        return f"{value:.{digits}f}"

    for column in frame.columns:
        lower = str(column).lower()
        if "msa_percent" in lower or "sb_percent" in lower:
            frame[column] = frame[column].map(lambda value: format_metric(value, 2))
        elif (
            re.search(r"(^|_)mae($|_)", lower)
            or re.search(r"(^|_)rmse($|_)", lower)
            or lower.startswith(("mean_insitu_rrs", "mean_eo_rrs", "mean_satellite_rrs", "sat_rrs_mean_"))
        ):
            frame[column] = frame[column].map(lambda value: format_metric(value, 5))
    return frame
