import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


W_MIN, W_MAX = 350, 900
TITLE_FONTSIZE = 40
SUBTITLE_FONTSIZE = 34
AXIS_LABEL_FONTSIZE = 36
TICK_FONTSIZE = 30
LEGEND_FONTSIZE = 30
COMBINED_RRS_W_MIN, COMBINED_RRS_W_MAX = 400, 750
COMBINED_FIG_WIDTH_CM, COMBINED_FIG_HEIGHT_CM = 14.86, 7.73
CM_TO_IN = 1 / 2.54
FIGURE_WIDTH_CM, FIGURE_HEIGHT_CM = 33.02, 22.86
WORKSPACE = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu")
METADATA_DIR = WORKSPACE / "metadata"
DEFAULT_CAMPAIGN_ROOT = Path(r"C:\Users\puehrifi\Documents\insitu")
DEFAULT_SAVE_ROOT = WORKSPACE / "insitu_spectra"

CAMPAIGNS = [
    "20230610_CST",
    "20231008_ZRH",
    "20240618_BIE",
    "20240619_WAL",
    "20250303_CST",
    "20260226_ZRH",
    "20260227_ZRH",
    "20260423_ZRH",
    "20260430_ZRH",
]

PLOT_CHOICES = ("irradiance", "radiance", "rrs", "combined-rrs")
RRS_VARIANTS = {
    "original": ("original_3C", "original 3C"),
    "updated": ("updated_3C", "3C-O25"),
}
SENSOR_PRIORITY = ["S2B", "L9", "L8", "S2A"]

STATS_MODULE_DIR = Path(r"C:\Users\puehrifi\Documents\AE_personal_migration\results\stats")
if str(STATS_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(STATS_MODULE_DIR))

TRANSECTS_MODULE_DIR = Path(
    r"C:\Users\puehrifi\Documents\AE_personal_migration\insitu\quality_control"
)
if str(TRANSECTS_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSECTS_MODULE_DIR))

from campaign_locations import extract_location
from campaign_transects import CAMPAIGN_TRANSECTS


def extract_campaign(path):
    match = re.search(r"\d{8}_[A-Z]+", str(path))
    return match.group(0) if match else "UNKNOWN"


def read_spectral_csv(path, retain_datetimes=None):
    df = pd.read_csv(path, index_col=0, sep=None, engine="python")
    drop_columns = [c for c in df.columns if str(c).strip().lower() in {"median", "q25", "q75"}]
    if drop_columns:
        df = df.drop(columns=drop_columns)
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.loc[df.index.notna()].sort_index()
    df = df.loc[(df.index >= W_MIN) & (df.index <= W_MAX)]
    if retain_datetimes is not None:
        # Duplicate raw column labels (e.g. minute-only timestamps repeated
        # across several measurements) get a pandas-appended ".1", ".2", ...
        # suffix on read; strip it before parsing so every column still
        # resolves to its real timestamp instead of becoming NaT.
        cleaned_columns = [re.sub(r"\.\d+$", "", str(c)) for c in df.columns]
        sample = next((c for c in cleaned_columns if c), "")
        use_dayfirst = not re.match(r"^\d{4}-", sample)
        column_datetimes = pd.to_datetime(cleaned_columns, errors="coerce", dayfirst=use_dayfirst)
        keep = column_datetimes.isin(retain_datetimes)
        if keep.sum() == 0:
            # Some Rrs exports only retain minute-level timestamp precision;
            # fall back to minute-level matching rather than dropping everything.
            retain_minutes = retain_datetimes.floor("min")
            keep = column_datetimes.floor("min").isin(retain_minutes)
        df = df.loc[:, keep]
    return df.apply(pd.to_numeric, errors="coerce")


def compute_stats(df):
    return pd.DataFrame(
        {
            "median": df.median(axis=1, skipna=True),
            "q25": df.quantile(0.25, axis=1, numeric_only=True),
            "q75": df.quantile(0.75, axis=1, numeric_only=True),
        },
        index=df.index,
    ).dropna(how="all")


def compute_median(df):
    return pd.DataFrame(
        {"median": df.median(axis=1, skipna=True)},
        index=df.index,
    ).dropna(how="all")


def find_spectra(campaign_dir):
    campaign = extract_campaign(campaign_dir)
    candidate_dirs = [
        campaign_dir / "preprocessing" / f"{campaign}_ramses_spectra",
        campaign_dir / f"{campaign}_ramses_spectra",
    ]
    spectra_dir = next((d for d in candidate_dirs if d.exists()), candidate_dirs[0])
    ed = sorted(spectra_dir.glob(f"{campaign}_*_Ed.csv"))
    ld = sorted(spectra_dir.glob(f"{campaign}_*_Ld.csv"))
    lu = sorted(spectra_dir.glob(f"{campaign}_*_Lu.csv"))
    return {
        "Ed": ed[0] if ed else None,
        "Lw": lu[0] if lu else None,
        "Lsky": ld[0] if ld else None,
    }


def find_rrs(campaign_dir, variant):
    if variant == "original":
        candidates = [
            campaign_dir / "preprocessing" / "3C_output" / "Rrs_output_3C.csv",
            campaign_dir / "3C_output" / "Rrs_output_3C.csv",
        ]
    else:
        candidates = sorted(campaign_dir.glob("**/current_like_bounds/Rrs_output_3C_O25.csv"))
        candidates = [p for p in candidates if "test" not in str(p).lower()]
        corrected = [p for p in candidates if "corrected" in str(p).lower()]
        if corrected:
            candidates = corrected
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(f"No {variant} Rrs file found in {campaign_dir}")
    return sorted(existing, key=lambda p: (len(p.parts), str(p)))[0]


def sensor_rank(path):
    name = path.stem.upper()
    for idx, sensor in enumerate(SENSOR_PRIORITY):
        if re.search(rf"(?:^|_)({sensor})(?:_|$)", name):
            return idx
    return len(SENSOR_PRIORITY)


def final_qc_score(path):
    is_current = "current" in path.stem.lower()
    return 0 if not is_current else 1, sensor_rank(path), path.name


def find_final_qc_metadata(campaign_dir):
    campaign = extract_campaign(campaign_dir)
    qc_dir = campaign_dir / "visualization_analysis" / "quality_control"
    candidates = sorted(qc_dir.glob(f"{campaign}_metadata_with_Rrs_*_final_qc.csv"))
    if not candidates:
        raise FileNotFoundError(f"No final QC Rrs metadata file found in {qc_dir}")
    return sorted(candidates, key=final_qc_score)[0]


def load_retain_datetimes(campaign_dir):
    campaign = extract_campaign(campaign_dir)
    qc_path = find_final_qc_metadata(campaign_dir)
    df = pd.read_csv(
        qc_path, sep=None, engine="python",
        usecols=["datetime (CET)", "hard_filter_retain", "transect_nr"],
    )
    df["datetime (CET)"] = pd.to_datetime(df["datetime (CET)"]).dt.tz_localize(None)
    mask = df["hard_filter_retain"] == True  # noqa: E712
    allowed_transects = CAMPAIGN_TRANSECTS.get(campaign)
    if allowed_transects is not None:
        mask &= df["transect_nr"].isin(allowed_transects)
    return pd.DatetimeIndex(df.loc[mask, "datetime (CET)"])


def read_final_qc_rrs_stats(path):
    df = pd.read_csv(path, sep=None, engine="python")
    if "hard_filter_retain" not in df.columns:
        raise ValueError(f"No hard_filter_retain column found in {path}")
    mask = df["hard_filter_retain"] == True  # noqa: E712
    allowed_transects = CAMPAIGN_TRANSECTS.get(extract_campaign(path))
    if allowed_transects is not None:
        mask &= df["transect_nr"].isin(allowed_transects)
    df = df.loc[mask]

    pairs = []
    for column in df.columns:
        match = re.fullmatch(r"Rrs_(\d+(?:\.\d+)?)", str(column))
        if match:
            wavelength = float(match.group(1))
            if COMBINED_RRS_W_MIN <= wavelength <= COMBINED_RRS_W_MAX:
                pairs.append((column, wavelength))

    if not pairs:
        raise ValueError(f"No Rrs_* columns in plotting range found in {path}")

    pairs.sort(key=lambda item: item[1])
    columns = [column for column, _ in pairs]
    wavelengths = [wavelength for _, wavelength in pairs]
    spectra = df[columns].apply(pd.to_numeric, errors="coerce")
    stats = pd.DataFrame(
        {
            "median": spectra.median(axis=0, skipna=True).to_numpy(),
            "q25": spectra.quantile(0.25, axis=0, numeric_only=True).to_numpy(),
            "q75": spectra.quantile(0.75, axis=0, numeric_only=True).to_numpy(),
        },
        index=pd.Index(wavelengths, name="wavelength"),
    )
    return stats.dropna(how="all")


def style_axis(ax, wl_min=W_MIN, wl_max=W_MAX):
    ax.tick_params(axis="both", labelsize=30, width=1.6, length=9)
    ax.grid(True, which="both", axis="both", color="lightgrey", alpha=0.5, linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_xlim(wl_min, wl_max)


def plot_band(ax, stats, color, label, lw=4):
    ax.plot(stats.index, stats["median"], color=color, lw=lw, label=label)
    ax.fill_between(stats.index, stats["q25"], stats["q75"], alpha=0.28, facecolor=color)


def apply_combined_plot_style():
    plt.rcParams.update({
        "font.family": "Helvetica",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def style_combined_axis(ax, wl_min, wl_max):
    ax.tick_params(axis="both", labelsize=8, width=0.5, length=2.5)
    ax.grid(True, which="both", axis="both", color="lightgrey", alpha=0.5, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_xlim(wl_min, wl_max)


def apply_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
    })


def set_two_row_title(ax, top_text, bottom_text):
    ax.text(
        0.5, 1.19, top_text,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=TITLE_FONTSIZE, fontweight="bold",
    )
    ax.text(
        0.5, 1.05, bottom_text,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=SUBTITLE_FONTSIZE, fontstyle="italic", fontweight="normal",
    )


def plot_irradiance(campaign_dir, ed_save_path):
    campaign = extract_campaign(campaign_dir)
    spectra = find_spectra(campaign_dir)
    if spectra["Ed"] is None or not spectra["Ed"].exists():
        print(f"Skipped irradiance figure for {campaign}: missing Ed input")
        return False

    try:
        retain_datetimes = load_retain_datetimes(campaign_dir)
    except FileNotFoundError as exc:
        print(f"Skipped irradiance figure for {campaign}: {exc}")
        return False

    apply_plot_style()
    ed = compute_stats(read_spectral_csv(spectra["Ed"], retain_datetimes))
    if ed.empty:
        print(f"Skipped irradiance figure for {campaign}: no hard_filter_retain measurements found")
        return False
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_CM * CM_TO_IN, FIGURE_HEIGHT_CM * CM_TO_IN))
    plot_band(ax, ed, "k", r"$E_d$")
    ax.set_xlabel(r"$\lambda\ [nm]$", labelpad=18, fontstyle="italic")
    ax.set_ylabel(r"$E\ [mW\ m^{-2}\ nm^{-1}]$", labelpad=22, fontstyle="italic")
    set_two_row_title(ax, "Irradiance Spectra", extract_location(campaign_dir))
    ax.legend(loc="best", frameon=False)
    style_axis(ax)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.76, bottom=0.20)
    ed_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ed_save_path, dpi=600, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return True


def plot_radiance(campaign_dir, radiance_save_path):
    campaign = extract_campaign(campaign_dir)
    spectra = find_spectra(campaign_dir)
    if (
        spectra["Lw"] is None
        or not spectra["Lw"].exists()
        or spectra["Lsky"] is None
        or not spectra["Lsky"].exists()
    ):
        print(f"Skipped radiance figure for {campaign}: missing Lw/Lsky inputs")
        return False

    try:
        retain_datetimes = load_retain_datetimes(campaign_dir)
    except FileNotFoundError as exc:
        print(f"Skipped radiance figure for {campaign}: {exc}")
        return False

    apply_plot_style()
    lw = compute_stats(read_spectral_csv(spectra["Lw"], retain_datetimes))
    lsky = compute_stats(read_spectral_csv(spectra["Lsky"], retain_datetimes))
    if lw.empty or lsky.empty:
        print(f"Skipped radiance figure for {campaign}: no hard_filter_retain measurements found")
        return False

    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_CM * CM_TO_IN, FIGURE_HEIGHT_CM * CM_TO_IN))
    plot_band(ax, lw, "tab:blue", r"$L_w$")
    plot_band(ax, lsky, "tab:red", r"$L_{sky}$")
    ax.set_xlabel(r"$\lambda\ [nm]$", labelpad=18, fontstyle="italic")
    ax.set_ylabel(r"$L\ [mW\ m^{-2}\ nm^{-1}\ sr^{-1}]$", labelpad=22, fontstyle="italic")
    set_two_row_title(ax, "Radiance Spectra", extract_location(campaign_dir))
    ax.legend(loc="best", frameon=False)
    style_axis(ax)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.76, bottom=0.20)
    radiance_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(radiance_save_path, dpi=600, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return True


def plot_input_spectra(campaign_dir, ed_save_path, radiance_save_path):
    plotted = False
    if plot_irradiance(campaign_dir, ed_save_path):
        plotted = True
    if plot_radiance(campaign_dir, radiance_save_path):
        plotted = True
    return plotted


def plot_rrs(campaign_dir, rrs_path, rrs_save_path, title_suffix):
    campaign = extract_campaign(campaign_dir)
    try:
        retain_datetimes = load_retain_datetimes(campaign_dir)
    except FileNotFoundError as exc:
        print(f"Skipped Rrs figure for {campaign}: {exc}")
        return False

    apply_plot_style()
    rrs = compute_stats(read_spectral_csv(rrs_path, retain_datetimes))
    if rrs.empty:
        print(f"Skipped Rrs figure for {campaign}: no hard_filter_retain measurements found")
        return False
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_CM * CM_TO_IN, FIGURE_HEIGHT_CM * CM_TO_IN))
    plot_band(ax, rrs, "tab:green", r"$R_{rs}$")
    ax.set_xlabel(r"$\lambda\ [nm]$", labelpad=18)
    ax.set_ylabel(r"$R_{rs}\ [sr^{-1}]$", labelpad=22)
    ax.set_title(f"{campaign} Rrs - {title_suffix}", pad=28)
    ax.legend(loc="best", frameon=False)
    style_axis(ax)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.84, bottom=0.20)
    rrs_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(rrs_save_path, dpi=600, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return True


def plot_combined_rrs(campaign_dirs, variant, combined_save_path, title_suffix):
    apply_combined_plot_style()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COMBINED_FIG_WIDTH_CM * CM_TO_IN, COMBINED_FIG_HEIGHT_CM * CM_TO_IN),
    )
    cmap = plt.get_cmap("tab10")
    plotted = 0

    for index, campaign_dir in enumerate(campaign_dirs):
        campaign = extract_campaign(campaign_dir)
        try:
            rrs_path = find_final_qc_metadata(campaign_dir)
        except FileNotFoundError as exc:
            print(f"Skipped combined Rrs for {campaign}: {exc}")
            continue

        stats = read_final_qc_rrs_stats(rrs_path)
        if stats.empty:
            print(f"Skipped combined Rrs for {campaign}: no numeric Rrs values in plotting range")
            continue
        color = cmap(index % cmap.N)
        plot_band(ax, stats, color, extract_location(campaign_dir), lw=1.2)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"Skipped combined Rrs - {title_suffix}: no Rrs inputs found")
        return False

    ax.set_xlabel(r"$\lambda\ [nm]$", labelpad=4)
    ax.set_ylabel(r"$R_{rs}\ [sr^{-1}]$", labelpad=4)
    ax.set_title("In Situ Rrs Spectra", fontweight="bold", pad=6)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
        fontsize=8,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.4,
        borderaxespad=0.15,
    )
    style_combined_axis(ax, COMBINED_RRS_W_MIN, COMBINED_RRS_W_MAX)
    ax.set_xticks(range(COMBINED_RRS_W_MIN, COMBINED_RRS_W_MAX + 1, 100))
    # Three-digit wavelength ticks fit horizontally, so they stay unrotated and
    # the bottom margin only has to cover one label row plus the axis label.
    fig.subplots_adjust(left=0.111, right=0.649, top=0.922, bottom=0.135)
    combined_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(combined_save_path, dpi=800, pad_inches=0)
    plt.close(fig)
    return True


def normalize_plots(plots):
    selected = set(plots)
    if "all" in selected:
        return set(PLOT_CHOICES)
    return selected


def run_batch(save_root, campaign_root, plots, rrs_variants):
    selected_plots = normalize_plots(plots)
    campaign_dirs = [campaign_root / campaign for campaign in CAMPAIGNS]

    for campaign in CAMPAIGNS:
        campaign_dir = campaign_root / campaign
        ed_save_path = save_root / "input_spectra" / f"{campaign}_Ed.png"
        radiance_save_path = save_root / "input_spectra" / f"{campaign}_radiance.png"

        if "irradiance" in selected_plots and plot_irradiance(campaign_dir, ed_save_path):
            print(f"Saved irradiance: {ed_save_path}")

        if "radiance" in selected_plots and plot_radiance(campaign_dir, radiance_save_path):
            print(f"Saved radiance: {radiance_save_path}")

        if "rrs" not in selected_plots:
            continue

        for variant in rrs_variants:
            subdir, title_suffix = RRS_VARIANTS[variant]
            rrs_path = find_rrs(campaign_dir, variant)
            rrs_save_path = save_root / subdir / f"{campaign}_Rrs_{variant}.png"
            plot_rrs(campaign_dir, rrs_path, rrs_save_path, title_suffix)
            print(f"Saved Rrs: {rrs_save_path}")

    if "combined-rrs" in selected_plots:
        for variant in rrs_variants:
            _, title_suffix = RRS_VARIANTS[variant]
            combined_save_path = save_root / f"all_campaigns_Rrs_{variant}.png"
            if plot_combined_rrs(campaign_dirs, variant, combined_save_path, title_suffix):
                print(f"Saved combined Rrs: {combined_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot RAMSES spectra and Rrs campaign figures.")
    parser.add_argument("--batch", action="store_true", help="Plot all configured campaigns.")
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
        help="Root containing campaign folders. Default: C:\\Users\\puehrifi\\Documents\\insitu.",
    )
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--rrs", type=Path)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--title-suffix", default="Rrs")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("all",) + PLOT_CHOICES,
        default=["irradiance", "radiance", "rrs"],
        help=(
            "Plots to create. Use any combination of irradiance, radiance, rrs, "
            "combined-rrs, or all. Default: irradiance radiance rrs."
        ),
    )
    parser.add_argument(
        "--rrs-variants",
        nargs="+",
        choices=tuple(RRS_VARIANTS),
        default=list(RRS_VARIANTS),
        help="Rrs correction variants to plot in batch mode. Default: original updated.",
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.save_root, args.campaign_root, args.plots, args.rrs_variants)
        return

    selected_plots = normalize_plots(args.plots)
    if "combined-rrs" in selected_plots:
        raise SystemExit("Use --batch for combined-rrs plots.")
    if args.campaign_dir is None:
        raise SystemExit("Use --batch, or provide --campaign-dir.")
    if "rrs" in selected_plots and args.rrs is None:
        raise SystemExit("Provide --rrs when requesting the rrs plot outside --batch.")

    campaign = extract_campaign(args.campaign_dir)
    ed_save_path = args.save_path or args.save_root / f"{campaign}_Ed.png"
    radiance_save_path = args.save_root / f"{campaign}_radiance.png"
    rrs_save_path = args.save_root / f"{campaign}_Rrs.png"

    if "irradiance" in selected_plots and plot_irradiance(args.campaign_dir, ed_save_path):
        print(f"Saved: {ed_save_path}")

    if "radiance" in selected_plots and plot_radiance(args.campaign_dir, radiance_save_path):
        print(f"Saved: {radiance_save_path}")

    if "rrs" in selected_plots:
        plot_rrs(args.campaign_dir, args.rrs, rrs_save_path, args.title_suffix)
        print(f"Saved: {rrs_save_path}")


if __name__ == "__main__":
    main()
