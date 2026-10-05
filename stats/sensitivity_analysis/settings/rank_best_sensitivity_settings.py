from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DEFAULT_INPUT_ROOT / "best_sensitivity_settings_by_lake_processor.csv"
LUTS_FOR_RANKING = ("MOD1", "MOD2")
TOP_N_PER_WAVELENGTH = 3
RANK_WEIGHT = {1: 3, 2: 2, 3: 1}
DEFAULT_EQUIVALENCE_TOLERANCE_MSA = 1.0
PROCESSOR_ALIASES = {
    "acolite": "acolite",
    "acolite_dsf": "acolite",
    "tmart": "acolite_tmart",
    "acolite_tmart": "acolite_tmart",
    "acolite+tmart": "acolite_tmart",
    "radcor": "acolite_radcor",
    "acolite_radcor": "acolite_radcor",
    "acolite+radcor": "acolite_radcor",
    "c2rcc": "c2rcc",
    "c2rcc_tmart": "c2rcc_tmart",
    "c2rcc+tmart": "c2rcc_tmart",
    "c2rcc_radcor": "c2rcc_radcor",
    "c2rcc+radcor": "c2rcc_radcor",
    "polymer": "polymer",
    "polymer_tmart": "polymer_tmart",
    "polymer+tmart": "polymer_tmart",
    "polymer_radcor": "polymer_radcor",
    "polymer+radcor": "polymer_radcor",
}
KERNEL_PARAMETER_PROCESSORS = {
    "acolite_radcor",
    "c2rcc_radcor",
    "polymer_radcor",
}
COMMON_EXTENTS_KM = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0]


def canonical_processor(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    key = text.lower().replace("-", "_").replace(" ", "_")
    return PROCESSOR_ALIASES.get(key, key)


def parse_root_processors(items: list[str] | None) -> dict[Path, str]:
    mapping = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Root processor mapping must use ROOT=PROCESSOR syntax: {item}")
        root, processor = item.split("=", 1)
        mapping[Path(root).resolve()] = canonical_processor(processor)
    return mapping


def infer_processor(path: Path, frame: pd.DataFrame, root_processors: dict[Path, str]) -> str:
    resolved_path = path.resolve()
    matches = [
        (root, processor)
        for root, processor in root_processors.items()
        if root == resolved_path or root in resolved_path.parents
    ]
    if matches:
        return sorted(matches, key=lambda item: len(str(item[0])), reverse=True)[0][1]

    if "processor" in frame.columns:
        values = [canonical_processor(value) for value in frame["processor"].dropna().unique()]
        values = [value for value in values if value]
        if len(values) == 1:
            return values[0]

    text = " ".join([str(path), " ".join(path.parts)])
    if "product" in frame.columns and not frame["product"].dropna().empty:
        text += " " + str(frame["product"].dropna().iloc[0])
    low = text.lower()
    if "c2rcc_radcor" in low:
        return "c2rcc_radcor"
    if "c2rcc_tmart" in low:
        return "c2rcc_tmart"
    if "polymer_tmart" in low:
        return "polymer_tmart"
    if "radcor_polymer" in low or "polymer_radcor" in low:
        return "polymer_radcor"
    if "acolite_radcor" in low or "radcor" in low:
        return "acolite_radcor"
    if "acolite_tmart" in low or "tmart" in low:
        return "acolite_tmart"
    if "acolite_dsf" in low or "acolite" in low:
        return "acolite"
    if "c2rcc" in low:
        return "c2rcc"
    if "polymer" in low:
        return "polymer"
    return "unknown"


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def discover_tables(input_roots: list[Path]) -> list[Path]:
    tables = []
    for root in input_roots:
        if root.is_file():
            tables.append(root)
        else:
            tables.extend(root.rglob("*_all_settings_by_transect_visible_wavelength.csv"))
    return sorted(set(tables))


def campaign_from_path(path: Path) -> str:
    return path.parent.name


def lake_from_campaign(campaign: str) -> str:
    match = re.match(r"([A-Za-z]+)", campaign)
    if match:
        return match.group(1).upper()
    if len(campaign) >= 3:
        return campaign[:3].upper()
    return campaign.upper()


def unique_join(values, numeric: bool = False) -> str:
    seen = []
    for value in values:
        if pd.isna(value):
            continue
        for part in str(value).split(","):
            text = part.strip()
            if text and text not in seen:
                seen.append(text)
    if numeric:
        try:
            seen = sorted(seen, key=lambda item: float(item))
        except ValueError:
            seen = sorted(seen)
    else:
        seen = sorted(seen)
    return ",".join(seen)


def fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.10g}"


def parse_extent_km(value: object) -> float:
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not numbers:
        return float("nan")
    extent = float(numbers[0])
    nearest = min(COMMON_EXTENTS_KM, key=lambda candidate: abs(candidate - extent))
    if abs(nearest - extent) <= 0.08:
        return nearest
    return extent


def load_tables(input_roots: list[Path], root_processors: dict[Path, str]) -> pd.DataFrame:
    frames = []
    for path in discover_tables(input_roots):
        frame = read_csv_auto(path)
        if frame.empty:
            continue
        campaign = campaign_from_path(path)
        frame["source_table"] = str(path)
        frame["campaign"] = campaign
        frame["lake"] = lake_from_campaign(campaign)
        frame["processor"] = infer_processor(path, frame, root_processors)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_rank_columns(data: pd.DataFrame) -> pd.DataFrame:
    rank_columns = ["processor", "campaign", "sensor", "scenario", "selected_transects", "wavelength_nm"]
    if "rank_msa_within_transect_wavelength" in data.columns:
        data["rank_int"] = pd.to_numeric(data["rank_msa_within_transect_wavelength"], errors="coerce")
    else:
        sort_columns = [*rank_columns, "msa_percent"]
        for optional in ["spectral_angle_deg_median", "sam_deg_median", "rmse"]:
            if optional in data.columns:
                sort_columns.append(optional)
        data = data.sort_values(sort_columns, na_position="last")
        data["rank_int"] = data.groupby(rank_columns, dropna=False).cumcount() + 1
    data = data.dropna(subset=["rank_int"]).copy()
    data["rank_int"] = data["rank_int"].astype(int)
    data["rank_weight"] = data["rank_int"].map(RANK_WEIGHT).fillna(0).astype(int)
    return data


def setting_parameter_family(processor: str) -> str:
    if processor in KERNEL_PARAMETER_PROCESSORS:
        return "lut_kernel_radius"
    return "lut_dsf_extent"


def quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if not series.dropna().empty else float("nan")


def write_rankings(data: pd.DataFrame, output: Path, scores_output: Path, equivalence_tolerance_msa: float) -> None:
    for column in ["msa_percent", "wavelength_nm"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for optional in ["rmse", "spectral_angle_deg_median", "sam_deg_median"]:
        if optional in data.columns:
            data[optional] = pd.to_numeric(data[optional], errors="coerce")

    data = data[data["lut"].isin(LUTS_FOR_RANKING)].copy()
    data = data.dropna(subset=["msa_percent", "wavelength_nm"])
    data = add_rank_columns(data)
    data["parameter_family"] = data["processor"].map(setting_parameter_family)
    for column in ["tile_dimensions", "physical_tile_extent", "kernel_radius"]:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    if "selected_transects" not in data.columns:
        data["selected_transects"] = ""
    if "scenario" not in data.columns:
        data["scenario"] = ""
    data["extent_km"] = data["physical_tile_extent"].map(parse_extent_km)
    data.loc[data["processor"].isin(KERNEL_PARAMETER_PROCESSORS), ["tile_dimensions", "physical_tile_extent"]] = ""
    data.loc[data["processor"].isin(KERNEL_PARAMETER_PROCESSORS), "extent_km"] = pd.NA
    data.loc[~data["processor"].isin(KERNEL_PARAMETER_PROCESSORS), "kernel_radius"] = ""

    rows = []
    setting_columns = ["processor", "lake", "lut", "extent_km", "kernel_radius"]
    for keys, group in data.groupby(setting_columns, dropna=False):
        processor, lake, lut, extent_km, kernel_radius = keys
        first = group.iloc[0]
        ranks = group["rank_int"]
        top3 = group[group["rank_int"].le(TOP_N_PER_WAVELENGTH)]
        spectral_angle_column = "spectral_angle_deg_median"
        if spectral_angle_column not in group.columns and "sam_deg_median" in group.columns:
            spectral_angle_column = "sam_deg_median"
        spectral_angles = pd.to_numeric(group.get(spectral_angle_column, pd.Series(dtype=float)), errors="coerce")
        msi_tiles = unique_join(group.loc[group["sensor"].eq("MSI"), "tile_dimensions"].unique())
        oli_tiles = unique_join(group.loc[group["sensor"].eq("OLI_TIRS"), "tile_dimensions"].unique())
        rows.append(
            {
                "processor": processor,
                "lake": lake,
                "sensors": unique_join(group["sensor"].unique()),
                "parameter_family": first["parameter_family"],
                "lut": lut,
                "extent_km": fmt(extent_km),
                "msi_tile_dimensions": msi_tiles,
                "oli_tirs_tile_dimensions": oli_tiles,
                "kernel_radius": kernel_radius,
                "atomic_group_count": int(len(group)),
                "median_msa": group["msa_percent"].median(),
                "p90_msa": quantile(group["msa_percent"], 0.90),
                "median_spectral_angle_deg": spectral_angles.median(),
                "p90_spectral_angle_deg": quantile(spectral_angles, 0.90),
                "weighted_top3_score": int(top3["rank_weight"].sum()),
                "top3_appearances": int(len(top3)),
                "rank1": int((ranks == 1).sum()),
                "rank2": int((ranks == 2).sum()),
                "rank3": int((ranks == 3).sum()),
                "mean_rank": group["rank_int"].mean(),
                "campaigns": unique_join(group["campaign"].unique()),
                "scenarios": unique_join(group["scenario"].unique(), numeric=True),
                "transects": unique_join(group["selected_transects"].unique(), numeric=True),
                "wavelengths": unique_join(group["wavelength_nm"].astype(int).astype(str).unique(), numeric=True),
                "source_tables": unique_join(group["source_table"].unique()),
            }
        )

    scores = pd.DataFrame(rows)
    if scores.empty:
        raise SystemExit("No rankable MOD1/MOD2 settings found in the input tables.")
    scores = scores.sort_values(
        ["processor", "lake", "median_msa", "p90_msa", "median_spectral_angle_deg"],
        ascending=[True, True, True, True, True],
    )
    scores["ranking_method"] = "median_msa_then_p90_msa_then_median_spectral_angle"
    scores["equivalence_tolerance_msa"] = equivalence_tolerance_msa
    scores["setting_rank"] = scores.groupby(["processor", "lake"], dropna=False).cumcount() + 1
    best_msa = scores.groupby(["processor", "lake"], dropna=False)["median_msa"].transform("min")
    scores["delta_median_msa_from_best"] = scores["median_msa"] - best_msa
    scores["within_equivalence_tolerance"] = scores["delta_median_msa_from_best"].le(equivalence_tolerance_msa)
    best = scores[scores["within_equivalence_tolerance"]].copy()
    best["is_primary_best"] = best["setting_rank"].eq(1)
    for frame in (scores, best):
        frame["mean_rank"] = frame["mean_rank"].map(fmt)
        frame["median_msa"] = frame["median_msa"].map(fmt)
        frame["p90_msa"] = frame["p90_msa"].map(fmt)
        frame["median_spectral_angle_deg"] = frame["median_spectral_angle_deg"].map(fmt)
        frame["p90_spectral_angle_deg"] = frame["p90_spectral_angle_deg"].map(fmt)
        frame["delta_median_msa_from_best"] = frame["delta_median_msa_from_best"].map(fmt)

    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_output, index=False)
    best.to_csv(output, index=False)
    print(f"Wrote all setting scores to {scores_output}")
    print(f"Wrote best settings to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank best sensitivity settings per lake and processor from "
            "*_all_settings_by_transect_visible_wavelength.csv validation outputs."
        )
    )
    parser.add_argument(
        "--input-roots",
        nargs="+",
        default=[str(DEFAULT_INPUT_ROOT)],
        help="Directories or CSV files to scan. Defaults to this sensitivity_analysis folder.",
    )
    parser.add_argument(
        "--processors",
        nargs="+",
        help="Optional processor filter, e.g. acolite acolite_tmart acolite_radcor. Defaults to all inferred processors.",
    )
    parser.add_argument(
        "--root-processor",
        action="append",
        help="Optional ROOT=PROCESSOR mapping used before processor inference. Can be supplied multiple times.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--scores-output",
        help="Optional detailed score table path. Defaults to OUTPUT with _all_setting_scores suffix.",
    )
    parser.add_argument(
        "--equivalence-tolerance-msa",
        type=float,
        default=DEFAULT_EQUIVALENCE_TOLERANCE_MSA,
        help="Settings within this absolute median-MSA percentage-point distance from the best are marked equivalent. Defaults to 1.0.",
    )
    args = parser.parse_args()

    input_roots = [Path(root) for root in args.input_roots]
    output = Path(args.output)
    scores_output = (
        Path(args.scores_output)
        if args.scores_output
        else output.with_name(f"{output.stem}_all_setting_scores{output.suffix}")
    )
    root_processors = parse_root_processors(args.root_processor)
    data = load_tables(input_roots, root_processors)
    if data.empty:
        raise SystemExit("No all-settings wavelength tables found.")

    if args.processors:
        processors = {canonical_processor(processor) for processor in args.processors}
        data = data[data["processor"].isin(processors)].copy()
        if data.empty:
            raise SystemExit(f"No tables matched requested processors: {', '.join(sorted(processors))}")

    write_rankings(data, output, scores_output, args.equivalence_tolerance_msa)


if __name__ == "__main__":
    main()
