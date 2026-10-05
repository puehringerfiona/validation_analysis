from __future__ import annotations

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOCUMENTS_ROOT = PROJECT_ROOT.parent


DEFAULT_REPORT = SCRIPT_DIR / "aot550_report_best_only_with_log_aot.csv"
DEFAULT_EO_OUTPUT = DOCUMENTS_ROOT / "eo_data" / "output"
DEFAULT_INVENTORY = PROJECT_ROOT / "eo_data" / "analysis" / "parameter_settings_inventory"

PROCESSOR_ROOT_NAMES = {
    "acolite_radcor": "acolite_radcor",
    "acolite": "acolite",
    "acolite_tmart": "acolite_tmart",
    "radcor_acolite": "radcor_acolite",
}

LOG_DIR_NAMES = {
    "acolite": ["run_logs"],
    "acolite_tmart": ["run_logs_acolite_tmart"],
    "acolite_radcor": ["run_logs_acolite_radcor"],
    # These products are ACOLITE runs after RAdCor; there is no separate
    # radcor_acolite log inventory, so use the ACOLITE log directory.
    "radcor_acolite": ["run_logs"],
}

FIELDNAMES = [
    "campaign",
    "processor",
    "aot_550_best_available",
    "aot_550_best_source",
    "product_dir",
    "nc_file",
    "ac_aot_550",
    "aot_550_var_min",
    "aot_550_var_mean",
    "aot_550_var_median",
    "aot_550_var_max",
    "dsf_aot_estimate",
    "radcor_aot_estimate",
    "status",
    "error",
    "log_aot_550",
    "log_aot_model",
    "log_aot_line",
    "log_aot_text",
    "log_path",
    "log_processor_source",
]

CAMPAIGN_RE = re.compile(r"([A-Z]{3}\d{8}|\d{8}[A-Z]{3})")
SELECTED_RE = re.compile(
    r"Selected\s+(?P<model>\S+).*?mean\s+aot\s*=\s*(?P<aot>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
SUBSET_RE = re.compile(
    r"(?P<model>\S+):\s+.*?mean\s+aot\s+of\s+subset\s*=\s*(?P<aot>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def long_path(path: Path) -> str:
    text = str(path)
    return text if text.startswith(r"\\?\\") else r"\\?\\" + text


def read_text(path: Path) -> str:
    data = Path(long_path(path)).read_bytes()
    for encoding in ("utf-8", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def product_dir(path: Path) -> str:
    parts = path.parts
    if "L2ACOLITE" in parts:
        return str(Path(*parts[: parts.index("L2ACOLITE")]))
    return str(path.parent)


def campaign_name(path: Path) -> str:
    match = CAMPAIGN_RE.search(str(path))
    return match.group(1) if match else ""


def nc_attribute(ds: Dataset, name: str) -> str:
    if name not in ds.ncattrs():
        return ""
    raw = getattr(ds, name)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def empty_row(path: Path, processor: str) -> dict[str, str]:
    return {
        "campaign": campaign_name(path),
        "processor": processor,
        "aot_550_best_available": "",
        "aot_550_best_source": "",
        "product_dir": product_dir(path),
        "nc_file": str(path),
        "ac_aot_550": "",
        "aot_550_var_min": "",
        "aot_550_var_mean": "",
        "aot_550_var_median": "",
        "aot_550_var_max": "",
        "dsf_aot_estimate": "",
        "radcor_aot_estimate": "",
        "status": "ok",
        "error": "",
        "log_aot_550": "",
        "log_aot_model": "",
        "log_aot_line": "",
        "log_aot_text": "",
        "log_path": "",
        "log_processor_source": "",
    }


def update_from_netcdf(row: dict[str, str]) -> None:
    path = Path(row["nc_file"])
    try:
        with Dataset(long_path(path), "r") as ds:
            row["ac_aot_550"] = nc_attribute(ds, "ac_aot_550")
            row["dsf_aot_estimate"] = nc_attribute(ds, "dsf_aot_estimate")
            row["radcor_aot_estimate"] = nc_attribute(ds, "radcor_aot_estimate")
            if "aot_550" in ds.variables:
                data = np.ma.masked_invalid(ds.variables["aot_550"][:])
                if data.count() > 0:
                    row["aot_550_var_min"] = str(float(data.min()))
                    row["aot_550_var_mean"] = str(float(data.mean()))
                    row["aot_550_var_median"] = str(float(np.ma.median(data)))
                    row["aot_550_var_max"] = str(float(data.max()))
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"


def parse_log(path: Path, source: str) -> dict[str, str] | None:
    try:
        text = read_text(path)
    except OSError:
        return None
    selected = None
    subset = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        if selected is None:
            match = SELECTED_RE.search(line)
            if match:
                selected = (line_no, line.strip(), match.group("model"), match.group("aot"))
                continue
        if subset is None:
            match = SUBSET_RE.search(line)
            if match:
                subset = (line_no, line.strip(), match.group("model"), match.group("aot"))
    chosen = selected or subset
    if not chosen:
        return None
    line_no, text_line, model, aot = chosen
    return {
        "log_aot_550": aot,
        "log_aot_model": model,
        "log_aot_line": str(line_no),
        "log_aot_text": text_line,
        "log_path": str(path),
        "log_processor_source": source,
    }


def row_tokens(row: dict[str, str]) -> set[str]:
    text = Path(row["product_dir"]).name.lower()
    tokens = {row["campaign"].lower()}
    for pattern in (r"oli_tirs", r"msi", r"mod\d+", r"\d+x\d+", r"lb\d+", r"kr\d+km", r"\d+km"):
        tokens.update(re.findall(pattern, text))
    return {token for token in tokens if token}


def log_tokens(path: Path) -> set[str]:
    name = re.sub(r"^\d+_", "", path.stem.lower())
    tokens: set[str] = set()
    for pattern in (r"[a-z]{3}\d{8}", r"oli_tirs", r"msi", r"mod\d+", r"\d+x\d+", r"lb\d+", r"kr\d+km", r"\d+km"):
        tokens.update(re.findall(pattern, name))
    return tokens


def local_log_record(row: dict[str, str]) -> dict[str, str] | None:
    log_dir = Path(row["product_dir"]) / "Logs"
    if not log_dir.exists():
        return None
    for path in sorted(list(log_dir.glob("*.txt")) + list(log_dir.glob("*.log"))):
        record = parse_log(path, row["processor"] + "_product_log")
        if record:
            return record
    return None


def inventory_log_records(inventory_root: Path) -> dict[str, list[tuple[Path, dict[str, str], set[str]]]]:
    records: dict[str, list[tuple[Path, dict[str, str], set[str]]]] = {}
    for processor, dir_names in LOG_DIR_NAMES.items():
        for dir_name in dir_names:
            directory = inventory_root / dir_name
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.log")):
                record = parse_log(path, processor)
                if record:
                    records.setdefault(processor, []).append((path, record, log_tokens(path)))
    return records


def exact_inventory_match(
    row: dict[str, str],
    records: dict[str, list[tuple[Path, dict[str, str], set[str]]]],
) -> dict[str, str] | None:
    tokens = row_tokens(row)
    for _, record, candidate_tokens in records.get(row["processor"], []):
        if tokens and tokens.issubset(candidate_tokens):
            return record
    return None


def recompute_best(row: dict[str, str]) -> None:
    if row["status"] == "error":
        row["aot_550_best_available"] = ""
        row["aot_550_best_source"] = ""
    elif row["ac_aot_550"]:
        row["aot_550_best_available"] = row["ac_aot_550"]
        row["aot_550_best_source"] = "netcdf:ac_aot_550"
        row["status"] = "ok"
    elif row["aot_550_var_mean"]:
        row["aot_550_best_available"] = row["aot_550_var_mean"]
        row["aot_550_best_source"] = "netcdf:aot_550_mean"
        row["status"] = "aot_550_variable_only"
    elif row["log_aot_550"]:
        row["aot_550_best_available"] = row["log_aot_550"]
        row["aot_550_best_source"] = "log:mean_aot"
        row["status"] = "filled_from_log"
    else:
        row["aot_550_best_available"] = ""
        row["aot_550_best_source"] = ""
        row["status"] = "missing_ac_aot_550"


def collect_product_rows(eo_output_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    product_roots = {
        processor: eo_output_root / root_name
        for processor, root_name in PROCESSOR_ROOT_NAMES.items()
    }
    for processor, root in product_roots.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.nc")):
            if any(part.lower() == "non_best" for part in path.parts):
                continue
            row = empty_row(path, processor)
            update_from_netcdf(row)
            rows.append(row)
    return rows


def write_report(rows: list[dict[str, str]], report_path: Path, backup: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and report_path.exists():
        backup_path = report_path.with_name(f"{report_path.stem}.backup_{datetime.now():%Y%m%dT%H%M%S}{report_path.suffix}")
        shutil.copy2(report_path, backup_path)
        print(f"Backed up previous report to {backup_path.resolve()}")
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the ACOLITE-family AOT550 report from NetCDF attributes/variables "
            "and fill missing values from ACOLITE run logs."
        )
    )
    parser.add_argument("--eo-output", type=Path, default=DEFAULT_EO_OUTPUT, help="Root containing ACOLITE output folders.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY, help="Parameter-settings inventory with run log folders.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="CSV report path to write.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up an existing report before overwriting it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_product_rows(args.eo_output)
    records = inventory_log_records(args.inventory)

    for row in rows:
        if not row["ac_aot_550"] and not row["aot_550_var_mean"]:
            record = local_log_record(row) or exact_inventory_match(row, records)
            if record:
                row.update(record)
        recompute_best(row)

    write_report(rows, args.report, backup=not args.no_backup)

    remaining = [row for row in rows if not row["aot_550_best_available"]]
    print(f"Wrote {args.report.resolve()}")
    print(f"Rows: {len(rows)}")
    print(f"Remaining without best AOT: {len(remaining)}")
    for processor in sorted(PROCESSOR_ROOT_NAMES):
        proc_rows = [row for row in rows if row["processor"] == processor]
        proc_missing = [row for row in proc_rows if not row["aot_550_best_available"]]
        print(f"{processor}\trows={len(proc_rows)}\tmissing={len(proc_missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
