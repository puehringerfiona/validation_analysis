"""
Download L2 surface-reflectance scenes matching the campaign/sensor L1 inputs.

Landsat is downloaded from public USGS Landsat Collection 2 Level-2 URLs. Only
the visible SR bands, QA_PIXEL, QA_RADSAT, and metadata files are fetched.

Sentinel-2 L2A is discovered through Copernicus Data Space OData and downloaded
as a product zip. Set CDSE_USERNAME and CDSE_PASSWORD before downloading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR.parents[1]
INPUT_ROOT = Path(r"C:\Users\puehrifi\Documents\eo_data\input")
OUTPUT_ROOT = Path(r"C:\Users\puehrifi\Documents\eo_data\input_l2sr")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_performance_metrics_batch import ACTIVE_RUNS  # noqa: E402


LANDSAT_VISIBLE_BANDS = {
    "L8": ["SR_B1", "SR_B2", "SR_B3", "SR_B4"],
    "L9": ["SR_B1", "SR_B2", "SR_B3", "SR_B4"],
}

LANDSAT_EXTRA_ASSETS = ["QA_PIXEL", "QA_RADSAT", "MTL.txt", "MTL.xml", "MTL.json"]
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_PRODUCTS_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"


@dataclass(frozen=True)
class DownloadItem:
    url: str
    path: Path
    auth: bool = False


def existing_l1_dir(campaign: str, sensor: str) -> Path | None:
    campaign_dir = INPUT_ROOT / campaign
    if not campaign_dir.exists():
        return None
    if sensor == "L8":
        prefixes = ["LC08"]
    elif sensor == "L9":
        prefixes = ["LC09", "LC08"]
    else:
        prefixes = [sensor]
    for prefix in prefixes:
        matches = sorted(path for path in campaign_dir.glob(f"{prefix}_L1*") if path.is_dir())
        if matches:
            return matches[0]
    return None


def existing_safe(campaign: str, sensor: str) -> Path | None:
    campaign_dir = INPUT_ROOT / campaign
    if not campaign_dir.exists():
        return None
    matches = sorted(path for path in campaign_dir.glob(f"{sensor}_MSIL1C*.SAFE") if path.is_dir())
    return matches[0] if matches else None


def landsat_l2_id_from_l1(l1_id: str) -> str:
    match = re.match(r"^(L[COTEM]\d{2})_L1\w{2}_(\d{6})_(\d{8})_(\d{8})_(\d{2})_(T\d|RT)$", l1_id)
    if not match:
        raise ValueError(f"Could not parse Landsat L1 product ID: {l1_id}")
    satellite, pathrow, acquired, processed, collection, tier = match.groups()
    return f"{satellite}_L2SP_{pathrow}_{acquired}_{processed}_{collection}_{tier}"


def landsat_items(campaign: str, sensor: str) -> list[DownloadItem]:
    l1_dir = existing_l1_dir(campaign, sensor)
    if l1_dir is None:
        raise FileNotFoundError(f"No Landsat L1 directory found for {campaign} {sensor}")
    l2_id = landsat_l2_id_from_l1(l1_dir.name)
    satellite = l2_id.split("_", 1)[0]
    year = l2_id.split("_")[3][:4]
    pathrow = l2_id.split("_")[2]
    wrs_path = pathrow[:3]
    wrs_row = pathrow[3:]
    base = (
        "https://landsatlook.usgs.gov/data/collection02/level-2/standard/oli-tirs/"
        f"{year}/{wrs_path}/{wrs_row}/{l2_id}"
    )
    target_dir = OUTPUT_ROOT / campaign / l2_id
    suffixes = [*LANDSAT_VISIBLE_BANDS[sensor], *LANDSAT_EXTRA_ASSETS]
    items = []
    for suffix in suffixes:
        ext = ".TIF" if suffix.startswith(("SR_", "QA_")) else f"_{suffix}"
        filename = f"{l2_id}_{suffix}.TIF" if suffix.startswith(("SR_", "QA_")) else f"{l2_id}_{suffix}"
        items.append(DownloadItem(f"{base}/{filename}", target_dir / filename))
    return items


def cdse_token(username: str, password: str) -> str:
    response = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def sentinel_l2a_product(campaign: str, sensor: str) -> dict:
    safe = existing_safe(campaign, sensor)
    if safe is None:
        raise FileNotFoundError(f"No Sentinel-2 L1C SAFE found for {campaign} {sensor}")
    l1_name = safe.name
    match = re.match(r"^(S2[AB])_MSIL1C_(\d{8}T\d{6})_.*_(T\d{2}[A-Z]{3})_\d{8}T\d{6}\.SAFE$", l1_name)
    if not match:
        raise ValueError(f"Could not parse Sentinel-2 SAFE name: {l1_name}")
    platform, sensing_time, tile = match.groups()
    prefix = f"{platform}_MSIL2A_{sensing_time}"
    filters = [
        "Collection/Name eq 'SENTINEL-2'",
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')",
        f"contains(Name,'{prefix}')",
        f"contains(Name,'_{tile}_')",
    ]
    params = {
        "$filter": " and ".join(filters),
        "$orderby": "ContentDate/Start asc",
        "$top": "10",
    }
    response = requests.get(CDSE_PRODUCTS_URL, params=params, timeout=60)
    response.raise_for_status()
    products = response.json().get("value", [])
    if not products:
        raise FileNotFoundError(f"No Sentinel-2 L2A product found for {l1_name}")
    products = sorted(products, key=lambda item: (item.get("Name", ""), item.get("Id", "")))
    return products[0]


def sentinel_items(campaign: str, sensor: str) -> list[DownloadItem]:
    product = sentinel_l2a_product(campaign, sensor)
    product_id = product["Id"]
    name = product["Name"]
    target = OUTPUT_ROOT / campaign / f"{name}.zip"
    url = f"{CDSE_DOWNLOAD_URL}({product_id})/$value"
    sidecar = target.with_suffix(target.suffix + ".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(product, indent=2), encoding="utf-8")
    return [DownloadItem(url, target, auth=True)]


def iter_items(campaign_filter: str | None, sensor_filter: str | None, include_landsat: bool, include_sentinel: bool):
    for cfg in ACTIVE_RUNS:
        if campaign_filter and cfg.campaign != campaign_filter:
            continue
        if sensor_filter and cfg.sensor.upper() != sensor_filter.upper():
            continue
        if cfg.sensor.startswith("L") and include_landsat:
            yield cfg.campaign, cfg.sensor, landsat_items(cfg.campaign, cfg.sensor)
        elif cfg.sensor.startswith("S2") and include_sentinel:
            yield cfg.campaign, cfg.sensor, sentinel_items(cfg.campaign, cfg.sensor)


def download(item: DownloadItem, token: str | None, overwrite: bool = False) -> str:
    if item.path.exists() and not overwrite and item.path.stat().st_size > 0:
        return "exists"
    item.path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if item.auth and token else {}
    with requests.get(item.url, headers=headers, stream=True, timeout=120, allow_redirects=True) as response:
        response.raise_for_status()
        tmp = item.path.with_suffix(item.path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp.replace(item.path)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="Optional campaign filter, e.g. 20230610_CST.")
    parser.add_argument("--sensor", help="Optional sensor filter, e.g. S2B.")
    parser.add_argument("--landsat-only", action="store_true")
    parser.add_argument("--sentinel-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without fetching files.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    include_landsat = not args.sentinel_only
    include_sentinel = not args.landsat_only
    token = None
    if include_sentinel and not args.dry_run:
        username = os.environ.get("CDSE_USERNAME")
        password = os.environ.get("CDSE_PASSWORD")
        if not username or not password:
            raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD to download Sentinel-2 L2A products.")
        token = cdse_token(username, password)

    total = 0
    for campaign, sensor, items in iter_items(args.campaign, args.sensor, include_landsat, include_sentinel):
        print(f"[l2sr] {campaign} {sensor}: {len(items)} file(s)")
        for item in items:
            total += 1
            if args.dry_run:
                print(f"  plan {item.path} <- {item.url}")
            else:
                status = download(item, token, overwrite=args.overwrite)
                print(f"  {status}: {item.path}")
    if total == 0:
        raise RuntimeError("No download items matched the requested filters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
