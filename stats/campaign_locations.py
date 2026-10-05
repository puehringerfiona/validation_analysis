from __future__ import annotations

import os
import re
from pathlib import Path


CAMPAIGN_LOCATIONS = {
    "20231008": "Lake Zurich, 08.10.2023",
    "20230610": "Lake Constance, 10.06.2023",
    "20240619": "Walensee, 19.06.2024",
    "20240618": "Lake Biel, 18.06.2024",
    "20250303": "Lake Constance, 03.03.2025",
    "20260226": "Lake Zurich, 26.02.2026",
    "20260227": "Lake Zurich, 27.02.2026",
    "20260423": "Lake Zurich, 23.04.2026",
    "20260430": "Lake Zurich, 30.04.2026",
}


def extract_campaign_date(value: str | os.PathLike) -> str | None:
    text = Path(value).name if isinstance(value, os.PathLike) else str(value)
    match = re.search(r"(20\d{6})", text)
    return match.group(1) if match else None


def location_for_campaign(campaign: str, default: str = "Unknown Location and Date") -> str:
    campaign_date = extract_campaign_date(campaign)
    if campaign_date is None:
        return default
    return CAMPAIGN_LOCATIONS.get(campaign_date, default)


def extract_location(csv_path: str | os.PathLike) -> str:
    return location_for_campaign(str(csv_path))
