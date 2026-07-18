"""Fundamental-data update pipeline for normalized USDA datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


WASDE_COLUMNS = [
    "date",
    "commodity",
    "marketing_year",
    "production",
    "exports",
    "ending_stocks",
    "total_use",
    "stocks_to_use",
]

CROP_CONDITION_COLUMNS = [
    "date",
    "commodity",
    "good_pct",
    "excellent_pct",
    "good_excellent_pct",
]


@dataclass(frozen=True)
class FundamentalUpdateResult:
    rows_appended: dict[str, int]
    updated_paths: list[Path]
    warnings: list[str] = field(default_factory=list)


class FundamentalDataProvider(Protocol):
    """Adapter interface for USDA fundamentals providers."""

    def fetch_latest_wasde(self, existing: pd.DataFrame) -> pd.DataFrame:
        """Return normalized WASDE rows using WASDE_COLUMNS."""

    def fetch_latest_crop_conditions(self, existing: pd.DataFrame) -> pd.DataFrame:
        """Return normalized crop-condition rows using CROP_CONDITION_COLUMNS."""


class UsdaApiFundamentalProvider:
    """USDA API adapter using credentials from environment variables.

    Environment variables:
      USDA_FAS_API_KEY or FAS_API_KEY
      USDA_FAS_API_URL or THRESH_FAS_API_URL
      USDA_NASS_API_KEY or NASS_API_KEY
      USDA_NASS_API_URL, defaulting to Quick Stats api_GET

    The adapter expects API responses to expose rows that already match the
    normalized output schemas. This keeps the update step separate from one-off
    historical normalizers and makes the provider easy to replace.
    """

    def __init__(self) -> None:
        self.fas_api_key = os.getenv("USDA_FAS_API_KEY") or os.getenv("FAS_API_KEY")
        self.fas_api_url = os.getenv("USDA_FAS_API_URL") or os.getenv("THRESH_FAS_API_URL")
        self.nass_api_key = os.getenv("USDA_NASS_API_KEY") or os.getenv("NASS_API_KEY")
        self.nass_api_url = os.getenv(
            "USDA_NASS_API_URL",
            "https://quickstats.nass.usda.gov/api/api_GET/",
        )

    def fetch_latest_wasde(self, existing: pd.DataFrame) -> pd.DataFrame:
        if not self.fas_api_key or not self.fas_api_url:
            raise ValueError("USDA_FAS_API_KEY and USDA_FAS_API_URL must be set to update WASDE data.")
        latest_date = _latest_date_string(existing)
        payload = self._get_json(
            self.fas_api_url,
            {
                "api_key": self.fas_api_key,
                "since": latest_date or "",
            },
        )
        rows = payload.get("data", payload if isinstance(payload, list) else [])
        return _normalize_wasde_api_rows(pd.DataFrame(rows))

    def fetch_latest_crop_conditions(self, existing: pd.DataFrame) -> pd.DataFrame:
        if not self.nass_api_key:
            raise ValueError("USDA_NASS_API_KEY must be set to update crop-condition data.")
        latest_date = _latest_date_string(existing)
        params = {
            "key": self.nass_api_key,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "agg_level_desc": "NATIONAL",
            "commodity_desc": "CORN",
            "statisticcat_desc": "CONDITION",
            "format": "JSON",
        }
        if latest_date:
            params["week_ending__GE"] = latest_date
        payload = self._get_json(self.nass_api_url, params)
        rows = payload.get("data", payload if isinstance(payload, list) else [])
        return _normalize_quickstats_crop_rows(pd.DataFrame(rows))

    @staticmethod
    def _get_json(url: str, params: dict[str, str]) -> object:
        query = urlencode({key: value for key, value in params.items() if value != ""})
        request_url = f"{url}?{query}" if query else url
        request = Request(request_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


def update_fundamental_data(
    provider: FundamentalDataProvider,
    wasde_path: str | Path = Path("data/fundamentals/wasde.csv"),
    crop_condition_path: str | Path = Path("data/fundamentals/crop_conditions.csv"),
) -> FundamentalUpdateResult:
    """Fetch latest USDA fundamentals and append unseen normalized rows."""
    wasde_output = Path(wasde_path)
    crop_output = Path(crop_condition_path)
    existing_wasde = _read_existing(wasde_output, WASDE_COLUMNS)
    existing_crop = _read_existing(crop_output, CROP_CONDITION_COLUMNS)
    _validate_wasde(existing_wasde)
    _validate_crop_conditions(existing_crop)

    # Fetch both before writing either file. If any API call fails, existing
    # good data is left untouched.
    new_wasde = provider.fetch_latest_wasde(existing_wasde)
    new_crop = provider.fetch_latest_crop_conditions(existing_crop)

    new_wasde = _coerce_wasde(new_wasde)
    new_crop = _coerce_crop_conditions(new_crop)
    _validate_wasde(new_wasde)
    _validate_crop_conditions(new_crop)

    combined_wasde = _append_unseen(
        existing_wasde,
        new_wasde,
        key_columns=["date", "commodity", "marketing_year"],
        output_columns=WASDE_COLUMNS,
    )
    combined_crop = _append_unseen(
        existing_crop,
        new_crop,
        key_columns=["date", "commodity"],
        output_columns=CROP_CONDITION_COLUMNS,
    )
    _validate_wasde(combined_wasde)
    _validate_crop_conditions(combined_crop)

    wasde_appended = len(combined_wasde) - len(existing_wasde)
    crop_appended = len(combined_crop) - len(existing_crop)
    updated_paths: list[Path] = []
    if wasde_appended > 0:
        _atomic_write_csv(combined_wasde, wasde_output)
        updated_paths.append(wasde_output)
    if crop_appended > 0:
        _atomic_write_csv(combined_crop, crop_output)
        updated_paths.append(crop_output)

    return FundamentalUpdateResult(
        rows_appended={
            "wasde": wasde_appended,
            "crop_conditions": crop_appended,
        },
        updated_paths=updated_paths,
    )


def _coerce_wasde(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=WASDE_COLUMNS)
    missing = set(WASDE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Fetched WASDE data missing columns: {sorted(missing)}")
    frame = data[WASDE_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["commodity"] = frame["commodity"].astype(str).str.lower().str.strip()
    frame["marketing_year"] = frame["marketing_year"].astype(str).str.strip()
    for column in ["production", "exports", "ending_stocks", "total_use", "stocks_to_use"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _coerce_crop_conditions(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=CROP_CONDITION_COLUMNS)
    missing = set(CROP_CONDITION_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Fetched crop-condition data missing columns: {sorted(missing)}")
    frame = data[CROP_CONDITION_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["commodity"] = frame["commodity"].astype(str).str.lower().str.strip()
    for column in ["good_pct", "excellent_pct", "good_excellent_pct"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _normalize_wasde_api_rows(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=WASDE_COLUMNS)
    if set(WASDE_COLUMNS).issubset(data.columns):
        return _coerce_wasde(data)

    required = {"ReportTitle", "Attribute", "Commodity", "Region", "MarketYear", "Value", "ReleaseDate"}
    if not required.issubset(data.columns):
        raise ValueError(
            "USDA FAS/WASDE response must contain normalized WASDE columns or "
            f"OCE-style columns: {sorted(required)}"
        )

    specs = [
        ("corn", "Corn", "U.S. Feed Grain and Corn Supply and Use"),
        ("soybeans", "Oilseed, Soybean", "U.S. Soybeans and Products Supply and Use (Domestic Measure)"),
    ]
    attribute_map = {
        "production": "production",
        "exports": "exports",
        "ending stocks": "ending_stocks",
        "use total": "total_use",
    }
    frames = []
    for commodity, raw_commodity, report_title in specs:
        subset = data[
            data["ReportTitle"].astype(str).eq(report_title)
            & data["Commodity"].astype(str).eq(raw_commodity)
            & data["Region"].map(_normalize_text).isin({"united states", "u s", "us"})
        ].copy()
        if subset.empty:
            continue
        subset["metric"] = subset["Attribute"].map(lambda value: attribute_map.get(_normalize_text(value)))
        subset = subset[subset["metric"].notna()].copy()
        subset["date"] = pd.to_datetime(subset["ReleaseDate"], errors="coerce").dt.date.astype(str)
        subset["commodity"] = commodity
        subset["marketing_year"] = subset["MarketYear"].astype(str).str.strip()
        subset["value"] = pd.to_numeric(subset["Value"], errors="coerce")
        long_data = subset[["date", "commodity", "marketing_year", "metric", "value"]]
        if long_data.duplicated(["date", "commodity", "marketing_year", "metric"], keep=False).any():
            raise ValueError("USDA FAS/WASDE response contains duplicate metric rows.")
        wide = (
            long_data.pivot(
                index=["date", "commodity", "marketing_year"],
                columns="metric",
                values="value",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        frames.append(wide)
    if not frames:
        return pd.DataFrame(columns=WASDE_COLUMNS)
    normalized = pd.concat(frames, ignore_index=True)
    missing = [column for column in ["production", "exports", "ending_stocks", "total_use"] if column not in normalized]
    if missing:
        raise ValueError(f"USDA FAS/WASDE response missing required metrics: {missing}")
    normalized["stocks_to_use"] = (normalized["ending_stocks"] / normalized["total_use"]) * 100
    return _coerce_wasde(normalized[WASDE_COLUMNS])


def _normalize_quickstats_crop_rows(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=CROP_CONDITION_COLUMNS)
    if set(CROP_CONDITION_COLUMNS).issubset(data.columns):
        return _coerce_crop_conditions(data)

    column_map = {_normalize_name(column): column for column in data.columns}
    date_column = column_map.get("weekending") or column_map.get("weekendingdate")
    commodity_column = column_map.get("commoditydesc") or column_map.get("commodity")
    item_column = column_map.get("shortdesc") or column_map.get("dataitem")
    value_column = column_map.get("value")
    agg_column = column_map.get("aggleveldesc")
    if not all([date_column, commodity_column, item_column, value_column]):
        raise ValueError("NASS Quick Stats response is missing crop-condition columns.")

    subset = data.copy()
    if agg_column is not None:
        subset = subset[subset[agg_column].map(_normalize_text).eq("national")].copy()
    subset["commodity"] = subset[commodity_column].map(_crop_commodity)
    subset = subset[subset["commodity"].isin({"corn", "soybeans"})].copy()
    subset["metric"] = subset[item_column].map(_crop_metric)
    subset = subset[subset["metric"].notna()].copy()
    subset["date"] = pd.to_datetime(subset[date_column], errors="coerce").dt.date.astype(str)
    subset["value"] = pd.to_numeric(subset[value_column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if subset.empty:
        return pd.DataFrame(columns=CROP_CONDITION_COLUMNS)
    long_data = subset[["date", "commodity", "metric", "value"]]
    if long_data.duplicated(["date", "commodity", "metric"], keep=False).any():
        raise ValueError("NASS Quick Stats response contains duplicate crop-condition metric rows.")
    wide = (
        long_data.pivot(index=["date", "commodity"], columns="metric", values="value")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    missing = [column for column in ["good_pct", "excellent_pct"] if column not in wide]
    if missing:
        raise ValueError(f"NASS Quick Stats response missing required condition metrics: {missing}")
    wide["good_excellent_pct"] = wide["good_pct"] + wide["excellent_pct"]
    return _coerce_crop_conditions(wide[CROP_CONDITION_COLUMNS])


def _validate_wasde(data: pd.DataFrame) -> None:
    missing = set(WASDE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"WASDE data missing columns: {sorted(missing)}")
    if data.empty:
        return
    if data[["date", "commodity", "marketing_year"]].isna().any().any():
        raise ValueError("WASDE data contains missing key values.")
    duplicate_keys = data.duplicated(["date", "commodity", "marketing_year"], keep=False)
    if duplicate_keys.any():
        examples = data.loc[duplicate_keys, ["date", "commodity", "marketing_year"]].head(8)
        raise ValueError(f"WASDE data has duplicate rows: {examples.to_dict(orient='records')}")
    for column in ["production", "exports", "ending_stocks", "total_use", "stocks_to_use"]:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"WASDE column {column} must be numeric.")


def _validate_crop_conditions(data: pd.DataFrame) -> None:
    missing = set(CROP_CONDITION_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Crop-condition data missing columns: {sorted(missing)}")
    if data.empty:
        return
    if data[["date", "commodity"]].isna().any().any():
        raise ValueError("Crop-condition data contains missing key values.")
    duplicate_keys = data.duplicated(["date", "commodity"], keep=False)
    if duplicate_keys.any():
        examples = data.loc[duplicate_keys, ["date", "commodity"]].head(8)
        raise ValueError(f"Crop-condition data has duplicate rows: {examples.to_dict(orient='records')}")
    for column in ["good_pct", "excellent_pct", "good_excellent_pct"]:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Crop-condition column {column} must be numeric.")


def _append_unseen(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    key_columns: list[str],
    output_columns: list[str],
) -> pd.DataFrame:
    if new_rows.empty:
        return existing[output_columns].copy()
    existing_keys = set(tuple(row) for row in existing[key_columns].astype(str).itertuples(index=False, name=None))
    key_frame = new_rows[key_columns].astype(str)
    unseen_mask = [tuple(row) not in existing_keys for row in key_frame.itertuples(index=False, name=None)]
    unseen = new_rows.loc[unseen_mask]
    combined = pd.concat([existing, unseen], ignore_index=True)
    return combined[output_columns].sort_values(key_columns).reset_index(drop=True)


def _read_existing(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)[columns]


def _latest_date_string(data: pd.DataFrame) -> str | None:
    if data.empty or "date" not in data.columns:
        return None
    latest = pd.to_datetime(data["date"], errors="coerce").max()
    if pd.isna(latest):
        return None
    return str(latest.date())


def _atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def _crop_commodity(value: object) -> str | None:
    normalized = _normalize_text(value)
    if normalized == "corn":
        return "corn"
    if normalized in {"soybean", "soybeans"}:
        return "soybeans"
    return None


def _crop_metric(value: object) -> str | None:
    normalized = _normalize_text(value)
    if "condition measured in pct excellent" in normalized:
        return "excellent_pct"
    if "condition measured in pct good" in normalized:
        return "good_pct"
    return None


def _normalize_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
