"""Normalize USDA crop-condition exports for corn and soybeans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


OUTPUT_COLUMNS = [
    "date",
    "commodity",
    "good_pct",
    "excellent_pct",
    "good_excellent_pct",
]

COMMODITY_MAP = {
    "corn": "corn",
    "soybean": "soybeans",
    "soybeans": "soybeans",
}

CONDITION_MAP = {
    "good": "good_pct",
    "excellent": "excellent_pct",
}


@dataclass(frozen=True)
class CropConditionNormalizerResult:
    data: pd.DataFrame
    summary: list[str]
    warnings: list[str]


def normalize_crop_condition_directory(input_dir: str | Path) -> CropConditionNormalizerResult:
    """Read NASS crop-condition CSV exports and return normalized weekly rows."""
    paths = sorted(Path(input_dir).rglob("*.csv"))
    if not paths:
        raise ValueError(f"No crop-condition CSV files found in {input_dir}")

    frames = []
    for path in paths:
        raw = pd.read_csv(path, low_memory=False)
        extracted = _extract_file(raw, path)
        if not extracted.empty:
            frames.append(extracted)

    if not frames:
        raise ValueError("No U.S. national corn or soybean crop-condition rows found.")

    long_data = pd.concat(frames, ignore_index=True)
    _validate_long_data(long_data)

    duplicate_metric = long_data.duplicated(
        subset=["date", "commodity", "condition"],
        keep=False,
    )
    if duplicate_metric.any():
        examples = (
            long_data.loc[duplicate_metric, ["date", "commodity", "condition", "source_file"]]
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(f"Duplicate crop-condition metric rows found: {examples}")

    data = (
        long_data.pivot(
            index=["date", "commodity"],
            columns="condition",
            values="value",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    missing_columns = [column for column in CONDITION_MAP.values() if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Crop-condition rows missing required metrics: {missing_columns}")

    missing_metrics = [column for column in CONDITION_MAP.values() if data[column].isna().any()]
    if missing_metrics:
        raise ValueError(f"Crop-condition rows missing required metrics: {missing_metrics}")

    data["good_excellent_pct"] = data["good_pct"] + data["excellent_pct"]
    data = data[OUTPUT_COLUMNS].sort_values(["commodity", "date"]).reset_index(drop=True)

    _validate_output(data)
    warnings = _warnings(long_data, data)
    summary = summarize_crop_conditions(data)
    return CropConditionNormalizerResult(data=data, summary=summary, warnings=warnings)


def write_normalized_crop_conditions(
    input_dir: str | Path,
    output_path: str | Path,
) -> CropConditionNormalizerResult:
    """Normalize crop conditions and write the processed CSV."""
    result = normalize_crop_condition_directory(input_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(output, index=False)
    return result


def summarize_crop_conditions(data: pd.DataFrame) -> list[str]:
    lines = []
    for commodity, group in data.groupby("commodity", sort=True):
        dates = pd.to_datetime(group["date"], errors="coerce")
        lines.append(
            f"{commodity}: {len(group)} rows, {dates.min().date()} to {dates.max().date()}"
        )
    return lines


def _extract_file(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {
        "Year",
        "Week Ending",
        "Geo Level",
        "State",
        "Commodity",
        "Data Item",
        "Value",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} missing crop-condition columns: {sorted(missing)}")

    data = raw[
        raw["Geo Level"].map(_normalize_name).eq("national")
        & raw["State"].map(_normalize_name).isin({"us total", "united states"})
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=["date", "commodity", "condition", "value", "year", "source_file"])

    data["commodity"] = data["Commodity"].map(_normalize_commodity)
    data = data[data["commodity"].isin({"corn", "soybeans"})].copy()
    if data.empty:
        return pd.DataFrame(columns=["date", "commodity", "condition", "value", "year", "source_file"])

    data["condition"] = data["Data Item"].map(_condition_from_data_item)
    data = data[data["condition"].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=["date", "commodity", "condition", "value", "year", "source_file"])

    data["date"] = pd.to_datetime(data["Week Ending"], errors="coerce").dt.date.astype(str)
    data["year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["value"] = pd.to_numeric(data["Value"], errors="coerce")
    data["source_file"] = str(path)
    data = data[data["year"].between(2010, 2026, inclusive="both")].copy()

    return data[["date", "commodity", "condition", "value", "year", "source_file"]]


def _validate_long_data(data: pd.DataFrame) -> None:
    if data["date"].eq("NaT").any():
        examples = data.loc[data["date"].eq("NaT"), ["source_file", "commodity", "condition"]].head(5)
        raise ValueError(f"Crop-condition rows contain invalid dates: {examples.to_dict(orient='records')}")

    if data["year"].isna().any():
        examples = data.loc[data["year"].isna(), ["source_file", "commodity", "date"]].head(5)
        raise ValueError(f"Crop-condition rows contain invalid years: {examples.to_dict(orient='records')}")

    if data["value"].isna().any():
        examples = data.loc[data["value"].isna(), ["source_file", "commodity", "date", "condition"]].head(5)
        raise ValueError(f"Crop-condition percentages must be numeric: {examples.to_dict(orient='records')}")

    outside_pct = ~data["value"].between(0, 100, inclusive="both")
    if outside_pct.any():
        examples = data.loc[outside_pct, ["source_file", "commodity", "date", "condition", "value"]].head(5)
        raise ValueError(f"Crop-condition percentages must be between 0 and 100: {examples.to_dict(orient='records')}")


def _validate_output(data: pd.DataFrame) -> None:
    duplicate_mask = data.duplicated(subset=["date", "commodity"], keep=False)
    if duplicate_mask.any():
        examples = data.loc[duplicate_mask, ["date", "commodity"]].head(8).to_dict(orient="records")
        raise ValueError(f"Duplicate crop-condition date + commodity rows found: {examples}")

    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Crop-condition output contains invalid dates.")

    for commodity, group in data.assign(_date=parsed_dates).groupby("commodity", sort=False):
        if not group["_date"].is_monotonic_increasing:
            raise ValueError(f"Crop-condition rows are not chronological for {commodity}.")

    numeric_columns = ["good_pct", "excellent_pct", "good_excellent_pct"]
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} must be numeric.")
        if not values.between(0, 100, inclusive="both").all():
            raise ValueError(f"{column} must be between 0 and 100.")


def _warnings(long_data: pd.DataFrame, output: pd.DataFrame) -> list[str]:
    warnings = []
    present = set(output["commodity"].unique())
    for commodity in ("corn", "soybeans"):
        if commodity not in present:
            warnings.append(f"No U.S. national {commodity} crop-condition rows found.")

    for commodity, commodity_group in long_data.groupby("commodity", sort=True):
        for year, year_group in commodity_group.groupby("year", sort=True):
            dates = pd.to_datetime(year_group["date"], errors="coerce").dropna().drop_duplicates().sort_values()
            if dates.empty:
                continue
            expected = pd.date_range(dates.min(), dates.max(), freq="7D")
            observed = set(dates.dt.date)
            missing = [date.date().isoformat() for date in expected if date.date() not in observed]
            if missing:
                warnings.append(
                    f"{commodity} {int(year)} missing crop-condition weeks: {', '.join(missing)}"
                )
    return warnings


def _normalize_commodity(value: object) -> str | None:
    return COMMODITY_MAP.get(_normalize_name(value))


def _condition_from_data_item(value: object) -> str | None:
    normalized = _normalize_name(value)
    if "condition measured in pct excellent" in normalized:
        return "excellent_pct"
    if "condition measured in pct good" in normalized:
        return "good_pct"
    return None


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
