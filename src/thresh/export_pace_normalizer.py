"""Normalize weekly export-sales pace data for corn and soybeans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


OUTPUT_COLUMNS = [
    "date",
    "commodity",
    "marketing_year",
    "weekly_exports",
    "cumulative_exports",
    "outstanding_sales",
    "total_commitments",
]

REQUIRED_METRICS = [
    "weekly_exports",
    "cumulative_exports",
    "outstanding_sales",
    "total_commitments",
]


@dataclass(frozen=True)
class ExportPaceNormalizerResult:
    data: pd.DataFrame
    summary: list[str]
    warnings: list[str]


def normalize_export_pace_files(
    corn_input: str | Path,
    soybean_input: str | Path,
) -> ExportPaceNormalizerResult:
    """Normalize corn and soybean export-sales files into one dataset."""
    frames = [
        _normalize_one_file(Path(corn_input), "corn"),
        _normalize_one_file(Path(soybean_input), "soybeans"),
    ]
    data = pd.concat(frames, ignore_index=True)
    data = data[OUTPUT_COLUMNS].sort_values(["commodity", "marketing_year", "date"]).reset_index(drop=True)

    _validate_output(data)
    warnings = _missing_week_warnings(data)
    summary = summarize_export_pace(data)
    return ExportPaceNormalizerResult(data=data, summary=summary, warnings=warnings)


def write_normalized_export_pace(
    corn_input: str | Path,
    soybean_input: str | Path,
    output_path: str | Path,
) -> ExportPaceNormalizerResult:
    """Normalize export pace data and write the processed CSV."""
    result = normalize_export_pace_files(
        corn_input=corn_input,
        soybean_input=soybean_input,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(output, index=False)
    return result


def summarize_export_pace(data: pd.DataFrame) -> list[str]:
    lines = []
    for commodity, group in data.groupby("commodity", sort=True):
        dates = pd.to_datetime(group["date"], errors="coerce")
        years = sorted(str(year) for year in group["marketing_year"].dropna().unique())
        lines.append(
            f"{commodity}: {len(group)} rows, {dates.min().date()} to {dates.max().date()}, "
            f"{len(years)} marketing years"
        )
    return lines


def _normalize_one_file(path: Path, commodity: str) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Export pace input path does not exist: {path}")

    raw = pd.read_csv(path)
    missing = {"instrument", "date", "actual"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path} missing export-pace columns: {sorted(missing)}")

    data = raw.copy()
    data["metric"] = data["instrument"].map(_metric_from_instrument)
    mapped = data[data["metric"].notna()].copy()

    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in set(mapped["metric"])]
    if missing_metrics:
        instruments = sorted(str(value) for value in raw["instrument"].dropna().unique())
        raise ValueError(
            f"{commodity} export sales file is missing required metrics {missing_metrics}. "
            f"Found instruments: {instruments}"
        )

    parsed_dates = pd.to_datetime(mapped["date"], errors="coerce")
    values = pd.to_numeric(mapped["actual"], errors="coerce")
    if parsed_dates.isna().any():
        examples = mapped.loc[parsed_dates.isna(), ["instrument", "date"]].head(5)
        raise ValueError(f"{commodity} export pace contains invalid dates: {examples.to_dict(orient='records')}")
    if values.isna().any():
        examples = mapped.loc[values.isna(), ["instrument", "date", "actual"]].head(5)
        raise ValueError(f"{commodity} export pace values must be numeric: {examples.to_dict(orient='records')}")

    mapped["date"] = parsed_dates.dt.date.astype(str)
    mapped["commodity"] = commodity
    mapped["marketing_year"] = parsed_dates.map(_marketing_year)
    mapped["value"] = values.astype(float)

    duplicate_metrics = mapped.duplicated(
        subset=["date", "commodity", "marketing_year", "metric"],
        keep=False,
    )
    if duplicate_metrics.any():
        examples = (
            mapped.loc[duplicate_metrics, ["date", "commodity", "marketing_year", "metric", "instrument"]]
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(f"Duplicate export-pace metric rows found: {examples}")

    output = (
        mapped.pivot(
            index=["date", "commodity", "marketing_year"],
            columns="metric",
            values="value",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    missing_columns = [metric for metric in REQUIRED_METRICS if metric not in output.columns]
    if missing_columns:
        raise ValueError(f"{commodity} export pace missing required metrics: {missing_columns}")

    incomplete_rows = output[REQUIRED_METRICS].isna().any(axis=1)
    if incomplete_rows.any():
        examples = output.loc[incomplete_rows, ["date", "commodity", "marketing_year"]].head(8)
        raise ValueError(f"{commodity} export pace has incomplete rows: {examples.to_dict(orient='records')}")

    return output[OUTPUT_COLUMNS]


def _validate_output(data: pd.DataFrame) -> None:
    duplicate_keys = data.duplicated(subset=["date", "commodity", "marketing_year"], keep=False)
    if duplicate_keys.any():
        examples = data.loc[duplicate_keys, ["date", "commodity", "marketing_year"]].head(8)
        raise ValueError(f"Duplicate export-pace date + commodity + marketing_year rows found: {examples.to_dict(orient='records')}")

    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Export pace output contains invalid dates.")

    for column in REQUIRED_METRICS:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} must be numeric.")

    for (commodity, marketing_year), group in data.assign(_date=parsed_dates).groupby(
        ["commodity", "marketing_year"],
        sort=False,
    ):
        if not group["_date"].is_monotonic_increasing:
            raise ValueError(f"Export pace rows are not chronological for {commodity} {marketing_year}.")


def _missing_week_warnings(data: pd.DataFrame) -> list[str]:
    warnings = []
    parsed = data.assign(_week=pd.to_datetime(data["date"], errors="coerce").dt.to_period("W-SUN"))
    for (commodity, marketing_year), group in parsed.groupby(["commodity", "marketing_year"], sort=True):
        weeks = group["_week"].dropna().drop_duplicates().sort_values()
        if weeks.empty:
            continue
        expected = pd.period_range(weeks.min(), weeks.max(), freq="W-SUN")
        observed = set(weeks)
        missing = [str(week) for week in expected if week not in observed]
        if missing:
            warnings.append(
                f"{commodity} {marketing_year} missing export-sales weeks: {', '.join(missing)}"
            )
    return warnings


def _metric_from_instrument(value: object) -> str | None:
    normalized = _normalize_name(value)
    if "outstanding" in normalized and "sales" in normalized:
        return "outstanding_sales"
    if "total" in normalized and ("commitment" in normalized or "commitments" in normalized):
        return "total_commitments"
    if ("cumulative" in normalized or "accumulated" in normalized or "year to date" in normalized) and (
        "export" in normalized or "shipment" in normalized
    ):
        return "cumulative_exports"
    if "weekly" in normalized and ("export" in normalized or "shipment" in normalized):
        return "weekly_exports"
    return None


def _marketing_year(date: pd.Timestamp) -> str:
    start_year = date.year if date.month >= 9 else date.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
