"""Normalize WASDE fundamentals for corn and soybeans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


OUTPUT_COLUMNS = [
    "date",
    "commodity",
    "marketing_year",
    "production",
    "exports",
    "ending_stocks",
    "total_use",
    "stocks_to_use",
]

ATTRIBUTE_MAP = {
    "production": "production",
    "exports": "exports",
    "ending stocks": "ending_stocks",
    "use total": "total_use",
}


@dataclass(frozen=True)
class WasdeNormalizerResult:
    data: pd.DataFrame
    summary: list[str]
    warnings: list[str]


def normalize_wasde_directory(input_dir: str | Path) -> WasdeNormalizerResult:
    """Read a WASDE CSV folder and return normalized corn/soybean fundamentals."""
    frames = []
    paths = sorted(Path(input_dir).rglob("*.csv"))
    if not paths:
        raise ValueError(f"No WASDE CSV files found in {input_dir}")

    for path in paths:
        raw = pd.read_csv(path, low_memory=False)
        frames.extend(_extract_file(raw, path))

    if not frames:
        raise ValueError("No corn or soybean WASDE supply/use rows found.")

    long_data = pd.concat(frames, ignore_index=True)
    _validate_long_data(long_data)

    duplicate_mask = long_data.duplicated(
        subset=["date", "commodity", "marketing_year", "metric"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            long_data.loc[duplicate_mask, ["date", "commodity", "marketing_year", "metric"]]
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(f"Duplicate WASDE metric rows found: {examples}")

    data = (
        long_data.pivot(
            index=["date", "commodity", "marketing_year"],
            columns="metric",
            values="value",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    required_metrics = list(dict.fromkeys(ATTRIBUTE_MAP.values()))
    missing_columns = [column for column in required_metrics if column not in data.columns]
    if missing_columns:
        raise ValueError(f"WASDE rows missing required metrics: {missing_columns}")

    missing_metrics = [column for column in required_metrics if data[column].isna().any()]
    if missing_metrics:
        raise ValueError(f"WASDE rows missing required metrics: {missing_metrics}")

    data["stocks_to_use"] = (data["ending_stocks"] / data["total_use"]) * 100
    data = data[OUTPUT_COLUMNS].sort_values(["date", "commodity", "marketing_year"]).reset_index(drop=True)

    warnings = _missing_report_warnings(data)
    summary = summarize_wasde(data)
    return WasdeNormalizerResult(data=data, summary=summary, warnings=warnings)


def write_normalized_wasde(
    input_dir: str | Path,
    output_path: str | Path,
) -> WasdeNormalizerResult:
    """Normalize WASDE fundamentals and write the processed CSV."""
    result = normalize_wasde_directory(input_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(output, index=False)
    return result


def summarize_wasde(data: pd.DataFrame) -> list[str]:
    lines = []
    for commodity, group in data.groupby("commodity", sort=True):
        dates = pd.to_datetime(group["date"], errors="coerce")
        marketing_years = sorted(str(year) for year in group["marketing_year"].dropna().unique())
        lines.append(
            f"{commodity}: {len(group)} rows, {dates.min().date()} to {dates.max().date()}, "
            f"{len(marketing_years)} marketing years"
        )
    return lines


def _extract_file(raw: pd.DataFrame, path: Path) -> list[pd.DataFrame]:
    required = {
        "ReportTitle",
        "Attribute",
        "Commodity",
        "Region",
        "MarketYear",
        "Value",
        "ReleaseDate",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} missing WASDE columns: {sorted(missing)}")

    specs = [
        (
            "corn",
            "Corn",
            "U.S. Feed Grain and Corn Supply and Use",
        ),
        (
            "soybeans",
            "Oilseed, Soybean",
            "U.S. Soybeans and Products Supply and Use (Domestic Measure)",
        ),
    ]

    frames = []
    for commodity, raw_commodity, report_title in specs:
        subset = raw[
            raw["ReportTitle"].astype(str).eq(report_title)
            & raw["Commodity"].astype(str).eq(raw_commodity)
            & raw["Region"].map(_normalize_region).eq("united states")
        ].copy()
        if subset.empty:
            continue

        subset["metric"] = subset["Attribute"].map(lambda value: ATTRIBUTE_MAP.get(_normalize_attribute(value)))
        subset = subset[subset["metric"].notna()].copy()
        if subset.empty:
            continue

        subset["date"] = pd.to_datetime(subset["ReleaseDate"], errors="coerce").dt.date.astype(str)
        subset["commodity"] = commodity
        subset["marketing_year"] = subset["MarketYear"].astype(str).str.strip()
        subset["value"] = pd.to_numeric(subset["Value"], errors="coerce")
        subset["source_file"] = str(path)
        frames.append(
            subset[
                [
                    "date",
                    "commodity",
                    "marketing_year",
                    "metric",
                    "value",
                    "source_file",
                ]
            ]
        )
    return frames


def _validate_long_data(data: pd.DataFrame) -> None:
    invalid_dates = data["date"].eq("NaT")
    if invalid_dates.any():
        examples = data.loc[invalid_dates, ["source_file", "commodity", "marketing_year"]].head(5)
        raise ValueError(f"WASDE rows contain invalid ReleaseDate values: {examples.to_dict(orient='records')}")

    if data["marketing_year"].eq("").any() or data["marketing_year"].str.lower().eq("nan").any():
        raise ValueError("WASDE rows contain missing marketing_year values.")

    if data["value"].isna().any():
        examples = data.loc[data["value"].isna(), ["source_file", "commodity", "marketing_year", "metric"]].head(5)
        raise ValueError(f"WASDE metric values must be numeric: {examples.to_dict(orient='records')}")


def _missing_report_warnings(data: pd.DataFrame) -> list[str]:
    warnings = []
    for commodity, group in data.groupby("commodity", sort=True):
        dates = pd.to_datetime(group["date"], errors="coerce").dropna().drop_duplicates().sort_values()
        if dates.empty:
            continue
        expected = pd.period_range(dates.min().to_period("M"), dates.max().to_period("M"), freq="M")
        observed = set(dates.dt.to_period("M"))
        missing_months = [str(period) for period in expected if period not in observed]
        if missing_months:
            warnings.append(
                f"{commodity} missing WASDE report months: {', '.join(missing_months)}"
            )
    return warnings


def _normalize_attribute(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _normalize_region(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    if normalized in {"u s", "us", "usa", "united states"}:
        return "united states"
    return normalized
