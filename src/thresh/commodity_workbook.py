"""Adapters for simple commodity-history workbooks.

This module normalizes workbooks where each sheet contains one commodity series
with date, settlement, volume, and open-interest columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class CommodityWorkbookSheetSpec:
    """Metadata for one sheet in a commodity-history workbook."""

    sheet_name: str
    commodity: str
    series_id: Optional[str] = None
    symbol: Optional[str] = None


DEFAULT_COMMODITY_SHEET_SPECS = (
    CommodityWorkbookSheetSpec(
        sheet_name="CORN",
        commodity="corn",
        series_id="CORN_CC1",
    ),
    CommodityWorkbookSheetSpec(
        sheet_name="SOY",
        commodity="soybeans",
        series_id="SOYBEANS_SC1",
    ),
)


def normalize_commodity_sheet(
    raw: pd.DataFrame,
    spec: CommodityWorkbookSheetSpec,
) -> pd.DataFrame:
    """Normalize one raw workbook sheet into Thresh's seasonal-agent schema."""
    if raw.empty:
        raise ValueError(f"Sheet {spec.sheet_name!r} is empty.")

    header_idx = _find_header_row(raw)
    headers = {
        _normalize_header(value): column
        for column, value in raw.iloc[header_idx].items()
        if not pd.isna(value)
    }

    date_column = _required_header(headers, "timestamp", spec.sheet_name)
    settle_column = _required_header(headers, "close", spec.sheet_name)
    volume_column = headers.get("volume")
    open_interest_column = headers.get("opint_1") or headers.get("open_interest")

    symbol = spec.symbol or _detect_symbol(raw, header_idx)
    if symbol is None:
        raise ValueError(f"Could not detect a symbol for sheet {spec.sheet_name!r}.")

    series_id = spec.series_id or f"{spec.commodity}_{symbol}".upper()
    data = raw.iloc[header_idx + 1 :].copy()

    parsed = pd.DataFrame(index=data.index)
    parsed["date"] = pd.to_datetime(data[date_column], errors="coerce")
    parsed["settle"] = pd.to_numeric(data[settle_column], errors="coerce")
    parsed["volume"] = _optional_numeric(data, volume_column)
    parsed["open_interest"] = _optional_numeric(data, open_interest_column)
    parsed = parsed.dropna(subset=["date", "settle"]).reset_index(drop=True)

    normalized = parsed[["date", "settle"]].copy()
    normalized["date"] = normalized["date"].dt.date.astype(str)
    normalized["year"] = pd.to_datetime(normalized["date"]).dt.year.astype(int)
    normalized["commodity"] = spec.commodity.lower().strip()
    normalized["spread_id"] = series_id.upper().strip()
    normalized["symbol"] = str(symbol).upper().strip()
    normalized["spread_value"] = normalized["settle"]
    normalized["volume"] = parsed["volume"]
    normalized["open_interest"] = parsed["open_interest"]

    columns = [
        "date",
        "year",
        "commodity",
        "spread_id",
        "symbol",
        "settle",
        "spread_value",
        "volume",
        "open_interest",
    ]
    return normalized[columns].sort_values("date").reset_index(drop=True)


def normalize_commodity_workbook(
    input_path: str | Path,
    specs: Iterable[CommodityWorkbookSheetSpec] = DEFAULT_COMMODITY_SHEET_SPECS,
) -> pd.DataFrame:
    """Read and normalize all configured sheets from a workbook."""
    path = Path(input_path)
    frames = []

    try:
        workbook = pd.ExcelFile(path)
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx files requires openpyxl. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc

    available_sheets = set(workbook.sheet_names)
    for spec in specs:
        if spec.sheet_name not in available_sheets:
            raise ValueError(
                f"Workbook {path} does not contain sheet {spec.sheet_name!r}. "
                f"Available sheets: {sorted(available_sheets)}"
            )
        raw = pd.read_excel(path, sheet_name=spec.sheet_name, header=None)
        frames.append(normalize_commodity_sheet(raw, spec))

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["commodity", "spread_id", "date"])
        .reset_index(drop=True)
    )


def write_normalized_commodity_history(
    input_path: str | Path,
    output_path: str | Path,
    specs: Iterable[CommodityWorkbookSheetSpec] = DEFAULT_COMMODITY_SHEET_SPECS,
) -> pd.DataFrame:
    """Normalize a commodity workbook and write the result to CSV."""
    normalized = normalize_commodity_workbook(input_path, specs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return normalized


def summarize_commodity_history(history: pd.DataFrame) -> list[str]:
    """Return compact summary lines for normalized commodity history."""
    required = {"date", "commodity", "spread_id", "settle"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing required normalized columns: {sorted(missing)}")

    df = history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["settle"] = pd.to_numeric(df["settle"], errors="raise")

    lines = []
    for (_, series_id), group in df.groupby(["commodity", "spread_id"], sort=True):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        volume = _format_optional(latest.get("volume"))
        open_interest = _format_optional(latest.get("open_interest"))
        lines.append(
            f"{series_id}: {len(group)} rows, "
            f"{group.iloc[0]['date'].date()} to {latest['date'].date()}, "
            f"latest settle {float(latest['settle']):.2f}, "
            f"volume {volume}, open interest {open_interest}"
        )
    return lines


def _find_header_row(raw: pd.DataFrame) -> int:
    for idx, row in raw.iterrows():
        headers = {_normalize_header(value) for value in row if not pd.isna(value)}
        if "timestamp" in headers and "close" in headers:
            return int(idx)
    raise ValueError("Could not find a header row containing Timestamp and CLOSE.")


def _required_header(headers: dict[str, object], name: str, sheet_name: str) -> object:
    try:
        return headers[name]
    except KeyError as exc:
        raise ValueError(f"Sheet {sheet_name!r} is missing required column {name!r}.") from exc


def _detect_symbol(raw: pd.DataFrame, header_idx: int) -> Optional[str]:
    first_column = raw.iloc[:header_idx, 0]
    for value in first_column:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and not text.lower().startswith("updated at"):
            return text
    return None


def _optional_numeric(raw: pd.DataFrame, column: object | None) -> pd.Series:
    if column is None:
        return pd.Series([pd.NA] * len(raw), index=raw.index, dtype="Float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _normalize_header(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _format_optional(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"
