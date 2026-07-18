"""Adapters for chart-exported calendar spread data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from thresh.spread_builder import contract_symbol


REQUIRED_CHART_COLUMNS = {"Timestamp (UTC)", "Open", "High", "Low", "Close"}


@dataclass(frozen=True)
class ChartCalendarSpreadSpec:
    """Metadata needed to normalize one exported calendar spread series."""

    commodity: str
    spread_id: str
    symbol_prefix: str
    nearby_month: str
    deferred_month: str
    contract_year: int


@dataclass(frozen=True)
class SpreadSeriesSummary:
    """Compact current-context summary for one spread time series."""

    date_start: str
    date_end: str
    observations: int
    latest_close: float
    previous_close: Optional[float]
    one_day_change: Optional[float]
    twenty_day_change: Optional[float]
    fifty_day_change: Optional[float]
    close_percentile: float
    latest_volume: Optional[float]
    latest_ema_20: Optional[float]
    latest_ema_50: Optional[float]
    latest_ema_200: Optional[float]
    trend_state: str

    def to_dict(self) -> dict:
        return {
            "date_start": self.date_start,
            "date_end": self.date_end,
            "observations": self.observations,
            "latest_close": self.latest_close,
            "previous_close": self.previous_close,
            "one_day_change": self.one_day_change,
            "twenty_day_change": self.twenty_day_change,
            "fifty_day_change": self.fifty_day_change,
            "close_percentile": self.close_percentile,
            "latest_volume": self.latest_volume,
            "latest_ema_20": self.latest_ema_20,
            "latest_ema_50": self.latest_ema_50,
            "latest_ema_200": self.latest_ema_200,
            "trend_state": self.trend_state,
        }


def normalize_chart_calendar_spread_csv(
    path: str | Path,
    spec: ChartCalendarSpreadSpec,
) -> pd.DataFrame:
    """Convert an OHLC chart export into Thresh's calendar spread format."""
    raw = pd.read_csv(path)
    missing = REQUIRED_CHART_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required chart-export columns: {sorted(missing)}")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw["Timestamp (UTC)"]).dt.date.astype(str)
    df["year"] = int(spec.contract_year)
    df["commodity"] = spec.commodity.lower().strip()
    df["spread_id"] = spec.spread_id.upper().strip()
    df["nearby_symbol"] = contract_symbol(
        spec.symbol_prefix,
        spec.nearby_month,
        spec.contract_year,
    )
    df["deferred_symbol"] = contract_symbol(
        spec.symbol_prefix,
        spec.deferred_month,
        spec.contract_year,
    )

    for output_column, raw_column in {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    }.items():
        df[output_column] = pd.to_numeric(raw[raw_column], errors="raise")

    df["spread_value"] = df["close"]
    df["volume"] = _optional_numeric(raw, _find_first_column(raw, ["volume"]))
    df["ema_20"] = _optional_numeric(raw, _find_first_column(raw, ["ma", "20", "ema"]))
    df["ema_50"] = _optional_numeric(raw, _find_first_column(raw, ["ma", "50", "ema"]))
    df["ema_200"] = _optional_numeric(raw, _find_first_column(raw, ["ma", "200", "ema"]))
    df["bb_top_20"] = _optional_numeric(raw, _find_first_column(raw, ["bollinger", "top"]))
    df["bb_mid_20"] = _optional_numeric(raw, _find_first_column(raw, ["bollinger", "median"]))
    df["bb_bottom_20"] = _optional_numeric(raw, _find_first_column(raw, ["bollinger", "bottom"]))

    columns = [
        "date",
        "year",
        "commodity",
        "spread_id",
        "nearby_symbol",
        "deferred_symbol",
        "open",
        "high",
        "low",
        "close",
        "spread_value",
        "volume",
        "ema_20",
        "ema_50",
        "ema_200",
        "bb_top_20",
        "bb_mid_20",
        "bb_bottom_20",
    ]
    return df[columns].sort_values("date").reset_index(drop=True)


def write_normalized_calendar_spread(
    input_path: str | Path,
    output_path: str | Path,
    spec: ChartCalendarSpreadSpec,
) -> pd.DataFrame:
    """Normalize a chart export and write it to CSV."""
    normalized = normalize_chart_calendar_spread_csv(input_path, spec)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return normalized


def summarize_spread_series(spread_series: pd.DataFrame) -> SpreadSeriesSummary:
    """Summarize one normalized calendar spread series."""
    required = {"date", "close", "spread_value"}
    missing = required - set(spread_series.columns)
    if missing:
        raise ValueError(f"Missing required normalized columns: {sorted(missing)}")

    df = spread_series.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    latest = df.iloc[-1]
    latest_close = float(latest["close"])

    return SpreadSeriesSummary(
        date_start=str(df.iloc[0]["date"].date()),
        date_end=str(latest["date"].date()),
        observations=int(len(df)),
        latest_close=round(latest_close, 2),
        previous_close=_rounded_value(_lagged_change(df["close"], 1, latest_close, previous=True)),
        one_day_change=_rounded_value(_lagged_change(df["close"], 1, latest_close)),
        twenty_day_change=_rounded_value(_lagged_change(df["close"], 20, latest_close)),
        fifty_day_change=_rounded_value(_lagged_change(df["close"], 50, latest_close)),
        close_percentile=round(_percentile_rank(latest_close, df["close"].to_numpy(dtype=float)), 1),
        latest_volume=_rounded_value(latest.get("volume")),
        latest_ema_20=_rounded_value(latest.get("ema_20")),
        latest_ema_50=_rounded_value(latest.get("ema_50")),
        latest_ema_200=_rounded_value(latest.get("ema_200")),
        trend_state=_classify_trend_state(latest),
    )


def _find_first_column(df: pd.DataFrame, tokens: list[str]) -> Optional[str]:
    for column in df.columns:
        lowered = column.lower()
        if all(token in lowered for token in tokens):
            return column
    return None


def _optional_numeric(raw: pd.DataFrame, column: Optional[str]) -> pd.Series:
    if column is None:
        return pd.Series([pd.NA] * len(raw), dtype="Float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _lagged_change(
    series: pd.Series,
    periods: int,
    latest_close: float,
    previous: bool = False,
) -> Optional[float]:
    if len(series) <= periods:
        return None
    lagged = float(series.iloc[-periods - 1])
    if previous:
        return lagged
    return latest_close - lagged


def _percentile_rank(value: float, sample: np.ndarray) -> float:
    return 100.0 * float(np.mean(sample <= value))


def _rounded_value(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def _classify_trend_state(latest: pd.Series) -> str:
    close = _rounded_value(latest.get("close"))
    ema_20 = _rounded_value(latest.get("ema_20"))
    ema_50 = _rounded_value(latest.get("ema_50"))
    ema_200 = _rounded_value(latest.get("ema_200"))

    if close is None or ema_20 is None or ema_50 is None:
        return "unknown"
    if close > ema_20 > ema_50:
        return "firming"
    if close < ema_20 < ema_50:
        if ema_200 is not None and close < ema_200:
            return "weak below major averages"
        return "weakening"
    return "mixed"
