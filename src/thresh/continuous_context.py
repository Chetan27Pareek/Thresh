"""Market-context reports for continuous commodity histories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CONTEXT_WARNING = "Continuous data is context only; it is not calendar spread seasonality."
REQUIRED_COLUMNS = {"date", "spread_id", "settle"}


@dataclass(frozen=True)
class ContinuousContextReport:
    """Simple current-market context for one continuous commodity series."""

    symbol: str
    latest_date: str
    latest_close: float
    one_day_change: Optional[float]
    twenty_day_change: Optional[float]
    fifty_day_change: Optional[float]
    current_percentile: float
    trend_label: str
    volume: Optional[float]
    open_interest: Optional[float]
    warning: str = CONTEXT_WARNING

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def build_continuous_context_report(
    data: pd.DataFrame,
    symbol: str,
) -> ContinuousContextReport:
    """Build a context report for one continuous series such as CORN_CC1."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required continuous-history columns: {sorted(missing)}")

    series_id = symbol.upper().strip()
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["spread_id"] = df["spread_id"].astype(str).str.upper().str.strip()
    df["settle"] = pd.to_numeric(df["settle"], errors="raise")
    df = df[df["spread_id"] == series_id].sort_values("date").reset_index(drop=True)

    if df.empty:
        available = sorted(data["spread_id"].astype(str).str.upper().unique())
        raise ValueError(f"No continuous history found for symbol {series_id!r}. Available: {available}")

    latest = df.iloc[-1]
    latest_close = float(latest["settle"])
    one_day_change = _lagged_change(df["settle"], 1, latest_close)
    twenty_day_change = _lagged_change(df["settle"], 20, latest_close)
    fifty_day_change = _lagged_change(df["settle"], 50, latest_close)

    return ContinuousContextReport(
        symbol=series_id,
        latest_date=str(latest["date"].date()),
        latest_close=round(latest_close, 2),
        one_day_change=_rounded_optional(one_day_change),
        twenty_day_change=_rounded_optional(twenty_day_change),
        fifty_day_change=_rounded_optional(fifty_day_change),
        current_percentile=round(_percentile_rank(latest_close, df["settle"].to_numpy(dtype=float)), 1),
        trend_label=_trend_label(twenty_day_change, fifty_day_change),
        volume=_rounded_optional(latest.get("volume")),
        open_interest=_rounded_optional(latest.get("open_interest")),
    )


def load_continuous_context_report(
    data_path: str | Path,
    symbol: str,
) -> ContinuousContextReport:
    """Read a normalized continuous-history CSV and build a context report."""
    data = pd.read_csv(data_path)
    return build_continuous_context_report(data, symbol)


def report_to_markdown(report: ContinuousContextReport) -> str:
    """Format a continuous-context report as Markdown."""
    volume_line = _optional_metric_line("Volume", report.volume)
    open_interest_line = _optional_metric_line("Open interest", report.open_interest)

    return f"""
# Continuous Context Report

**Symbol:** {report.symbol}  
**Latest date:** {report.latest_date}  
**Latest close:** {report.latest_close}  
**1-day change:** {_format_optional_number(report.one_day_change)}  
**20-day change:** {_format_optional_number(report.twenty_day_change)}  
**50-day change:** {_format_optional_number(report.fifty_day_change)}  
**Current percentile:** {report.current_percentile}%  
**Trend:** {report.trend_label}{volume_line}{open_interest_line}

> Warning: {report.warning}
""".strip()


def report_to_json(report: ContinuousContextReport, indent: Optional[int] = 2) -> str:
    """Format a continuous-context report as JSON."""
    return report.to_json(indent=indent)


def _lagged_change(
    series: pd.Series,
    periods: int,
    latest_close: float,
) -> Optional[float]:
    if len(series) <= periods:
        return None
    return latest_close - float(series.iloc[-periods - 1])


def _percentile_rank(value: float, sample: np.ndarray) -> float:
    return 100.0 * float(np.mean(sample <= value))


def _trend_label(
    twenty_day_change: Optional[float],
    fifty_day_change: Optional[float],
) -> str:
    if twenty_day_change is None or fifty_day_change is None:
        return "Neutral"
    if twenty_day_change > 0 and fifty_day_change > 0:
        return "Strong"
    if twenty_day_change < 0 and fifty_day_change < 0:
        return "Weak"
    return "Neutral"


def _rounded_optional(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def _optional_metric_line(label: str, value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"  \n**{label}:** {_format_optional_number(value)}"


def _format_optional_number(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"
