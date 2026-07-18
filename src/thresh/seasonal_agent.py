"""
Thresh Seasonal Agent v1
------------------------

This is the first specialist agent in the Thresh system.

Its job is intentionally narrow:
    For one calendar spread and one analysis date, ask what history usually did
    from this point in the crop year over the next N days.

Important learning idea:
    An "agent" is not just an LLM prompt. In trading systems, an agent should be
    a specialist decision unit with:
        1. a narrow job,
        2. clean inputs,
        3. repeatable calculations,
        4. structured output,
        5. explicit uncertainty.

This version is deterministic and data-driven. Later, we can add an LLM layer to
turn the structured output into a senior-trader narrative, but the numbers should
come from code first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "commodity",
    "spread_id",
    "spread_value",
}
MARKET_CONTEXT_NOTE = (
    "Seasonal signal is based on spread data only. "
    "Continuous data is shown only as background context."
)


@dataclass(frozen=True)
class SeasonalAgentConfig:
    """Configuration for the Seasonal Agent.

    Attributes
    ----------
    commodity:
        Commodity name used in the dataset, for example "corn" or "soybeans".
    spread_id:
        Spread identifier used in the dataset, for example "CORN_NZ".
        In this convention, N = July and Z = December.
    horizon_days:
        Calendar-day horizon for the seasonal test. 30 means:
        "What usually happened over the next 30 calendar days?"
    min_years:
        Minimum number of historical years required before trusting the result.
    bullish_threshold_cents:
        Minimum median positive move required before calling the signal bullish.
    bearish_threshold_cents:
        Minimum median negative move required before calling the signal bearish.
    """

    commodity: str
    spread_id: str
    horizon_days: int = 30
    min_years: int = 10
    bullish_threshold_cents: float = 2.0
    bearish_threshold_cents: float = -2.0

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be greater than zero.")
        if self.min_years <= 0:
            raise ValueError("min_years must be greater than zero.")


@dataclass(frozen=True)
class SeasonalAgentReport:
    """Structured output from the Seasonal Agent.

    This structure is designed so a future Supervisor Agent can consume it.
    """

    commodity: str
    spread_id: str
    analysis_date: str
    analysis_date_is_latest: bool
    analysis_date_message: str
    crop_stage: str
    horizon_days: int
    current_spread: float
    current_volume: Optional[float]
    current_open_interest: Optional[float]
    continuous_context: Optional[dict[str, Any]]
    move_already_priced: Optional[float]
    historical_years_used: int
    seasonal_signal: str
    median_move: float
    average_move: float
    win_rate: float
    percentile_25: float
    percentile_75: float
    best_move: float
    worst_move: float
    consistency_score: str
    current_level_percentile: float
    entry_assessment: str
    top_analog_years: list[dict]
    warning: Optional[str]
    narrative: str

    def to_dict(self) -> dict:
        """Return the report as plain Python data for APIs and supervisors."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Return the report as JSON."""
        return json.dumps(self.to_dict(), indent=indent)


class SeasonalAgent:
    """Data-driven seasonal spread agent.

    The agent expects a DataFrame with at least these columns:
        date, year, commodity, spread_id, spread_value

    spread_value should be in cents per bushel. For outright/continuous series,
    spread_value can be set equal to the settlement price.

    Example spread convention:
        Corn July-Dec = July corn settle - December corn settle
        If July = 445.00 and December = 430.00, spread_value = +15.00
    """

    def __init__(self, spread_history: pd.DataFrame, config: SeasonalAgentConfig):
        self.config = config
        self.history = self._prepare_history(spread_history)

    @staticmethod
    def _prepare_history(spread_history: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS - set(spread_history.columns)
        if "year" not in spread_history.columns and "contract_year" not in spread_history.columns:
            missing.add("year or contract_year")
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = spread_history.copy()
        df["date"] = pd.to_datetime(df["date"])
        if "year" not in df.columns:
            df["year"] = df["contract_year"]
        df["year"] = df["year"].astype(int)
        df["commodity"] = df["commodity"].str.lower()
        df["spread_id"] = df["spread_id"].str.upper()
        df = df.sort_values(["commodity", "spread_id", "year", "date"]).reset_index(drop=True)
        return df

    def analyze(
        self,
        analysis_date: str,
        current_spread: Optional[float] = None,
        continuous_context: Optional[object] = None,
        analysis_date_is_latest: bool = False,
    ) -> SeasonalAgentReport:
        """Run the seasonal analysis for one date.

        Parameters
        ----------
        analysis_date:
            Date being analyzed. Example: "2024-06-07".
        current_spread:
            Optional current spread value. If not supplied, the agent tries to
            read it from the dataset for analysis_date.
        continuous_context:
            Optional market-context payload to attach to the report. It is not
            used in any seasonal signal calculation.
        analysis_date_is_latest:
            True when the CLI selected the latest available date automatically.
        """
        dt = pd.Timestamp(analysis_date)
        df = self._filter_spread()
        try:
            current_row = self._get_current_row(df, dt)
        except ValueError:
            if current_spread is None:
                raise
            current_row = None

        if current_spread is None:
            current_spread = float(current_row["spread_value"])
        current_volume = self._optional_row_value(current_row, "volume")
        current_open_interest = self._optional_row_value(current_row, "open_interest")

        analogs = self._build_historical_analogs(df, dt)

        warning = None
        if len(analogs) < self.config.min_years:
            warning = (
                f"Only {len(analogs)} historical analog years found. "
                f"Minimum requested is {self.config.min_years}. Treat the signal cautiously."
            )

        moves = analogs["move_cents"].to_numpy(dtype=float)
        median_move = float(np.median(moves))
        average_move = float(np.mean(moves))
        p25 = float(np.percentile(moves, 25))
        p75 = float(np.percentile(moves, 75))
        best = float(np.max(moves))
        worst = float(np.min(moves))

        seasonal_signal = self._classify_signal(median_move, moves)
        win_rate = self._calculate_win_rate(seasonal_signal, moves)
        consistency = self._classify_consistency(win_rate)

        current_level_percentile = self._percentile_rank(
            value=current_spread,
            sample=analogs["entry_spread"].to_numpy(dtype=float),
        )
        move_already_priced = self._calculate_move_already_priced(current_spread, analogs)
        entry_assessment = self._classify_entry(seasonal_signal, current_level_percentile)
        crop_stage = self._simple_crop_stage(self.config.commodity, dt)
        top_analogs = self._rank_top_analogs(analogs, current_spread)

        narrative = self._build_narrative(
            current_spread=current_spread,
            crop_stage=crop_stage,
            seasonal_signal=seasonal_signal,
            median_move=median_move,
            average_move=average_move,
            win_rate=win_rate,
            p25=p25,
            p75=p75,
            current_level_percentile=current_level_percentile,
            entry_assessment=entry_assessment,
            current_volume=current_volume,
            current_open_interest=current_open_interest,
            years=len(analogs),
        )

        return SeasonalAgentReport(
            commodity=self.config.commodity,
            spread_id=self.config.spread_id,
            analysis_date=str(dt.date()),
            analysis_date_is_latest=bool(analysis_date_is_latest),
            analysis_date_message=self._analysis_date_message(dt, analysis_date_is_latest),
            crop_stage=crop_stage,
            horizon_days=self.config.horizon_days,
            current_spread=round(float(current_spread), 2),
            current_volume=self._rounded_optional(current_volume),
            current_open_interest=self._rounded_optional(current_open_interest),
            continuous_context=self._normalize_continuous_context(continuous_context),
            move_already_priced=self._rounded_optional(move_already_priced),
            historical_years_used=int(len(analogs)),
            seasonal_signal=seasonal_signal,
            median_move=round(median_move, 2),
            average_move=round(average_move, 2),
            win_rate=round(float(win_rate), 3),
            percentile_25=round(p25, 2),
            percentile_75=round(p75, 2),
            best_move=round(best, 2),
            worst_move=round(worst, 2),
            consistency_score=consistency,
            current_level_percentile=round(float(current_level_percentile), 1),
            entry_assessment=entry_assessment,
            top_analog_years=top_analogs,
            warning=warning,
            narrative=narrative,
        )

    def _filter_spread(self) -> pd.DataFrame:
        df = self.history[
            (self.history["commodity"] == self.config.commodity.lower())
            & (self.history["spread_id"] == self.config.spread_id.upper())
        ].copy()

        if df.empty:
            raise ValueError(
                f"No data found for commodity={self.config.commodity!r}, "
                f"spread_id={self.config.spread_id!r}"
            )
        return df

    @staticmethod
    def _get_current_row(df: pd.DataFrame, dt: pd.Timestamp) -> pd.Series:
        exact = df[df["date"] == dt]
        if not exact.empty:
            current_year = exact[exact["year"] == dt.year]
            if not current_year.empty:
                return current_year.sort_values(["year", "date"]).iloc[-1]
            eligible = exact[exact["year"] <= dt.year]
            if not eligible.empty:
                return eligible.sort_values(["year", "date"]).iloc[-1]
            return exact.sort_values(["year", "date"]).iloc[-1]

        # If the exact date is a weekend/holiday, use the nearest previous date.
        previous = df[df["date"] < dt]
        if previous.empty:
            raise ValueError(
                "current_spread was not supplied and no earlier spread value exists in the dataset."
            )
        current_year = previous[previous["year"] == dt.year]
        if not current_year.empty:
            return current_year.sort_values("date").iloc[-1]
        eligible = previous[previous["year"] <= dt.year]
        if not eligible.empty:
            return eligible.sort_values(["year", "date"]).iloc[-1]
        return previous.sort_values(["year", "date"]).iloc[-1]

    @staticmethod
    def _get_current_spread(df: pd.DataFrame, dt: pd.Timestamp) -> float:
        return float(SeasonalAgent._get_current_row(df, dt)["spread_value"])

    def _build_historical_analogs(self, df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
        """Find same-calendar-date historical examples and their future moves.

        For each historical year, we find the trading day nearest to the same
        month/day as analysis_date. Then we find the first available trading day
        on or after horizon_days later.
        """
        rows: list[dict] = []
        target_month = dt.month
        target_day = dt.day

        for year, year_df in df.groupby("year"):
            if year >= dt.year:
                # Avoid lookahead: only completed prior years can be analog history.
                continue

            entry_target = self._same_month_day(year, target_month, target_day)
            entry_row = self._nearest_trading_row(year_df, entry_target, max_distance_days=5)
            if entry_row is None:
                continue

            future_target = entry_row["date"] + pd.Timedelta(days=self.config.horizon_days)
            future_row = self._first_trading_row_on_or_after(year_df, future_target, max_distance_days=10)
            if future_row is None:
                continue

            entry_spread = float(entry_row["spread_value"])
            future_spread = float(future_row["spread_value"])
            rows.append(
                {
                    "year": int(year),
                    "entry_date": entry_row["date"],
                    "future_date": future_row["date"],
                    "entry_spread": entry_spread,
                    "future_spread": future_spread,
                    "move_cents": future_spread - entry_spread,
                }
            )

        analogs = pd.DataFrame(rows)
        if analogs.empty:
            prior_years = sorted(int(year) for year in df["year"].unique() if year < dt.year)
            if not prior_years:
                available_years = sorted(int(year) for year in df["year"].unique())
                raise ValueError(
                    "No prior contract years are available for seasonal analogs. "
                    f"Analysis year is {dt.year}; data contains years {available_years}. "
                    "For a seasonal signal, provide prior contract-year histories for this spread."
                )
            raise ValueError("No historical analogs found for this date and horizon.")
        return analogs.sort_values("year").reset_index(drop=True)

    @staticmethod
    def _same_month_day(year: int, month: int, day: int) -> pd.Timestamp:
        """Create a same-calendar-date target, handling leap-day gracefully."""
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            if month == 2 and day == 29:
                return pd.Timestamp(year=year, month=2, day=28)
            raise

    @staticmethod
    def _nearest_trading_row(
        year_df: pd.DataFrame,
        target_date: pd.Timestamp,
        max_distance_days: int,
    ) -> Optional[pd.Series]:
        temp = year_df.copy()
        temp["distance"] = (temp["date"] - target_date).abs().dt.days
        temp = temp[temp["distance"] <= max_distance_days]
        if temp.empty:
            return None
        return temp.sort_values(["distance", "date"]).iloc[0]

    @staticmethod
    def _first_trading_row_on_or_after(
        year_df: pd.DataFrame,
        target_date: pd.Timestamp,
        max_distance_days: int,
    ) -> Optional[pd.Series]:
        temp = year_df[year_df["date"] >= target_date].copy()
        if temp.empty:
            return None
        temp["distance"] = (temp["date"] - target_date).dt.days
        temp = temp[temp["distance"] <= max_distance_days]
        if temp.empty:
            return None
        return temp.sort_values("date").iloc[0]

    def _classify_signal(self, median_move: float, moves: np.ndarray) -> str:
        positive_win_rate = float(np.mean(moves > 0))
        negative_win_rate = float(np.mean(moves < 0))

        if median_move >= self.config.bullish_threshold_cents and positive_win_rate >= 0.55:
            return "Bullish"
        if median_move <= self.config.bearish_threshold_cents and negative_win_rate >= 0.55:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _calculate_win_rate(seasonal_signal: str, moves: np.ndarray) -> float:
        if seasonal_signal == "Bullish":
            return float(np.mean(moves > 0))
        if seasonal_signal == "Bearish":
            return float(np.mean(moves < 0))
        # For neutral signals, use the larger directional tendency but do not
        # pretend it is a tradeable edge.
        return float(max(np.mean(moves > 0), np.mean(moves < 0)))

    @staticmethod
    def _classify_consistency(win_rate: float) -> str:
        if win_rate >= 0.70:
            return "High"
        if win_rate >= 0.58:
            return "Medium"
        return "Low"

    @staticmethod
    def _percentile_rank(value: float, sample: np.ndarray) -> float:
        """Return percentile rank from 0 to 100."""
        return 100.0 * float(np.mean(sample <= value))

    @staticmethod
    def _classify_entry(seasonal_signal: str, current_level_percentile: float) -> str:
        """Simple first version of entry quality.

        For a bullish spread thesis, a low/medium percentile is usually better
        because there may be more room for the spread to rise.

        For a bearish spread thesis, a high/medium percentile is usually better
        because there may be more room for the spread to fall.
        """
        if seasonal_signal == "Bullish":
            if current_level_percentile <= 40:
                return "Attractive"
            if current_level_percentile <= 70:
                return "Fair"
            return "Stretched"

        if seasonal_signal == "Bearish":
            if current_level_percentile >= 60:
                return "Attractive"
            if current_level_percentile >= 30:
                return "Fair"
            return "Stretched"

        return "No clear entry"

    @staticmethod
    def _simple_crop_stage(commodity: str, dt: pd.Timestamp) -> str:
        """Very simple crop stage logic.

        This will later be replaced by the real Crop Calendar Engine described
        in the Thresh blueprint.
        """
        m, d = dt.month, dt.day
        commodity = commodity.lower()

        if commodity == "corn":
            if m in (3, 4):
                return "pre-planting / planting"
            if m == 5:
                return "planting / emergence"
            if m == 6 or (m == 7 and d <= 20):
                return "pre-pollination / pollination risk window"
            if (m == 7 and d > 20) or m == 8:
                return "grain fill / weather premium decay"
            if m in (9, 10):
                return "harvest approach"
            return "off-season / storage and carry focus"

        if commodity == "soybeans":
            if m in (2, 3, 4):
                return "South American harvest / US planting intentions"
            if m in (5, 6):
                return "US planting / emergence"
            if m in (7, 8):
                return "pod setting / pod fill risk window"
            if m in (9, 10):
                return "harvest approach"
            return "off-season / South American weather focus"

        return "unknown crop stage"

    @staticmethod
    def _rank_top_analogs(analogs: pd.DataFrame, current_spread: float, top_n: int = 5) -> list[dict]:
        ranked = analogs.copy()
        spread_std = float(ranked["entry_spread"].std(ddof=0))
        if spread_std == 0 or np.isnan(spread_std):
            spread_std = 1.0

        ranked["similarity_score"] = 100.0 - (
            (ranked["entry_spread"] - current_spread).abs() / spread_std * 20.0
        )
        ranked["similarity_score"] = ranked["similarity_score"].clip(lower=0, upper=100)
        ranked = ranked.sort_values("similarity_score", ascending=False).head(top_n)

        return [
            {
                "year": int(row["year"]),
                "similarity_score": round(float(row["similarity_score"]), 1),
                "entry_spread": round(float(row["entry_spread"]), 2),
                "move_cents": round(float(row["move_cents"]), 2),
            }
            for _, row in ranked.iterrows()
        ]

    @staticmethod
    def _optional_row_value(row: Optional[pd.Series], column: str) -> Optional[float]:
        if row is None or column not in row or pd.isna(row[column]):
            return None
        return float(row[column])

    @staticmethod
    def _rounded_optional(value: Optional[float]) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 2)

    @staticmethod
    def _calculate_move_already_priced(
        current_spread: float,
        analogs: pd.DataFrame,
    ) -> Optional[float]:
        if "entry_spread" not in analogs.columns or analogs.empty:
            return None
        return float(current_spread) - float(np.median(analogs["entry_spread"].to_numpy(dtype=float)))

    @staticmethod
    def _analysis_date_message(dt: pd.Timestamp, is_latest: bool) -> str:
        suffix = " (latest available data)" if is_latest else ""
        return f"Analysis date: {dt.date()}{suffix}"

    @staticmethod
    def _normalize_continuous_context(context: Optional[object]) -> Optional[dict[str, Any]]:
        if context is None:
            return None
        if hasattr(context, "to_dict"):
            payload = dict(context.to_dict())
        elif isinstance(context, dict):
            payload = dict(context)
        else:
            raise ValueError("continuous_context must be a dict-like object or expose to_dict().")
        payload.setdefault("seasonal_signal_note", MARKET_CONTEXT_NOTE)
        return payload

    @staticmethod
    def _build_narrative(
        current_spread: float,
        crop_stage: str,
        seasonal_signal: str,
        median_move: float,
        average_move: float,
        win_rate: float,
        p25: float,
        p75: float,
        current_level_percentile: float,
        entry_assessment: str,
        current_volume: Optional[float],
        current_open_interest: Optional[float],
        years: int,
    ) -> str:
        win_rate_pct = round(win_rate * 100, 1)
        liquidity_text = ""
        if current_volume is not None or current_open_interest is not None:
            volume = SeasonalAgent._format_optional_number(current_volume)
            open_interest = SeasonalAgent._format_optional_number(current_open_interest)
            liquidity_text = (
                f" Latest volume is {volume} and open interest is {open_interest}."
            )
        return (
            f"Seasonal Agent view: the current value is {current_spread:.2f} cents "
            f"during the {crop_stage} stage. Across {years} historical analog years, "
            f"the next-window median move was {median_move:.2f} cents and the average "
            f"move was {average_move:.2f} cents. The directional win rate is "
            f"{win_rate_pct}%. The middle outcome range is {p25:.2f} to {p75:.2f} "
            f"cents. Current level sits at the {current_level_percentile:.1f}th "
            f"historical percentile for this point in the crop calendar, so entry is "
            f"classified as {entry_assessment}.{liquidity_text} Final seasonal signal: "
            f"{seasonal_signal}."
        )

    @staticmethod
    def _format_optional_number(value: Optional[float]) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.2f}"


def report_to_markdown(report: SeasonalAgentReport) -> str:
    """Format a SeasonalAgentReport as readable Markdown."""
    analog_lines = "\n".join(
        f"- {a['year']}: similarity {a['similarity_score']}%, "
        f"entry {a['entry_spread']}c, next move {a['move_cents']}c"
        for a in report.top_analog_years
    )

    warning_text = f"\n> Warning: {report.warning}\n" if report.warning else ""
    market_context_lines = []
    if report.current_volume is not None:
        market_context_lines.append(
            f"**Current volume:** {_format_optional_number(report.current_volume)}"
        )
    if report.current_open_interest is not None:
        market_context_lines.append(
            f"**Current open interest:** {_format_optional_number(report.current_open_interest)}"
        )
    market_context = ""
    if market_context_lines:
        market_context = "\n" + "  \n".join(market_context_lines)
    continuous_context_section = _format_continuous_context_section(report.continuous_context)

    return f"""
# Seasonal Agent Report

**Commodity:** {report.commodity}  
**Spread/series:** {report.spread_id}  
{report.analysis_date_message}  
**Crop stage:** {report.crop_stage}  
**Horizon:** {report.horizon_days} calendar days  
**Current spread/value:** {report.current_spread} cents{market_context}

## Seasonal Signal

**Seasonal signal:** {report.seasonal_signal}  
**Historical win rate:** {report.win_rate:.1%}  
**Consistency:** {report.consistency_score}  
**Entry assessment:** {report.entry_assessment}  
**Current spread percentile:** {report.current_level_percentile}%  
**Move already priced:** {_format_optional_cents(report.move_already_priced)}

## Historical outcome distribution

Years used: {report.historical_years_used}

| Metric | Value |
|---|---:|
| Win rate | {report.win_rate:.1%} |
| Average move | {report.average_move}c |
| Median move | {report.median_move}c |
| 25th percentile move | {report.percentile_25}c |
| 75th percentile move | {report.percentile_75}c |
| Best move | {report.best_move}c |
| Worst move | {report.worst_move}c |

## Top analog years

{analog_lines}
{warning_text}
{continuous_context_section}
## Narrative

{report.narrative}
""".strip()


def report_to_dict(report: SeasonalAgentReport) -> dict:
    """Format a SeasonalAgentReport as plain Python data."""
    return report.to_dict()


def report_to_json(report: SeasonalAgentReport, indent: Optional[int] = 2) -> str:
    """Format a SeasonalAgentReport as JSON."""
    return report.to_json(indent=indent)


def _format_optional_number(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


def _format_optional_cents(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{_format_optional_number(value)}c"


def _format_continuous_context_section(context: Optional[dict[str, Any]]) -> str:
    if not context:
        return ""

    rows = [
        ("Symbol", context.get("symbol")),
        ("Latest date", context.get("latest_date")),
        ("Latest close", context.get("latest_close")),
        ("1-day change", context.get("one_day_change")),
        ("20-day change", context.get("twenty_day_change")),
        ("50-day change", context.get("fifty_day_change")),
        ("Current percentile", _format_percent(context.get("current_percentile"))),
        ("Trend", context.get("trend_label")),
        ("Volume", context.get("volume")),
        ("Open interest", context.get("open_interest")),
    ]
    lines = [
        f"**{label}:** {_format_context_value(value)}"
        for label, value in rows
        if value is not None and not pd.isna(value)
    ]
    warning = context.get("warning")
    warning_text = f"\n\n> Warning: {warning}" if warning else ""

    return (
        "## Market Context\n\n"
        f"{MARKET_CONTEXT_NOTE}\n\n"
        + "  \n".join(lines)
        + warning_text
        + "\n"
    )


def _format_percent(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    return f"{_format_context_value(value)}%"


def _format_context_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (float, int)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.2f}"
    return str(value)
