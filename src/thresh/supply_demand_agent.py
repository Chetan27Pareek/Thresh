"""Rule-based Supply/Demand Agent v1.

This agent is intentionally deterministic. It reads normalized WASDE and crop
condition data, then produces a small structured fundamentals report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Optional

import numpy as np
import pandas as pd


WASDE_REQUIRED_COLUMNS = {
    "date",
    "commodity",
    "marketing_year",
    "production",
    "exports",
    "ending_stocks",
    "total_use",
    "stocks_to_use",
}

CROP_REQUIRED_COLUMNS = {
    "date",
    "commodity",
    "good_pct",
    "excellent_pct",
    "good_excellent_pct",
}


@dataclass(frozen=True)
class SupplyDemandAgentConfig:
    commodity: str
    tight_percentile: float = 33.0
    ample_percentile: float = 67.0
    revision_stable_threshold: float = 0.25
    crop_condition_stable_threshold: float = 2.0
    poor_crop_threshold: float = 50.0
    strong_crop_threshold: float = 70.0

    def __post_init__(self) -> None:
        if self.tight_percentile <= 0 or self.ample_percentile >= 100:
            raise ValueError("Percentile thresholds must be inside 0-100.")
        if self.tight_percentile >= self.ample_percentile:
            raise ValueError("tight_percentile must be below ample_percentile.")
        if self.revision_stable_threshold < 0:
            raise ValueError("revision_stable_threshold must be non-negative.")
        if self.crop_condition_stable_threshold < 0:
            raise ValueError("crop_condition_stable_threshold must be non-negative.")


@dataclass(frozen=True)
class SupplyDemandAgentReport:
    commodity: str
    analysis_date: str
    fundamental_bias: str
    stocks_situation: str
    stocks_to_use: float
    stocks_to_use_percentile: float
    wasde_revision_trend: str
    crop_condition_good_excellent_pct: Optional[float]
    crop_condition_trajectory: str
    marketing_year: str
    latest_wasde_date: str
    latest_crop_condition_date: Optional[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SupplyDemandAgent:
    """Rule-based supply/demand fundamentals agent."""

    def __init__(
        self,
        wasde_data: pd.DataFrame,
        crop_condition_data: pd.DataFrame,
        config: SupplyDemandAgentConfig,
    ):
        self.config = config
        self.wasde = self._prepare_wasde(wasde_data)
        self.crop_conditions = self._prepare_crop_conditions(crop_condition_data)

    def analyze(self, analysis_date: str = "latest") -> SupplyDemandAgentReport:
        commodity = self.config.commodity.lower().strip()
        wasde = self.wasde[self.wasde["commodity"].eq(commodity)].copy()
        if wasde.empty:
            raise ValueError(f"No WASDE data found for commodity {commodity!r}.")

        crop = self.crop_conditions[self.crop_conditions["commodity"].eq(commodity)].copy()
        dt = self._resolve_analysis_date(analysis_date, wasde, crop)
        warnings: list[str] = []

        eligible_wasde = wasde[wasde["date"] <= dt]
        if eligible_wasde.empty:
            raise ValueError(f"No WASDE rows available on or before {dt.date()}.")

        latest_wasde_date = eligible_wasde["date"].max()
        latest_wasde = eligible_wasde[eligible_wasde["date"].eq(latest_wasde_date)]
        marketing_year = self._latest_marketing_year(latest_wasde)
        current_row = latest_wasde[latest_wasde["marketing_year"].eq(marketing_year)].iloc[0]
        current_stocks_to_use = float(current_row["stocks_to_use"])

        historical_stocks = wasde[wasde["date"] <= latest_wasde_date]["stocks_to_use"].to_numpy(dtype=float)
        stocks_percentile = self._percentile_rank(current_stocks_to_use, historical_stocks)
        stocks_situation = self._classify_stocks(stocks_percentile)

        revision_trend, revision_warning = self._revision_trend(
            wasde=eligible_wasde,
            marketing_year=marketing_year,
        )
        if revision_warning is not None:
            warnings.append(revision_warning)

        crop_value, crop_trend, crop_date, crop_warning = self._crop_condition(crop, dt)
        if crop_warning is not None:
            warnings.append(crop_warning)

        fundamental_bias = self._fundamental_bias(
            stocks_situation=stocks_situation,
            revision_trend=revision_trend,
            crop_value=crop_value,
            crop_trend=crop_trend,
        )

        return SupplyDemandAgentReport(
            commodity=commodity,
            analysis_date=str(dt.date()),
            fundamental_bias=fundamental_bias,
            stocks_situation=stocks_situation,
            stocks_to_use=round(current_stocks_to_use, 2),
            stocks_to_use_percentile=round(stocks_percentile, 2),
            wasde_revision_trend=revision_trend,
            crop_condition_good_excellent_pct=self._rounded_optional(crop_value),
            crop_condition_trajectory=crop_trend,
            marketing_year=marketing_year,
            latest_wasde_date=str(latest_wasde_date.date()),
            latest_crop_condition_date=None if crop_date is None else str(crop_date.date()),
            warnings=warnings,
        )

    @staticmethod
    def _prepare_wasde(wasde_data: pd.DataFrame) -> pd.DataFrame:
        missing = WASDE_REQUIRED_COLUMNS - set(wasde_data.columns)
        if missing:
            raise ValueError(f"Missing WASDE columns: {sorted(missing)}")

        df = wasde_data.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("WASDE data contains invalid dates.")
        df["commodity"] = df["commodity"].astype(str).str.lower().str.strip()
        df["marketing_year"] = df["marketing_year"].astype(str).str.strip()
        for column in ["production", "exports", "ending_stocks", "total_use", "stocks_to_use"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            if df[column].isna().any():
                raise ValueError(f"WASDE column {column} must be numeric.")
        duplicate_keys = df.duplicated(["date", "commodity", "marketing_year"], keep=False)
        if duplicate_keys.any():
            examples = df.loc[duplicate_keys, ["date", "commodity", "marketing_year"]].head(8)
            raise ValueError(f"Duplicate WASDE rows found: {examples.to_dict(orient='records')}")
        return df.sort_values(["commodity", "marketing_year", "date"]).reset_index(drop=True)

    @staticmethod
    def _prepare_crop_conditions(crop_condition_data: pd.DataFrame) -> pd.DataFrame:
        missing = CROP_REQUIRED_COLUMNS - set(crop_condition_data.columns)
        if missing:
            raise ValueError(f"Missing crop-condition columns: {sorted(missing)}")

        df = crop_condition_data.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("Crop-condition data contains invalid dates.")
        df["commodity"] = df["commodity"].astype(str).str.lower().str.strip()
        for column in ["good_pct", "excellent_pct", "good_excellent_pct"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            if df[column].isna().any():
                raise ValueError(f"Crop-condition column {column} must be numeric.")
        duplicate_keys = df.duplicated(["date", "commodity"], keep=False)
        if duplicate_keys.any():
            examples = df.loc[duplicate_keys, ["date", "commodity"]].head(8)
            raise ValueError(f"Duplicate crop-condition rows found: {examples.to_dict(orient='records')}")
        return df.sort_values(["commodity", "date"]).reset_index(drop=True)

    @staticmethod
    def _resolve_analysis_date(
        analysis_date: str,
        wasde: pd.DataFrame,
        crop: pd.DataFrame,
    ) -> pd.Timestamp:
        if analysis_date.lower().strip() != "latest":
            return pd.Timestamp(analysis_date)

        dates = [wasde["date"].max()]
        if not crop.empty:
            dates.append(crop["date"].max())
        return max(dates)

    @staticmethod
    def _latest_marketing_year(wasde_rows: pd.DataFrame) -> str:
        years = sorted(wasde_rows["marketing_year"].dropna().unique(), key=_marketing_year_start)
        if not years:
            raise ValueError("No marketing_year values found in latest WASDE report.")
        return str(years[-1])

    def _classify_stocks(self, percentile: float) -> str:
        if percentile <= self.config.tight_percentile:
            return "Tight"
        if percentile >= self.config.ample_percentile:
            return "Ample"
        return "Average"

    def _revision_trend(
        self,
        wasde: pd.DataFrame,
        marketing_year: str,
    ) -> tuple[str, Optional[str]]:
        rows = wasde[wasde["marketing_year"].eq(marketing_year)].sort_values("date")
        latest_dates = rows["date"].drop_duplicates().tail(3)
        recent = rows[rows["date"].isin(latest_dates)].sort_values("date")
        if len(recent) < 2:
            return "Stable", f"Fewer than 2 WASDE reports found for {marketing_year}; revision trend set to Stable."

        change = float(recent.iloc[-1]["stocks_to_use"] - recent.iloc[0]["stocks_to_use"])
        if change <= -self.config.revision_stable_threshold:
            return "Tightening", None
        if change >= self.config.revision_stable_threshold:
            return "Loosening", None
        return "Stable", None

    def _crop_condition(
        self,
        crop: pd.DataFrame,
        analysis_date: pd.Timestamp,
    ) -> tuple[Optional[float], str, Optional[pd.Timestamp], Optional[str]]:
        commodity = self.config.commodity.lower().strip()
        if crop.empty:
            return None, "Unavailable", None, f"Crop condition data unavailable for {commodity}; using WASDE only."

        rows = crop[crop["date"] <= analysis_date].sort_values("date")
        if rows.empty:
            return None, "Unavailable", None, f"No crop condition rows available on or before {analysis_date.date()}; using WASDE only."

        recent = rows.tail(3)
        crop_value = float(recent.iloc[-1]["good_excellent_pct"])
        crop_date = pd.Timestamp(recent.iloc[-1]["date"])
        if len(recent) < 2:
            return crop_value, "Stable", crop_date, "Fewer than 2 crop condition reports found; trajectory set to Stable."

        change = float(recent.iloc[-1]["good_excellent_pct"] - recent.iloc[0]["good_excellent_pct"])
        if change >= self.config.crop_condition_stable_threshold:
            return crop_value, "Improving", crop_date, None
        if change <= -self.config.crop_condition_stable_threshold:
            return crop_value, "Deteriorating", crop_date, None
        return crop_value, "Stable", crop_date, None

    def _fundamental_bias(
        self,
        stocks_situation: str,
        revision_trend: str,
        crop_value: Optional[float],
        crop_trend: str,
    ) -> str:
        score = 0
        if stocks_situation == "Tight":
            score += 1
        elif stocks_situation == "Ample":
            score -= 1

        if revision_trend == "Tightening":
            score += 1
        elif revision_trend == "Loosening":
            score -= 1

        if crop_value is not None:
            if crop_trend == "Deteriorating":
                score += 1
            elif crop_trend == "Improving":
                score -= 1

            if crop_value <= self.config.poor_crop_threshold:
                score += 1
            elif crop_value >= self.config.strong_crop_threshold:
                score -= 1

        if score >= 2:
            return "Bullish"
        if score <= -2:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _percentile_rank(value: float, sample: np.ndarray) -> float:
        if len(sample) == 0:
            raise ValueError("Cannot calculate percentile from an empty sample.")
        return float((np.sum(sample <= value) / len(sample)) * 100)

    @staticmethod
    def _rounded_optional(value: Optional[float]) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 2)


def report_to_json(report: SupplyDemandAgentReport) -> str:
    return report.to_json()


def report_to_markdown(report: SupplyDemandAgentReport) -> str:
    lines = [
        "# Supply/Demand Agent Report",
        "",
        f"**Commodity:** {report.commodity}",
        f"**Analysis date:** {report.analysis_date}",
        f"**Fundamental Bias:** {report.fundamental_bias}",
        "",
        "## WASDE",
        "",
        f"**Marketing year:** {report.marketing_year}",
        f"**Latest WASDE date:** {report.latest_wasde_date}",
        f"**Stocks Situation:** {report.stocks_situation}",
        f"**Stocks-to-Use:** {report.stocks_to_use}%",
        f"**Stocks-to-Use percentile:** {report.stocks_to_use_percentile}%",
        f"**WASDE Revision Trend:** {report.wasde_revision_trend}",
        "",
        "## Crop Condition",
        "",
    ]
    if report.crop_condition_good_excellent_pct is None:
        lines.append("**Good/Excellent %:** n/a")
    else:
        lines.append(f"**Good/Excellent %:** {report.crop_condition_good_excellent_pct}%")
    lines.extend(
        [
            f"**Latest crop condition date:** {report.latest_crop_condition_date or 'n/a'}",
            f"**Crop Condition Trajectory:** {report.crop_condition_trajectory}",
            "",
            "## Warnings",
            "",
        ]
    )
    if report.warnings:
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("- None")
    return "\n".join(lines)


def _marketing_year_start(marketing_year: str) -> int:
    match = str(marketing_year).split("/")[0]
    return int(match)
