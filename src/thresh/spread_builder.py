"""
Thresh Spread Builder v1
------------------------

This module converts individual futures contract prices into calendar spread prices.

Example:
    CN24 = Corn July 2024 settlement
    CZ24 = Corn December 2024 settlement

    CORN_NZ = CN24 - CZ24

This is infrastructure, not an AI agent. The Seasonal Agent needs clean spread
history, and this builder creates that history from contract prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_PRICE_COLUMNS = {"date", "symbol", "settle"}


@dataclass(frozen=True)
class SpreadSpec:
    """Definition of a futures calendar spread.

    Attributes
    ----------
    commodity:
        Human-readable commodity name, for example "corn".
    spread_id:
        Project identifier, for example "CORN_NZ".
    symbol_prefix:
        Futures root symbol. For our simplified convention:
        corn = "C", soybeans = "S".
    nearby_month:
        Month code for the nearby leg. July = "N".
    deferred_month:
        Month code for the deferred leg. December = "Z".
    """

    commodity: str
    spread_id: str
    symbol_prefix: str
    nearby_month: str
    deferred_month: str


def contract_symbol(symbol_prefix: str, month_code: str, year: int) -> str:
    """Create a simple futures symbol like CN24 or CZ25.

    This starter project uses a two-digit year convention.

    Example:
        contract_symbol("C", "N", 2024) -> "CN24"
    """
    year_two_digits = str(year)[-2:]
    return f"{symbol_prefix.upper()}{month_code.upper()}{year_two_digits}"


def prepare_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw futures price data."""
    missing = REQUIRED_PRICE_COLUMNS - set(prices.columns)

    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].str.upper().str.strip()
    df["settle"] = pd.to_numeric(df["settle"], errors="raise")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


def build_calendar_spread_for_year(
    prices: pd.DataFrame,
    spec: SpreadSpec,
    contract_year: int,
) -> pd.DataFrame:
    """Build one year's calendar spread series.

    For corn July-Dec 2024:
        nearby_symbol = CN24
        deferred_symbol = CZ24
        spread_value = CN24 settle - CZ24 settle
    """
    df = prepare_price_data(prices)

    nearby_symbol = contract_symbol(
        spec.symbol_prefix,
        spec.nearby_month,
        contract_year,
    )
    deferred_symbol = contract_symbol(
        spec.symbol_prefix,
        spec.deferred_month,
        contract_year,
    )

    nearby = df[df["symbol"] == nearby_symbol][["date", "settle"]].rename(
        columns={"settle": "nearby_settle"}
    )

    deferred = df[df["symbol"] == deferred_symbol][["date", "settle"]].rename(
        columns={"settle": "deferred_settle"}
    )

    if nearby.empty:
        raise ValueError(f"No price data found for nearby contract {nearby_symbol}")

    if deferred.empty:
        raise ValueError(f"No price data found for deferred contract {deferred_symbol}")

    spread = nearby.merge(deferred, on="date", how="inner")

    if spread.empty:
        raise ValueError(
            f"No overlapping dates found for {nearby_symbol} and {deferred_symbol}"
        )

    spread["year"] = int(contract_year)
    spread["commodity"] = spec.commodity.lower()
    spread["spread_id"] = spec.spread_id.upper()
    spread["nearby_symbol"] = nearby_symbol
    spread["deferred_symbol"] = deferred_symbol
    spread["spread_value"] = spread["nearby_settle"] - spread["deferred_settle"]

    columns = [
        "date",
        "year",
        "commodity",
        "spread_id",
        "nearby_symbol",
        "deferred_symbol",
        "nearby_settle",
        "deferred_settle",
        "spread_value",
    ]

    return spread[columns].sort_values("date").reset_index(drop=True)


def build_calendar_spread_history(
    prices: pd.DataFrame,
    spec: SpreadSpec,
    years: Iterable[int],
) -> pd.DataFrame:
    """Build a multi-year calendar spread history."""
    frames = []

    for year in years:
        yearly_spread = build_calendar_spread_for_year(
            prices=prices,
            spec=spec,
            contract_year=int(year),
        )
        frames.append(yearly_spread)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["year", "date"])
        .reset_index(drop=True)
    )