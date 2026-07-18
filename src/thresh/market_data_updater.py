"""End-of-day market-data update pipeline for calendar-spread histories."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Protocol

import pandas as pd


OUTPUT_COLUMNS = [
    "date",
    "contract_year",
    "commodity",
    "spread_id",
    "nearby_symbol",
    "deferred_symbol",
    "spread_value",
    "volume",
    "open_interest",
]


@dataclass(frozen=True)
class SpreadUpdateSpec:
    commodity: str
    spread_id: str
    symbol_root: str
    nearby_month: str
    deferred_month: str
    output_path: Path


@dataclass(frozen=True)
class MarketUpdateResult:
    rows_appended: dict[str, int]
    updated_paths: list[Path]
    warnings: list[str] = field(default_factory=list)


class EodMarketDataProvider(Protocol):
    """Adapter interface for daily settlement providers.

    The provider is intentionally end-of-day only. SeasonalAgent consumes the
    resulting spread history after the CBOT settlement is available; it does
    not use tick or minute data.
    """

    def fetch_eod_contract_data(
        self,
        symbols: list[str],
        start_date: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """Return date, symbol, open, high, low, settle, volume, open_interest rows."""


CORN_NZ_SPEC = SpreadUpdateSpec(
    commodity="corn",
    spread_id="CORN_NZ",
    symbol_root="ZC",
    nearby_month="N",
    deferred_month="Z",
    output_path=Path("data/processed/corn_july_dec_spread_history.csv"),
)

SOYBEANS_NX_SPEC = SpreadUpdateSpec(
    commodity="soybeans",
    spread_id="SOYBEANS_NX",
    symbol_root="ZS",
    nearby_month="N",
    deferred_month="X",
    output_path=Path("data/processed/soy_july_nov_spread_history.csv"),
)

DEFAULT_SPECS = [CORN_NZ_SPEC, SOYBEANS_NX_SPEC]


class CsvEodMarketDataProvider:
    """Simple provider adapter backed by one CSV per futures symbol.

    Configure with THRESH_EOD_MARKET_DATA_DIR, or THRESH_MARKET_DATA_DIR for
    backward compatibility. Each file should be named SYMBOL.csv and contain
    date, open, high, low, settlement/close, volume, open_interest.
    """

    def __init__(self, data_dir: str | Path | None = None):
        configured_dir = (
            data_dir
            or os.getenv("THRESH_EOD_MARKET_DATA_DIR")
            or os.getenv("THRESH_MARKET_DATA_DIR")
        )
        if not configured_dir:
            raise ValueError("THRESH_EOD_MARKET_DATA_DIR must be set for CsvEodMarketDataProvider.")
        self.data_dir = Path(configured_dir)

    def fetch_eod_contract_data(
        self,
        symbols: list[str],
        start_date: pd.Timestamp | None,
    ) -> pd.DataFrame:
        frames = []
        for symbol in symbols:
            path = self.data_dir / f"{symbol}.csv"
            if not path.exists():
                raise ValueError(f"Missing market data file for {symbol}: {path}")
            raw = pd.read_csv(path)
            frame = _normalize_provider_frame(raw, symbol)
            if start_date is not None:
                frame = frame[frame["date"] >= start_date]
            frames.append(frame)
        if not frames:
            return pd.DataFrame(
                columns=["date", "symbol", "open", "high", "low", "settle", "volume", "open_interest"]
            )
        return pd.concat(frames, ignore_index=True)


MarketDataProvider = EodMarketDataProvider
CsvMarketDataProvider = CsvEodMarketDataProvider


def update_eod_market_data(
    provider: EodMarketDataProvider,
    specs: list[SpreadUpdateSpec] | None = None,
) -> MarketUpdateResult:
    """Fetch latest EOD market data and append unseen processed spread rows."""
    update_specs = specs or DEFAULT_SPECS
    staged: list[tuple[SpreadUpdateSpec, pd.DataFrame, int]] = []
    rows_appended: dict[str, int] = {}
    warnings: list[str] = []

    for spec in update_specs:
        existing = _read_existing_history(spec.output_path)
        _validate_processed_history(existing, spec)
        contract_year = _latest_contract_year(existing)
        symbols = _symbols_for_year(spec, contract_year)
        latest_date = _latest_date_for_year(existing, contract_year)
        start_date = None if latest_date is None else latest_date + pd.Timedelta(days=1)

        raw_quotes = provider.fetch_eod_contract_data(symbols=symbols, start_date=start_date)
        new_rows = _build_spread_rows(raw_quotes, spec, contract_year)
        if not new_rows.empty:
            _validate_new_rows(new_rows, spec)

        combined = _append_unseen(existing, new_rows)
        _validate_processed_history(combined, spec)
        appended = len(combined) - len(existing)
        staged.append((spec, combined, appended))
        rows_appended[spec.spread_id] = appended
        if combined["volume"].isna().any() or combined["open_interest"].isna().any():
            warnings.append(f"{spec.spread_id} contains historical rows with missing volume/open_interest.")

    updated_paths: list[Path] = []
    for spec, combined, appended in staged:
        if appended > 0:
            _atomic_write_csv(combined, spec.output_path)
            updated_paths.append(spec.output_path)

    return MarketUpdateResult(rows_appended=rows_appended, updated_paths=updated_paths, warnings=warnings)


def update_market_data(
    provider: EodMarketDataProvider,
    specs: list[SpreadUpdateSpec] | None = None,
) -> MarketUpdateResult:
    """Backward-compatible wrapper for the EOD market-data updater."""
    return update_eod_market_data(provider, specs=specs)


def _normalize_provider_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    settle_column = _first_existing(raw, ["settle", "settlement", "close"])
    open_column = _first_existing(raw, ["open"])
    high_column = _first_existing(raw, ["high"])
    low_column = _first_existing(raw, ["low"])
    volume_column = _first_existing(raw, ["volume"])
    open_interest_column = _first_existing(raw, ["open_interest", "open interest", "oi"])
    required = {
        "date",
        open_column,
        high_column,
        low_column,
        settle_column,
        volume_column,
        open_interest_column,
    }
    missing = {column for column in required if column not in raw.columns}
    if missing:
        raise ValueError(f"Provider data for {symbol} missing columns: {sorted(missing)}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce"),
            "symbol": raw.get("symbol", symbol),
            "open": pd.to_numeric(raw[open_column], errors="coerce"),
            "high": pd.to_numeric(raw[high_column], errors="coerce"),
            "low": pd.to_numeric(raw[low_column], errors="coerce"),
            "settle": pd.to_numeric(raw[settle_column], errors="coerce"),
            "volume": pd.to_numeric(raw[volume_column], errors="coerce"),
            "open_interest": pd.to_numeric(raw[open_interest_column], errors="coerce"),
        }
    )
    eod_columns = ["date", "symbol", "open", "high", "low", "settle", "volume", "open_interest"]
    if frame[eod_columns].isna().any().any():
        raise ValueError(f"Provider data for {symbol} has missing or nonnumeric values.")
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    expected_symbol = symbol.upper().strip()
    unexpected_symbols = sorted(set(frame["symbol"].unique()) - {expected_symbol})
    if unexpected_symbols:
        raise ValueError(f"Provider data for {symbol} contains unexpected symbols: {unexpected_symbols}")
    return frame


def _build_spread_rows(raw_quotes: pd.DataFrame, spec: SpreadUpdateSpec, contract_year: int) -> pd.DataFrame:
    if raw_quotes.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    required = {"date", "symbol", "open", "high", "low", "settle", "volume", "open_interest"}
    missing = required - set(raw_quotes.columns)
    if missing:
        raise ValueError(f"Market provider response missing columns: {sorted(missing)}")

    quotes = raw_quotes.copy()
    quotes["date"] = pd.to_datetime(quotes["date"], errors="coerce")
    quotes["symbol"] = quotes["symbol"].astype(str).str.upper().str.strip()
    for column in ["open", "high", "low", "settle", "volume", "open_interest"]:
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    if quotes[list(required)].isna().any().any():
        raise ValueError("Market provider response contains missing values.")

    nearby_symbol, deferred_symbol = _symbols_for_year(spec, contract_year)
    expected_symbols = {nearby_symbol, deferred_symbol}
    unexpected_symbols = sorted(set(quotes["symbol"].unique()) - expected_symbols)
    if unexpected_symbols:
        raise ValueError(f"{spec.spread_id} provider response contains unexpected symbols: {unexpected_symbols}")
    duplicate_quotes = quotes.duplicated(["date", "symbol"], keep=False)
    if duplicate_quotes.any():
        examples = quotes.loc[duplicate_quotes, ["date", "symbol"]].head(8)
        raise ValueError(
            f"{spec.spread_id} provider response has duplicate date + symbol rows: "
            f"{examples.to_dict(orient='records')}"
        )

    nearby = quotes[quotes["symbol"].eq(nearby_symbol)]
    deferred = quotes[quotes["symbol"].eq(deferred_symbol)]
    if not nearby.empty or not deferred.empty:
        missing_symbols = sorted(expected_symbols - set(quotes["symbol"].unique()))
        if missing_symbols:
            raise ValueError(f"{spec.spread_id} provider response missing symbols: {missing_symbols}")

    nearby_dates = set(nearby["date"])
    deferred_dates = set(deferred["date"])
    if nearby_dates != deferred_dates:
        nearby_only = sorted(date.date().isoformat() for date in nearby_dates - deferred_dates)
        deferred_only = sorted(date.date().isoformat() for date in deferred_dates - nearby_dates)
        raise ValueError(
            f"{spec.spread_id} EOD leg dates do not match; "
            f"nearby-only dates={nearby_only[:8]}, deferred-only dates={deferred_only[:8]}"
        )

    merged = nearby.merge(
        deferred,
        on="date",
        suffixes=("_nearby", "_deferred"),
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    output = pd.DataFrame()
    output["date"] = merged["date"].dt.date.astype(str)
    output["contract_year"] = int(contract_year)
    output["commodity"] = spec.commodity
    output["spread_id"] = spec.spread_id
    output["nearby_symbol"] = nearby_symbol
    output["deferred_symbol"] = deferred_symbol
    output["spread_value"] = merged["settle_nearby"] - merged["settle_deferred"]
    output["volume"] = merged["volume_nearby"]
    output["open_interest"] = merged["open_interest_nearby"]
    return output[OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)


def _append_unseen(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows.empty:
        return existing.copy()
    existing_keys = set(zip(existing["date"].astype(str), existing["contract_year"].astype(int)))
    unseen_mask = [
        (str(row.date), int(row.contract_year)) not in existing_keys
        for row in new_rows.itertuples(index=False)
    ]
    unseen = new_rows.loc[unseen_mask]
    combined = pd.concat([existing, unseen], ignore_index=True)
    return combined[OUTPUT_COLUMNS].sort_values(["contract_year", "date"]).reset_index(drop=True)


def _validate_processed_history(data: pd.DataFrame, spec: SpreadUpdateSpec) -> None:
    missing_columns = set(OUTPUT_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"{spec.spread_id} history missing columns: {sorted(missing_columns)}")
    if data.empty:
        return
    duplicate_keys = data.duplicated(["date", "contract_year"], keep=False)
    if duplicate_keys.any():
        examples = data.loc[duplicate_keys, ["date", "contract_year"]].head(8)
        raise ValueError(
            f"{spec.spread_id} has duplicate date + contract_year rows: "
            f"{examples.to_dict(orient='records')}"
        )
    required_columns = [
        "date",
        "contract_year",
        "commodity",
        "spread_id",
        "nearby_symbol",
        "deferred_symbol",
        "spread_value",
    ]
    if data[required_columns].isna().any().any():
        raise ValueError(f"{spec.spread_id} has missing required processed values.")
    if not data["commodity"].astype(str).str.lower().eq(spec.commodity).all():
        raise ValueError(f"{spec.spread_id} has unexpected commodity values.")
    if not data["spread_id"].astype(str).str.upper().eq(spec.spread_id).all():
        raise ValueError(f"{spec.spread_id} has unexpected spread_id values.")
    for row in data[["contract_year", "nearby_symbol", "deferred_symbol"]].itertuples(index=False):
        expected_nearby, expected_deferred = _symbols_for_year(spec, int(row.contract_year))
        if row.nearby_symbol != expected_nearby or row.deferred_symbol != expected_deferred:
            raise ValueError(f"{spec.spread_id} has invalid contract symbols for {row.contract_year}.")
    spread_values = pd.to_numeric(data["spread_value"], errors="coerce")
    if spread_values.isna().any():
        raise ValueError(f"{spec.spread_id} spread_value must be numeric.")


def _validate_new_rows(data: pd.DataFrame, spec: SpreadUpdateSpec) -> None:
    _validate_processed_history(data, spec)
    if data[["volume", "open_interest"]].isna().any().any():
        raise ValueError(f"{spec.spread_id} new market rows must include volume and open_interest.")


def _read_existing_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = pd.read_csv(path)
    return data[OUTPUT_COLUMNS]


def _latest_contract_year(existing: pd.DataFrame) -> int:
    if existing.empty:
        return pd.Timestamp.today().year
    return int(pd.to_numeric(existing["contract_year"], errors="coerce").max())


def _latest_date_for_year(existing: pd.DataFrame, contract_year: int) -> pd.Timestamp | None:
    if existing.empty:
        return None
    rows = existing[pd.to_numeric(existing["contract_year"], errors="coerce").eq(contract_year)]
    if rows.empty:
        return None
    return pd.to_datetime(rows["date"], errors="coerce").max()


def _symbols_for_year(spec: SpreadUpdateSpec, contract_year: int) -> tuple[str, str]:
    suffix = str(contract_year)[-2:]
    nearby = f"{spec.symbol_root.upper()}{spec.nearby_month.upper()}{suffix}"
    deferred = f"{spec.symbol_root.upper()}{spec.deferred_month.upper()}{suffix}"
    return nearby, deferred


def _first_existing(data: pd.DataFrame, columns: list[str]) -> str:
    normalized = {_normalize_name(column): column for column in data.columns}
    for column in columns:
        found = normalized.get(_normalize_name(column))
        if found is not None:
            return found
    return columns[0]


def _atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
