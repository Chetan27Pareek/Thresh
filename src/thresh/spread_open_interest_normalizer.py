"""Merge historical spread values with matching open-interest history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import pandas as pd

from thresh.spread_history_normalizer import OUTPUT_COLUMNS


@dataclass(frozen=True)
class MergeSpreadOpenInterestSpec:
    """Metadata for one calendar-spread dataset."""

    commodity: str
    spread_id: str
    symbol_root: str
    nearby_month: str
    deferred_month: str


@dataclass(frozen=True)
class MergedSpreadOpenInterestResult:
    """Merged spread/open-interest data with validation summary."""

    data: pd.DataFrame
    summary: list[str]
    warnings: list[str]


CORN_JULY_DEC_SPEC = MergeSpreadOpenInterestSpec(
    commodity="corn",
    spread_id="CORN_NZ",
    symbol_root="ZC",
    nearby_month="N",
    deferred_month="Z",
)

SOYBEAN_JULY_NOV_SPEC = MergeSpreadOpenInterestSpec(
    commodity="soybeans",
    spread_id="SOYBEANS_NX",
    symbol_root="ZS",
    nearby_month="N",
    deferred_month="X",
)


def merge_spread_and_open_interest_history(
    spread_input: str | Path,
    open_interest_input: str | Path,
    spec: MergeSpreadOpenInterestSpec,
) -> MergedSpreadOpenInterestResult:
    """Normalize spread and open-interest histories, then merge by date/year."""
    spread = normalize_spread_input(spread_input, spec)
    open_interest = normalize_open_interest_input(open_interest_input)

    _validate_unique_keys(spread, "spread history")
    _validate_unique_keys(open_interest, "open-interest history")

    merged = spread.drop(columns=["open_interest"]).merge(
        open_interest,
        on=["date", "contract_year"],
        how="left",
        validate="one_to_one",
    )
    merged = merged[OUTPUT_COLUMNS].sort_values(["contract_year", "date"]).reset_index(drop=True)

    warnings = validate_merged_spread_open_interest(merged)
    summary = summarize_merged_spread_open_interest(merged)
    return MergedSpreadOpenInterestResult(data=merged, summary=summary, warnings=warnings)


def write_merged_spread_open_interest_history(
    spread_input: str | Path,
    open_interest_input: str | Path,
    output_path: str | Path,
    spec: MergeSpreadOpenInterestSpec,
) -> MergedSpreadOpenInterestResult:
    """Merge histories and write the processed CSV."""
    result = merge_spread_and_open_interest_history(
        spread_input=spread_input,
        open_interest_input=open_interest_input,
        spec=spec,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(output, index=False)
    return result


def normalize_spread_input(
    source: str | Path,
    spec: MergeSpreadOpenInterestSpec,
) -> pd.DataFrame:
    """Normalize spread-value history from a CSV file or directory of CSVs."""
    raw = _read_csv_source(source)
    wide = _normalize_wide_spread_input(raw, spec)
    if wide is not None:
        _validate_unique_keys(wide, "spread history")
        return wide

    date_column = _required_column(raw, [["timestamp"], ["date"]], "date")
    spread_column = _required_column(
        raw,
        [["spread", "value"], ["settle"], ["settlement"], ["settling"], ["close"]],
        "spread value",
    )
    contract_year_column = _optional_column(raw, ["contract", "year"]) or _optional_column(raw, ["year"])
    volume_column = _optional_column(raw, ["volume"])

    parsed_dates = pd.to_datetime(raw[date_column], errors="coerce")
    spread_values = _to_numeric(raw[spread_column])
    has_date = _has_value(raw[date_column])
    has_spread = _has_value(raw[spread_column])
    has_data = has_date & has_spread
    if (has_date & has_spread & parsed_dates.isna()).any():
        raise ValueError("Spread history contains invalid dates.")
    if (has_data & spread_values.isna()).any():
        raise ValueError("Spread history spread_value must be numeric.")

    normalized = pd.DataFrame()
    normalized["date"] = parsed_dates[has_data].dt.date.astype(str).to_numpy()
    normalized["contract_year"] = _contract_year_values(raw, has_data, parsed_dates, contract_year_column)
    normalized["commodity"] = spec.commodity.lower().strip()
    normalized["spread_id"] = spec.spread_id.upper().strip()
    normalized["nearby_symbol"] = normalized["contract_year"].map(
        lambda year: _contract_symbol(spec.symbol_root, spec.nearby_month, int(year))
    )
    normalized["deferred_symbol"] = normalized["contract_year"].map(
        lambda year: _contract_symbol(spec.symbol_root, spec.deferred_month, int(year))
    )
    normalized["spread_value"] = spread_values[has_data].astype(float).to_numpy()
    normalized["volume"] = _optional_numeric(raw, volume_column)[has_data].to_numpy()
    normalized["open_interest"] = pd.Series([pd.NA] * len(normalized), dtype="Float64")
    normalized = normalized[OUTPUT_COLUMNS].sort_values(["contract_year", "date"]).reset_index(drop=True)

    _validate_unique_keys(normalized, "spread history")
    return normalized


def normalize_open_interest_input(source: str | Path) -> pd.DataFrame:
    """Normalize open-interest history from a CSV file or directory of CSVs."""
    raw = _read_csv_source(source)
    date_column = _required_column(raw, [["timestamp"], ["date"]], "date")
    open_interest_column = _required_column(
        raw,
        [["open", "interest"], ["openinterest"], ["openint"], ["opint"], ["oi"]],
        "open interest",
    )
    contract_year_column = _optional_column(raw, ["contract", "year"]) or _optional_column(raw, ["year"])
    instrument_column = _optional_column(raw, ["instrument"]) or _optional_column(raw, ["symbol"])

    parsed_dates = pd.to_datetime(raw[date_column], errors="coerce")
    open_interest = _to_numeric(raw[open_interest_column])
    has_date = _has_value(raw[date_column])
    has_open_interest = _has_value(raw[open_interest_column])
    has_data = has_date & has_open_interest
    if (has_date & has_open_interest & parsed_dates.isna()).any():
        raise ValueError("Open-interest history contains invalid dates.")
    if (has_data & open_interest.isna()).any():
        raise ValueError("Open-interest history open_interest must be numeric.")

    normalized = pd.DataFrame()
    normalized["date"] = parsed_dates[has_data].dt.date.astype(str).to_numpy()
    normalized["contract_year"] = _contract_year_values(
        raw,
        has_data,
        parsed_dates,
        contract_year_column,
        instrument_column=instrument_column,
    )
    normalized["open_interest"] = open_interest[has_data].astype(float).to_numpy()
    normalized = normalized.sort_values(["contract_year", "date"]).reset_index(drop=True)

    _validate_unique_keys(normalized, "open-interest history")
    return normalized


def _normalize_wide_spread_input(
    raw: pd.DataFrame,
    spec: MergeSpreadOpenInterestSpec,
) -> Optional[pd.DataFrame]:
    pairs = _wide_year_pairs(raw)
    if not pairs:
        return None

    frames = []
    for contract_year, date_column, spread_column in pairs:
        parsed_dates = pd.to_datetime(raw[date_column], errors="coerce")
        spread_values = _to_numeric(raw[spread_column])
        has_date = _has_value(raw[date_column])
        has_spread = _has_value(raw[spread_column])
        has_data = has_date & has_spread
        if (has_date & has_spread & parsed_dates.isna()).any():
            raise ValueError(f"Spread history contains invalid dates for contract_year {contract_year}.")
        if (has_data & spread_values.isna()).any():
            raise ValueError(f"Spread history spread_value must be numeric for contract_year {contract_year}.")
        if not has_data.any():
            continue

        frame = pd.DataFrame()
        frame["date"] = parsed_dates[has_data].dt.date.astype(str).to_numpy()
        frame["contract_year"] = int(contract_year)
        frame["commodity"] = spec.commodity.lower().strip()
        frame["spread_id"] = spec.spread_id.upper().strip()
        frame["nearby_symbol"] = _contract_symbol(spec.symbol_root, spec.nearby_month, int(contract_year))
        frame["deferred_symbol"] = _contract_symbol(spec.symbol_root, spec.deferred_month, int(contract_year))
        frame["spread_value"] = spread_values[has_data].astype(float).to_numpy()
        frame["volume"] = pd.Series([pd.NA] * len(frame), dtype="Float64")
        frame["open_interest"] = pd.Series([pd.NA] * len(frame), dtype="Float64")
        frames.append(frame)

    if not frames:
        raise ValueError("Wide spread history contains no dated numeric spread rows.")

    normalized = pd.concat(frames, ignore_index=True)
    return normalized[OUTPUT_COLUMNS].sort_values(["contract_year", "date"]).reset_index(drop=True)


def validate_merged_spread_open_interest(data: pd.DataFrame) -> list[str]:
    """Validate merged processed history and return non-fatal warnings."""
    warnings = []
    missing = set(OUTPUT_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Merged history missing columns: {sorted(missing)}")

    _validate_unique_keys(data, "merged history")

    spread_values = _to_numeric(data["spread_value"])
    if spread_values.isna().any():
        raise ValueError("Merged history spread_value must be numeric.")

    open_interest_present = data["open_interest"].notna()
    open_interest = _to_numeric(data.loc[open_interest_present, "open_interest"])
    if open_interest.isna().any():
        raise ValueError("Merged history open_interest must be numeric when present.")
    missing_open_interest = int((~open_interest_present).sum())
    if missing_open_interest:
        warnings.append(
            f"{missing_open_interest} spread rows did not match an open-interest row."
        )

    return warnings


def summarize_merged_spread_open_interest(data: pd.DataFrame) -> list[str]:
    """Return summary lines showing available years and row counts."""
    lines = []
    for contract_year, group in data.groupby("contract_year", sort=True):
        lines.append(
            f"{int(contract_year)}: {len(group)} rows "
            f"({group['date'].min()} to {group['date'].max()})"
        )
    return lines


def _read_csv_source(source: str | Path) -> pd.DataFrame:
    path = Path(source)
    if not path.exists():
        raise ValueError(f"Input path does not exist: {path}")
    if path.is_dir():
        paths = sorted(candidate for candidate in path.iterdir() if candidate.suffix.lower() == ".csv")
        if not paths:
            raise ValueError(f"No CSV files found in {path}")
        frames = [_read_csv_file_with_source_year(candidate) for candidate in paths]
        return pd.concat(frames, ignore_index=True)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file or directory of CSV files: {path}")
    return _read_csv_file_with_source_year(path)


def _read_csv_file_with_source_year(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    source_year = _infer_contract_year_from_path(path)
    if source_year is not None:
        df["_source_contract_year"] = int(source_year)
    return df


def _contract_year_values(
    raw: pd.DataFrame,
    mask: pd.Series,
    parsed_dates: pd.Series,
    contract_year_column: Optional[str],
    instrument_column: Optional[str] = None,
) -> pd.Series:
    if contract_year_column is None:
        if "_source_contract_year" in raw.columns:
            source_year = _to_numeric(raw.loc[mask, "_source_contract_year"])
            if not source_year.isna().any():
                return source_year.astype(int).reset_index(drop=True)
        if instrument_column is not None:
            years = raw.loc[mask, instrument_column].map(_contract_year_from_instrument)
            if years.isna().any():
                raise ValueError("Could not infer contract_year from instrument values.")
            return years.astype(int).reset_index(drop=True)
        return parsed_dates[mask].dt.year.astype(int).reset_index(drop=True)
    contract_year = _to_numeric(raw.loc[mask, contract_year_column])
    if contract_year.isna().any():
        raise ValueError("contract_year must be numeric when provided.")
    return contract_year.astype(int).reset_index(drop=True)


def _validate_unique_keys(data: pd.DataFrame, label: str) -> None:
    duplicate_mask = data.duplicated(subset=["date", "contract_year"], keep=False)
    if duplicate_mask.any():
        examples = data.loc[duplicate_mask, ["date", "contract_year"]].head(5).to_dict(orient="records")
        raise ValueError(f"{label} has duplicate date + contract_year rows: {examples}")


def _required_column(df: pd.DataFrame, token_options: list[list[str]], label: str) -> str:
    for tokens in token_options:
        column = _optional_column(df, tokens)
        if column is not None:
            return column
    raise ValueError(f"Missing required {label} column.")


def _optional_column(df: pd.DataFrame, tokens: list[str]) -> Optional[str]:
    normalized_tokens = [_normalize_name(token) for token in tokens]
    for column in df.columns:
        normalized_column = _normalize_name(column)
        if all(token in normalized_column for token in normalized_tokens):
            return column
    return None


def _wide_year_pairs(raw: pd.DataFrame) -> list[tuple[int, str, str]]:
    columns_by_name = {_normalize_name(column): column for column in raw.columns}
    pairs = []
    for column in raw.columns:
        match = re.fullmatch(r"timestamp(?P<year>\d{4})", _normalize_name(column))
        if not match:
            continue
        year = int(match.group("year"))
        value_column = columns_by_name.get(str(year))
        if value_column is None:
            continue
        pairs.append((year, column, value_column))
    return sorted(pairs, key=lambda item: item[0])


def _optional_numeric(raw: pd.DataFrame, column: Optional[str]) -> pd.Series:
    if column is None:
        return pd.Series([pd.NA] * len(raw), dtype="Float64")
    return _to_numeric(raw[column])


def _has_value(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    return values.notna() & ~normalized.isin({"", "na", "na-na", "nan", "none", "null"})


def _to_numeric(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _contract_symbol(root: str, month: str, contract_year: int) -> str:
    return f"{root.upper()}{month.upper()}{str(contract_year)[-2:]}"


def _infer_contract_year_from_path(path: Path) -> Optional[int]:
    stem = path.stem.upper().replace(" ", "")
    futures_year = re.search(r"[A-Z]{1,3}[FGHJKMNQUVXZ](?P<year>\d{2,4})", stem)
    if futures_year:
        return _expand_year(futures_year.group("year"))

    calendar_year = re.search(r"(?<!\d)(?P<year>20\d{2}|19\d{2})(?!\d)", stem)
    if calendar_year:
        return int(calendar_year.group("year"))
    return None


def _contract_year_from_instrument(value: object) -> float:
    match = re.search(r"[A-Z]{1,3}[FGHJKMNQUVXZ](?P<year>\d{2,4})", str(value).upper())
    if not match:
        return float("nan")
    return float(_expand_year(match.group("year")))


def _expand_year(year_text: str) -> int:
    if len(year_text) == 4:
        return int(year_text)
    year = int(year_text)
    return 2000 + year if year < 80 else 1900 + year


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
