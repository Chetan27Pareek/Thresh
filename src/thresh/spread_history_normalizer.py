"""Normalize historical calendar spread CSV files into one spread-history file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

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

ROOT_COMMODITIES = {
    "ZC": "corn",
    "C": "corn",
    "ZS": "soybeans",
    "S": "soybeans",
}


@dataclass(frozen=True)
class SpreadHistoryNormalizationResult:
    """Normalized spread history plus non-fatal validation warnings."""

    data: pd.DataFrame
    warnings: list[str]


@dataclass(frozen=True)
class SpreadFileMetadata:
    """Metadata inferred from one raw spread CSV filename."""

    nearby_symbol: str
    deferred_symbol: str
    contract_year: int
    nearby_month: str
    deferred_month: str
    root: str


def normalize_spread_history_directory(
    input_dir: str | Path,
    commodity: Optional[str] = None,
    spread_id: Optional[str] = None,
    min_rows_per_contract_year: int = 100,
    min_contract_years_warning: int = 10,
) -> SpreadHistoryNormalizationResult:
    """Normalize all CSV files in a directory into one spread-history table."""
    directory = Path(input_dir)
    if not directory.exists():
        raise ValueError(f"Input directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Input path is not a directory: {directory}")

    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".csv")
    if not paths:
        raise ValueError(f"No CSV files found in {directory}")

    frames = [
        normalize_spread_history_file(path, commodity=commodity, spread_id=spread_id)
        for path in paths
    ]
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["contract_year", "date"])
        .reset_index(drop=True)
    )
    warnings = validate_spread_history(
        combined,
        min_rows_per_contract_year=min_rows_per_contract_year,
        min_contract_years_warning=min_contract_years_warning,
    )
    return SpreadHistoryNormalizationResult(data=combined, warnings=warnings)


def normalize_spread_history_file(
    path: str | Path,
    commodity: Optional[str] = None,
    spread_id: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize one raw calendar spread CSV file."""
    csv_path = Path(path)
    metadata = parse_spread_filename(csv_path)
    raw = pd.read_csv(csv_path)

    date_column = _find_required_column(raw, [["timestamp"], ["date"]], "date")
    value_column = _find_required_column(
        raw,
        [["spread", "value"], ["settle"], ["settlement"], ["settling"], ["close"]],
        "spread value",
    )
    volume_column = _find_optional_column(raw, ["volume"])
    open_interest_column = (
        _find_optional_column(raw, ["open", "interest"])
        or _find_optional_column(raw, ["open_interest"])
        or _find_optional_column(raw, ["opint"])
    )

    parsed_dates = pd.to_datetime(raw[date_column], errors="coerce")
    spread_values = pd.to_numeric(raw[value_column], errors="coerce")
    has_any_data = raw[[date_column, value_column]].notna().any(axis=1)

    bad_dates = has_any_data & parsed_dates.isna()
    if bad_dates.any():
        raise ValueError(f"{csv_path.name}: found rows with invalid dates.")

    bad_values = has_any_data & spread_values.isna()
    if bad_values.any():
        raise ValueError(f"{csv_path.name}: spread_value must be numeric.")

    valid_rows = has_any_data
    if not valid_rows.any():
        raise ValueError(f"{csv_path.name}: no usable spread rows found.")

    normalized = pd.DataFrame()
    normalized["date"] = parsed_dates[valid_rows].dt.date.astype(str).to_numpy()
    normalized["contract_year"] = metadata.contract_year
    normalized["commodity"] = (
        commodity.lower().strip() if commodity else _infer_commodity(metadata.root)
    )
    normalized["spread_id"] = (
        spread_id.upper().strip()
        if spread_id
        else _build_spread_id(normalized["commodity"].iloc[0], metadata)
    )
    normalized["nearby_symbol"] = metadata.nearby_symbol
    normalized["deferred_symbol"] = metadata.deferred_symbol
    normalized["spread_value"] = spread_values[valid_rows].astype(float).to_numpy()
    normalized["volume"] = _optional_numeric(raw, volume_column)[valid_rows].to_numpy()
    normalized["open_interest"] = _optional_numeric(raw, open_interest_column)[valid_rows].to_numpy()

    return normalized[OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)


def write_normalized_spread_history(
    input_dir: str | Path,
    output_path: str | Path,
    commodity: Optional[str] = None,
    spread_id: Optional[str] = None,
) -> SpreadHistoryNormalizationResult:
    """Normalize a directory of spread CSVs and write the combined CSV."""
    result = normalize_spread_history_directory(
        input_dir=input_dir,
        commodity=commodity,
        spread_id=spread_id,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(output, index=False)
    return result


def validate_spread_history(
    data: pd.DataFrame,
    min_rows_per_contract_year: int = 100,
    min_contract_years_warning: int = 10,
) -> list[str]:
    """Validate combined spread history and return non-fatal warnings."""
    missing = set(OUTPUT_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing normalized spread-history columns: {sorted(missing)}")

    df = data.copy()
    duplicate_mask = df.duplicated(subset=["date", "contract_year"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = df.loc[duplicate_mask, ["date", "contract_year"]].drop_duplicates()
        examples = duplicate_rows.head(5).to_dict(orient="records")
        raise ValueError(f"Duplicate date + contract_year rows found: {examples}")

    spread_values = pd.to_numeric(df["spread_value"], errors="coerce")
    if spread_values.isna().any():
        raise ValueError("spread_value must be numeric.")

    counts = df.groupby("contract_year").size()
    short_years = counts[counts < min_rows_per_contract_year]
    if not short_years.empty:
        details = ", ".join(f"{int(year)}={int(count)}" for year, count in short_years.items())
        raise ValueError(
            f"Each contract_year must have at least {min_rows_per_contract_year} rows. "
            f"Short years: {details}"
        )

    warnings = []
    contract_year_count = int(df["contract_year"].nunique())
    if contract_year_count < min_contract_years_warning:
        warnings.append(
            f"Only {contract_year_count} contract years found. "
            f"At least {min_contract_years_warning} are recommended for seasonal analysis."
        )
    return warnings


def parse_spread_filename(path: str | Path) -> SpreadFileMetadata:
    """Infer symbols and contract year from filenames like ZCN26-ZCZ26.csv."""
    stem = Path(path).stem.upper().replace(" ", "")
    match = re.search(
        r"(?P<nearby>[A-Z]{1,3}[FGHJKMNQUVXZ]\d{2,4})[-_](?P<deferred>[A-Z]{1,3}[FGHJKMNQUVXZ]\d{2,4})",
        stem,
    )
    if not match:
        raise ValueError(f"Could not infer spread symbols from filename: {Path(path).name}")

    nearby_symbol = match.group("nearby")
    deferred_symbol = match.group("deferred")
    nearby_parts = _split_symbol(nearby_symbol)
    deferred_parts = _split_symbol(deferred_symbol)

    if nearby_parts["root"] != deferred_parts["root"]:
        raise ValueError(f"Spread legs have different roots in filename: {Path(path).name}")
    if nearby_parts["year"] != deferred_parts["year"]:
        raise ValueError(f"Spread legs have different contract years in filename: {Path(path).name}")

    return SpreadFileMetadata(
        nearby_symbol=nearby_symbol,
        deferred_symbol=deferred_symbol,
        contract_year=_expand_year(nearby_parts["year"]),
        nearby_month=nearby_parts["month"],
        deferred_month=deferred_parts["month"],
        root=nearby_parts["root"],
    )


def _split_symbol(symbol: str) -> dict[str, str]:
    match = re.fullmatch(
        r"(?P<root>[A-Z]{1,3})(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{2,4})",
        symbol,
    )
    if not match:
        raise ValueError(f"Invalid futures symbol: {symbol}")
    return match.groupdict()


def _expand_year(year_text: str) -> int:
    if len(year_text) == 4:
        return int(year_text)
    year = int(year_text)
    return 2000 + year if year < 80 else 1900 + year


def _infer_commodity(root: str) -> str:
    try:
        return ROOT_COMMODITIES[root]
    except KeyError as exc:
        raise ValueError(f"Unknown commodity root {root!r}; pass --commodity explicitly.") from exc


def _build_spread_id(commodity: str, metadata: SpreadFileMetadata) -> str:
    return f"{commodity}_{metadata.nearby_month}{metadata.deferred_month}".upper()


def _find_required_column(
    df: pd.DataFrame,
    token_options: list[list[str]],
    label: str,
) -> str:
    column = None
    for tokens in token_options:
        column = _find_optional_column(df, tokens)
        if column is not None:
            return column
    raise ValueError(f"Missing required {label} column.")


def _find_optional_column(df: pd.DataFrame, tokens: list[str]) -> Optional[str]:
    normalized_tokens = [_normalize_name(token) for token in tokens]
    for column in df.columns:
        normalized_column = _normalize_name(column)
        if all(token in normalized_column for token in normalized_tokens):
            return column
    return None


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _optional_numeric(raw: pd.DataFrame, column: Optional[str]) -> pd.Series:
    if column is None:
        return pd.Series([pd.NA] * len(raw), dtype="Float64")
    return pd.to_numeric(raw[column], errors="coerce")
