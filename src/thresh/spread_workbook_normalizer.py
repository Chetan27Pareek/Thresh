"""Normalize spread-history workbook sheets directly into processed CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pandas as pd

from thresh.spread_history_normalizer import OUTPUT_COLUMNS


XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


@dataclass(frozen=True)
class SpreadWorkbookSheetSpec:
    """Workbook sheet mapping for one calendar spread history."""

    sheet_name: str
    commodity: str
    spread_id: str
    root: str
    nearby_month: str
    deferred_month: str
    output_filename: str


@dataclass(frozen=True)
class SpreadWorkbookNormalizationResult:
    """One normalized workbook sheet and its validation warnings."""

    sheet_name: str
    output_path: Path
    data: pd.DataFrame
    warnings: list[str]


DEFAULT_SPREAD_WORKBOOK_SHEETS = (
    SpreadWorkbookSheetSpec(
        sheet_name="CORN J-D",
        commodity="corn",
        spread_id="CORN_NZ",
        root="ZC",
        nearby_month="N",
        deferred_month="Z",
        output_filename="corn_july_dec_spread_history.csv",
    ),
    SpreadWorkbookSheetSpec(
        sheet_name="SOY J-N",
        commodity="soybeans",
        spread_id="SOYBEANS_NX",
        root="ZS",
        nearby_month="N",
        deferred_month="X",
        output_filename="soy_july_nov_spread_history.csv",
    ),
)


def write_normalized_spread_workbook(
    input_path: str | Path,
    output_dir: str | Path = "data/processed",
) -> list[SpreadWorkbookNormalizationResult]:
    """Normalize configured spread sheets from one workbook and write CSVs."""
    workbook_path = resolve_workbook_input_path(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in DEFAULT_SPREAD_WORKBOOK_SHEETS:
        raw = read_xlsx_sheet(workbook_path, spec.sheet_name)
        normalized, warnings = normalize_spread_workbook_sheet(raw, spec)
        output_path = output_root / spec.output_filename
        normalized.to_csv(output_path, index=False)
        results.append(
            SpreadWorkbookNormalizationResult(
                sheet_name=spec.sheet_name,
                output_path=output_path,
                data=normalized,
                warnings=warnings,
            )
        )
    return results


def normalize_spread_workbook_sheet(
    raw: pd.DataFrame,
    spec: SpreadWorkbookSheetSpec,
    min_rows_per_contract_year: int = 100,
    min_contract_years_warning: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize one workbook sheet into the shared spread-history schema."""
    if raw.empty:
        raise ValueError(f"Sheet {spec.sheet_name!r} is empty.")

    header_idx = _find_header_row(raw)
    headers = {
        _normalize_name(value): column
        for column, value in raw.iloc[header_idx].items()
        if not pd.isna(value)
    }

    date_column = _required_header(headers, ["timestamp", "date"], spec.sheet_name)
    spread_column = _required_header(
        headers,
        ["spreadvalue", "settle", "settlement", "settling", "close"],
        spec.sheet_name,
    )
    contract_year_column = _optional_header(headers, ["contractyear", "year"])
    volume_column = _optional_header(headers, ["volume"])
    open_interest_column = _optional_header(headers, ["openinterest", "openint", "opint"])

    data = raw.iloc[header_idx + 1 :].copy()
    parsed = pd.DataFrame(index=data.index)
    parsed["date"] = pd.to_datetime(data[date_column], errors="coerce")
    parsed["spread_value"] = pd.to_numeric(data[spread_column], errors="coerce")
    parsed["volume"] = _optional_numeric(data, volume_column)
    parsed["open_interest"] = _optional_numeric(data, open_interest_column)

    has_any_data = data[[date_column, spread_column]].notna().any(axis=1)
    if (has_any_data & parsed["date"].isna()).any():
        raise ValueError(f"{spec.sheet_name}: found rows with invalid dates.")
    if (has_any_data & parsed["spread_value"].isna()).any():
        raise ValueError(f"{spec.sheet_name}: spread_value must be numeric.")

    parsed = parsed[has_any_data].dropna(subset=["date", "spread_value"]).reset_index(drop=True)
    if parsed.empty:
        raise ValueError(f"{spec.sheet_name}: no usable spread rows found.")

    if contract_year_column is not None:
        contract_year = pd.to_numeric(data.loc[has_any_data, contract_year_column], errors="coerce")
        if contract_year.isna().any():
            raise ValueError(f"{spec.sheet_name}: contract_year must be numeric when provided.")
        parsed["contract_year"] = contract_year.astype(int).to_numpy()
    else:
        parsed["contract_year"] = parsed["date"].dt.year.astype(int)

    normalized = pd.DataFrame()
    normalized["date"] = parsed["date"].dt.date.astype(str)
    normalized["contract_year"] = parsed["contract_year"].astype(int)
    normalized["commodity"] = spec.commodity.lower().strip()
    normalized["spread_id"] = spec.spread_id.upper().strip()
    normalized["nearby_symbol"] = normalized["contract_year"].map(
        lambda year: _contract_symbol(spec.root, spec.nearby_month, int(year))
    )
    normalized["deferred_symbol"] = normalized["contract_year"].map(
        lambda year: _contract_symbol(spec.root, spec.deferred_month, int(year))
    )
    normalized["spread_value"] = parsed["spread_value"].astype(float)
    normalized["volume"] = parsed["volume"]
    normalized["open_interest"] = parsed["open_interest"]
    normalized = normalized[OUTPUT_COLUMNS].sort_values(["contract_year", "date"]).reset_index(drop=True)

    warnings = validate_spread_workbook_history(
        normalized,
        sheet_name=spec.sheet_name,
        min_rows_per_contract_year=min_rows_per_contract_year,
        min_contract_years_warning=min_contract_years_warning,
    )
    return normalized, warnings


def validate_spread_workbook_history(
    data: pd.DataFrame,
    sheet_name: str,
    min_rows_per_contract_year: int = 100,
    min_contract_years_warning: int = 10,
) -> list[str]:
    """Validate normalized workbook spread history and return warnings."""
    missing = set(OUTPUT_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"{sheet_name}: missing normalized columns: {sorted(missing)}")

    duplicate_mask = data.duplicated(subset=["date", "contract_year"], keep=False)
    if duplicate_mask.any():
        examples = data.loc[duplicate_mask, ["date", "contract_year"]].head(5).to_dict(orient="records")
        raise ValueError(f"{sheet_name}: duplicate date + contract_year rows found: {examples}")

    spread_values = pd.to_numeric(data["spread_value"], errors="coerce")
    if spread_values.isna().any():
        raise ValueError(f"{sheet_name}: spread_value must be numeric.")

    warnings = []
    counts = data.groupby("contract_year").size()
    short_years = counts[counts < min_rows_per_contract_year]
    for contract_year, row_count in short_years.items():
        warnings.append(
            f"{sheet_name}: contract_year {int(contract_year)} has only {int(row_count)} rows."
        )

    contract_year_count = int(data["contract_year"].nunique())
    if contract_year_count < min_contract_years_warning:
        warnings.append(
            f"{sheet_name}: only {contract_year_count} contract years found. "
            f"At least {min_contract_years_warning} are recommended for seasonal analysis."
        )
    return warnings


def read_xlsx_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Read one xlsx sheet using the standard library."""
    workbook_path = Path(path)
    with ZipFile(workbook_path) as archive:
        shared_strings = _read_shared_strings(archive)
        date_style_ids = _read_date_style_ids(archive)
        sheet_paths = _read_sheet_paths(archive)
        if sheet_name not in sheet_paths:
            raise ValueError(
                f"Workbook {workbook_path} does not contain sheet {sheet_name!r}. "
                f"Available sheets: {sorted(sheet_paths)}"
            )
        return _read_sheet_frame(
            archive,
            sheet_paths[sheet_name],
            shared_strings=shared_strings,
            date_style_ids=date_style_ids,
        )


def resolve_workbook_input_path(input_path: str | Path) -> Path:
    """Resolve direct, Downloads, and whitespace-normalized workbook paths."""
    path = Path(input_path)
    if path.exists():
        return path

    candidates = []
    if not path.is_absolute():
        candidates.append(Path.home() / "Downloads" / path)

    normalized_name = _squash_spaces(path.name)
    for directory in [path.parent if path.parent != Path("") else Path("."), Path.home() / "Downloads"]:
        if not directory.exists():
            continue
        for candidate in directory.glob("*.xlsx"):
            if _squash_spaces(candidate.name) == normalized_name:
                candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise ValueError(f"Workbook not found: {input_path}")


def _read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in item.findall(".//m:t", XLSX_NS))
        for item in root.findall("m:si", XLSX_NS)
    ]


def _read_date_style_ids(archive: ZipFile) -> set[int]:
    if "xl/styles.xml" not in archive.namelist():
        return set()
    root = ET.fromstring(archive.read("xl/styles.xml"))
    custom_formats = {
        int(item.attrib["numFmtId"]): item.attrib.get("formatCode", "")
        for item in root.findall("m:numFmts/m:numFmt", XLSX_NS)
    }
    builtin_date_formats = set(range(14, 23)) | {45, 46, 47}
    date_style_ids = set()
    cell_styles = root.find("m:cellXfs", XLSX_NS)
    if cell_styles is None:
        return date_style_ids
    for index, style in enumerate(cell_styles.findall("m:xf", XLSX_NS)):
        format_id = int(style.attrib.get("numFmtId", 0))
        format_code = custom_formats.get(format_id, "").lower()
        if format_id in builtin_date_formats or re.search(r"[ymd]", format_code):
            date_style_ids.add(index)
    return date_style_ids


def _read_sheet_paths(archive: ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

    sheet_paths = {}
    for sheet in workbook.findall("m:sheets/m:sheet", XLSX_NS):
        target = rel_map[sheet.attrib[REL_ID]]
        sheet_paths[sheet.attrib["name"]] = target if target.startswith("xl/") else f"xl/{target}"
    return sheet_paths


def _read_sheet_frame(
    archive: ZipFile,
    sheet_path: str,
    shared_strings: list[str],
    date_style_ids: set[int],
) -> pd.DataFrame:
    root = ET.fromstring(archive.read(sheet_path))
    rows = []
    max_col = 0
    for row in root.findall("m:sheetData/m:row", XLSX_NS):
        values = {}
        for cell in row.findall("m:c", XLSX_NS):
            col_idx = _column_index(cell.attrib.get("r", "A1"))
            max_col = max(max_col, col_idx)
            values[col_idx] = _cell_value(cell, shared_strings, date_style_ids)
        rows.append(values)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [[row.get(col_idx) for col_idx in range(max_col + 1)] for row in rows]
    )


def _cell_value(
    cell: ET.Element,
    shared_strings: list[str],
    date_style_ids: set[int],
) -> object:
    inline_text = cell.find("m:is/m:t", XLSX_NS)
    if inline_text is not None:
        return inline_text.text or ""

    value_node = cell.find("m:v", XLSX_NS)
    if value_node is None:
        return None

    raw_value = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(raw_value)]
    if cell.attrib.get("t") == "str":
        return raw_value

    style_id = int(cell.attrib.get("s", -1)) if "s" in cell.attrib else -1
    if style_id in date_style_ids:
        try:
            return _excel_date(raw_value)
        except ValueError:
            return raw_value
    return raw_value


def _find_header_row(raw: pd.DataFrame) -> int:
    for idx, row in raw.iterrows():
        headers = {_normalize_name(value) for value in row if not pd.isna(value)}
        if ("timestamp" in headers or "date" in headers) and (
            "close" in headers
            or "settle" in headers
            or "settlement" in headers
            or "settling" in headers
            or "spreadvalue" in headers
        ):
            return int(idx)
    raise ValueError("Could not find a header row containing date and spread value columns.")


def _required_header(headers: dict[str, object], names: list[str], sheet_name: str) -> object:
    column = _optional_header(headers, names)
    if column is None:
        raise ValueError(f"{sheet_name}: missing required column matching {names}.")
    return column


def _optional_header(headers: dict[str, object], names: list[str]) -> object | None:
    for name in names:
        if name in headers:
            return headers[name]
    return None


def _optional_numeric(raw: pd.DataFrame, column: object | None) -> pd.Series:
    if column is None:
        return pd.Series([pd.NA] * len(raw), index=raw.index, dtype="Float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _contract_symbol(root: str, month: str, contract_year: int) -> str:
    return f"{root.upper()}{month.upper()}{str(contract_year)[-2:]}"


def _excel_date(value: str) -> str:
    return (datetime(1899, 12, 30) + timedelta(days=float(value))).date().isoformat()


def _column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter.upper()) - ord("A") + 1
    return index - 1


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _squash_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
