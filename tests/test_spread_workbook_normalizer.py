from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import build_parser, run_normalize_spread_workbook  # noqa: E402
from thresh.spread_history_normalizer import OUTPUT_COLUMNS  # noqa: E402
from thresh.spread_workbook_normalizer import (  # noqa: E402
    DEFAULT_SPREAD_WORKBOOK_SHEETS,
    normalize_spread_workbook_sheet,
    read_xlsx_sheet,
    write_normalized_spread_workbook,
)


class SpreadWorkbookNormalizerTests(unittest.TestCase):
    def test_writes_expected_outputs_from_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workbook = self._write_workbook(tmp_path / "SOY AND CORN (2).xlsx")
            output_dir = tmp_path / "processed"

            results = write_normalized_spread_workbook(workbook, output_dir=output_dir)

            self.assertEqual(len(results), 2)
            corn_path = output_dir / "corn_july_dec_spread_history.csv"
            soy_path = output_dir / "soy_july_nov_spread_history.csv"
            self.assertTrue(corn_path.exists())
            self.assertTrue(soy_path.exists())

            corn = pd.read_csv(corn_path)
            soy = pd.read_csv(soy_path)
            self.assertEqual(list(corn.columns), OUTPUT_COLUMNS)
            self.assertEqual(list(soy.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(corn), 210)
            self.assertEqual(set(corn["contract_year"]), {2024, 2025})
            self.assertEqual(set(corn["nearby_symbol"]), {"ZCN24", "ZCN25"})
            self.assertEqual(set(corn["deferred_symbol"]), {"ZCZ24", "ZCZ25"})
            self.assertEqual(set(soy["spread_id"]), {"SOYBEANS_NX"})
            self.assertEqual(set(soy["nearby_symbol"]), {"ZSN24", "ZSN25"})
            self.assertEqual(set(soy["deferred_symbol"]), {"ZSX24", "ZSX25"})
            self.assertTrue(any("only 2 contract years found" in warning for warning in results[0].warnings))

    def test_cli_normalize_spread_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workbook = self._write_workbook(tmp_path / "SOY AND CORN (2).xlsx")
            output_dir = tmp_path / "processed"
            args = build_parser().parse_args(
                [
                    "normalize-spread-workbook",
                    "--input",
                    str(workbook),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            message = run_normalize_spread_workbook(args)

            self.assertIn("CORN J-D", message)
            self.assertIn("SOY J-N", message)
            self.assertIn("corn_july_dec_spread_history.csv", message)
            self.assertIn("soy_july_nov_spread_history.csv", message)
            self.assertTrue((output_dir / "corn_july_dec_spread_history.csv").exists())
            self.assertTrue((output_dir / "soy_july_nov_spread_history.csv").exists())

    def test_duplicate_date_contract_year_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workbook = self._write_workbook(
                Path(tmp) / "SOY AND CORN (2).xlsx",
                duplicate_corn_date=True,
            )
            raw = read_xlsx_sheet(workbook, "CORN J-D")

            with self.assertRaisesRegex(ValueError, "duplicate date"):
                normalize_spread_workbook_sheet(raw, DEFAULT_SPREAD_WORKBOOK_SHEETS[0])

    def test_nonnumeric_spread_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workbook = self._write_workbook(
                Path(tmp) / "SOY AND CORN (2).xlsx",
                bad_corn_close=True,
            )
            raw = read_xlsx_sheet(workbook, "CORN J-D")

            with self.assertRaisesRegex(ValueError, "spread_value must be numeric"):
                normalize_spread_workbook_sheet(raw, DEFAULT_SPREAD_WORKBOOK_SHEETS[0])

    @classmethod
    def _write_workbook(
        cls,
        path: Path,
        duplicate_corn_date: bool = False,
        bad_corn_close: bool = False,
    ) -> Path:
        with ZipFile(path, "w") as archive:
            archive.writestr("xl/workbook.xml", cls._workbook_xml())
            archive.writestr("xl/_rels/workbook.xml.rels", cls._workbook_rels_xml())
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                cls._sheet_xml("SOYBEAN JUL-NOV", "SOY J-N", root_value=20.0),
            )
            archive.writestr(
                "xl/worksheets/sheet2.xml",
                cls._sheet_xml(
                    "CON JUL-DEC",
                    "CORN J-D",
                    root_value=-10.0,
                    duplicate_date=duplicate_corn_date,
                    bad_close=bad_corn_close,
                ),
            )
        return path

    @staticmethod
    def _workbook_xml() -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="SOY J-N" sheetId="1" r:id="rId1"/>
    <sheet name="CORN J-D" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>
"""

    @staticmethod
    def _workbook_rels_xml() -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>
"""

    @classmethod
    def _sheet_xml(
        cls,
        title: str,
        sheet_name: str,
        root_value: float,
        duplicate_date: bool = False,
        bad_close: bool = False,
    ) -> str:
        rows = [
            cls._row(1, ["", title]),
            cls._row(2, ["", "Timestamp", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]),
        ]
        dates = list(pd.bdate_range("2024-01-02", periods=105)) + list(
            pd.bdate_range("2025-01-02", periods=105)
        )
        if duplicate_date:
            dates[-1] = dates[-2]
        for offset, date in enumerate(dates, start=3):
            close: object = root_value + (offset - 3) * 0.25
            if bad_close and offset == len(dates) + 2:
                close = "not-a-number"
            rows.append(
                cls._row(
                    offset,
                    [
                        "",
                        date.date().isoformat(),
                        close,
                        close,
                        close,
                        close,
                        1000 + offset,
                    ],
                )
            )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
            "  <sheetData>\n"
            + "\n".join(rows)
            + "\n  </sheetData>\n"
            "</worksheet>\n"
        )

    @classmethod
    def _row(cls, row_number: int, values: list[object]) -> str:
        cells = []
        for index, value in enumerate(values):
            if value == "":
                continue
            ref = f"{cls._column_name(index + 1)}{row_number}"
            if isinstance(value, str):
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
        return f'    <row r="{row_number}">' + "".join(cells) + "</row>"

    @staticmethod
    def _column_name(index: int) -> str:
        letters = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return letters


if __name__ == "__main__":
    unittest.main()
