from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import build_parser, run_normalize_spread_history  # noqa: E402
from thresh.spread_history_normalizer import (  # noqa: E402
    OUTPUT_COLUMNS,
    normalize_spread_history_directory,
    normalize_spread_history_file,
    parse_spread_filename,
    validate_spread_history,
)


class SpreadHistoryNormalizerTests(unittest.TestCase):
    def test_parse_spread_filename(self) -> None:
        metadata = parse_spread_filename("ZCN26-ZCZ26.csv")

        self.assertEqual(metadata.nearby_symbol, "ZCN26")
        self.assertEqual(metadata.deferred_symbol, "ZCZ26")
        self.assertEqual(metadata.contract_year, 2026)
        self.assertEqual(metadata.nearby_month, "N")
        self.assertEqual(metadata.deferred_month, "Z")

    def test_normalizes_one_spread_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_spread_csv(Path(tmp) / "ZCN26-ZCZ26.csv", 2026)

            normalized = normalize_spread_history_file(
                path,
                commodity="corn",
                spread_id="CORN_NZ",
            )

            self.assertEqual(list(normalized.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(normalized), 105)
            self.assertEqual(set(normalized["contract_year"]), {2026})
            self.assertEqual(set(normalized["commodity"]), {"corn"})
            self.assertEqual(set(normalized["spread_id"]), {"CORN_NZ"})
            self.assertEqual(set(normalized["nearby_symbol"]), {"ZCN26"})
            self.assertEqual(set(normalized["deferred_symbol"]), {"ZCZ26"})
            self.assertEqual(float(normalized.iloc[0]["spread_value"]), 10.0)
            self.assertEqual(float(normalized.iloc[-1]["volume"]), 1104.0)
            self.assertEqual(float(normalized.iloc[-1]["open_interest"]), 5104.0)

    def test_normalizes_directory_and_warns_for_fewer_than_ten_years(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_spread_csv(tmp_path / "ZCN24-ZCZ24.csv", 2024)
            self._write_spread_csv(tmp_path / "ZCN25-ZCZ25.csv", 2025)
            self._write_spread_csv(tmp_path / "ZCN26-ZCZ26.csv", 2026)

            result = normalize_spread_history_directory(
                tmp_path,
                commodity="corn",
                spread_id="CORN_NZ",
            )

            self.assertEqual(len(result.data), 315)
            self.assertEqual(set(result.data["contract_year"]), {2024, 2025, 2026})
            self.assertEqual(len(result.warnings), 1)
            self.assertIn("Only 3 contract years found", result.warnings[0])

    def test_duplicate_date_contract_year_rows_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_spread_csv(
                Path(tmp) / "ZCN26-ZCZ26.csv",
                2026,
                duplicate_date=True,
            )
            normalized = normalize_spread_history_file(path)

            with self.assertRaisesRegex(ValueError, "Duplicate date"):
                validate_spread_history(normalized)

    def test_nonnumeric_spread_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_spread_csv(
                Path(tmp) / "ZCN26-ZCZ26.csv",
                2026,
                bad_spread_value=True,
            )

            with self.assertRaisesRegex(ValueError, "spread_value must be numeric"):
                normalize_spread_history_file(path)

    def test_short_contract_year_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_spread_csv(Path(tmp) / "ZCN26-ZCZ26.csv", 2026, rows=99)
            normalized = normalize_spread_history_file(path)

            with self.assertRaisesRegex(ValueError, "at least 100 rows"):
                validate_spread_history(normalized)

    def test_cli_normalize_spread_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "raw"
            output_path = tmp_path / "processed" / "corn_july_dec_spread_history.csv"
            input_dir.mkdir()
            self._write_spread_csv(input_dir / "ZCN25-ZCZ25.csv", 2025)
            self._write_spread_csv(input_dir / "ZCN26-ZCZ26.csv", 2026)

            args = build_parser().parse_args(
                [
                    "normalize-spread-history",
                    "--input-dir",
                    str(input_dir),
                    "--output",
                    str(output_path),
                ]
            )
            message = run_normalize_spread_history(args)

            self.assertTrue(output_path.exists())
            output = pd.read_csv(output_path)
            self.assertEqual(len(output), 210)
            self.assertIn("Wrote 210 rows", message)
            self.assertIn("Warning: Only 2 contract years found", message)

    @staticmethod
    def _write_spread_csv(
        path: Path,
        contract_year: int,
        rows: int = 105,
        duplicate_date: bool = False,
        bad_spread_value: bool = False,
    ) -> Path:
        dates = pd.bdate_range(f"{contract_year - 1}-10-01", periods=rows)
        if duplicate_date and rows > 1:
            dates = dates.to_list()
            dates[-1] = dates[-2]
        close_values: list[object] = [10.0 + index * 0.25 for index in range(rows)]
        if bad_spread_value:
            close_values[-1] = "not-a-number"

        pd.DataFrame(
            {
                "Timestamp (UTC)": dates,
                "Close": close_values,
                "ZC Calendar: Volume": [1000 + index for index in range(rows)],
                "Open Interest": [5000 + index for index in range(rows)],
            }
        ).to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
