from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import build_parser, run_normalize_wasde  # noqa: E402
from thresh.wasde_normalizer import (  # noqa: E402
    OUTPUT_COLUMNS,
    normalize_wasde_directory,
)


class WasdeNormalizerTests(unittest.TestCase):
    def test_normalizes_corn_and_soybeans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_wasde_csv(input_dir / "oce-wasde-report-data-2026-01.csv", "2026-01-12")

            result = normalize_wasde_directory(input_dir)

            self.assertEqual(list(result.data.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(result.data), 2)
            self.assertEqual(set(result.data["commodity"]), {"corn", "soybeans"})
            corn = result.data[result.data["commodity"].eq("corn")].iloc[0]
            self.assertEqual(corn["marketing_year"], "2025/26")
            self.assertEqual(float(corn["production"]), 15000.0)
            self.assertEqual(float(corn["exports"]), 2500.0)
            self.assertEqual(float(corn["ending_stocks"]), 1500.0)
            self.assertEqual(float(corn["total_use"]), 15000.0)
            self.assertEqual(float(corn["stocks_to_use"]), 10.0)

    def test_missing_months_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_wasde_csv(input_dir / "oce-wasde-report-data-2026-01.csv", "2026-01-12")
            self._write_wasde_csv(input_dir / "oce-wasde-report-data-2026-03.csv", "2026-03-12")

            result = normalize_wasde_directory(input_dir)

            self.assertIn("corn missing WASDE report months: 2026-02", result.warnings)
            self.assertIn("soybeans missing WASDE report months: 2026-02", result.warnings)

    def test_duplicate_metric_rows_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_wasde_csv(
                input_dir / "oce-wasde-report-data-2026-01.csv",
                "2026-01-12",
                duplicate_metric=True,
            )

            with self.assertRaisesRegex(ValueError, "Duplicate WASDE metric rows"):
                normalize_wasde_directory(input_dir)

    def test_nonnumeric_metric_values_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_wasde_csv(
                input_dir / "oce-wasde-report-data-2026-01.csv",
                "2026-01-12",
                bad_value=True,
            )

            with self.assertRaisesRegex(ValueError, "must be numeric"):
                normalize_wasde_directory(input_dir)

    def test_cli_normalize_wasde(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "wasde"
            input_dir.mkdir()
            self._write_wasde_csv(input_dir / "oce-wasde-report-data-2026-01.csv", "2026-01-12")
            output_path = tmp_path / "data" / "fundamentals" / "wasde.csv"
            args = build_parser().parse_args(
                [
                    "normalize-wasde",
                    "--input-dir",
                    str(input_dir),
                    "--output",
                    str(output_path),
                ]
            )

            message = run_normalize_wasde(args)

            self.assertTrue(output_path.exists())
            output = pd.read_csv(output_path)
            self.assertEqual(list(output.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(output), 2)
            self.assertIn("Wrote 2 rows", message)

    @staticmethod
    def _write_wasde_csv(
        path: Path,
        release_date: str,
        duplicate_metric: bool = False,
        bad_value: bool = False,
    ) -> Path:
        rows = []
        specs = [
            (
                "Corn",
                "U.S. Feed Grain and Corn Supply and Use",
                {
                    "Production": 15000,
                    "Exports": 2500,
                    "Ending Stocks": 1500,
                    "Use, Total": 15000,
                },
            ),
            (
                "Oilseed, Soybean",
                "U.S. Soybeans and Products Supply and Use (Domestic Measure)",
                {
                    "Production": 4500,
                    "Exports": 1800,
                    "Ending Stocks": 450,
                    "Use, Total": 4500,
                },
            ),
        ]
        for commodity, title, values in specs:
            for attribute, value in values.items():
                if bad_value and commodity == "Corn" and attribute == "Exports":
                    value = "bad"
                rows.append(
                    {
                        "WasdeNumber": 1,
                        "ReportDate": "January 2026",
                        "ReportTitle": title,
                        "Attribute": attribute,
                        "ReliabilityProjection": None,
                        "Commodity": commodity,
                        "Region": "United States",
                        "MarketYear": "2025/26",
                        "ProjEstFlag": "Est.",
                        "AnnualQuarterFlag": "Annual",
                        "Value": value,
                        "Unit": "Million Bushels",
                        "ReleaseDate": release_date,
                        "ReleaseTime": "12:00:00.0000000",
                        "ForecastYear": 2026,
                        "ForecastMonth": int(release_date[5:7]),
                    }
                )
        if duplicate_metric:
            rows.append(rows[0].copy())
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
