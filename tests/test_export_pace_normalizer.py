from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import build_parser, run_normalize_export_pace  # noqa: E402
from thresh.export_pace_normalizer import (  # noqa: E402
    OUTPUT_COLUMNS,
    normalize_export_pace_files,
)


class ExportPaceNormalizerTests(unittest.TestCase):
    def test_normalizes_corn_and_soybean_export_pace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = self._write_export_sales_csv(tmp_path / "corn.csv", "Corn")
            soy_path = self._write_export_sales_csv(tmp_path / "soybeans.csv", "Soybean")

            result = normalize_export_pace_files(corn_path, soy_path)

            self.assertEqual(list(result.data.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(result.data), 4)
            self.assertEqual(set(result.data["commodity"]), {"corn", "soybeans"})
            corn = result.data[
                result.data["commodity"].eq("corn") & result.data["date"].eq("2026-06-04")
            ].iloc[0]
            self.assertEqual(corn["marketing_year"], "2025/26")
            self.assertEqual(float(corn["weekly_exports"]), 42.0)
            self.assertEqual(float(corn["cumulative_exports"]), 1800.0)
            self.assertEqual(float(corn["outstanding_sales"]), 700.0)
            self.assertEqual(float(corn["total_commitments"]), 2500.0)
            self.assertEqual(result.warnings, [])

    def test_missing_weeks_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = self._write_export_sales_csv(
                tmp_path / "corn.csv",
                "Corn",
                dates=("2026-06-04", "2026-06-18"),
            )
            soy_path = self._write_export_sales_csv(
                tmp_path / "soybeans.csv",
                "Soybean",
                dates=("2026-06-04", "2026-06-18"),
            )

            result = normalize_export_pace_files(corn_path, soy_path)

            self.assertIn(
                "corn 2025/26 missing export-sales weeks: 2026-06-08/2026-06-14",
                result.warnings,
            )
            self.assertIn(
                "soybeans 2025/26 missing export-sales weeks: 2026-06-08/2026-06-14",
                result.warnings,
            )

    def test_duplicate_metric_rows_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = self._write_export_sales_csv(
                tmp_path / "corn.csv",
                "Corn",
                duplicate_metric=True,
            )
            soy_path = self._write_export_sales_csv(tmp_path / "soybeans.csv", "Soybean")

            with self.assertRaisesRegex(ValueError, "Duplicate export-pace metric rows"):
                normalize_export_pace_files(corn_path, soy_path)

    def test_nonnumeric_values_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = self._write_export_sales_csv(
                tmp_path / "corn.csv",
                "Corn",
                bad_value=True,
            )
            soy_path = self._write_export_sales_csv(tmp_path / "soybeans.csv", "Soybean")

            with self.assertRaisesRegex(ValueError, "values must be numeric"):
                normalize_export_pace_files(corn_path, soy_path)

    def test_missing_required_metrics_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = self._write_export_sales_csv(
                tmp_path / "corn.csv",
                "Corn",
                metrics=("Weekly Exports",),
            )
            soy_path = self._write_export_sales_csv(tmp_path / "soybeans.csv", "Soybean")

            with self.assertRaisesRegex(ValueError, "missing required metrics"):
                normalize_export_pace_files(corn_path, soy_path)

    def test_cli_normalize_export_pace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = self._write_export_sales_csv(tmp_path / "corn.csv", "Corn")
            soy_path = self._write_export_sales_csv(tmp_path / "soybeans.csv", "Soybean")
            output_path = tmp_path / "data" / "fundamentals" / "export_pace.csv"
            args = build_parser().parse_args(
                [
                    "normalize-export-pace",
                    "--corn-input",
                    str(corn_path),
                    "--soybean-input",
                    str(soy_path),
                    "--output",
                    str(output_path),
                ]
            )

            message = run_normalize_export_pace(args)

            self.assertTrue(output_path.exists())
            output = pd.read_csv(output_path)
            self.assertEqual(list(output.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(output), 4)
            self.assertIn("Wrote 4 rows", message)

    @staticmethod
    def _write_export_sales_csv(
        path: Path,
        commodity_label: str,
        dates: tuple[str, ...] = ("2026-06-04", "2026-06-11"),
        metrics: tuple[str, ...] = (
            "Weekly Exports",
            "Cumulative Exports",
            "Outstanding Sales",
            "Total Commitments",
        ),
        duplicate_metric: bool = False,
        bad_value: bool = False,
    ) -> Path:
        values = {
            "Weekly Exports": 42.0,
            "Cumulative Exports": 1800.0,
            "Outstanding Sales": 700.0,
            "Total Commitments": 2500.0,
        }
        rows = []
        for date_index, date in enumerate(dates):
            for metric in metrics:
                value: object = values[metric] + date_index
                if bad_value and metric == "Weekly Exports":
                    value = "bad"
                rows.append(
                    {
                        "instrument": f"US {commodity_label} {metric}",
                        "date": date,
                        "releasedate": date,
                        "actual": value,
                        "estimates": None,
                    }
                )
        if duplicate_metric:
            rows.append(rows[0].copy())
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
