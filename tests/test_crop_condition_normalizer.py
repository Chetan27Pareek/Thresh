from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import build_parser, run_normalize_crop_conditions  # noqa: E402
from thresh.crop_condition_normalizer import (  # noqa: E402
    OUTPUT_COLUMNS,
    normalize_crop_condition_directory,
)


class CropConditionNormalizerTests(unittest.TestCase):
    def test_normalizes_corn_and_soybean_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_crop_condition_csv(input_dir / "conditions.csv")

            result = normalize_crop_condition_directory(input_dir)

            self.assertEqual(list(result.data.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(result.data), 4)
            self.assertEqual(set(result.data["commodity"]), {"corn", "soybeans"})
            corn = result.data[
                result.data["commodity"].eq("corn") & result.data["date"].eq("2026-06-07")
            ].iloc[0]
            self.assertEqual(float(corn["good_pct"]), 60.0)
            self.assertEqual(float(corn["excellent_pct"]), 12.0)
            self.assertEqual(float(corn["good_excellent_pct"]), 72.0)
            self.assertEqual(result.warnings, [])

    def test_missing_weeks_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_crop_condition_csv(
                input_dir / "conditions.csv",
                dates=("2026-06-07", "2026-06-21"),
            )

            result = normalize_crop_condition_directory(input_dir)

            self.assertIn(
                "corn 2026 missing crop-condition weeks: 2026-06-14",
                result.warnings,
            )
            self.assertIn(
                "soybeans 2026 missing crop-condition weeks: 2026-06-14",
                result.warnings,
            )

    def test_missing_commodity_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_crop_condition_csv(input_dir / "conditions.csv", commodities=("CORN",))

            result = normalize_crop_condition_directory(input_dir)

            self.assertEqual(set(result.data["commodity"]), {"corn"})
            self.assertIn("No U.S. national soybeans crop-condition rows found.", result.warnings)

    def test_duplicate_condition_rows_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_crop_condition_csv(
                input_dir / "conditions.csv",
                duplicate_metric=True,
            )

            with self.assertRaisesRegex(ValueError, "Duplicate crop-condition metric rows"):
                normalize_crop_condition_directory(input_dir)

    def test_nonnumeric_percentages_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            self._write_crop_condition_csv(
                input_dir / "conditions.csv",
                bad_value=True,
            )

            with self.assertRaisesRegex(ValueError, "percentages must be numeric"):
                normalize_crop_condition_directory(input_dir)

    def test_cli_normalize_crop_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "conditions"
            input_dir.mkdir()
            self._write_crop_condition_csv(input_dir / "conditions.csv")
            output_path = tmp_path / "data" / "fundamentals" / "crop_conditions.csv"
            args = build_parser().parse_args(
                [
                    "normalize-crop-conditions",
                    "--input-dir",
                    str(input_dir),
                    "--output",
                    str(output_path),
                ]
            )

            message = run_normalize_crop_conditions(args)

            self.assertTrue(output_path.exists())
            output = pd.read_csv(output_path)
            self.assertEqual(list(output.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(output), 4)
            self.assertIn("Wrote 4 rows", message)

    @staticmethod
    def _write_crop_condition_csv(
        path: Path,
        dates: tuple[str, ...] = ("2026-06-07", "2026-06-14"),
        commodities: tuple[str, ...] = ("CORN", "SOYBEANS"),
        duplicate_metric: bool = False,
        bad_value: bool = False,
    ) -> Path:
        rows = []
        for commodity in commodities:
            for index, date in enumerate(dates):
                for condition, value in {
                    "EXCELLENT": 12 + index,
                    "GOOD": 60 - index,
                    "FAIR": 20,
                }.items():
                    if bad_value and commodity == "CORN" and condition == "GOOD":
                        value = "bad"
                    rows.append(
                        {
                            "Program": "SURVEY",
                            "Year": int(date[:4]),
                            "Period": "WEEK #23",
                            "Week Ending": date,
                            "Geo Level": "NATIONAL",
                            "State": "US TOTAL",
                            "State ANSI": None,
                            "Ag District": None,
                            "Ag District Code": None,
                            "County": None,
                            "County ANSI": None,
                            "Zip Code": None,
                            "Region": None,
                            "watershed_code": 0,
                            "Watershed": None,
                            "Commodity": commodity,
                            "Data Item": f"{commodity} - CONDITION, MEASURED IN PCT {condition}",
                            "Domain": "TOTAL",
                            "Domain Category": "NOT SPECIFIED",
                            "Value": value,
                            "CV (%)": None,
                        }
                    )
        if duplicate_metric:
            rows.append(rows[0].copy())
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
