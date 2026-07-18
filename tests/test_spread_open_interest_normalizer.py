from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import (  # noqa: E402
    build_parser,
    run_normalize_complete_spread_datasets,
    run_normalize_merged_spread_history,
)
from thresh.spread_history_normalizer import OUTPUT_COLUMNS  # noqa: E402
from thresh.spread_open_interest_normalizer import (  # noqa: E402
    CORN_JULY_DEC_SPEC,
    merge_spread_and_open_interest_history,
)


class SpreadOpenInterestNormalizerTests(unittest.TestCase):
    def test_merges_spread_and_open_interest_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spread_path = self._write_spread_csv(tmp_path / "corn_spread.csv")
            oi_path = self._write_open_interest_csv(tmp_path / "corn_oi.csv")

            result = merge_spread_and_open_interest_history(
                spread_input=spread_path,
                open_interest_input=oi_path,
                spec=CORN_JULY_DEC_SPEC,
            )

            self.assertEqual(list(result.data.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(result.data), 6)
            self.assertEqual(result.warnings, [])
            self.assertEqual(set(result.data["contract_year"]), {2025, 2026})
            self.assertEqual(set(result.data["commodity"]), {"corn"})
            self.assertEqual(set(result.data["spread_id"]), {"CORN_NZ"})
            self.assertEqual(set(result.data["nearby_symbol"]), {"ZCN25", "ZCN26"})
            self.assertEqual(set(result.data["deferred_symbol"]), {"ZCZ25", "ZCZ26"})
            self.assertEqual(float(result.data.iloc[0]["spread_value"]), -12.0)
            self.assertEqual(float(result.data.iloc[0]["open_interest"]), 5000.0)
            self.assertIn("2025: 3 rows", result.summary[0])

    def test_nonnumeric_open_interest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spread_path = self._write_spread_csv(tmp_path / "corn_spread.csv")
            oi_path = self._write_open_interest_csv(
                tmp_path / "corn_oi.csv",
                bad_open_interest=True,
            )

            with self.assertRaisesRegex(ValueError, "open_interest must be numeric"):
                merge_spread_and_open_interest_history(spread_path, oi_path, CORN_JULY_DEC_SPEC)

    def test_missing_open_interest_warns_and_preserves_spread_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spread_path = self._write_spread_csv(tmp_path / "corn_spread.csv")
            oi_path = self._write_open_interest_csv(
                tmp_path / "corn_oi.csv",
                skip_last=True,
            )

            result = merge_spread_and_open_interest_history(spread_path, oi_path, CORN_JULY_DEC_SPEC)

            self.assertEqual(len(result.data), 6)
            self.assertEqual(int(result.data["open_interest"].isna().sum()), 1)
            self.assertEqual(result.warnings, ["1 spread rows did not match an open-interest row."])

    def test_duplicate_date_contract_year_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spread_path = self._write_spread_csv(
                tmp_path / "corn_spread.csv",
                duplicate_key=True,
            )
            oi_path = self._write_open_interest_csv(tmp_path / "corn_oi.csv")

            with self.assertRaisesRegex(ValueError, "duplicate date \\+ contract_year"):
                merge_spread_and_open_interest_history(spread_path, oi_path, CORN_JULY_DEC_SPEC)

    def test_directory_inputs_infer_contract_year_from_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spread_dir = tmp_path / "spread"
            oi_dir = tmp_path / "oi"
            spread_dir.mkdir()
            oi_dir.mkdir()
            for contract_year in (2025, 2026):
                self._write_single_year_spread_csv(
                    spread_dir / f"ZCN{str(contract_year)[-2:]}-ZCZ{str(contract_year)[-2:]}.csv",
                )
                self._write_single_year_open_interest_csv(
                    oi_dir / f"ZCN{str(contract_year)[-2:]}-ZCZ{str(contract_year)[-2:]}_OI.csv",
                    contract_year=contract_year,
                )

            result = merge_spread_and_open_interest_history(
                spread_input=spread_dir,
                open_interest_input=oi_dir,
                spec=CORN_JULY_DEC_SPEC,
            )

            self.assertEqual(set(result.data["contract_year"]), {2025, 2026})
            self.assertEqual(set(result.data["nearby_symbol"]), {"ZCN25", "ZCN26"})

    def test_wide_spread_and_instrument_open_interest_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spread_path = tmp_path / "corn jul-dec 20 years.csv"
            oi_path = tmp_path / "corn jul OI 20 years.csv"
            pd.DataFrame(
                {
                    "Timestamp.2026": ["2025-02-21", "2025-02-22"],
                    "2026": ["24.25", "NA"],
                    "Timestamp.2025": ["2024-02-21", "2024-02-22"],
                    "2025": ["5.5", "5.75"],
                }
            ).to_csv(spread_path, index=False)
            pd.DataFrame(
                {
                    "timestamp": ["2025-02-21", "2024-02-21", "2024-02-22"],
                    "instrument": ["CNN26", "CNN25", "CNN25"],
                    "oi": ["2030.0", "1,500.0", "1600.0"],
                }
            ).to_csv(oi_path, index=False)

            result = merge_spread_and_open_interest_history(
                spread_input=spread_path,
                open_interest_input=oi_path,
                spec=CORN_JULY_DEC_SPEC,
            )

            self.assertEqual(len(result.data), 3)
            self.assertEqual(set(result.data["contract_year"]), {2025, 2026})
            self.assertEqual(float(result.data.iloc[0]["open_interest"]), 1500.0)
            self.assertEqual(set(result.data["nearby_symbol"]), {"ZCN25", "ZCN26"})

    def test_cli_normalize_merged_spread_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spread_path = self._write_spread_csv(tmp_path / "corn_spread.csv")
            oi_path = self._write_open_interest_csv(tmp_path / "corn_oi.csv")
            output_path = tmp_path / "processed" / "corn_july_dec_spread_history.csv"
            args = build_parser().parse_args(
                [
                    "normalize-merged-spread-history",
                    "--spread-input",
                    str(spread_path),
                    "--open-interest-input",
                    str(oi_path),
                    "--output",
                    str(output_path),
                    "--commodity",
                    "corn",
                    "--spread-id",
                    "CORN_NZ",
                    "--symbol-root",
                    "ZC",
                    "--nearby-month",
                    "N",
                    "--deferred-month",
                    "Z",
                ]
            )

            message = run_normalize_merged_spread_history(args)

            self.assertTrue(output_path.exists())
            output = pd.read_csv(output_path)
            self.assertEqual(len(output), 6)
            self.assertIn("Wrote 6 rows", message)
            self.assertIn("Years available:", message)
            self.assertIn("2026: 3 rows", message)

    def test_cli_normalize_complete_spread_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_spread = self._write_spread_csv(tmp_path / "corn_spread.csv")
            corn_oi = self._write_open_interest_csv(tmp_path / "corn_oi.csv")
            soy_spread = self._write_spread_csv(tmp_path / "soy_spread.csv")
            soy_oi = self._write_open_interest_csv(tmp_path / "soy_oi.csv")
            output_dir = tmp_path / "processed"
            args = build_parser().parse_args(
                [
                    "normalize-complete-spread-datasets",
                    "--corn-spread-input",
                    str(corn_spread),
                    "--corn-open-interest-input",
                    str(corn_oi),
                    "--soy-spread-input",
                    str(soy_spread),
                    "--soy-open-interest-input",
                    str(soy_oi),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            message = run_normalize_complete_spread_datasets(args)

            corn_output = output_dir / "corn_july_dec_spread_history.csv"
            soy_output = output_dir / "soy_july_nov_spread_history.csv"
            self.assertTrue(corn_output.exists())
            self.assertTrue(soy_output.exists())
            self.assertIn("Corn July-Dec", message)
            self.assertIn("Soybean July-Nov", message)
            self.assertEqual(set(pd.read_csv(corn_output)["spread_id"]), {"CORN_NZ"})
            self.assertEqual(set(pd.read_csv(soy_output)["spread_id"]), {"SOYBEANS_NX"})

    @staticmethod
    def _write_spread_csv(path: Path, duplicate_key: bool = False) -> Path:
        rows = []
        for contract_year in (2025, 2026):
            for index, date in enumerate(pd.bdate_range(f"{contract_year}-06-16", periods=3)):
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "contract_year": contract_year,
                        "close": -12.0 - index,
                        "volume": 1000 + index,
                    }
                )
        if duplicate_key:
            rows[-1]["date"] = rows[-2]["date"]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    @staticmethod
    def _write_open_interest_csv(
        path: Path,
        bad_open_interest: bool = False,
        skip_last: bool = False,
    ) -> Path:
        rows = []
        for contract_year in (2025, 2026):
            for index, date in enumerate(pd.bdate_range(f"{contract_year}-06-16", periods=3)):
                if skip_last and contract_year == 2026 and index == 2:
                    continue
                value: object = f"{5000 + index:,}"
                if bad_open_interest and contract_year == 2026 and index == 2:
                    value = "bad-oi"
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "contract_year": contract_year,
                        "open_interest": value,
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    @staticmethod
    def _write_single_year_spread_csv(path: Path) -> Path:
        rows = []
        for index, date in enumerate(pd.bdate_range("2024-06-16", periods=3)):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "close": -10.0 - index,
                    "volume": 1200 + index,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    @staticmethod
    def _write_single_year_open_interest_csv(path: Path, contract_year: int) -> Path:
        rows = []
        for index, date in enumerate(pd.bdate_range("2024-06-16", periods=3)):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "open_interest": 7000 + contract_year + index,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
