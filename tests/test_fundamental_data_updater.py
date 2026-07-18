from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.fundamental_data_updater import update_fundamental_data  # noqa: E402


class MockFundamentalProvider:
    def __init__(
        self,
        wasde_rows: list[dict] | None = None,
        crop_rows: list[dict] | None = None,
        fail_wasde: bool = False,
        fail_crop: bool = False,
    ):
        self.wasde_rows = wasde_rows or []
        self.crop_rows = crop_rows or []
        self.fail_wasde = fail_wasde
        self.fail_crop = fail_crop

    def fetch_latest_wasde(self, existing: pd.DataFrame) -> pd.DataFrame:
        if self.fail_wasde:
            raise ValueError("wasde api failed")
        return pd.DataFrame(self.wasde_rows)

    def fetch_latest_crop_conditions(self, existing: pd.DataFrame) -> pd.DataFrame:
        if self.fail_crop:
            raise ValueError("crop api failed")
        return pd.DataFrame(self.crop_rows)


class FundamentalDataUpdaterTests(unittest.TestCase):
    def test_appends_unseen_fundamental_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wasde_path = tmp_path / "wasde.csv"
            crop_path = tmp_path / "crop_conditions.csv"
            self._write_existing_wasde(wasde_path)
            self._write_existing_crop(crop_path)
            provider = MockFundamentalProvider(
                wasde_rows=[
                    self._wasde_row("2026-07-10", "corn", "2026/27", 12.0),
                    self._wasde_row("2026-07-10", "soybeans", "2026/27", 7.0),
                ],
                crop_rows=[
                    self._crop_row("2026-07-05", "corn", 52, 14),
                ],
            )

            result = update_fundamental_data(provider, wasde_path=wasde_path, crop_condition_path=crop_path)

            self.assertEqual(result.rows_appended, {"wasde": 2, "crop_conditions": 1})
            self.assertEqual(len(pd.read_csv(wasde_path)), 3)
            self.assertEqual(len(pd.read_csv(crop_path)), 2)

    def test_api_failure_does_not_overwrite_existing_fundamentals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wasde_path = tmp_path / "wasde.csv"
            crop_path = tmp_path / "crop_conditions.csv"
            self._write_existing_wasde(wasde_path)
            self._write_existing_crop(crop_path)
            wasde_before = wasde_path.read_text()
            crop_before = crop_path.read_text()

            with self.assertRaisesRegex(ValueError, "crop api failed"):
                update_fundamental_data(
                    MockFundamentalProvider(
                        wasde_rows=[self._wasde_row("2026-07-10", "corn", "2026/27", 12.0)],
                        fail_crop=True,
                    ),
                    wasde_path=wasde_path,
                    crop_condition_path=crop_path,
                )

            self.assertEqual(wasde_path.read_text(), wasde_before)
            self.assertEqual(crop_path.read_text(), crop_before)

    def test_duplicate_fetched_rows_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wasde_path = tmp_path / "wasde.csv"
            crop_path = tmp_path / "crop_conditions.csv"
            self._write_existing_wasde(wasde_path)
            self._write_existing_crop(crop_path)

            with self.assertRaisesRegex(ValueError, "duplicate rows"):
                update_fundamental_data(
                    MockFundamentalProvider(
                        wasde_rows=[
                            self._wasde_row("2026-07-10", "corn", "2026/27", 12.0),
                            self._wasde_row("2026-07-10", "corn", "2026/27", 12.0),
                        ],
                        crop_rows=[],
                    ),
                    wasde_path=wasde_path,
                    crop_condition_path=crop_path,
                )

    @staticmethod
    def _write_existing_wasde(path: Path) -> None:
        pd.DataFrame([FundamentalDataUpdaterTests._wasde_row("2026-06-10", "corn", "2026/27", 10.0)]).to_csv(
            path,
            index=False,
        )

    @staticmethod
    def _write_existing_crop(path: Path) -> None:
        pd.DataFrame([FundamentalDataUpdaterTests._crop_row("2026-06-28", "corn", 53, 14)]).to_csv(
            path,
            index=False,
        )

    @staticmethod
    def _wasde_row(date: str, commodity: str, marketing_year: str, stocks_to_use: float) -> dict:
        return {
            "date": date,
            "commodity": commodity,
            "marketing_year": marketing_year,
            "production": 1000.0,
            "exports": 200.0,
            "ending_stocks": stocks_to_use * 10,
            "total_use": 1000.0,
            "stocks_to_use": stocks_to_use,
        }

    @staticmethod
    def _crop_row(date: str, commodity: str, good: float, excellent: float) -> dict:
        return {
            "date": date,
            "commodity": commodity,
            "good_pct": good,
            "excellent_pct": excellent,
            "good_excellent_pct": good + excellent,
        }


if __name__ == "__main__":
    unittest.main()
