from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.market_data_updater import (  # noqa: E402
    SpreadUpdateSpec,
    update_eod_market_data,
)


class MockMarketProvider:
    def __init__(self, rows: list[dict] | None = None, fail: bool = False):
        self.rows = rows or []
        self.fail = fail

    def fetch_eod_contract_data(self, symbols: list[str], start_date: pd.Timestamp | None) -> pd.DataFrame:
        if self.fail:
            raise ValueError("download failed")
        data = pd.DataFrame(self.rows)
        if data.empty:
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "settle", "volume", "open_interest"])
        data["date"] = pd.to_datetime(data["date"])
        data = data[data["symbol"].isin(symbols)]
        if start_date is not None:
            data = data[data["date"] >= start_date]
        return data.reset_index(drop=True)


class MarketDataUpdaterTests(unittest.TestCase):
    def test_appends_unseen_market_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = tmp_path / "corn.csv"
            soy_path = tmp_path / "soy.csv"
            self._write_existing(corn_path, "corn", "CORN_NZ", "ZCN26", "ZCZ26")
            self._write_existing(soy_path, "soybeans", "SOYBEANS_NX", "ZSN26", "ZSX26")
            specs = [
                SpreadUpdateSpec("corn", "CORN_NZ", "ZC", "N", "Z", corn_path),
                SpreadUpdateSpec("soybeans", "SOYBEANS_NX", "ZS", "N", "X", soy_path),
            ]
            provider = MockMarketProvider(
                [
                    self._quote("2026-07-08", "ZCN26", 410, 100, 1000),
                    self._quote("2026-07-08", "ZCZ26", 430, 80, 900),
                    self._quote("2026-07-08", "ZSN26", 1010, 90, 700),
                    self._quote("2026-07-08", "ZSX26", 1020, 75, 650),
                ]
            )

            result = update_eod_market_data(provider, specs=specs)

            self.assertEqual(result.rows_appended, {"CORN_NZ": 1, "SOYBEANS_NX": 1})
            corn = pd.read_csv(corn_path)
            self.assertEqual(len(corn), 2)
            self.assertEqual(float(corn.iloc[-1]["spread_value"]), -20.0)
            self.assertEqual(float(corn.iloc[-1]["open_interest"]), 1000.0)

    def test_provider_failure_does_not_overwrite_existing_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = tmp_path / "corn.csv"
            self._write_existing(corn_path, "corn", "CORN_NZ", "ZCN26", "ZCZ26")
            before = corn_path.read_text()
            specs = [SpreadUpdateSpec("corn", "CORN_NZ", "ZC", "N", "Z", corn_path)]

            with self.assertRaisesRegex(ValueError, "download failed"):
                update_eod_market_data(MockMarketProvider(fail=True), specs=specs)

            self.assertEqual(corn_path.read_text(), before)

    def test_missing_eod_ohlc_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = tmp_path / "corn.csv"
            self._write_existing(corn_path, "corn", "CORN_NZ", "ZCN26", "ZCZ26")
            specs = [SpreadUpdateSpec("corn", "CORN_NZ", "ZC", "N", "Z", corn_path)]
            provider = MockMarketProvider(
                [
                    {
                        "date": "2026-07-08",
                        "symbol": "ZCN26",
                        "settle": 410,
                        "volume": 100,
                        "open_interest": 1000,
                    },
                    {
                        "date": "2026-07-08",
                        "symbol": "ZCZ26",
                        "settle": 430,
                        "volume": 80,
                        "open_interest": 900,
                    },
                ]
            )

            with self.assertRaisesRegex(ValueError, "missing columns"):
                update_eod_market_data(provider, specs=specs)

    def test_leg_date_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corn_path = tmp_path / "corn.csv"
            self._write_existing(corn_path, "corn", "CORN_NZ", "ZCN26", "ZCZ26")
            before = corn_path.read_text()
            specs = [SpreadUpdateSpec("corn", "CORN_NZ", "ZC", "N", "Z", corn_path)]
            provider = MockMarketProvider(
                [
                    self._quote("2026-07-08", "ZCN26", 410, 100, 1000),
                    self._quote("2026-07-09", "ZCZ26", 430, 80, 900),
                ]
            )

            with self.assertRaisesRegex(ValueError, "EOD leg dates do not match"):
                update_eod_market_data(provider, specs=specs)

            self.assertEqual(corn_path.read_text(), before)

    @staticmethod
    def _write_existing(
        path: Path,
        commodity: str,
        spread_id: str,
        nearby_symbol: str,
        deferred_symbol: str,
    ) -> None:
        pd.DataFrame(
            [
                {
                    "date": "2026-07-07",
                    "contract_year": 2026,
                    "commodity": commodity,
                    "spread_id": spread_id,
                    "nearby_symbol": nearby_symbol,
                    "deferred_symbol": deferred_symbol,
                    "spread_value": -10.0,
                    "volume": 1,
                    "open_interest": 2,
                }
            ]
        ).to_csv(path, index=False)

    @staticmethod
    def _quote(date: str, symbol: str, settle: float, volume: int, open_interest: int) -> dict:
        return {
            "date": date,
            "symbol": symbol,
            "open": settle - 1,
            "high": settle + 2,
            "low": settle - 3,
            "settle": settle,
            "volume": volume,
            "open_interest": open_interest,
        }


if __name__ == "__main__":
    unittest.main()
