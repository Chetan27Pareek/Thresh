from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.commodity_workbook import (  # noqa: E402
    CommodityWorkbookSheetSpec,
    normalize_commodity_sheet,
    summarize_commodity_history,
)


class CommodityWorkbookTests(unittest.TestCase):
    def test_normalizes_offset_sheet_and_preserves_liquidity_fields(self) -> None:
        raw = pd.DataFrame(
            [
                ["CC1", None, None, None, None],
                ["Updated at 13:21:35", None, None, None, None],
                [None, "Timestamp", "CLOSE", "VOLUME", "OPINT_1"],
                [None, "2026-06-23", "412", "19302", None],
                [None, None, "411", "100", "200"],
                [None, "2026-06-22", "411.5", "211745", "279642"],
            ]
        )
        spec = CommodityWorkbookSheetSpec(
            sheet_name="CORN",
            commodity="corn",
            series_id="CORN_CC1",
        )

        normalized = normalize_commodity_sheet(raw, spec)

        self.assertEqual(list(normalized["date"]), ["2026-06-22", "2026-06-23"])
        self.assertEqual(set(normalized["year"]), {2026})
        self.assertEqual(set(normalized["commodity"]), {"corn"})
        self.assertEqual(set(normalized["spread_id"]), {"CORN_CC1"})
        self.assertEqual(set(normalized["symbol"]), {"CC1"})
        self.assertEqual(float(normalized.iloc[0]["settle"]), 411.5)
        self.assertEqual(float(normalized.iloc[0]["spread_value"]), 411.5)
        self.assertEqual(float(normalized.iloc[0]["volume"]), 211745)
        self.assertEqual(float(normalized.iloc[0]["open_interest"]), 279642)
        self.assertTrue(pd.isna(normalized.iloc[1]["open_interest"]))

    def test_summarizes_normalized_history(self) -> None:
        history = pd.DataFrame(
            {
                "date": ["2026-06-22", "2026-06-23"],
                "year": [2026, 2026],
                "commodity": ["corn", "corn"],
                "spread_id": ["CORN_CC1", "CORN_CC1"],
                "symbol": ["CC1", "CC1"],
                "settle": [411.5, 412.0],
                "spread_value": [411.5, 412.0],
                "volume": [211745, 19302],
                "open_interest": [279642, pd.NA],
            }
        )

        summary = summarize_commodity_history(history)

        self.assertEqual(len(summary), 1)
        self.assertIn("CORN_CC1: 2 rows", summary[0])
        self.assertIn("latest settle 412.00", summary[0])
        self.assertIn("volume 19302", summary[0])
        self.assertIn("open interest n/a", summary[0])


if __name__ == "__main__":
    unittest.main()
