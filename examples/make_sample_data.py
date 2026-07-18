"""Create synthetic Corn July-Dec spread data for learning the agent workflow.

This data is not real CBOT data. It is intentionally generated to look broadly
seasonal so the first agent can be tested without a paid data vendor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_sample_data(output_path: str | Path) -> None:
    rng = np.random.default_rng(42)
    rows = []

    for year in range(2005, 2025):
        dates = pd.bdate_range(f"{year}-03-01", f"{year}-09-30")

        # Year-specific structure: some years are tight, some are burdensome.
        supply_shock = rng.normal(0, 7)
        weather_shock = rng.normal(0, 8)

        # A few intentionally unusual years so the analog logic has variation.
        if year in {2012, 2021}:
            weather_shock += 18  # drought/tightness style years
        if year in {2014, 2016, 2018}:
            weather_shock -= 10  # easier supply style years

        noise = rng.normal(0, 1.3, size=len(dates)).cumsum() * 0.25

        for i, date in enumerate(dates):
            day_of_year = date.dayofyear

            # Seasonal behavior:
            # - modest carry early spring
            # - weather premium builds into late June / early July
            # - premium fades as harvest becomes more certain
            pollination_peak = 190
            weather_premium = weather_shock * np.exp(-((day_of_year - pollination_peak) ** 2) / (2 * 28**2))
            harvest_fade = -0.16 * max(day_of_year - 210, 0)
            spring_carry = -18 + 0.06 * (day_of_year - 60)
            spread_value = spring_carry + supply_shock + weather_premium + harvest_fade + noise[i]

            rows.append(
                {
                    "date": date.date().isoformat(),
                    "year": year,
                    "commodity": "corn",
                    "spread_id": "CORN_NZ",
                    "nearby_month": "N",  # July
                    "deferred_month": "Z",  # December
                    "spread_value": round(float(spread_value), 2),
                }
            )

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_sample_data(root / "data" / "sample_corn_july_dec_spread.csv")
    print("Wrote sample data to data/sample_corn_july_dec_spread.csv")
