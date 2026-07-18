"""Run the first Thresh AI agent: Seasonal Agent."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Allow this example to run without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.seasonal_agent import SeasonalAgent, SeasonalAgentConfig, report_to_markdown


def main() -> None:
    data_path = ROOT / "data" / "sample_corn_july_dec_spread.csv"
    df = pd.read_csv(data_path)

    config = SeasonalAgentConfig(
        commodity="corn",
        spread_id="CORN_NZ",
        horizon_days=30,
        min_years=10,
    )

    agent = SeasonalAgent(spread_history=df, config=config)

    # We analyze June 7, 2024 using prior years as history.
    # June is in the corn pre-pollination / pollination risk window.
    report = agent.analyze(analysis_date="2024-06-07")

    print(report_to_markdown(report))


if __name__ == "__main__":
    main()
