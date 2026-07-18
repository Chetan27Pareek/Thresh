"""Plot the synthetic Corn July-Dec spread paths.

This is optional, but visualizing spread paths helps you learn seasonality.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "sample_corn_july_dec_spread.csv", parse_dates=["date"])
    df["day_of_year"] = df["date"].dt.dayofyear

    fig, ax = plt.subplots(figsize=(11, 6))
    for year, group in df.groupby("year"):
        ax.plot(group["day_of_year"], group["spread_value"], alpha=0.35, linewidth=1)

    avg = df.groupby("day_of_year", as_index=False)["spread_value"].mean()
    ax.plot(avg["day_of_year"], avg["spread_value"], linewidth=3, label="Average path")

    ax.set_title("Synthetic Corn July-Dec Spread Paths")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Spread value: July - December, cents/bu")
    ax.legend()

    output_path = ROOT / "outputs" / "sample_corn_july_dec_paths.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()
