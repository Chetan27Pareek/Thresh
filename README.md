# Thresh v1 - First AI Agent: Seasonal Agent

This is the first working Thresh agent scaffold.

The first agent is the **Seasonal Agent**. It answers:

> What has this agricultural calendar spread usually done from this point in the crop year, and is the current setup bullish, bearish, or neutral?

This version uses a sample/mock historical spread dataset so you can learn the agent structure before plugging in real CBOT data.

The first agent is deliberately deterministic: the market math, thresholds, analog years, and uncertainty all come from code. A future LLM layer can turn the structured report into a richer trader narrative, but the numeric view should stay auditable.

## Project layout

```text
thresh_v1/
  data/
    sample_corn_july_dec_spread.csv
    zc_n26_z26_calendar_daily.csv
  src/
    thresh/
      cli.py
      seasonal_agent.py
      spread_builder.py
  examples/
    run_seasonal_agent.py
    plot_sample_spread.py
  tests/
    test_seasonal_agent.py
  outputs/
```

## Install

From inside this folder:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the first agent

```bash
python examples/run_seasonal_agent.py
```

Or run it through the CLI:

```bash
PYTHONPATH=src python -m thresh seasonal --analysis-date 2024-06-07
```

For machine-readable output:

```bash
PYTHONPATH=src python -m thresh seasonal --analysis-date 2024-06-07 --format json
```

## Import a chart-exported spread

The project can normalize an OHLC chart export like `ZC Jul26-Dec26 Calendar_Daily.csv`:

```bash
PYTHONPATH=src python -m thresh normalize-calendar \
  --input "/home/chetanpareek/Downloads/ZC Jul26-Dec26 Calendar_Daily.csv" \
  --output data/zc_n26_z26_calendar_daily.csv \
  --commodity corn \
  --spread-id CORN_NZ \
  --symbol-prefix ZC \
  --nearby-month N \
  --deferred-month Z \
  --contract-year 2026
```

The normalized file uses `spread_value = close` and keeps the whole export as contract year `2026`. That is intentional: rows from 2024 and 2025 are still observations of the Jul26-Dec26 2026 spread, not separate historical analog years.

With only this one contract-year spread, Thresh can summarize current context, but it cannot produce a true multi-year seasonal analog signal yet. For that, we need prior Jul-Dec corn spread histories such as `ZCN25-ZCZ25`, `ZCN24-ZCZ24`, and older equivalents.

## Import corn and soybean history

The project can also normalize a workbook with `CORN` and `SOY` sheets containing `Timestamp`, `CLOSE`, `VOLUME`, and `OPINT_1` columns:

```bash
PYTHONPATH=src python -m thresh normalize-commodity-workbook \
  --input "/home/chetanpareek/Downloads/SOY AND CORN .xlsx" \
  --output data/soy_corn_continuous_history.csv
```

This creates one continuous outright series per crop:

- `CORN_CC1`
- `SOYBEANS_SC1`

The normalized file keeps settlement, volume, and open interest. For compatibility with the Seasonal Agent, `spread_value` is set equal to the settlement price.

Run the seasonal agent on the normalized corn series:

```bash
PYTHONPATH=src python -m thresh seasonal \
  --data data/soy_corn_continuous_history.csv \
  --commodity corn \
  --spread-id CORN_CC1 \
  --analysis-date 2026-06-23 \
  --min-years 10
```

Run the seasonal agent on the normalized soybean series:

```bash
PYTHONPATH=src python -m thresh seasonal \
  --data data/soy_corn_continuous_history.csv \
  --commodity soybeans \
  --spread-id SOYBEANS_SC1 \
  --analysis-date 2026-06-23 \
  --min-years 10
```

## Test

```bash
python -m unittest discover
```

## Important note

The sample CSV is synthetic data. It is useful for learning and testing code structure, but it is **not** suitable for trading.
