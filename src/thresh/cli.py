"""Command-line entry points for Thresh agents."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from thresh.chart_export import (
    ChartCalendarSpreadSpec,
    summarize_spread_series,
    write_normalized_calendar_spread,
)
from thresh.commodity_workbook import (
    summarize_commodity_history,
    write_normalized_commodity_history,
)
from thresh.continuous_context import (
    load_continuous_context_report,
    report_to_json as continuous_context_to_json,
    report_to_markdown as continuous_context_to_markdown,
)
from thresh.crop_condition_normalizer import write_normalized_crop_conditions
from thresh.export_pace_normalizer import write_normalized_export_pace
from thresh.fundamental_data_updater import (
    UsdaApiFundamentalProvider,
    update_fundamental_data,
)
from thresh.market_data_updater import CsvEodMarketDataProvider, update_eod_market_data
from thresh.seasonal_agent import (
    SeasonalAgent,
    SeasonalAgentConfig,
    report_to_json,
    report_to_markdown,
)
from thresh.spread_open_interest_normalizer import (
    CORN_JULY_DEC_SPEC,
    SOYBEAN_JULY_NOV_SPEC,
    MergeSpreadOpenInterestSpec,
    write_merged_spread_open_interest_history,
)
from thresh.spread_history_normalizer import write_normalized_spread_history
from thresh.spread_workbook_normalizer import write_normalized_spread_workbook
from thresh.supply_demand_agent import (
    SupplyDemandAgent,
    SupplyDemandAgentConfig,
    report_to_json as supply_demand_report_to_json,
    report_to_markdown as supply_demand_report_to_markdown,
)
from thresh.wasde_normalizer import write_normalized_wasde


DEFAULT_DATA_PATH = Path("data/sample_corn_july_dec_spread.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thresh",
        description="Run Thresh specialist agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seasonal = subparsers.add_parser(
        "seasonal",
        help="Run the Seasonal Agent for one calendar spread.",
    )
    seasonal.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Spread history CSV path. Default: {DEFAULT_DATA_PATH}",
    )
    seasonal.add_argument(
        "--analysis-date",
        default=None,
        help="Date to analyze, formatted as YYYY-MM-DD. Defaults to latest available spread date.",
    )
    seasonal.add_argument(
        "--commodity",
        default="corn",
        help="Commodity name in the spread history.",
    )
    seasonal.add_argument(
        "--spread-id",
        default="CORN_NZ",
        help="Spread identifier in the spread history.",
    )
    seasonal.add_argument(
        "--horizon-days",
        type=int,
        default=30,
        help="Forward calendar-day horizon for the seasonal test.",
    )
    seasonal.add_argument(
        "--min-years",
        type=int,
        default=10,
        help="Minimum analog years requested before warning.",
    )
    seasonal.add_argument(
        "--current-spread",
        type=float,
        default=None,
        help="Override the current spread value in cents.",
    )
    seasonal.add_argument(
        "--continuous-context",
        type=Path,
        default=None,
        help="Optional continuous-history CSV to attach as background market context.",
    )
    seasonal.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )

    supply_demand = subparsers.add_parser(
        "supply-demand",
        help="Run the Supply/Demand Agent for corn or soybeans.",
    )
    supply_demand.add_argument(
        "--commodity",
        required=True,
        help="Commodity to analyze, for example corn or soybeans.",
    )
    supply_demand.add_argument(
        "--analysis-date",
        default="latest",
        help="Date to analyze as YYYY-MM-DD, or latest.",
    )
    supply_demand.add_argument(
        "--wasde-data",
        type=Path,
        default=Path("data/fundamentals/wasde.csv"),
        help="Normalized WASDE fundamentals CSV path.",
    )
    supply_demand.add_argument(
        "--crop-condition-data",
        type=Path,
        default=Path("data/fundamentals/crop_conditions.csv"),
        help="Normalized crop-condition CSV path.",
    )
    supply_demand.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )

    normalize = subparsers.add_parser(
        "normalize-calendar",
        help="Normalize an OHLC calendar-spread chart export.",
    )
    normalize.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw chart-export CSV path.",
    )
    normalize.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Normalized Thresh CSV path to write.",
    )
    normalize.add_argument(
        "--commodity",
        default="corn",
        help="Commodity name for the normalized series.",
    )
    normalize.add_argument(
        "--spread-id",
        default="CORN_NZ",
        help="Spread identifier for the normalized series.",
    )
    normalize.add_argument(
        "--symbol-prefix",
        default="ZC",
        help="Futures symbol prefix.",
    )
    normalize.add_argument(
        "--nearby-month",
        default="N",
        help="Nearby leg futures month code. July is N.",
    )
    normalize.add_argument(
        "--deferred-month",
        default="Z",
        help="Deferred leg futures month code. December is Z.",
    )
    normalize.add_argument(
        "--contract-year",
        type=int,
        required=True,
        help="Contract year represented by this spread series.",
    )

    commodity_workbook = subparsers.add_parser(
        "normalize-commodity-workbook",
        help="Normalize a workbook with CORN and SOY commodity-history sheets.",
    )
    commodity_workbook.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw commodity workbook path.",
    )
    commodity_workbook.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Normalized Thresh CSV path to write.",
    )

    continuous_context = subparsers.add_parser(
        "continuous-context",
        help="Summarize market context for a continuous corn or soybean series.",
    )
    continuous_context.add_argument(
        "--symbol",
        required=True,
        help="Continuous series identifier, for example CORN_CC1 or SOYBEANS_SC1.",
    )
    continuous_context.add_argument(
        "--data",
        type=Path,
        default=Path("data/soy_corn_continuous_history.csv"),
        help="Normalized continuous-history CSV path.",
    )
    continuous_context.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )

    spread_history = subparsers.add_parser(
        "normalize-spread-history",
        help="Normalize historical calendar spread CSV files into one combined history.",
    )
    spread_history.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing one raw calendar spread CSV per contract year.",
    )
    spread_history.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Combined normalized spread-history CSV path to write.",
    )
    spread_history.add_argument(
        "--commodity",
        default="corn",
        help="Commodity name for the normalized spread history.",
    )
    spread_history.add_argument(
        "--spread-id",
        default="CORN_NZ",
        help="Spread identifier for the normalized spread history.",
    )

    spread_workbook = subparsers.add_parser(
        "normalize-spread-workbook",
        help="Normalize spread-history sheets directly from the corn/soy workbook.",
    )
    spread_workbook.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Workbook path containing CORN J-D and SOY J-N sheets.",
    )
    spread_workbook.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed spread-history CSV outputs.",
    )

    merged_spread = subparsers.add_parser(
        "normalize-merged-spread-history",
        help="Merge one spread history with matching open-interest history.",
    )
    merged_spread.add_argument(
        "--spread-input",
        type=Path,
        required=True,
        help="CSV file or directory containing spread history.",
    )
    merged_spread.add_argument(
        "--open-interest-input",
        type=Path,
        required=True,
        help="CSV file or directory containing open-interest history.",
    )
    merged_spread.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Processed merged CSV path to write.",
    )
    merged_spread.add_argument("--commodity", required=True)
    merged_spread.add_argument("--spread-id", required=True)
    merged_spread.add_argument("--symbol-root", required=True)
    merged_spread.add_argument("--nearby-month", required=True)
    merged_spread.add_argument("--deferred-month", required=True)

    complete_spreads = subparsers.add_parser(
        "normalize-complete-spread-datasets",
        help="Merge complete corn and soybean spread/open-interest histories.",
    )
    complete_spreads.add_argument("--corn-spread-input", type=Path, required=True)
    complete_spreads.add_argument("--corn-open-interest-input", type=Path, required=True)
    complete_spreads.add_argument("--soy-spread-input", type=Path, required=True)
    complete_spreads.add_argument("--soy-open-interest-input", type=Path, required=True)
    complete_spreads.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed corn and soybean outputs.",
    )

    wasde = subparsers.add_parser(
        "normalize-wasde",
        help="Normalize WASDE corn and soybean fundamentals.",
    )
    wasde.add_argument(
        "--input-dir",
        type=Path,
        default=Path("WASDE 2010-2026"),
        help="Directory containing WASDE CSV files.",
    )
    wasde.add_argument(
        "--output",
        type=Path,
        default=Path("data/fundamentals/wasde.csv"),
        help="Normalized WASDE fundamentals CSV path.",
    )

    crop_conditions = subparsers.add_parser(
        "normalize-crop-conditions",
        help="Normalize U.S. national corn and soybean crop-condition data.",
    )
    crop_conditions.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Crop condition"),
        help="Directory containing crop-condition CSV files.",
    )
    crop_conditions.add_argument(
        "--output",
        type=Path,
        default=Path("data/fundamentals/crop_conditions.csv"),
        help="Normalized crop-condition CSV path.",
    )

    export_pace = subparsers.add_parser(
        "normalize-export-pace",
        help="Normalize corn and soybean weekly export-sales pace data.",
    )
    export_pace.add_argument(
        "--corn-input",
        type=Path,
        default=Path("corn export sales last 20 years.csv"),
        help="Corn export-sales CSV path.",
    )
    export_pace.add_argument(
        "--soybean-input",
        "--soy-input",
        dest="soybean_input",
        type=Path,
        default=Path("soyabean export sales last 20 years.csv"),
        help="Soybean export-sales CSV path.",
    )
    export_pace.add_argument(
        "--output",
        type=Path,
        default=Path("data/fundamentals/export_pace.csv"),
        help="Normalized export-pace CSV path.",
    )

    update_all = subparsers.add_parser(
        "update-all",
        help="Update market/fundamental data and write latest agent reports.",
    )
    update_all.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/latest"),
        help="Directory for latest markdown and JSON reports.",
    )
    update_all.add_argument(
        "--seasonal-horizon-days",
        type=int,
        default=5,
        help="SeasonalAgent horizon used for latest reports.",
    )
    update_all.add_argument(
        "--wasde-data",
        type=Path,
        default=Path("data/fundamentals/wasde.csv"),
        help="Normalized WASDE fundamentals CSV path.",
    )
    update_all.add_argument(
        "--crop-condition-data",
        type=Path,
        default=Path("data/fundamentals/crop_conditions.csv"),
        help="Normalized crop-condition CSV path.",
    )

    update_eod_market = subparsers.add_parser(
        "update-eod-market",
        help="Update EOD market data and write latest SeasonalAgent reports.",
    )
    update_eod_market.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/latest"),
        help="Directory for latest SeasonalAgent markdown and JSON reports.",
    )
    update_eod_market.add_argument(
        "--seasonal-horizon-days",
        type=int,
        default=5,
        help="SeasonalAgent horizon used for latest reports.",
    )

    return parser


def run_seasonal_agent(args: argparse.Namespace) -> str:
    spread_history = pd.read_csv(args.data)
    analysis_date, analysis_date_is_latest = _resolve_analysis_date(
        spread_history=spread_history,
        commodity=args.commodity,
        spread_id=args.spread_id,
        requested_analysis_date=args.analysis_date,
    )
    continuous_context = None
    if args.continuous_context is not None:
        continuous_context = load_continuous_context_report(
            data_path=args.continuous_context,
            symbol=_continuous_context_symbol(args.commodity),
        )
    config = SeasonalAgentConfig(
        commodity=args.commodity,
        spread_id=args.spread_id,
        horizon_days=args.horizon_days,
        min_years=args.min_years,
    )
    agent = SeasonalAgent(spread_history=spread_history, config=config)
    report = agent.analyze(
        analysis_date=analysis_date,
        current_spread=args.current_spread,
        continuous_context=continuous_context,
        analysis_date_is_latest=analysis_date_is_latest,
    )

    if args.format == "json":
        return report_to_json(report)
    return report_to_markdown(report)


def run_supply_demand_agent(args: argparse.Namespace) -> str:
    wasde_data = pd.read_csv(args.wasde_data)
    crop_condition_data = pd.read_csv(args.crop_condition_data)
    agent = SupplyDemandAgent(
        wasde_data=wasde_data,
        crop_condition_data=crop_condition_data,
        config=SupplyDemandAgentConfig(commodity=args.commodity),
    )
    report = agent.analyze(args.analysis_date)

    if args.format == "json":
        return supply_demand_report_to_json(report)
    return supply_demand_report_to_markdown(report)


def _resolve_analysis_date(
    spread_history: pd.DataFrame,
    commodity: str,
    spread_id: str,
    requested_analysis_date: str | None,
) -> tuple[str, bool]:
    if spread_history.empty:
        raise ValueError("Spread history dataset is empty; cannot run seasonal analysis.")
    if requested_analysis_date:
        return str(pd.Timestamp(requested_analysis_date).date()), False

    required = {"date", "commodity", "spread_id"}
    missing = required - set(spread_history.columns)
    if missing:
        raise ValueError(f"Missing columns required to choose latest analysis date: {sorted(missing)}")

    df = spread_history.copy()
    df["commodity"] = df["commodity"].astype(str).str.lower()
    df["spread_id"] = df["spread_id"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    matching = df[
        (df["commodity"] == commodity.lower().strip())
        & (df["spread_id"] == spread_id.upper().strip())
    ].dropna(subset=["date"])

    if matching.empty:
        raise ValueError(
            f"No dated rows found for commodity={commodity!r}, spread_id={spread_id!r}; "
            "cannot choose latest analysis date."
        )
    return str(matching["date"].max().date()), True


def _continuous_context_symbol(commodity: str) -> str:
    normalized = commodity.lower().strip()
    if normalized == "corn":
        return "CORN_CC1"
    if normalized in {"soy", "soybean", "soybeans"}:
        return "SOYBEANS_SC1"
    raise ValueError(f"No default continuous-context symbol for commodity {commodity!r}.")


def run_normalize_calendar(args: argparse.Namespace) -> str:
    spec = ChartCalendarSpreadSpec(
        commodity=args.commodity,
        spread_id=args.spread_id,
        symbol_prefix=args.symbol_prefix,
        nearby_month=args.nearby_month,
        deferred_month=args.deferred_month,
        contract_year=args.contract_year,
    )
    normalized = write_normalized_calendar_spread(
        input_path=args.input,
        output_path=args.output,
        spec=spec,
    )
    summary = summarize_spread_series(normalized)

    return "\n".join(
        [
            f"Wrote {len(normalized)} rows to {args.output}",
            f"Date range: {summary.date_start} to {summary.date_end}",
            f"Latest close: {summary.latest_close} cents",
            f"One-day change: {summary.one_day_change} cents",
            f"20-day change: {summary.twenty_day_change} cents",
            f"50-day change: {summary.fifty_day_change} cents",
            f"Close percentile in file: {summary.close_percentile}%",
            f"Trend state: {summary.trend_state}",
        ]
    )


def run_normalize_commodity_workbook(args: argparse.Namespace) -> str:
    normalized = write_normalized_commodity_history(
        input_path=args.input,
        output_path=args.output,
    )
    return "\n".join(
        [
            f"Wrote {len(normalized)} rows to {args.output}",
            *summarize_commodity_history(normalized),
        ]
    )


def run_continuous_context(args: argparse.Namespace) -> str:
    report = load_continuous_context_report(
        data_path=args.data,
        symbol=args.symbol,
    )
    if args.format == "json":
        return continuous_context_to_json(report)
    return continuous_context_to_markdown(report)


def run_normalize_spread_history(args: argparse.Namespace) -> str:
    result = write_normalized_spread_history(
        input_dir=args.input_dir,
        output_path=args.output,
        commodity=args.commodity,
        spread_id=args.spread_id,
    )
    years = sorted(int(year) for year in result.data["contract_year"].unique())
    lines = [
        f"Wrote {len(result.data)} rows to {args.output}",
        f"Contract years: {', '.join(str(year) for year in years)}",
    ]
    lines.extend(f"Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def run_normalize_spread_workbook(args: argparse.Namespace) -> str:
    results = write_normalized_spread_workbook(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    lines = []
    for result in results:
        years = sorted(int(year) for year in result.data["contract_year"].unique())
        lines.append(
            f"Wrote {len(result.data)} rows from {result.sheet_name} to {result.output_path}"
        )
        lines.append(f"Contract years: {', '.join(str(year) for year in years)}")
        lines.extend(f"Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def run_normalize_merged_spread_history(args: argparse.Namespace) -> str:
    spec = MergeSpreadOpenInterestSpec(
        commodity=args.commodity,
        spread_id=args.spread_id,
        symbol_root=args.symbol_root,
        nearby_month=args.nearby_month,
        deferred_month=args.deferred_month,
    )
    result = write_merged_spread_open_interest_history(
        spread_input=args.spread_input,
        open_interest_input=args.open_interest_input,
        output_path=args.output,
        spec=spec,
    )
    return _format_merged_spread_result(args.output, result.summary, result.warnings, len(result.data))


def run_normalize_complete_spread_datasets(args: argparse.Namespace) -> str:
    output_dir = Path(args.output_dir)
    corn_output = output_dir / "corn_july_dec_spread_history.csv"
    soy_output = output_dir / "soy_july_nov_spread_history.csv"

    corn = write_merged_spread_open_interest_history(
        spread_input=args.corn_spread_input,
        open_interest_input=args.corn_open_interest_input,
        output_path=corn_output,
        spec=CORN_JULY_DEC_SPEC,
    )
    soy = write_merged_spread_open_interest_history(
        spread_input=args.soy_spread_input,
        open_interest_input=args.soy_open_interest_input,
        output_path=soy_output,
        spec=SOYBEAN_JULY_NOV_SPEC,
    )

    return "\n".join(
        [
            "Corn July-Dec",
            _format_merged_spread_result(corn_output, corn.summary, corn.warnings, len(corn.data)),
            "Soybean July-Nov",
            _format_merged_spread_result(soy_output, soy.summary, soy.warnings, len(soy.data)),
        ]
    )


def _format_merged_spread_result(
    output_path: Path,
    summary: list[str],
    warnings: list[str],
    row_count: int,
) -> str:
    lines = [f"Wrote {row_count} rows to {output_path}", "Years available:"]
    lines.extend(f"- {line}" for line in summary)
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def run_normalize_wasde(args: argparse.Namespace) -> str:
    result = write_normalized_wasde(
        input_dir=args.input_dir,
        output_path=args.output,
    )
    lines = [f"Wrote {len(result.data)} rows to {args.output}"]
    lines.extend(result.summary)
    lines.extend(f"Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def run_normalize_crop_conditions(args: argparse.Namespace) -> str:
    result = write_normalized_crop_conditions(
        input_dir=args.input_dir,
        output_path=args.output,
    )
    lines = [f"Wrote {len(result.data)} rows to {args.output}"]
    lines.extend(result.summary)
    lines.extend(f"Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def run_normalize_export_pace(args: argparse.Namespace) -> str:
    result = write_normalized_export_pace(
        corn_input=args.corn_input,
        soybean_input=args.soybean_input,
        output_path=args.output,
    )
    lines = [f"Wrote {len(result.data)} rows to {args.output}"]
    lines.extend(result.summary)
    lines.extend(f"Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def run_update_all(args: argparse.Namespace) -> str:
    market_result = update_eod_market_data(CsvEodMarketDataProvider())
    fundamental_result = update_fundamental_data(
        UsdaApiFundamentalProvider(),
        wasde_path=args.wasde_data,
        crop_condition_path=args.crop_condition_data,
    )
    report_paths = _write_latest_reports(
        output_dir=args.output_dir,
        seasonal_horizon_days=args.seasonal_horizon_days,
        wasde_data=args.wasde_data,
        crop_condition_data=args.crop_condition_data,
    )
    lines = [
        "Update complete.",
        f"Market rows appended: {market_result.rows_appended}",
        f"Fundamental rows appended: {fundamental_result.rows_appended}",
        f"Reports written: {', '.join(str(path) for path in report_paths)}",
    ]
    lines.extend(f"Warning: {warning}" for warning in market_result.warnings)
    lines.extend(f"Warning: {warning}" for warning in fundamental_result.warnings)
    return "\n".join(lines)


def run_update_eod_market(args: argparse.Namespace) -> str:
    market_result = update_eod_market_data(CsvEodMarketDataProvider())
    report_paths = _write_latest_seasonal_reports(
        output_dir=args.output_dir,
        seasonal_horizon_days=args.seasonal_horizon_days,
    )
    lines = [
        "EOD market update complete.",
        f"Market rows appended: {market_result.rows_appended}",
        f"Seasonal reports written: {', '.join(str(path) for path in report_paths)}",
    ]
    lines.extend(f"Warning: {warning}" for warning in market_result.warnings)
    return "\n".join(lines)


def _write_latest_reports(
    output_dir: Path,
    seasonal_horizon_days: int,
    wasde_data: Path,
    crop_condition_data: Path,
) -> list[Path]:
    paths = _write_latest_seasonal_reports(output_dir, seasonal_horizon_days)
    paths.extend(
        _write_latest_supply_demand_reports(
            output_dir=output_dir,
            wasde_data=wasde_data,
            crop_condition_data=crop_condition_data,
        )
    )
    return paths


def _write_latest_seasonal_reports(
    output_dir: Path,
    seasonal_horizon_days: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    seasonal_specs = [
        ("corn", "CORN_NZ", Path("data/processed/corn_july_dec_spread_history.csv"), "seasonal_corn"),
        ("soybeans", "SOYBEANS_NX", Path("data/processed/soy_july_nov_spread_history.csv"), "seasonal_soybeans"),
    ]
    for commodity, spread_id, data_path, prefix in seasonal_specs:
        spread_history = pd.read_csv(data_path)
        analysis_date, analysis_date_is_latest = _resolve_analysis_date(
            spread_history=spread_history,
            commodity=commodity,
            spread_id=spread_id,
            requested_analysis_date=None,
        )
        report = SeasonalAgent(
            spread_history=spread_history,
            config=SeasonalAgentConfig(
                commodity=commodity,
                spread_id=spread_id,
                horizon_days=seasonal_horizon_days,
            ),
        ).analyze(
            analysis_date=analysis_date,
            analysis_date_is_latest=analysis_date_is_latest,
        )
        paths.extend(
            _write_report_pair(
                output_dir=output_dir,
                prefix=prefix,
                markdown=report_to_markdown(report),
                json_text=report_to_json(report),
            )
        )

    return paths


def _write_latest_supply_demand_reports(
    output_dir: Path,
    wasde_data: Path,
    crop_condition_data: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    wasde = pd.read_csv(wasde_data)
    crop_conditions = pd.read_csv(crop_condition_data)
    for commodity, prefix in [("corn", "supply_demand_corn"), ("soybeans", "supply_demand_soybeans")]:
        report = SupplyDemandAgent(
            wasde_data=wasde,
            crop_condition_data=crop_conditions,
            config=SupplyDemandAgentConfig(commodity=commodity),
        ).analyze("latest")
        paths.extend(
            _write_report_pair(
                output_dir=output_dir,
                prefix=prefix,
                markdown=supply_demand_report_to_markdown(report),
                json_text=supply_demand_report_to_json(report),
            )
        )
    return paths


def _write_report_pair(output_dir: Path, prefix: str, markdown: str, json_text: str) -> list[Path]:
    markdown_path = output_dir / f"{prefix}.md"
    json_path = output_dir / f"{prefix}.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")
    return [markdown_path, json_path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "seasonal":
            print(run_seasonal_agent(args))
            return 0
        if args.command == "supply-demand":
            print(run_supply_demand_agent(args))
            return 0
        if args.command == "normalize-calendar":
            print(run_normalize_calendar(args))
            return 0
        if args.command == "normalize-commodity-workbook":
            print(run_normalize_commodity_workbook(args))
            return 0
        if args.command == "continuous-context":
            print(run_continuous_context(args))
            return 0
        if args.command == "normalize-spread-history":
            print(run_normalize_spread_history(args))
            return 0
        if args.command == "normalize-spread-workbook":
            print(run_normalize_spread_workbook(args))
            return 0
        if args.command == "normalize-merged-spread-history":
            print(run_normalize_merged_spread_history(args))
            return 0
        if args.command == "normalize-complete-spread-datasets":
            print(run_normalize_complete_spread_datasets(args))
            return 0
        if args.command == "normalize-wasde":
            print(run_normalize_wasde(args))
            return 0
        if args.command == "normalize-crop-conditions":
            print(run_normalize_crop_conditions(args))
            return 0
        if args.command == "normalize-export-pace":
            print(run_normalize_export_pace(args))
            return 0
        if args.command == "update-all":
            print(run_update_all(args))
            return 0
        if args.command == "update-eod-market":
            print(run_update_eod_market(args))
            return 0
    except (ImportError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
