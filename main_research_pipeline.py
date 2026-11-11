
"""
Central Automated Research Orchestration Pipeline - Single Source Edition.

Optimized to run exclusively off a single processed options matrix file,
dynamically extracting underlying price indices directly from inline fields.

Research Pipeline:
1. Data Ingestion and Cleaning
2. Black-Scholes Inversion and SVI Calibration
3. Rolling GJR-GARCH and Variance Risk Premium Extraction
4. Delta-Hedged Straddle Backtesting and Transaction Frictions
5. Performance Evaluation, Regime Analysis and Reporting

Outputs:
- Historical Regime Stability CSV
- Performance Dashboard Excel
- Daily Execution Logs Excel
- Analytical Visualization PNG
- Quantitative Research Report Markdown
"""

import os
import sys
import importlib.util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: PROJECT INITIALIZATION
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def import_from_numeric_dir(module_name, file_path):

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load module: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ==============================================================================
# SECTION 2: IMPORT RESEARCH MODULES
# ==============================================================================

data_loader_mod = import_from_numeric_dir(
    "data_loader",
    os.path.join(
        BASE_DIR,
        "1_data_processing",
        "data_loader.py"
    )
)

svi_mod = import_from_numeric_dir(
    "svi_surface_fitter",
    os.path.join(
        BASE_DIR,
        "2_volatility_math",
        "svi_surface_fitter.py"
    )
)

vrp_mod = import_from_numeric_dir(
    "vrp_calculator",
    os.path.join(
        BASE_DIR,
        "3_statistical_strategy",
        "vrp_calculator.py"
    )
)

backtest_mod = import_from_numeric_dir(
    "straddle_backtester",
    os.path.join(
        BASE_DIR,
        "4_simulation_and_risk",
        "straddle_backtester.py"
    )
)

friction_mod = import_from_numeric_dir(
    "friction_overlay",
    os.path.join(
        BASE_DIR,
        "4_simulation_and_risk",
        "friction_overlay.py"
    )
)

evaluation_mod = import_from_numeric_dir(
    "performance_metrics",
    os.path.join(
        BASE_DIR,
        "5_evaluation",
        "performance_metrics.py"
    )
)


# ==============================================================================
# SECTION 3: MAP MODULE FUNCTIONS
# ==============================================================================

load_and_clean_options_data = (
    data_loader_mod.load_and_clean_options_data
)

calibrate_svi_smile = (
    svi_mod.calibrate_svi_smile
)

extract_implied_variance = (
    svi_mod.extract_implied_variance
)

calculate_rolling_gjr_garch = (
    vrp_mod.calculate_rolling_gjr_garch
)

extract_variance_risk_premium = (
    vrp_mod.extract_variance_risk_premium
)

run_vectorized_straddle_backtest = (
    backtest_mod.run_vectorized_straddle_backtest
)

apply_transaction_frictions = (
    friction_mod.apply_transaction_frictions
)

calculate_baseline_metrics = (
    evaluation_mod.calculate_baseline_metrics
)

run_time_regime_stability_test = (
    evaluation_mod.run_time_regime_stability_test
)


# ==============================================================================
# SECTION 4: MAIN RESEARCH ORCHESTRATION
# ==============================================================================

def execute_comprehensive_research_run(
    raw_options_path,
    output_root="output_reports"
):

    """
    Executes the complete SPY Variance Risk Premium research pipeline
    using a single processed options data source.
    """

    figures_dir = os.path.join(
        output_root,
        "figures"
    )

    tables_dir = os.path.join(
        output_root,
        "tables"
    )

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("CENTRAL QUANTITATIVE RESEARCH ORCHESTRATION PIPELINE")
    print("=" * 70)


    # ==========================================================================
    # STAGE 1: DATA INGESTION AND UNDERLYING PRICE EXTRACTION
    # ==========================================================================

    print("\n📂 Stage 1: Initializing Data Ingestion & Single-Source Filtration...")

    options_df = load_and_clean_options_data(
        raw_options_path,
        target_symbol="SPY"
    )

    if options_df.empty:

        raise ValueError(
            "Critical Failure: Options data is empty after ingestion."
        )

    print(
        f"Successfully loaded {len(options_df):,} cleaned option observations."
    )

    print(
        "Extracting underlying price observations from inline options data..."
    )

    if "underlying_price" not in options_df.columns:

        raise KeyError(
            "Missing required 'underlying_price' column."
        )

    spot_extraction = (
        options_df
        .groupby("quote_date")["underlying_price"]
        .last()
        .reset_index()
    )

    spot_extraction.columns = [
        "date",
        "close"
    ]

    spot_extraction = (
        spot_extraction
        .sort_values("date")
        .reset_index(drop=True)
    )

    spot_extraction["log_return"] = np.log(
        spot_extraction["close"]
        /
        spot_extraction["close"].shift(1)
    )

    spot_extraction["is_negative_shock"] = (
        spot_extraction["log_return"] < 0
    ).astype(int)

    aligned_underlying_df = (
        spot_extraction
        .dropna(subset=["log_return"])
        .reset_index(drop=True)
    )

    valid_dates = aligned_underlying_df["date"].unique()

    options_df = (
        options_df[
            options_df["quote_date"].isin(valid_dates)
        ]
        .reset_index(drop=True)
    )

    print(
        f"Underlying observations extracted: "
        f"{len(aligned_underlying_df):,}"
    )


    # ==========================================================================
    # STAGE 2: BLACK-SCHOLES INVERSION AND SVI CALIBRATION
    # ==========================================================================

    print("\n📐 Stage 2: Processing Black-Scholes Inversion & SVI Calibration...")

    daily_svi_records = []

    options_by_date = dict(
        tuple(options_df.groupby("quote_date"))
    )

    aligned_spot_lookup = (
        aligned_underlying_df
        .set_index("date")["close"]
    )

    for current_date in sorted(valid_dates):

        day_slice = options_by_date.get(
            current_date,
            pd.DataFrame()
        )

        if day_slice.empty:
            continue

        spot_price = float(
            aligned_spot_lookup.loc[current_date]
        )

        day_slice = day_slice.copy()

        day_slice["dte"] = (
            day_slice["expiration"]
            -
            day_slice["quote_date"]
        ).dt.days

        # Select liquid near-term options:
        # 14-45 DTE and within +/- 3% of ATM.

        liquid_slice = day_slice[
            (day_slice["dte"] >= 14)
            &
            (day_slice["dte"] <= 45)
            &
            (
                np.abs(
                    day_slice["strike"] - spot_price
                ) / spot_price <= 0.03
            )
        ]

        if liquid_slice.empty:
            continue

        target_expiry = liquid_slice["expiration"].min()

        atm_options = liquid_slice[
            liquid_slice["expiration"] == target_expiry
        ]

        day_variances = []
        day_moneyness = []

        T_slice = max(
            (target_expiry - current_date).days / 365.25,
            1e-4
        )

        for _, row in atm_options.iterrows():

            implied_v = extract_implied_variance(
                S=spot_price,
                K=row["strike"],
                T=T_slice,
                target_price=row["mid_price"],
                option_type=row["option_type"]
            )

            if (
                implied_v is not None
                and np.isfinite(implied_v)
                and implied_v > 0
            ):

                day_variances.append(
                    implied_v
                )

                day_moneyness.append(
                    np.log(
                        row["strike"] / spot_price
                    )
                )

        if len(day_variances) >= 3:

            svi_params = calibrate_svi_smile(
                day_moneyness,
                day_variances
            )

            if svi_params is None:
                continue

            # SVI parameter a is the minimum total variance
            # under the current parameterization.
            # Confirm this interpretation against svi_surface_fitter.py.

            annualized_implied_var = (
                float(svi_params[0]) / T_slice
            )

            if (
                not np.isfinite(annualized_implied_var)
                or annualized_implied_var <= 0
            ):
                continue

            daily_svi_records.append(
                {
                    "quote_date": current_date,
                    "svi_implied_variance": annualized_implied_var,
                    "underlying_price": spot_price
                }
            )

    if not daily_svi_records:

        raise ValueError(
            "Critical Failure: Unable to fit daily SVI volatility surfaces."
        )

    svi_summary_df = pd.DataFrame(
        daily_svi_records
    )

    final_analysis_df = pd.merge(
        svi_summary_df,
        aligned_underlying_df,
        left_on="quote_date",
        right_on="date",
        how="inner"
    )

    final_analysis_df = (
        final_analysis_df
        .sort_values("quote_date")
        .reset_index(drop=True)
    )

    print(
        f"Successfully calibrated SVI for "
        f"{len(final_analysis_df):,} trading observations."
    )


    # ==========================================================================
    # STAGE 3: ROLLING GJR-GARCH AND VRP EXTRACTION
    # ==========================================================================

    print("\n📉 Stage 3: Running Rolling Asymmetric GJR-GARCH(1,1)...")

    # One-year rolling estimation window.

    daily_gjr_variance = calculate_rolling_gjr_garch(
        final_analysis_df["log_return"],
        lookback_window=252
    )

    daily_gjr_variance = np.asarray(
        daily_gjr_variance,
        dtype=float
    )

    if len(daily_gjr_variance) != len(final_analysis_df):

        raise ValueError(
            "GJR-GARCH forecast length does not match analysis observations."
        )

    # Convert daily physical variance into annualized variance.

    annualized_gjr_variance = (
        daily_gjr_variance * 252.0
    )

    signal_matrix_df = extract_variance_risk_premium(
        final_analysis_df,
        annualized_gjr_variance
    )

    signal_matrix_df = (
        signal_matrix_df
        .reset_index(drop=True)
    )

    required_signal_columns = [
        "quote_date",
        "underlying_price",
        "svi_implied_variance",
        "physical_variance_forecast",
        "vrp_spread",
        "is_premium_positive"
    ]

    missing_columns = [
        col for col in required_signal_columns
        if col not in signal_matrix_df.columns
    ]

    if missing_columns:

        raise KeyError(
            f"Missing required VRP columns: {missing_columns}"
        )

    if signal_matrix_df.empty:

        raise ValueError(
            "Critical Failure: VRP signal matrix is empty."
        )

    print(
        f"VRP observations generated: {len(signal_matrix_df):,}"
    )


    # ==========================================================================
    # STAGE 4: STRADDLE BACKTEST AND TRANSACTION FRICTIONS
    # ==========================================================================

    print("\n📈 Stage 4: Executing Straddle Simulation & Rehedging Sequences...")

    raw_strategy_returns, active_trades_mask = (
        run_vectorized_straddle_backtest(
            signal_matrix_df,
            options_df
        )
    )

    net_strategy_returns = apply_transaction_frictions(
        raw_strategy_returns,
        active_trades_mask,
        signal_matrix_df,
        options_df
    )

    raw_strategy_returns = np.asarray(
        raw_strategy_returns,
        dtype=float
    )

    net_strategy_returns = np.asarray(
        net_strategy_returns,
        dtype=float
    )

    active_trades_mask = np.asarray(
        active_trades_mask,
        dtype=bool
    )

    expected_length = len(signal_matrix_df)

    if not (
        len(net_strategy_returns)
        == len(active_trades_mask)
        == expected_length
    ):

        raise ValueError(
            "Backtest output lengths do not match the VRP signal matrix."
        )


    # ==========================================================================
    # STAGE 5: PERFORMANCE EVALUATION
    # ==========================================================================

    print("\n📊 Stage 5: Evaluating Performance Profiles...")

    baseline_metrics = calculate_baseline_metrics(
        net_strategy_returns
    )

    print("\n🏆 --- SIMULATION TRIAL NET METRICS RESULTS ---")

    for metric_name, value in baseline_metrics.items():

        print(
            f"   • {metric_name.replace('_', ' ').title()}: "
            f"{value:,.4f}"
        )

    print("-" * 55)


    # ==========================================================================
    # STAGE 6: HISTORICAL REGIME STABILITY
    # ==========================================================================

    historical_regimes = {

        "Full Window Analysis": (
            str(signal_matrix_df["quote_date"].min()),
            str(signal_matrix_df["quote_date"].max())
        ),

        "2020 Volatility Shock": (
            "2020-01-01",
            "2020-12-31"
        ),

        "2022 Rates Regime Shift": (
            "2022-01-01",
            "2022-12-31"
        )
    }

    regime_df = run_time_regime_stability_test(
        daily_strategy_returns=net_strategy_returns,
        date_index=signal_matrix_df["quote_date"].to_numpy(),
        periods_dict=historical_regimes
    )

    regime_csv_path = os.path.join(
        tables_dir,
        "historical_regime_stability.csv"
    )

    regime_df.to_csv(
        regime_csv_path,
        index=False
    )

    print(
        f"Historical regime analysis saved: {regime_csv_path}"
    )


    # ==========================================================================
    # STAGE 7: ANALYTICAL VISUALIZATION
    # ==========================================================================

    print("\n📈 Stage 7: Generating Analytical Visualization...")

    cum_wealth = np.cumprod(
        1.0 + net_strategy_returns
    )

    fig, ax1 = plt.subplots(
        figsize=(11, 6)
    )

    ax1.plot(
        signal_matrix_df["quote_date"],
        cum_wealth,
        color="#1f77b4",
        lw=2,
        label="Strategy Net Wealth"
    )

    ax1.set_xlabel(
        "Timeline",
        fontsize=11
    )

    ax1.set_ylabel(
        "Growth Curve Factor",
        color="#1f77b4",
        fontsize=11
    )

    ax1.tick_params(
        axis="y",
        labelcolor="#1f77b4"
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    ax2 = ax1.twinx()

    ax2.fill_between(
        signal_matrix_df["quote_date"],
        signal_matrix_df["vrp_spread"],
        color="gray",
        alpha=0.15,
        label="VRP Spread Tracking"
    )

    ax2.set_ylabel(
        "Annualized VRP Spread",
        color="gray",
        fontsize=11
    )

    ax2.tick_params(
        axis="y",
        labelcolor="gray"
    )

    plt.title(
        "Systematic VRP Short Straddle Strategy Performance Dashboard",
        fontsize=12,
        fontweight="bold"
    )

    fig.tight_layout()

    chart_destination = os.path.join(
        figures_dir,
        "analytical_visuals.png"
    )

    plt.savefig(
        chart_destination,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Dashboard visualization saved: {chart_destination}"
    )


    # ==========================================================================
    # STAGE 8: MULTI-TAB EXCEL DASHBOARD
    # ==========================================================================

    print("\n📊 Stage 8: Generating Multi-Tab Excel Dashboard...")

    excel_path = os.path.join(
        output_root,
        "performance_dashboard.xlsx"
    )

    historical_log = pd.DataFrame(
        {
            "Date": signal_matrix_df["quote_date"],

            "Spot Price": signal_matrix_df["underlying_price"],

            "Annualized Implied Variance (Q)": (
                signal_matrix_df["svi_implied_variance"]
            ),

            "Annualized Physical Variance Forecast (P)": (
                signal_matrix_df["physical_variance_forecast"]
            ),

            "Extracted VRP Signal": (
                signal_matrix_df["vrp_spread"]
            ),

            "Position Active": active_trades_mask,

            "Daily Strategy Net Return": net_strategy_returns
        }
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        # Sheet 1: Performance Summary

        pd.DataFrame(
            list(baseline_metrics.items()),
            columns=[
                "Metric",
                "Metric Value"
            ]
        ).to_excel(
            writer,
            sheet_name="Performance Summary",
            index=False
        )

        # Sheet 2: Regime Analysis Matrix

        regime_df.to_excel(
            writer,
            sheet_name="Regime Analysis Matrix",
            index=False
        )

        # Sheet 3: Daily Execution Logs

        historical_log.to_excel(
            writer,
            sheet_name="Daily Execution Logs",
            index=False
        )

    print(
        f"💾 Multi-tab production spreadsheet generated: {excel_path}"
    )


    # ==========================================================================
    # STAGE 9: AUTOMATED MARKDOWN RESEARCH NOTE
    # ==========================================================================

    print("\n📄 Stage 9: Generating Quantitative Research Note...")

    report_path = os.path.join(
        output_root,
        "research_report.md"
    )

    positive_signal_pct = (
        signal_matrix_df["is_premium_positive"].mean()
    ) * 100.0

    markdown_text = f"""# Quantitative Research Note: SPY Variance Risk Premium Capturing

## Research Overview

This report summarizes the SPY Variance Risk Premium research pipeline
for the period {signal_matrix_df['quote_date'].min().date()}
to {signal_matrix_df['quote_date'].max().date()}.

The analysis compares annualized implied variance (Q), extracted from
SPY options using Stochastic Volatility Inspired (SVI) calibration,
with the annualized physical variance forecast (P), generated by a
rolling GJR-GARCH(1,1) model.

## Signal Summary

- Out-of-sample observations: {len(signal_matrix_df):,}
- Percentage of observations with positive VRP signal: {positive_signal_pct:.2f}%
- Active trading observations: {int(active_trades_mask.sum()):,}
- Total observations with an inactive position: {int((~active_trades_mask).sum()):,}

## Net Performance Metrics (After Transaction Frictions)

"""

    for metric_name, value in baseline_metrics.items():

        markdown_text += (
            f"- **{metric_name.replace('_', ' ').title()}:** "
            f"{value:,.4f}\n"
        )

    markdown_text += """

## Historical Regime Analysis

The separate `historical_regime_stability.csv` file contains the
performance breakdown for:

- Full sample period
- 2020 volatility-shock period
- 2022 rates-regime-shift period

## Generated Research Outputs

| File | Description |
|---|---|
| `performance_dashboard.xlsx` | Three-sheet performance and execution dashboard |
| `historical_regime_stability.csv` | Historical regime performance breakdown |
| `analytical_visuals.png` | Cumulative strategy wealth and VRP visualization |
| `research_report.md` | Automatically generated research summary |

## Methodological Notes

The research pipeline uses historical SPY options observations,
Black-Scholes implied variance inversion, SVI calibration,
rolling GJR-GARCH physical variance forecasts, and a backtested
straddle strategy with transaction-friction adjustments.

The reported results are historical simulation outputs and do not
constitute evidence of guaranteed future profitability.

The validity of the results depends on the quality of the option
quotes, model calibration, execution assumptions, and backtesting
implementation.
"""

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report_file:

        report_file.write(
            markdown_text
        )

    print(
        f"📄 Quantitative research report dynamically compiled: {report_path}"
    )


    # ==========================================================================
    # FINAL RESEARCH COMPLETION
    # ==========================================================================

    print("\n" + "=" * 70)
    print("QUANTITATIVE RESEARCH PIPELINE COMPLETED")
    print("=" * 70)

    print("\nGENERATED OUTPUTS:")

    print(f"1. Excel Dashboard : {excel_path}")
    print(f"2. Regime CSV      : {regime_csv_path}")
    print(f"3. Research Note   : {report_path}")
    print(f"4. Visualization   : {chart_destination}")

    print("\n" + "=" * 70)


# ==============================================================================
# SECTION 10: EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    OPTIONS_DATA_SOURCE = os.path.join(
        BASE_DIR,
        "data_inputs",
        "processed",
        "spy_options_combined.csv"
    )

    execute_comprehensive_research_run(
        raw_options_path=OPTIONS_DATA_SOURCE,
        output_root=os.path.join(
            BASE_DIR,
            "output_reports"
        )
    )