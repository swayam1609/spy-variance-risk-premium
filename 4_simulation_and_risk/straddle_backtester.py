"""
Consolidated Vectorized Options Execution and Delta-Hedging Simulation Engine.

Programmatically targets At-The-Money (ATM) contract strikes, manages capital 
accounting vectors, and executes daily underlying share rebalancing.
"""

import numpy as np
import pandas as pd


def select_atm_straddle_contracts(options_chain_df, current_spot):
    """
    Identifies the specific call and put contract records closest to spot (ATM).
    Targets short-dated liquid contracts (between 7 and 45 days to expiration).
    """
    if options_chain_df.empty:
        return None, None

    chain = options_chain_df.copy()
    chain["dte"] = (chain["expiration"] - chain["quote_date"]).dt.days

    # Filter for standard liquid monthly/weekly expirations (7 to 45 DTE)
    valid_dte = chain[(chain["dte"] >= 7) & (chain["dte"] <= 45)]
    if valid_dte.empty:
        valid_dte = chain[chain["dte"] > 0]
    if valid_dte.empty:
        return None, None

    nearest_expiry = valid_dte["expiration"].min()
    expiry_slice = valid_dte[valid_dte["expiration"] == nearest_expiry].copy()

    # Locate the strike closest to the underlying spot price
    expiry_slice["strike_distance"] = np.abs(expiry_slice["strike"] - current_spot)
    closest_idx = expiry_slice["strike_distance"].idxmin()
    atm_strike = expiry_slice.loc[closest_idx, "strike"]

    strike_slice = expiry_slice[expiry_slice["strike"] == atm_strike]
    
    # Handle both 'C'/'P' and 'CALL'/'PUT' formats cleanly
    opt_col = strike_slice["option_type"].astype(str).str.upper().str[0]
    atm_call = strike_slice[opt_col == "C"]
    atm_put = strike_slice[opt_col == "P"]

    if atm_call.empty or atm_put.empty:
        return None, None

    return atm_call.iloc[0], atm_put.iloc[0]


def run_vectorized_straddle_backtest(
    vrp_signals_df, options_master_df, initial_capital=1000000.0
):
    """
    Executes a historical backtest tracking short ATM straddle positions 
    under systematic daily underlying delta rebalancing constraints.
    """
    unique_dates = sorted(vrp_signals_df["quote_date"].unique())
    n_days = len(unique_dates)

    portfolio_value = np.full(n_days, initial_capital, dtype=float)
    daily_strategy_returns = np.zeros(n_days, dtype=float)
    active_trades_mask = np.zeros(n_days, dtype=int)

    pos_days = int(vrp_signals_df["is_premium_positive"].sum())
    print(f"📈 Initiating simulation engine across {n_days} trading dates (Positive VRP signal days: {pos_days})...")

    current_capital = initial_capital
    active_position = None
    trades_executed = 0

    options_by_date = dict(tuple(options_master_df.groupby("quote_date")))

    for t, target_date in enumerate(unique_dates):
        day_signal = vrp_signals_df[vrp_signals_df["quote_date"] == target_date].iloc[0]
        day_options = options_by_date.get(target_date, pd.DataFrame())

        spot_price = float(day_signal["underlying_price"])

        # 1. Mark-to-Market and Re-hedge Active Position
        if active_position is not None and not day_options.empty:
            opt_col = day_options["option_type"].astype(str).str.upper().str[0]
            current_call = day_options[
                (day_options["strike"] == active_position["strike"])
                & (day_options["expiration"] == active_position["expiry"])
                & (opt_col == "C")
            ]
            current_put = day_options[
                (day_options["strike"] == active_position["strike"])
                & (day_options["expiration"] == active_position["expiry"])
                & (opt_col == "P")
            ]

            if not current_call.empty and not current_put.empty:
                c_mid = float(current_call.iloc[0]["mid_price"])
                p_mid = float(current_put.iloc[0]["mid_price"])
                current_straddle_cost = c_mid + p_mid

                # Short straddle PnL (100 multiplier per standard equity option contract)
                option_pnl = (
                    (active_position["last_cost"] - current_straddle_cost)
                    * 100.0
                    * active_position["contracts"]
                )

                # Underlying delta-hedging share PnL
                stock_pnl = active_position["stock_shares"] * (
                    spot_price - active_position["last_spot"]
                )

                current_capital += option_pnl + stock_pnl
                active_trades_mask[t] = 1

                # Approximate net straddle delta as spot drifts from strike
                moneyness_pct = (spot_price - active_position["strike"]) / active_position["strike"]
                approx_call_delta = np.clip(0.50 + (moneyness_pct * 2.5), 0.05, 0.95)
                approx_put_delta = approx_call_delta - 1.0
                net_short_delta = -(approx_call_delta + approx_put_delta)

                # Rebalance SPY shares to neutralize directional delta
                active_position["stock_shares"] = (
                    -net_short_delta * 100.0 * active_position["contracts"]
                )
                active_position["last_spot"] = spot_price
                active_position["last_cost"] = current_straddle_cost

                # Close position 1 day before expiration or if VRP signal turns negative
                days_left = (active_position["expiry"] - target_date).days
                if days_left <= 1 or int(day_signal["is_premium_positive"]) == 0:
                    active_position = None
            else:
                active_position = None

        # 2. Open New ATM Short Straddle if Flat and VRP Signal is Positive
        if active_position is None and int(day_signal["is_premium_positive"]) == 1 and not day_options.empty:
            call_contract, put_contract = select_atm_straddle_contracts(
                day_options, spot_price
            )

            if call_contract is not None and put_contract is not None:
                entry_premium = float(call_contract["mid_price"]) + float(put_contract["mid_price"])
                num_contracts = max(1, int((current_capital * 0.05) / (spot_price * 100.0)))

                active_position = {
                    "strike": float(call_contract["strike"]),
                    "expiry": call_contract["expiration"],
                    "contracts": num_contracts,
                    "last_cost": entry_premium,
                    "stock_shares": 0.0,
                    "last_spot": spot_price,
                }
                active_trades_mask[t] = 1
                trades_executed += 1

        portfolio_value[t] = current_capital
        if t > 0 and portfolio_value[t - 1] > 0:
            daily_strategy_returns[t] = (
                portfolio_value[t] - portfolio_value[t - 1]
            ) / portfolio_value[t - 1]

    print(f"✅ Simulation complete. Total straddles traded: {trades_executed} | Gross Ending Capital: ${current_capital:,.2f}")
    return daily_strategy_returns, active_trades_mask
