"""
Production Transaction Friction and Slippage Overlay Engine.

Applies bid-ask spread penalties, execution slippage, and clearing commissions 
only on verified trade entry and exit transitions.
"""

import numpy as np
import pandas as pd


def apply_transaction_frictions(
    daily_strategy_returns,
    active_trades_mask,
    vrp_signals_df,
    options_master_df,
    commission_per_contract=0.65,
    execution_slippage_pct=0.05,
    allocation_weight=0.05,
):
    """
    Applies realistic transaction friction penalties on trade entries and exits.
    """
    net_returns = np.array(daily_strategy_returns, dtype=float)
    unique_dates = sorted(vrp_signals_df["quote_date"].unique())

    print("✂️ Applying transaction frictions on verified trade transitions...")
    options_by_date = dict(tuple(options_master_df.groupby("quote_date")))
    friction_events = 0

    for t in range(1, len(unique_dates)):
        is_entry = (active_trades_mask[t] == 1) and (active_trades_mask[t - 1] == 0)
        is_exit = (active_trades_mask[t] == 0) and (active_trades_mask[t - 1] == 1)

        if is_entry or is_exit:
            target_date = unique_dates[t]
            day_options = options_by_date.get(target_date, pd.DataFrame())

            if not day_options.empty:
                spot_price = float(
                    vrp_signals_df[vrp_signals_df["quote_date"] == target_date].iloc[0]["underlying_price"]
                )
                # Filter to ATM options to get accurate spread width
                atm_slice = day_options[
                    np.abs(day_options["strike"] - spot_price) / spot_price <= 0.01
                ]
                if atm_slice.empty:
                    atm_slice = day_options

                avg_spread = float(atm_slice["bid_ask_spread"].median())

                if spot_price > 0:
                    # Half-spread crossing + slippage for 2 legs (call + put), scaled by capital allocation
                    per_share_cost = (avg_spread * (0.5 + execution_slippage_pct)) * 2.0
                    commission_per_share = (commission_per_contract * 2.0) / 100.0
                    total_cost_pct = ((per_share_cost + commission_per_share) / spot_price) * allocation_weight

                    net_returns[t] -= total_cost_pct
                    friction_events += 1

    net_returns = np.clip(net_returns, -0.99, None)
    print(f"✅ Friction overlay complete. Applied realistic costs across {friction_events} entry/exit events.")
    return net_returns