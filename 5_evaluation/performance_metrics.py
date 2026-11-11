"""
Production Strategy Evaluation and Robustness Assessment Engine.

Provides vectorized mathematical modules to evaluate risk-adjusted return profiles,
drawdown distribution spaces, and multi-regime timeline stability metrics.
"""

import numpy as np
import pandas as pd


def calculate_baseline_metrics(daily_strategy_returns, risk_free_rate=0.0):
    """
    Computes annualized risk-adjusted and tail performance attributes 
    from a continuous vector of daily trading returns.
    
    Parameters:
    -----------
    daily_strategy_returns : np.array or pd.Series
        Continuous chronological array of daily net portfolio returns.
    risk_free_rate : float
        Annualized risk-free benchmark rate (default is 0.0).
    """
    returns = np.array(daily_strategy_returns, dtype=float)
    if len(returns) == 0:
        return {
            "total_return": 0.0, "annualized_return": 0.0, "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0
        }
        
    # 1. Trajectory and Annualized Return Vectors
    total_return = np.prod(1.0 + returns) - 1.0
    n_days = len(returns)
    annualized_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    
    # 2. Continuous Volatility Scaling
    daily_vol = np.std(returns, ddof=1)
    annualized_vol = daily_vol * np.sqrt(252.0)
    
    # 3. Sharpe Ratio Math (Arithmetic Daily Downscaling)
    excess_returns = returns - (risk_free_rate / 252.0)
    mean_excess = np.mean(excess_returns)
    sharpe_ratio = (mean_excess / daily_vol) * np.sqrt(252.0) if daily_vol > 0 else 0.0
    
    # 4. Sortino Ratio Math (Downside Deviations Only)
    downside_returns = excess_returns[excess_returns < 0.0]
    downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(252.0) if len(downside_returns) > 1 else 0.0
    sortino_ratio = (mean_excess * 252.0) / downside_vol if downside_vol > 0 else 0.0
    
    # 5. Peak-to-Trough Drawdown Array Mechanics (Vectorized)
    cum_wealth = np.cumprod(1.0 + returns)
    running_peaks = np.maximum.accumulate(cum_wealth)
    drawdowns = (cum_wealth - running_peaks) / running_peaks
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": max_drawdown
    }


def run_time_regime_stability_test(daily_strategy_returns, date_index, periods_dict):
    """
    Sub-slices historical strategy performance into discrete time buckets 
    (e.g., 2020 Crash, 2022 Rate Hikes) to isolate structural performance regime decay.
    
    Parameters:
    -----------
    daily_strategy_returns : np.array
        Chronological array of net strategy returns from the backtest engine.
    date_index : np.array or pd.DatetimeIndex
        The matching array of timestamps mapping directly to the returns.
    periods_dict : dict
        A map of specific performance periods to slice. Format:
        {"Period Name": ("YYYY-MM-DD", "YYYY-MM-DD")}
    """
    # Force bind returns data to a cleanly indexed Pandas structure
    returns_series = pd.Series(daily_strategy_returns, index=pd.to_datetime(date_index))
    regime_results = {}
    
    for period_name, (start_dt, end_dt) in periods_dict.items():
        try:
            # Cleanly slice the target window interval from our performance track
            sliced_subset = returns_series.loc[start_dt:end_dt]
            
            # Enforce a minimum sample threshold of 10 observations to protect calculation metrics
            if len(sliced_subset) > 10:
                regime_results[period_name] = calculate_baseline_metrics(sliced_subset.to_numpy())
            else:
                print(f"⚠️ Insufficient data observations to compile window: {period_name}")
        except Exception as e:
            print(f"⚠️ Slicing window failed for period {period_name}. Error: {str(e)}")
            continue
            
    # Return as an organized, scannable summary matrix
    return pd.DataFrame(regime_results).T