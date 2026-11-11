"""
Production Asymmetric Volatility Forecasting and VRP Extraction Engine.

Implements a rolling out-of-sample GJR-GARCH(1,1) econometrics framework 
to isolate physical variance profiles alongside cross-sectional VRP spreads.
"""

import numpy as np
import pandas as pd
from arch import arch_model  # Industry-standard time-series core


def calculate_rolling_gjr_garch(daily_log_returns, lookback_window=500, forecast_horizon=1):
    """
    Executes a rolling out-of-sample forecasting loop to calculate the daily 
    one-step-ahead physical variance vector using a GJR-GARCH(1,1) model.
    
    Parameters:
    -----------
    daily_log_returns : pd.Series or np.array
        Continuous time-series matrix of underlying log stock returns.
    lookback_window : int
        The size of the historical estimation window (default is 2 trading years).
    forecast_horizon : int
        Number of steps ahead to project variance (default is 1 day).
    """
    returns = np.array(daily_log_returns)
    n_obs = len(returns)
    
    # Pre-allocate array for our daily out-of-sample forecasts
    # Filled with NaN for historical periods within the initial lookback window
    forecasted_variance = np.full(n_obs, np.nan)
    
    print(f"📉 Initializing GJR-GARCH(1,1) rolling estimation loop across {n_obs} observations...")
    
    # We step day-by-day to simulate a rigorous out-of-sample production baseline
    for t in range(lookback_window, n_obs):
        # Extract the historical training subset matrix (Fixed rolling lookback)
        train_window = returns[t - lookback_window : t]
        
        try:
            # Enforce an explicit scale factor (*100) to optimize the underlying 
            # optimization functions and prevent numeric underflow faults.
            scaled_train = train_window * 100.0
            
            # Instantiating the asymmetric GJR-GARCH(1,1) model structure
            # p=1 (GARCH lag), o=1 (Asymmetric indicator dummy), q=1 (ARCH innovation lag)
            # dist='t' handles fat-tailed index distributions robustly
            model = arch_model(scaled_train, p=1, o=1, q=1, vol='GARCH', dist='t', mean='Constant')
            
            # fit(disp='off') hides tracking logs to protect compute memory loops
            fitted_model = model.fit(disp='off')
            
            # Run the forward projection forecast
            fc = fitted_model.forecast(horizon=forecast_horizon)
            
            # Extract the tomorrow variance value and descale it back to raw variance terms
            raw_var_forecast = fc.variance.iloc[-1, 0]
            forecasted_variance[t] = raw_var_forecast / 10000.0
            
        except Exception as e:
            # Fallback routine: if optimization parameters fail to converge, 
            # use a simple rolling standard deviation baseline to prevent pipeline halt
            daily_sample_variance = np.var(train_window, ddof=1)
            forecasted_variance[t] = daily_sample_variance
            
    return forecasted_variance


def extract_variance_risk_premium(aligned_data_df, physical_variance_vector):
    """
    Isolates the historical Variance Risk Premium matrix by calculating the structural 
    spread between implied options expectation (Q) and forecast physical drift (P).
    
    Parameters:
    -----------
    aligned_data_df : pd.DataFrame
        Dataframe containing your calibrated SVI implied variance parameters mapped by date.
    physical_variance_vector : np.array
        The output vector derived from calculate_rolling_gjr_garch.
    """
    output_matrix = aligned_data_df.copy()
    
    # Ensure critical implied metrics column from the SVI tier is available
    if 'svi_implied_variance' not in output_matrix.columns:
        raise KeyError("❌ Missing 'svi_implied_variance' column. Calibrate the SVI surface first.")
        
    output_matrix['physical_variance_forecast'] = physical_variance_vector
    
    # Calculate the continuous raw Variance Risk Premium spread
    output_matrix['vrp_spread'] = output_matrix['svi_implied_variance'] - output_matrix['physical_variance_forecast']
    
    # Identify alpha directional signals based on premium expansion regimes
    # 1 if option implied variance is higher than forecast physical variance, 0 otherwise
    output_matrix['is_premium_positive'] = (output_matrix['vrp_spread'] > 0).astype(int)
    
    # Drop lookback rows where the GJR forecast is unpopulated to ensure clean signal series
    output_matrix = output_matrix.dropna(subset=['vrp_spread']).reset_index(drop=True)
    
    print(f"✅ VRP premium extraction finalized. {len(output_matrix)} clean trading signals derived.")
    return output_matrix