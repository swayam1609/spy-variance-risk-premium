"""
Production Underlying Price Alignment and Return Extraction Module.

Extracts historical underlying equity prices, calculates synchronized log returns,
and enforces explicit date intersections to prevent index alignment shift bias.
"""

import os
import numpy as np
import pandas as pd


def extract_and_align_prices(underlying_path, clean_options_df):
    """
    Extracts raw historical underlying prices, calculates daily continuously 
    compounded log returns, and inner-joins with options dates to remove 
    asymmetric holiday gap edge cases.
    
    Parameters:
    -----------
    underlying_path : str
        Path to the CSV containing historical SPY spot index closing prices.
    clean_options_df : pd.DataFrame
        The output matrix from data_loader.py containing filtered option records.
    """
    if not os.path.exists(underlying_path):
        raise FileNotFoundError(f"❌ Underlying price file not found at: {underlying_path}")
        
    print(f"📥 Loading raw underlying price feed from: {underlying_path}")
    spot_df = pd.read_csv(underlying_path)
    spot_df.columns = [col.lower().strip() for col in spot_df.columns]
    
    if 'date' not in spot_df.columns or 'close' not in spot_df.columns:
        raise KeyError("❌ Underlying CSV must contain explicit 'Date' and 'Close' columns.")
        
    spot_df['date'] = pd.to_datetime(spot_df['date'])
    spot_df['close'] = spot_df['close'].astype(float)
    
    # Enforce strict chronological order before return calculations
    spot_df = spot_df.sort_values('date').reset_index(drop=True)
    
    # 2. Asymmetric Leverage Return Calculation
    # use log returns rather than simple returns because log returns are additive 
    # over time and are the mandatory standard input for GJR-GARCH econometrics.
    spot_df['log_return'] = np.log(spot_df['close'] / spot_df['close'].shift(1))
    spot_df['is_negative_shock'] = (spot_df['log_return'] < 0).astype(int)
    spot_df = spot_df.dropna(subset=['log_return']).reset_index(drop=True)
    
    # 3. Synchronous Temporal Date Alignment
    # Extract unique, pristine trading dates present in the processed options file
    unique_options_dates = pd.DataFrame({'date': clean_options_df['quote_date'].unique()})
    unique_options_dates['date'] = pd.to_datetime(unique_options_dates['date'])
    
    print(f"🔄 Executing defensive temporal alignment across datasets...")
    print(f"   - Raw underlying dates: {len(spot_df)}")
    print(f"   - Options chain dates:  {len(unique_options_dates)}")
    
    # The Inner-Join Guard: Drops asymmetric edge cases (e.g. the October 23rd Ghost Row)
    aligned_underlying = pd.merge(spot_df, unique_options_dates, on='date', how='inner')
    aligned_underlying = aligned_underlying.sort_values('date').reset_index(drop=True)
    
    print(f"✅ Alignment locked. {len(aligned_underlying)} common trading dates verified.")
    
    return aligned_underlying