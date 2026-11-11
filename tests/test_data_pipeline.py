"""
Unified Validation Suite: Data Ingestion and Alignment Pipelines.

Provides automated test assertions to verify file ingestion constraints, 
data type schema compliance, and order book filtering sanity.
"""

import io
import os
import unittest
import importlib.util
import numpy as np
import pandas as pd

# Dynamic path resolution to bypass Python numeric folder naming constraints
def import_from_numeric_dir(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(module)
    return module

# Resolve module hooks cleanly using relative script locations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_loader_mod = import_from_numeric_dir("data_loader", os.path.join(BASE_DIR, "1_data_processing", "data_loader.py"))
price_extractor_mod = import_from_numeric_dir("underlying_price_extractor", os.path.join(BASE_DIR, "1_data_processing", "underlying_price_extractor.py"))

load_and_clean_options_data = data_loader_mod.load_and_clean_options_data
extract_and_align_prices = price_extractor_mod.extract_and_align_prices


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        """
        Constructs mock csv data files in memory to isolate pipeline modules 
        from disk file dependencies during testing.
        """
        self.mock_options_csv = (
            "underlying_symbol,quote_date,expiration,strike,option_type,bid,ask,underlying_price\n"
            "SPY,2026-10-20,2026-11-20,450.0,C,5.20,5.40,450.10\n"
            "SPY,2026-10-20,2026-11-20,450.0,P,4.80,4.95,450.10\n"
            "SPY,2026-10-21,2026-11-20,452.0,C,3.10,3.25,451.85\n"
            "SPY,2026-10-21,2026-11-20,452.0,P,0.00,4.10,451.85\n"       # Toxic Row: Zero Bid -> Dropped
            "SPY,2026-10-22,2026-11-20,455.0,C,6.50,5.10,453.20\n"       # Toxic Row: Inverted/Crossed -> Dropped
        )
        
        self.mock_underlying_csv = (
            "Date,Close\n"
            "2026-10-19,449.00\n" 
            "2026-10-20,450.10\n"
            "2026-10-21,451.85\n"
            "2026-10-22,453.20\n"
            "2026-10-23,456.00\n" 
        )

    def test_data_loader_quote_filtration(self):
        """
        Asserts that load_and_clean_options_data strips out zero bids, 
        flags crossed order books, and accurately derives option mid-quotes.
        """
        temp_filepath = "test_options_temp.csv"
        with open(temp_filepath, "w") as f:
            f.write(self.mock_options_csv)
            
        try:
            result_df = load_and_clean_options_data(temp_filepath, target_symbol="SPY", chunk_size=10)
            
            # Exactly 3 clean rows should survive (Rows 1, 2, and 3)
            self.assertEqual(len(result_df), 3)
            
            # Verify mid-quote mathematical precision on the first clean row
            # Bid = 5.20, Ask = 5.40 -> Mid = 5.30
            self.assertAlmostEqual(result_df.loc[0, "mid_price"], 5.30, places=4)
            self.assertAlmostEqual(result_df.loc[0, "bid_ask_spread"], 0.20, places=4)
            
            # Verify explicit schema data type enforcement rules
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(result_df["quote_date"]))
            self.assertTrue(pd.api.types.is_float_dtype(result_df["strike"]))
            
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    def test_asymmetric_date_alignment_guard(self):
        """
        Asserts that extract_and_align_prices cleanly handles the 
        inner-join gate and removes non-overlapping trailing timeline rows.
        """
        temp_options_path = "test_opt_align.csv"
        temp_spot_path = "test_spot_align.csv"
        
        with open(temp_options_path, "w") as f: f.write(self.mock_options_csv)
        with open(temp_spot_path, "w") as f: f.write(self.mock_underlying_csv)
        
        try:
            clean_options = load_and_clean_options_data(temp_options_path, target_symbol="SPY")
            aligned_spot = extract_and_align_prices(temp_spot_path, clean_options)
            
            # Verify that the asynchronous entry (Oct 23rd) was successfully dropped via inner join
            self.assertNotIn(pd.to_datetime("2026-10-23"), aligned_spot["date"].values)
            self.assertFalse(aligned_spot["log_return"].isnull().any())
            
        finally:
            if os.path.exists(temp_options_path): os.remove(temp_options_path)
            if os.path.exists(temp_spot_path): os.remove(temp_spot_path)


if __name__ == "__main__":
    unittest.main()