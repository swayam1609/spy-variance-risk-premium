"""
OptionsDX SPY Data Loader and Quote Cleaner
===========================================

Converts the OptionsDX wide call/put layout into a standardized long table.

Input columns used:
    QUOTE_DATE, EXPIRE_DATE, UNDERLYING_LAST, STRIKE,
    C_BID, C_ASK, P_BID, P_ASK

The source file in this project is already SPY-specific, so a ticker column
is not required. If a symbol column exists, target_symbol is applied.

Output columns include:
    quote_date, expiration, underlying_price, strike, option_type,
    bid, ask, mid_price, bid_ask_spread

Call and put quotes become separate rows. The loader reads the source in
chunks, but the returned DataFrame is ultimately held in memory because the
current research pipeline expects a complete DataFrame.
"""

import os
import pandas as pd


def _normalise_column_names(columns):
    """Remove optional brackets/whitespace and normalize to uppercase."""
    return [
        str(column).strip().strip("[]").strip().upper()
        for column in columns
    ]


def _numeric(series):
    """Convert a column to numeric; malformed values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def _make_option_side(chunk, side, target_symbol):
    """
    Convert one side (C or P) of the OptionsDX row into option-level records.
    Returns an empty DataFrame if required source columns are unavailable.
    """
    prefix = f"{side}_"
    required = [
        "QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE",
        f"{prefix}BID", f"{prefix}ASK"
    ]
    missing = [column for column in required if column not in chunk.columns]
    if missing:
        raise KeyError(
            "OptionsDX source is missing required columns: "
            f"{missing}. Available columns: {list(chunk.columns)}"
        )

    # Filter by ticker only when the source actually provides a ticker field.
    symbol_column = None
    for candidate in ("UNDERLYING_SYMBOL", "SYMBOL", "ROOT"):
        if candidate in chunk.columns:
            symbol_column = candidate
            break

    working = chunk
    if symbol_column is not None:
        working = working[
            working[symbol_column].astype(str).str.upper() == target_symbol.upper()
        ]

    if working.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=working.index)
    result["quote_date"] = pd.to_datetime(
        working["QUOTE_DATE"], errors="coerce"
    )
    result["expiration"] = pd.to_datetime(
        working["EXPIRE_DATE"], errors="coerce"
    )
    result["underlying_price"] = _numeric(working["UNDERLYING_LAST"])
    result["strike"] = _numeric(working["STRIKE"])
    result["option_type"] = "call" if side == "C" else "put"
    result["bid"] = _numeric(working[f"{prefix}BID"])
    result["ask"] = _numeric(working[f"{prefix}ASK"])

    # Retain useful side-specific fields when present, for downstream
    # diagnostics or strategy modules that may use observed Greeks/IV.
    optional_map = {
        "delta": f"{prefix}DELTA",
        "gamma": f"{prefix}GAMMA",
        "vega": f"{prefix}VEGA",
        "theta": f"{prefix}THETA",
        "rho": f"{prefix}RHO",
        "implied_volatility": f"{prefix}IV",
        "volume": f"{prefix}VOLUME",
        "last_price": f"{prefix}LAST",
        "quote_size": f"{prefix}SIZE",
    }
    for output_name, source_name in optional_map.items():
        if source_name in working.columns:
            result[output_name] = _numeric(working[source_name])

    # Remove malformed records and invalid markets. A zero bid is excluded
    # because this loader's downstream mid-price strategy requires a two-sided
    # positive quote; this is a research filter, not a claim that zero bids
    # are universally invalid market data.
    valid = (
        result["quote_date"].notna()
        & result["expiration"].notna()
        & result["underlying_price"].gt(0)
        & result["strike"].gt(0)
        & result["bid"].gt(0)
        & result["ask"].gt(0)
        & result["ask"].ge(result["bid"])
    )
    result = result.loc[valid].copy()

    if result.empty:
        return result

    result["mid_price"] = (result["bid"] + result["ask"]) / 2.0
    result["bid_ask_spread"] = result["ask"] - result["bid"]

    return result.reset_index(drop=True)


def load_and_clean_options_data(
    file_path,
    target_symbol="SPY",
    chunk_size=100000
):
    """
    Read a large OptionsDX options-chain CSV in chunks and return a cleaned
    long-format DataFrame containing separate call and put observations.

    Parameters
    ----------
    file_path : str
        Path to the combined OptionsDX CSV.
    target_symbol : str
        Underlying ticker to retain if the CSV has a ticker column.
        The project's SPY-only OptionsDX file may not contain such a column.
    chunk_size : int
        Number of source rows read per chunk.

    Returns
    -------
    pandas.DataFrame
        Standardized option-level observations.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Target options data file not found at: {file_path}"
        )

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    cleaned_chunks = []
    print(f"Beginning chunked ingestion of options file: {file_path}")

    total_source_rows = 0
    total_call_rows = 0
    total_put_rows = 0

    reader = pd.read_csv(
        file_path,
        chunksize=chunk_size,
        low_memory=False
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_source_rows += len(chunk)

        # Normalize headers, including optional OptionsDX square brackets.
        chunk.columns = _normalise_column_names(chunk.columns)

        call_records = _make_option_side(chunk, "C", target_symbol)
        put_records = _make_option_side(chunk, "P", target_symbol)

        total_call_rows += len(call_records)
        total_put_rows += len(put_records)

        if not call_records.empty:
            cleaned_chunks.append(call_records)
        if not put_records.empty:
            cleaned_chunks.append(put_records)

        if chunk_number % 10 == 0:
            print(
                f"Processed {total_source_rows:,} source rows; "
                f"retained {total_call_rows + total_put_rows:,} option records."
            )

    if not cleaned_chunks:
        print("Ingestion warning: no rows survived quote validation.")
        return pd.DataFrame(
            columns=[
                "quote_date", "expiration", "underlying_price", "strike",
                "option_type", "bid", "ask", "mid_price", "bid_ask_spread"
            ]
        )

    final_options_matrix = pd.concat(cleaned_chunks, ignore_index=True)

    # Keep the output order deterministic for later date/contract operations.
    final_options_matrix.sort_values(
        by=["quote_date", "expiration", "strike", "option_type"],
        inplace=True,
        kind="mergesort"
    )
    final_options_matrix.reset_index(drop=True, inplace=True)

    print("OptionsDX ingestion completed.")
    print(f"Source rows read : {total_source_rows:,}")
    print(f"Call records kept: {total_call_rows:,}")
    print(f"Put records kept : {total_put_rows:,}")
    print(f"Total clean rows : {len(final_options_matrix):,}")

    return final_options_matrix
