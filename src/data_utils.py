"""
data_utils.py
-------------
Shared data loading and cleaning pipeline for CFFEX IF futures research.
All notebooks should call load_clean_main_contract() instead of
replicating the inline pipeline.
"""

import pathlib
import numpy as np
import pandas as pd
import load_data
from load_data import load_date_range


# ---------------------------------------------------------------------------
# Session boundary constants (seconds since midnight)
# These match the CFFEX IF continuous trading windows used throughout the
# project. Do not change without updating all notebooks.
# ---------------------------------------------------------------------------
MORNING_START   = 33300   # 09:15:00
MORNING_END     = 41280   # 11:28:00
AFTERNOON_START = 46800   # 13:00:00
AFTERNOON_END   = 54780   # 15:13:00


def load_clean_main_contract(
    start_date: int,
    end_date: int,
    columns: list[str],
    db_path: "pathlib.Path | str | None" = None,
) -> pd.DataFrame:
    """
    Load raw CTP snapshot data and return a cleaned, session-labelled
    DataFrame containing only the most-liquid IF contract per day.

    Steps applied in order
    ----------------------
    1. Load raw data via load_date_range().
    2. Parse ActionDateTime; derive time_seconds (float seconds since midnight).
    3. Filter to IF contracts only.
    4. Filter to morning (09:15–11:28) and afternoon (13:00–15:13) sessions.
    5. Drop rows with crossed best quotes (Bid1 >= Ask1) or zero best quotes.
    6. Select the main contract per day using the previous-day highest
       cumulative volume rule (no look-ahead bias).
    7. Add derived columns: MidPrice, Spread, session ('morning'/'afternoon').

    Parameters
    ----------
    start_date : int
        First trading day to load, inclusive, in YYYYMMDD format.
    end_date : int
        Last trading day to load, inclusive, in YYYYMMDD format.
    columns : list[str]
        Columns to pass to load_date_range(). Must include at minimum:
        'InstruID', 'TradDay', 'ActionDateTime', 'BidPrice1', 'AskPrice1',
        'BidVolume1', 'AskVolume1', 'Volume'.
        Include additional columns (e.g. all 5 LOB levels, Turnover,
        OpenInt) as required by the calling notebook.
    db_path : pathlib.Path or str, optional
        Override load_data.DB_PATH before loading. If None, the caller
        is responsible for setting load_data.DB_PATH before calling this
        function.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with the following additional columns added:
        - time_seconds  : float, seconds since midnight
        - session       : str, 'morning' or 'afternoon'
        - MidPrice      : float, (Bid1 + Ask1) / 2
        - Spread        : float, Ask1 - Bid1
        Index is reset (0-based integer).
    """
    if db_path is not None:
        load_data.DB_PATH = pathlib.Path(db_path)

    # 1. Load
    df = load_date_range(start_date, end_date, columns=columns)

    # 2. Parse datetime
    df['ActionDateTime'] = pd.to_datetime(
        df['ActionDateTime'], format='mixed'
    )
    df['time_seconds'] = (
        df['ActionDateTime'].dt.hour        * 3600
        + df['ActionDateTime'].dt.minute    * 60
        + df['ActionDateTime'].dt.second
        + df['ActionDateTime'].dt.microsecond / 1e6
    )

    # 3. IF contracts only
    df_clean = df[df['InstruID'].str.startswith('IF')].reset_index(drop=True)

    # 4. Session windows
    morning_mask   = (df_clean['time_seconds'] >= MORNING_START)   & (df_clean['time_seconds'] < MORNING_END)
    afternoon_mask = (df_clean['time_seconds'] >= AFTERNOON_START)  & (df_clean['time_seconds'] < AFTERNOON_END)
    df_clean = df_clean[morning_mask | afternoon_mask].reset_index(drop=True)

    # 5. Crossed / zero best quotes
    crossed  = df_clean['BidPrice1'] >= df_clean['AskPrice1']
    zero_bid = df_clean['BidPrice1'] == 0
    zero_ask = df_clean['AskPrice1'] == 0
    df_clean = df_clean[~crossed & ~zero_bid & ~zero_ask].reset_index(drop=True)

    # 6. Main contract selection: previous-day highest cumulative volume
    #    CTP Volume is cumulative intraday; daily total = max(Volume) per contract per day.
    daily_vol = (
        df_clean.groupby(['TradDay', 'InstruID'])['Volume']
        .max()
        .unstack('InstruID')
    )
    prev_day_vol = daily_vol.shift(1)
    main_contract_by_day = (
        prev_day_vol[prev_day_vol.notna().any(axis=1)]
        .idxmax(axis=1)
        .dropna()
        .rename('main_contract')
    )
    df_main = df_clean[
        df_clean['TradDay'].isin(main_contract_by_day.index)
        & (df_clean['InstruID'] == df_clean['TradDay'].map(main_contract_by_day))
    ].reset_index(drop=True)

    # 7. Derived columns
    df_main['MidPrice'] = (df_main['BidPrice1'] + df_main['AskPrice1']) / 2
    df_main['Spread']   =  df_main['AskPrice1'] - df_main['BidPrice1']
    df_main['session']  = np.where(
        df_main['time_seconds'] < AFTERNOON_START, 'morning', 'afternoon'
    )

    return df_main
