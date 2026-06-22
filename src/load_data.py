"""
load_data.py
============
Data loading utilities for CFFEX tick data research.
Import this module in Jupyter notebooks or analysis scripts.

Example usage:
    from load_data import load_day, load_days, query

    df = load_day(20260202)
    df_multi = load_days([20260202, 20260203])
    df_custom = query("SELECT * FROM tick_data WHERE TradDay = 20260202 LIMIT 100")
"""

import duckdb
import pandas as pd
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path("data/market_data.db")


@contextmanager
def _connect():
    """Internal: open and automatically close a read-only DuckDB connection."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        yield con
    finally:
        con.close()


def load_day(date: int, columns: list = None) -> pd.DataFrame:
    """
    Load all tick data for a single trading day, sorted by time.

    Parameters
    ----------
    date    : int  Trading day in YYYYMMDD format, e.g. 20260202.
    columns : list, optional  Column subset to load. If None, all columns are returned.
              e.g. columns=["BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"]

    Returns
    -------
    pd.DataFrame
    """
    cols = ", ".join(columns) if columns else "*"
    with _connect() as con:
        return con.execute(f"""
            SELECT {cols}
            FROM tick_data
            WHERE TradDay = {date}
            ORDER BY UpdateTime, UpdateMillisec
        """).df()


def load_days(dates: list, columns: list = None) -> pd.DataFrame:
    """
    Load and concatenate data for multiple trading days, sorted by date and time.

    Parameters
    ----------
    dates   : list  List of trading days, e.g. [20260202, 20260203, 20260204].
    columns : list, optional  Column subset to load.

    Returns
    -------
    pd.DataFrame
    """
    cols = ", ".join(columns) if columns else "*"
    dates_str = ", ".join(str(d) for d in dates)
    with _connect() as con:
        return con.execute(f"""
            SELECT {cols}
            FROM tick_data
            WHERE TradDay IN ({dates_str})
            ORDER BY TradDay, UpdateTime, UpdateMillisec
        """).df()


def load_date_range(start: int, end: int, columns: list = None) -> pd.DataFrame:
    """
    Load all tick data within a date range (inclusive on both ends).

    Parameters
    ----------
    start   : int  Start date in YYYYMMDD format, e.g. 20260202.
    end     : int  End date in YYYYMMDD format, e.g. 20260228.
    columns : list, optional  Column subset to load.

    Returns
    -------
    pd.DataFrame
    """
    cols = ", ".join(columns) if columns else "*"
    with _connect() as con:
        return con.execute(f"""
            SELECT {cols}
            FROM tick_data
            WHERE TradDay BETWEEN {start} AND {end}
            ORDER BY TradDay, UpdateTime, UpdateMillisec
        """).df()


def query(sql: str) -> pd.DataFrame:
    """
    Execute a raw SQL query and return the result as a DataFrame.
    Useful for ad-hoc exploratory analysis.

    Example:
        query("SELECT TradDay, COUNT(*) as rows FROM tick_data GROUP BY TradDay ORDER BY TradDay")
    """
    with _connect() as con:
        return con.execute(sql).df()


def get_trading_days() -> list:
    """Return a sorted list of all trading days present in the database."""
    with _connect() as con:
        result = con.execute(
            "SELECT DISTINCT TradDay FROM tick_data ORDER BY TradDay"
        ).fetchall()
    return [row[0] for row in result]


def get_instruments(date: int = None) -> list:
    """
    Return a list of instrument IDs.
    If date is provided, return instruments active on that day; otherwise return all.
    """
    where = f"WHERE TradDay = {date}" if date else ""
    with _connect() as con:
        result = con.execute(
            f"SELECT DISTINCT InstruID FROM tick_data {where} ORDER BY InstruID"
        ).fetchall()
    return [row[0] for row in result]


def db_summary():
    """Print a brief summary of the database contents."""
    with _connect() as con:
        row = con.execute("""
            SELECT
                COUNT(*)                 AS total_rows,
                COUNT(DISTINCT TradDay)  AS trading_days,
                COUNT(DISTINCT InstruID) AS instruments,
                MIN(TradDay)             AS first_day,
                MAX(TradDay)             AS last_day
            FROM tick_data
        """).fetchone()

    print("Database summary")
    print(f"   Total rows     : {row[0]:,}")
    print(f"   Trading days   : {row[1]}")
    print(f"   Instruments    : {row[2]}")
    print(f"   First day      : {row[3]}")
    print(f"   Last day       : {row[4]}")
    print(f"   DB file        : {DB_PATH} ({DB_PATH.stat().st_size/1e6:.1f} MB)")
