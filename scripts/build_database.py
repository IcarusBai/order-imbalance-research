"""
build_database.py
=================
Import raw CFFEX tick data (.tar / .tar.gz / .tar.bz2) into a local DuckDB database.

Usage
-----
Import all available years (outputs data/market_data.db):
    python build_database.py

Import a specific year only (outputs data/market_data_{YEAR}.db):
    python build_database.py --year 2018
    python build_database.py --year 2026

Run once per database. All subsequent analysis reads from the .db file.
"""

import argparse
import io
import tarfile
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("raw_data")

DTYPES = {
    "InstruID":         "str",
    "ActionDateTime":   "str",
    "TradDay":          "int32",
    "ActionDay":        "int32",
    "UpdateTime":       "str",
    "UpdateMillisec":   "int16",
    "LastPrice":        "float32",
    "PreSetPrice":      "float32",
    "PreCloPrice":      "float32",
    "PreOpenInt":       "float32",
    "OpenPrice":        "float32",
    "HighPrice":        "float32",
    "LowPrice":         "float32",
    "Volume":           "int32",
    "Turnover":         "float64",
    "OpenInt":          "float32",
    "ClosePrice":       "float32",
    "SetPrice":         "float32",
    "ULimitPrice":      "float32",
    "LLimitPrice":      "float32",
    "PreDelta":         "float32",
    "CurrDelta":        "float32",
    "BidPrice1":        "float32",  "BidVolume1":  "int32",
    "BidPrice2":        "float32",  "BidVolume2":  "int32",
    "BidPrice3":        "float32",  "BidVolume3":  "int32",
    "BidPrice4":        "float32",  "BidVolume4":  "int32",
    "BidPrice5":        "float32",  "BidVolume5":  "int32",
    "AskPrice1":        "float32",  "AskVolume1":  "int32",
    "AskPrice2":        "float32",  "AskVolume2":  "int32",
    "AskPrice3":        "float32",  "AskVolume3":  "int32",
    "AskPrice4":        "float32",  "AskVolume4":  "int32",
    "AskPrice5":        "float32",  "AskVolume5":  "int32",
    "AveragePrice":     "float64",
    "RecvTime":         "str",
}


def find_tar_files(year: str = None) -> list[Path]:
    """
    Return a sorted list of tar files under RAW_DIR.
    If year is given, only include subdirectories whose name starts with that year.
    """
    if year:
        files = []
        for subdir in sorted(RAW_DIR.iterdir()):
            if subdir.is_dir() and subdir.name.startswith(year):
                files += sorted(subdir.rglob("*.tar"))
                files += sorted(subdir.rglob("*.tar.gz"))
                files += sorted(subdir.rglob("*.tar.bz2"))
        return files
    return (
        sorted(RAW_DIR.rglob("*.tar"))
        + sorted(RAW_DIR.rglob("*.tar.gz"))
        + sorted(RAW_DIR.rglob("*.tar.bz2"))
    )


def read_csv_from_tar(tar_path: Path) -> pd.DataFrame:
    """Read all CSV files inside a tar archive and return them concatenated."""
    all_dfs = []
    with tarfile.open(tar_path, "r:*") as tar:
        csv_members = [m for m in tar.getmembers() if m.name.endswith(".csv")]
        if not csv_members:
            print(f"  [WARN] no CSV found in {tar_path.name}, skipping")
            return pd.DataFrame()
        for member in csv_members:
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                all_dfs.append(pd.read_csv(io.BytesIO(f.read()), dtype=DTYPES))
            except Exception as e:
                print(f"    [WARN] failed to read {member.name}: {e}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def setup_database(con: duckdb.DuckDBPyConnection, sample_df: pd.DataFrame):
    """Create the tick_data table from a sample DataFrame (runs only on first import)."""
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "tick_data" not in tables:
        con.from_df(sample_df).limit(0).create("tick_data")


def get_already_imported(con: duckdb.DuckDBPyConnection) -> set:
    """Return the set of TradDay values already in the database."""
    try:
        return {row[0] for row in con.execute("SELECT DISTINCT TradDay FROM tick_data").fetchall()}
    except Exception:
        return set()


def import_tar(tar_path: Path, con: duckdb.DuckDBPyConnection, imported_days: set) -> int:
    """Insert one tar file's data into the database. Returns the number of rows inserted."""
    df = read_csv_from_tar(tar_path)
    if df.empty:
        return 0
    already = set(df["TradDay"].unique()) & imported_days
    if already:
        print(f"  [SKIP] {tar_path.name} already imported (TradDay={already})")
        return 0
    con.execute("INSERT INTO tick_data SELECT * FROM df")
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Import CFFEX tick data into DuckDB.")
    parser.add_argument(
        "--year", type=str, default=None,
        help="Import only subdirectories starting with this year (e.g. 2018). "
             "Omit to import all available data."
    )
    args = parser.parse_args()

    db_path = Path(f"data/market_data_{args.year}.db") if args.year else Path("data/market_data.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    tar_files = find_tar_files(args.year)
    if not tar_files:
        print(f"[ERR] no tar files found under {RAW_DIR}" + (f" for year {args.year}" if args.year else ""))
        return

    print(f"Found {len(tar_files)} tar file(s)")
    print(f"Database: {db_path}\n")

    con = duckdb.connect(str(db_path))

    first_df = None
    for p in tar_files:
        first_df = read_csv_from_tar(p)
        if not first_df.empty:
            break
    if first_df is None or first_df.empty:
        print("[ERR] could not read any CSV, aborting")
        con.close()
        return

    setup_database(con, first_df)
    imported_days = get_already_imported(con)
    if imported_days:
        print(f"{len(imported_days)} trading day(s) already in DB — duplicates will be skipped\n")

    total_rows = 0
    for tar_path in tqdm(tar_files, desc="Importing"):
        try:
            rows = import_tar(tar_path, con, imported_days)
            if rows > 0:
                print(f"  [OK] {tar_path.name}: {rows:,} rows inserted")
                total_rows += rows
        except Exception as e:
            print(f"  [ERR] {tar_path.name} failed: {e}")

    print("\nBuilding indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tradday ON tick_data(TradDay)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_instru_day ON tick_data(InstruID, TradDay)")

    summary = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT TradDay), COUNT(DISTINCT InstruID),
               MIN(TradDay), MAX(TradDay)
        FROM tick_data
    """).fetchone()

    print("\n" + "=" * 50)
    print("Done. Database summary:")
    print(f"   Total rows     : {summary[0]:,}")
    print(f"   Trading days   : {summary[1]}")
    print(f"   Instruments    : {summary[2]}")
    print(f"   First day      : {summary[3]}")
    print(f"   Last day       : {summary[4]}")
    print(f"   Rows added     : {total_rows:,}")
    print(f"   File size      : {db_path.stat().st_size / 1e6:.1f} MB")
    print("=" * 50)

    con.close()


if __name__ == "__main__":
    main()
