"""
diag_test5.py - Test 5: VOI autocorrelation (lags 1-10)
Computes ACF of raw VOI within each session, then averages across sessions.
Higher ACF means more persistent order flow (less efficient market).
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "results_diagnostic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_2026 = "data/market_data.db"
DB_2018 = "data/market_data_2018.db"

COLS = ["InstruID", "TradDay", "ActionDateTime",
        "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"]

# ── VOI construction (raw, no normalisation) ─────────────────────────────────

def compute_VOI(df):
    bid_p, bid_v = df['BidPrice1'], df['BidVolume1']
    ask_p, ask_v = df['AskPrice1'], df['AskVolume1']
    d_bid = bid_p.diff();  d_ask = ask_p.diff()
    sub_b = bid_v.shift(1).where(d_bid == 0, other=0.0)
    dVB   = (bid_v - sub_b) * (d_bid >= 0)
    sub_a = ask_v.shift(1).where(d_ask == 0, other=0.0)
    dVA   = (ask_v - sub_a) * (d_ask <= 0)
    voi   = (dVB - dVA).astype(float)
    voi.iloc[0] = np.nan
    return voi

def session_acf(voi_series, max_lag=10):
    """Compute ACF for lags 1..max_lag on a single session's VOI."""
    v = voi_series.dropna().values
    if len(v) < max_lag + 10:
        return np.full(max_lag, np.nan)
    v = v - v.mean()
    var = np.var(v)
    if var == 0:
        return np.full(max_lag, np.nan)
    acf = np.array([np.mean(v[lag:] * v[:-lag]) / var for lag in range(1, max_lag+1)])
    return acf

# ── load and compute ──────────────────────────────────────────────────────────

def load_and_acf(db_path, label, max_lag=10):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        f"SELECT {', '.join(COLS)} FROM tick_data "
        "WHERE InstruID LIKE 'IF%' ORDER BY TradDay, ActionDateTime"
    ).df()
    con.close()

    try:
        df['ActionDateTime'] = pd.to_datetime(df['ActionDateTime'],
                                              format='%Y%m%d %H:%M:%S.%f')
    except Exception:
        df['ActionDateTime'] = pd.to_datetime(df['ActionDateTime'],
                                              format='%Y-%m-%d %H:%M:%S.%f')
    df['ts'] = (df['ActionDateTime'].dt.hour * 3600
                + df['ActionDateTime'].dt.minute * 60
                + df['ActionDateTime'].dt.second
                + df['ActionDateTime'].dt.microsecond / 1e6)
    am = (df['ts'] >= 33300) & (df['ts'] < 41280)
    pm = (df['ts'] >= 46800) & (df['ts'] < 54780)
    df = df[am | pm].reset_index(drop=True)
    df = df[(df['BidPrice1'] > 0) & (df['AskPrice1'] > 0)
            & (df['BidPrice1'] < df['AskPrice1'])].reset_index(drop=True)
    df['session'] = np.where(df['ts'] < 46800, 'morning', 'afternoon')

    # Use main contract (prev-day highest volume)
    daily_vol = df.groupby(['TradDay','InstruID'])['Volume' if 'Volume' in df.columns
                            else 'BidVolume1'].max().unstack('InstruID') \
                if 'Volume' not in df.columns else \
                df.groupby(['TradDay','InstruID'])['BidVolume1'].max().unstack('InstruID')

    # reload with volume for main contract selection
    con2 = duckdb.connect(db_path, read_only=True)
    vol_df = con2.execute(
        "SELECT InstruID, TradDay, Volume FROM tick_data "
        "WHERE InstruID LIKE 'IF%' ORDER BY TradDay, ActionDateTime"
    ).df()
    con2.close()
    dv = vol_df.groupby(['TradDay','InstruID'])['Volume'].max().unstack('InstruID')
    prev_vol = dv.shift(1)
    main_map = prev_vol.dropna(how='all').idxmax(axis=1).dropna()
    df = df[df['TradDay'].isin(main_map.index) &
            (df['InstruID'] == df['TradDay'].map(main_map))].reset_index(drop=True)
    print(f"  {label}: {len(df):,} rows, {df['TradDay'].nunique()} days")

    # Compute VOI per session
    df['VOI'] = df.groupby(['TradDay','session'], group_keys=False).apply(compute_VOI)

    # Compute ACF per session, then average
    all_acf = []
    for (day, sess), grp in df.groupby(['TradDay','session']):
        acf_vals = session_acf(grp['VOI'], max_lag=max_lag)
        if not np.isnan(acf_vals).all():
            all_acf.append(acf_vals)

    acf_arr = np.array(all_acf)
    mean_acf = np.nanmean(acf_arr, axis=0)
    std_acf  = np.nanstd(acf_arr, axis=0)
    n_sess   = acf_arr.shape[0]

    print(f"  {label}: {n_sess} sessions used for ACF")
    return mean_acf, std_acf, n_sess


MAX_LAG = 10
LAGS    = np.arange(1, MAX_LAG + 1)

print("Computing VOI ACF for 2026...")
acf26, std26, n26 = load_and_acf(DB_2026, "2026", max_lag=MAX_LAG)
print("Computing VOI ACF for 2018...")
acf18, std18, n18 = load_and_acf(DB_2018, "2018", max_lag=MAX_LAG)

# Bartlett 95% CI bound
ci26 = 1.96 / np.sqrt(n26 * 1000)   # rough sessions × ticks per session
ci18 = 1.96 / np.sqrt(n18 * 1000)

print()
print("=" * 65)
print("TEST 5: VOI Autocorrelation (raw, main contract, averaged across sessions)")
print("=" * 65)
print(f"  {'Lag':>5}  {'ACF 2026':>12}  {'ACF 2018':>12}  {'Ratio 18/26':>14}")
print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*14}")
for lag in LAGS:
    a26 = acf26[lag-1]
    a18 = acf18[lag-1]
    ratio = f"{a18/a26:.3f}" if (a26 != 0 and not np.isnan(a26)) else "N/A"
    print(f"  {lag:>5}  {a26:>12.5f}  {a18:>12.5f}  {ratio:>14}")

print()
print(f"  Mean ACF lags 1-5 : 2026={acf26[:5].mean():.5f}  2018={acf18[:5].mean():.5f}")
print(f"  Mean ACF lags 1-10: 2026={acf26.mean():.5f}  2018={acf18.mean():.5f}")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Test 5: VOI Autocorrelation (raw, main contract, per-session average)", fontsize=13)

ax = axes[0]
ax.bar(LAGS - 0.2, acf26, 0.35, label='2026', color='steelblue', alpha=0.8)
ax.bar(LAGS + 0.2, acf18, 0.35, label='2018', color='darkorange', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Lag (ticks)')
ax.set_ylabel('Autocorrelation')
ax.set_title('VOI ACF — side by side')
ax.set_xticks(LAGS)
ax.legend()

ax = axes[1]
ax.plot(LAGS, acf26, 'o-', color='steelblue', linewidth=2, markersize=7, label='2026')
ax.plot(LAGS, acf18, 's-', color='darkorange', linewidth=2, markersize=7, label='2018')
ax.fill_between(LAGS, acf26 - std26, acf26 + std26, alpha=0.15, color='steelblue')
ax.fill_between(LAGS, acf18 - std18, acf18 + std18, alpha=0.15, color='darkorange')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Lag (ticks)')
ax.set_ylabel('Autocorrelation (mean +/- 1 std across sessions)')
ax.set_title('VOI ACF with cross-session variability')
ax.set_xticks(LAGS)
ax.legend()

plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, "test5_voi_acf.png")
plt.savefig(outpath, dpi=150)
plt.close()
print(f"\nPlot saved: {outpath}")
print("Test 5 complete.")
