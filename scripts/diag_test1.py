"""
diag_test1.py - Test 1: Basic market structure comparison
Runs standalone; saves output to results_diagnostic/
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
        "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",
        "Volume", "Turnover"]

# ── data loading ────────────────────────────────────────────────────────────

def load_if_session(db_path):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        f"SELECT {', '.join(COLS)} FROM tick_data "
        "WHERE InstruID LIKE 'IF%' "
        "ORDER BY TradDay, ActionDateTime"
    ).df()
    con.close()

    # 2026 fmt: '20260202 09:29:00.400'   2018 fmt: '2018-01-02 09:15:00.500'
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
    return df


# ── compute metrics ──────────────────────────────────────────────────────────

def compute_metrics(df, label):
    df = df.copy()
    df['MidPrice']  = (df['BidPrice1'] + df['AskPrice1']) / 2
    df['Spread']    =  df['AskPrice1'] - df['BidPrice1']
    df['SpreadPct'] =  df['Spread'] / df['MidPrice'] * 100

    # within-group diffs to avoid cross-contract / cross-session artefacts
    grp = df.groupby(['TradDay', 'InstruID'])
    df['dMid'] = grp['MidPrice'].diff().abs()
    df['dVol'] = grp['Volume'].diff().clip(lower=0)

    sp = df['Spread']
    dm = df['dMid'].dropna()

    res = {
        'label':          label,
        'n_rows':         len(df),
        'n_days':         df['TradDay'].nunique(),
        'mean_spread':    float(sp.mean()),
        'median_spread':  float(sp.median()),
        'spread_pct':     float(df['SpreadPct'].mean()),
        'mean_dMid':      float(dm.mean()),
        'pct_dMid_zero':  float((dm == 0).mean() * 100),
        'mean_dVol':      float(df['dVol'].dropna().mean()),
        'sp_p10':  float(sp.quantile(0.10)),
        'sp_p25':  float(sp.quantile(0.25)),
        'sp_p50':  float(sp.quantile(0.50)),
        'sp_p75':  float(sp.quantile(0.75)),
        'sp_p90':  float(sp.quantile(0.90)),
        'sp_p99':  float(sp.quantile(0.99)),
    }
    return res, df


# ── run ──────────────────────────────────────────────────────────────────────

print("Loading 2026 IF data...")
df26 = load_if_session(DB_2026)
print(f"  {len(df26):,} rows, {df26['TradDay'].nunique()} days")

print("Loading 2018 IF data...")
df18 = load_if_session(DB_2018)
print(f"  {len(df18):,} rows, {df18['TradDay'].nunique()} days")

r26, df26 = compute_metrics(df26, "2026")
r18, df18 = compute_metrics(df18, "2018")


# ── print table ──────────────────────────────────────────────────────────────

rows = [
    ("Rows (all IF, in session)",   "n_rows",        "{:>14,.0f}", ""),
    ("Trading days",                "n_days",         "{:>14,.0f}", ""),
    ("Mean bid-ask spread",         "mean_spread",    "{:>14.4f}", "index pts"),
    ("Median spread",               "median_spread",  "{:>14.4f}", "index pts"),
    ("Spread as % of mid",          "spread_pct",     "{:>13.4f}%", ""),
    ("Mean |dMid| per tick",        "mean_dMid",      "{:>14.4f}", "index pts"),
    ("  (% of ticks with dMid=0)", "pct_dMid_zero",  "{:>13.1f}%", ""),
    ("Mean dVol per tick",          "mean_dVol",      "{:>14.2f}", "lots"),
    ("Spread p10",                  "sp_p10",         "{:>14.4f}", ""),
    ("Spread p25",                  "sp_p25",         "{:>14.4f}", ""),
    ("Spread p50",                  "sp_p50",         "{:>14.4f}", ""),
    ("Spread p75",                  "sp_p75",         "{:>14.4f}", ""),
    ("Spread p90",                  "sp_p90",         "{:>14.4f}", ""),
    ("Spread p99",                  "sp_p99",         "{:>14.4f}", ""),
]

print()
print("=" * 70)
print("TEST 1: Basic Market Structure Comparison (all IF contracts)")
print("=" * 70)
print(f"  {'Metric':<35}  {'2026':>12}  {'2018':>12}")
print(f"  {'-'*35}  {'-'*12}  {'-'*12}")
for name, key, fmt, unit in rows:
    v26 = fmt.format(r26[key])
    v18 = fmt.format(r18[key])
    suffix = f"  {unit}" if unit else ""
    print(f"  {name:<35}  {v26}  {v18}{suffix}")

print()

# ratio helper
def ratio(a, b):
    return f"{a/b:.2f}x" if b != 0 else "N/A"

print("  Interpretive ratios:")
print(f"    Spread 2026/2018     : {ratio(r26['mean_spread'], r18['mean_spread'])}")
print(f"    |dMid| 2026/2018     : {ratio(r26['mean_dMid'],   r18['mean_dMid'])}")
print(f"    dVol/tick 2026/2018  : {ratio(r26['mean_dVol'],   r18['mean_dVol'])}")


# ── plots ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Test 1: Market Structure — 2018 vs 2026 (all IF contracts)", fontsize=13)

# 1. Spread distribution
ax = axes[0]
cap = max(r26['sp_p99'], r18['sp_p99']) * 1.1
ax.hist(df26['Spread'].clip(upper=cap), bins=80, color='steelblue',
        alpha=0.65, density=True, label='2026')
ax.hist(df18['Spread'].clip(upper=cap), bins=80, color='darkorange',
        alpha=0.65, density=True, label='2018')
ax.axvline(r26['mean_spread'], color='steelblue', linewidth=2, linestyle='--')
ax.axvline(r18['mean_spread'], color='darkorange', linewidth=2, linestyle='--')
ax.set_xlabel('Bid-Ask Spread (index pts)')
ax.set_ylabel('Density')
ax.set_title('Spread Distribution')
ax.legend()

# 2. |dMid| distribution
ax = axes[1]
dm26 = df26['dMid'].dropna()
dm18 = df18['dMid'].dropna()
cap_d = max(float(dm26.quantile(0.99)), float(dm18.quantile(0.99))) * 1.1
ax.hist(dm26.clip(upper=cap_d), bins=80, color='steelblue',
        alpha=0.65, density=True, label='2026')
ax.hist(dm18.clip(upper=cap_d), bins=80, color='darkorange',
        alpha=0.65, density=True, label='2018')
ax.set_xlabel('|dMid| per tick (index pts)')
ax.set_ylabel('Density')
ax.set_title('|dMid| per tick Distribution')
ax.legend()

# 3. Spread percentile bar chart side-by-side
ax = axes[2]
pcts = ['p10','p25','p50','p75','p90','p99']
vals26 = [r26[f'sp_{p}'] for p in pcts]
vals18 = [r18[f'sp_{p}'] for p in pcts]
x = np.arange(len(pcts))
w = 0.35
ax.bar(x - w/2, vals26, w, label='2026', color='steelblue', alpha=0.8)
ax.bar(x + w/2, vals18, w, label='2018', color='darkorange', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(pcts)
ax.set_xlabel('Percentile')
ax.set_ylabel('Spread (index pts)')
ax.set_title('Spread Percentiles')
ax.legend()

plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, "test1_market_structure.png")
plt.savefig(outpath, dpi=150)
plt.close()
print(f"Plot saved: {outpath}")
print("Test 1 complete.")
