"""
diag_test4.py - Test 4: Signal decay across forecast horizons
For k = 5, 10, 20, 40, 60 steps:
  - Refit Strategy B with that k as forecast horizon
  - Run walk-forward backtest (paper cost, q=0.20)
  - Record annualized Sharpe for both periods
Plots Sharpe vs k side by side.
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

from backtest_engine import run_backtest, compute_performance_metrics

OUTPUT_DIR = "results_diagnostic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_2026 = "data/market_data.db"
DB_2018 = "data/market_data_2018.db"

COLS = ["InstruID", "TradDay", "ActionDateTime",
        "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",
        "Volume", "Turnover"]

# ── signal construction ───────────────────────────────────────────────────────

def compute_VOI(df):
    bid_p, bid_v = df['BidPrice1'], df['BidVolume1']
    ask_p, ask_v = df['AskPrice1'], df['AskVolume1']
    d_bid = bid_p.diff();  d_ask = ask_p.diff()
    sub_b = bid_v.shift(1).where(d_bid == 0, other=0.0)
    dVB   = (bid_v - sub_b) * (d_bid >= 0)
    sub_a = ask_v.shift(1).where(d_ask == 0, other=0.0)
    dVA   = (ask_v - sub_a) * (d_ask <= 0)
    voi   = (dVB - dVA).rename('VOI')
    voi.iloc[0] = np.nan
    return voi

def compute_MPB(df):
    mid  = df['MidPrice']
    d_vo = df['Volume'].diff()
    d_to = df['Turnover'].diff()
    avg  = (d_to / d_vo / 300).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    rm   = mid.rolling(2).mean().fillna(mid.iloc[0])
    return (avg - rm).rename('MPB')

def compute_response(df, k):
    mid = df['MidPrice']
    return (mid.rolling(k).mean().shift(-k) - mid).rename('y')

L_B = 5
FEAT_B = ([f'VOI_t{j}' for j in range(L_B+1)] +
          [f'OIR_t{j}' for j in range(L_B+1)] +
          ['MPB_norm'])

def build_features_B(df, k):
    n = len(df)
    sp  = df['Spread']
    voi = df['VOI']
    oir = df['OIR']
    X_v = pd.DataFrame({f'VOI_t{j}': voi.shift(j)/sp for j in range(L_B+1)}, index=df.index)
    X_o = pd.DataFrame({f'OIR_t{j}': oir.shift(j)/sp for j in range(L_B+1)}, index=df.index)
    X_m = (df['MPB']/sp).rename('MPB_norm')
    y   = compute_response(df, k=k)
    out = pd.concat([X_v, X_o, X_m, y], axis=1).iloc[L_B+1 : n-k]
    assert out.isna().sum().sum() == 0
    return out.copy()

def load_raw(db_path, label):
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
    df['MidPrice'] = (df['BidPrice1'] + df['AskPrice1']) / 2
    df['Spread']   =  df['AskPrice1'] - df['BidPrice1']
    df['session']  = np.where(df['ts'] < 46800, 'morning', 'afternoon')
    daily_vol = df.groupby(['TradDay','InstruID'])['Volume'].max().unstack('InstruID')
    prev_vol  = daily_vol.shift(1)
    main_map  = prev_vol.dropna(how='all').idxmax(axis=1).dropna()
    df = df[df['TradDay'].isin(main_map.index) &
            (df['InstruID'] == df['TradDay'].map(main_map))].reset_index(drop=True)
    print(f"  {label}: {len(df):,} rows, {df['TradDay'].nunique()} days")
    df['VOI'] = df.groupby(['TradDay','session'], group_keys=False).apply(compute_VOI)
    df['OIR'] = (df['BidVolume1'] - df['AskVolume1']) / (df['BidVolume1'] + df['AskVolume1'])
    df['MPB'] = df.groupby(['TradDay','session'], group_keys=False).apply(compute_MPB)
    return df

def run_at_k(df_raw, k, label):
    feats = (
        df_raw.groupby(['TradDay','session'], group_keys=False)
        .apply(lambda g: build_features_B(g, k=k))
        .join(df_raw[['TradDay','session','BidPrice1','AskPrice1','MidPrice','ts']])
        .reset_index(drop=True)
    )
    bt = run_backtest(
        features_df=feats, feature_cols=FEAT_B, target_col='y',
        day_col='TradDay', session_col='session',
        bid_col='BidPrice1', ask_col='AskPrice1', time_col='ts',
        threshold=0.20, tr_cost=2.5e-5, contract_multiplier=300,
        trading_hours={
            'morning':   {'open': 33960, 'close': 40800, 'end': 41280},
            'afternoon': {'open': 46860, 'close': 54000, 'end': 54780},
        },
    )
    m = compute_performance_metrics(bt, verbose=False, output_dir=OUTPUT_DIR)
    return m['annualized_sharpe'], m['mean_daily_pnl']


# ── run ───────────────────────────────────────────────────────────────────────

K_LIST = [5, 10, 20, 40, 60]

print("Loading raw data for 2026...")
raw26 = load_raw(DB_2026, "2026")
print("Loading raw data for 2018...")
raw18 = load_raw(DB_2018, "2018")

records = []
for k in K_LIST:
    print(f"\n--- k={k} ---")
    sh26, pnl26 = run_at_k(raw26, k, "2026")
    print(f"  2026: Sharpe={sh26:.3f}  PnL={pnl26:.1f}")
    sh18, pnl18 = run_at_k(raw18, k, "2018")
    print(f"  2018: Sharpe={sh18:.3f}  PnL={pnl18:.1f}")
    records.append({'k': k, 'sharpe_2026': sh26, 'sharpe_2018': sh18,
                    'pnl_2026': pnl26, 'pnl_2018': pnl18})

res = pd.DataFrame(records)

print()
print("=" * 65)
print("TEST 4: Signal Decay — Annualized Sharpe vs Forecast Horizon k")
print("=" * 65)
print(f"  {'k':>5}  {'Sharpe 2026':>14}  {'Sharpe 2018':>14}  {'PnL 2026':>12}  {'PnL 2018':>12}")
print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}")
for _, row in res.iterrows():
    print(f"  {int(row['k']):>5}  {row['sharpe_2026']:>14.3f}  {row['sharpe_2018']:>14.3f}"
          f"  {row['pnl_2026']:>12.1f}  {row['pnl_2018']:>12.1f}")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Test 4: Signal Decay — Strategy B, q=0.20, paper cost", fontsize=13)

ax = axes[0]
ax.plot(res['k'], res['sharpe_2026'], 'o-', color='steelblue',
        linewidth=2, markersize=8, label='2026')
ax.plot(res['k'], res['sharpe_2018'], 's-', color='darkorange',
        linewidth=2, markersize=8, label='2018')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Forecast horizon k (ticks, ~0.5s each)')
ax.set_ylabel('Annualized Sharpe')
ax.set_title('Sharpe vs k')
ax.legend()
ax.set_xticks(K_LIST)

ax = axes[1]
ax.plot(res['k'], res['pnl_2026'] / 1e3, 'o-', color='steelblue',
        linewidth=2, markersize=8, label='2026')
ax.plot(res['k'], res['pnl_2018'] / 1e3, 's-', color='darkorange',
        linewidth=2, markersize=8, label='2018')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Forecast horizon k (ticks, ~0.5s each)')
ax.set_ylabel('Mean daily PnL (k CNY)')
ax.set_title('Mean daily PnL vs k')
ax.legend()
ax.set_xticks(K_LIST)

plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, "test4_signal_decay.png")
plt.savefig(outpath, dpi=150)
plt.close()
print(f"\nPlot saved: {outpath}")
print("Test 4 complete.")
