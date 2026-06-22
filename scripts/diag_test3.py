"""
diag_test3.py - Test 3: Profit decomposition
Runs the actual Strategy B backtest (paper cost, q=0.20) for both periods.
Decomposes the per-trade expected profit gap into:
  win_rate, avg_win, avg_loss contributions.
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

from backtest_engine import (fit_ols_model, compute_signals,
                              run_backtest, compute_performance_metrics)

OUTPUT_DIR = "results_diagnostic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_2026 = "data/market_data.db"
DB_2018 = "data/market_data_2018.db"

COLS = ["InstruID", "TradDay", "ActionDateTime",
        "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",
        "Volume", "Turnover"]

# ── re-use signal construction from test2 ────────────────────────────────────

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

def compute_response(df, k=5):
    mid = df['MidPrice']
    return (mid.rolling(k).mean().shift(-k) - mid).rename('resp')

L_B, k_B = 5, 5
FEAT_B = ([f'VOI_t{j}' for j in range(L_B+1)] +
          [f'OIR_t{j}' for j in range(L_B+1)] +
          ['MPB_norm'])

def build_features_B(df, L=L_B, k=k_B):
    n = len(df)
    sp  = df['Spread']
    voi = df['VOI']
    oir = df['OIR']
    X_v = pd.DataFrame({f'VOI_t{j}': voi.shift(j)/sp for j in range(L+1)}, index=df.index)
    X_o = pd.DataFrame({f'OIR_t{j}': oir.shift(j)/sp for j in range(L+1)}, index=df.index)
    X_m = (df['MPB']/sp).rename('MPB_norm')
    y   = compute_response(df, k=k).rename('y')
    out = pd.concat([X_v, X_o, X_m, y], axis=1).iloc[L+1 : n-k]
    assert out.isna().sum().sum() == 0
    return out.copy()

def load_and_build(db_path, label):
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
    df['dMid_response'] = df.groupby(['TradDay','session'], group_keys=False).apply(
        lambda g: compute_response(g, k=k_B))
    feats = (
        df.groupby(['TradDay','session'], group_keys=False)
        .apply(build_features_B)
        .join(df[['TradDay','session','BidPrice1','AskPrice1','MidPrice','ts']])
        .reset_index(drop=True)
    )
    return feats


# ── run backtests ─────────────────────────────────────────────────────────────

TR_COST  = 2.5e-5
MULT     = 300
Q        = 0.20
HOURS = {
    'morning':   {'open': 33960, 'close': 40800, 'end': 41280},
    'afternoon': {'open': 46860, 'close': 54000, 'end': 54780},
}

COMMON = dict(
    feature_cols=FEAT_B, target_col='y', day_col='TradDay',
    session_col='session', bid_col='BidPrice1', ask_col='AskPrice1',
    time_col='ts', threshold=Q, tr_cost=TR_COST,
    contract_multiplier=MULT, trading_hours=HOURS,
)

print("Building features for 2026...")
f26 = load_and_build(DB_2026, "2026")
print("Building features for 2018...")
f18 = load_and_build(DB_2018, "2018")

print(f"\nRunning backtest 2026 (q={Q}, paper cost)...")
bt26 = run_backtest(features_df=f26, **COMMON)
m26  = compute_performance_metrics(bt26, verbose=False, output_dir=OUTPUT_DIR)

print(f"Running backtest 2018 (q={Q}, paper cost)...")
bt18 = run_backtest(features_df=f18, **COMMON)
m18  = compute_performance_metrics(bt18, verbose=False, output_dir=OUTPUT_DIR)

# ── extract per-trade PnL (in CNY) ───────────────────────────────────────────

all_trades26 = [p for d in bt26 for p in bt26[d]['daily_trade_pnl_list']]
all_trades18 = [p for d in bt18 for p in bt18[d]['daily_trade_pnl_list']]

t26 = np.array(all_trades26)
t18 = np.array(all_trades18)

def decompose(t, label):
    wins = t[t > 0]
    loss = t[t < 0]
    wr   = (t > 0).mean()
    lr   = 1 - wr
    avg_w = wins.mean() if len(wins) > 0 else np.nan
    avg_l = loss.mean() if len(loss) > 0 else np.nan
    e_pt  = t.mean()
    return {
        'label':    label,
        'n_trades': len(t),
        'E_pt':     float(e_pt),
        'win_rate': float(wr),
        'loss_rate': float(lr),
        'avg_win':  float(avg_w),
        'avg_loss': float(avg_l),
        'payoff':   float(abs(avg_w/avg_l)) if (not np.isnan(avg_w) and avg_l!=0) else np.nan,
    }

d26 = decompose(t26, "2026")
d18 = decompose(t18, "2018")

# ── profit decomposition ──────────────────────────────────────────────────────
# E[pt] = wr * avg_win + lr * avg_loss
# Delta = E[pt]_18 - E[pt]_26
# Broken into 4 terms:
#   T1 = delta_wr * avg_win_18            (win-rate contribution)
#   T2 = wr_26   * delta_avg_win          (avg-win-size contribution)
#   T3 = delta_lr * avg_loss_18           (loss-rate contribution, sign: -)
#   T4 = lr_26   * delta_avg_loss         (avg-loss-size contribution)

dwr    = d18['win_rate']  - d26['win_rate']
dlr    = d18['loss_rate'] - d26['loss_rate']
daw    = d18['avg_win']   - d26['avg_win']
dal    = d18['avg_loss']  - d26['avg_loss']

T1 = dwr  * d18['avg_win']
T2 = d26['win_rate'] * daw
T3 = dlr  * d18['avg_loss']   # negative contributes positively to 2018
T4 = d26['loss_rate'] * dal   # dal is negative (2018 loss is more negative)
total = T1 + T2 + T3 + T4
actual_delta = d18['E_pt'] - d26['E_pt']

print()
print("=" * 65)
print(f"TEST 3: Profit Decomposition  (Strategy B, q={Q}, paper cost)")
print("=" * 65)

rows = [
    ("Trades (total)",              "n_trades",   "{:>14,.0f}", ""),
    ("E[PnL per trade] (CNY)",      "E_pt",       "{:>14.2f}", "CNY"),
    ("Win rate",                    "win_rate",   "{:>13.2%}",  ""),
    ("Loss rate",                   "loss_rate",  "{:>13.2%}",  ""),
    ("Avg win (CNY)",               "avg_win",    "{:>14.2f}", "CNY"),
    ("Avg loss (CNY)",              "avg_loss",   "{:>14.2f}", "CNY"),
    ("Payoff ratio |win/loss|",     "payoff",     "{:>14.4f}", ""),
]
print(f"  {'Metric':<35}  {'2026':>12}  {'2018':>12}")
print(f"  {'-'*35}  {'-'*12}  {'-'*12}")
for name, key, fmt, unit in rows:
    v26 = fmt.format(d26[key])
    v18 = fmt.format(d18[key])
    suf = f"  {unit}" if unit else ""
    print(f"  {name:<35}  {v26}  {v18}{suf}")

print()
print("  Decomposition of E[PnL/trade] gap (2018 - 2026):")
print(f"    Actual gap                  : {actual_delta:>+.4f} CNY")
print(f"    Explained by decomposition  : {total:>+.4f} CNY")
print(f"    (T1) Delta win_rate         : {T1:>+.4f} CNY  (dwr={dwr:+.4f})")
print(f"    (T2) Delta avg_win size     : {T2:>+.4f} CNY  (daw={daw:+.2f})")
print(f"    (T3) Delta loss_rate        : {T3:>+.4f} CNY  (dlr={dlr:+.4f})")
print(f"    (T4) Delta avg_loss size    : {T4:>+.4f} CNY  (dal={dal:+.2f})")
print(f"    Dominant driver             : {max([(abs(T1),'T1 win_rate'),(abs(T2),'T2 avg_win'),(abs(T3),'T3 loss_rate'),(abs(T4),'T4 avg_loss')], key=lambda x: x[0])[1]}")

# ── plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Test 3: Profit Decomposition (Strategy B, q={Q}, paper cost)", fontsize=13)

# 1. Per-trade PnL distributions
ax = axes[0]
cap = max(float(np.percentile(np.abs(t26), 99)), float(np.percentile(np.abs(t18), 99)))
ax.hist(t26.clip(-cap, cap), bins=60, color='steelblue', alpha=0.65, density=True,
        label=f'2026 (n={len(t26):,})')
ax.hist(t18.clip(-cap, cap), bins=60, color='darkorange', alpha=0.65, density=True,
        label=f'2018 (n={len(t18):,})')
ax.axvline(0, color='black', linewidth=1)
ax.axvline(d26['E_pt'], color='steelblue', linewidth=2, linestyle='--',
           label=f'mean 2026={d26["E_pt"]:.1f}')
ax.axvline(d18['E_pt'], color='darkorange', linewidth=2, linestyle='--',
           label=f'mean 2018={d18["E_pt"]:.1f}')
ax.set_xlabel('Per-trade PnL (CNY)')
ax.set_ylabel('Density')
ax.set_title('Per-trade PnL distribution')
ax.legend(fontsize=8)

# 2. Win/loss bar chart
ax = axes[1]
cats = ['Win rate', 'Avg win (CNY)', '|Avg loss| (CNY)', 'Payoff ratio']
v26_bar = [d26['win_rate']*100, d26['avg_win'], abs(d26['avg_loss']), d26['payoff']]
v18_bar = [d18['win_rate']*100, d18['avg_win'], abs(d18['avg_loss']), d18['payoff']]
x = np.arange(len(cats))
w = 0.35
b1 = ax.bar(x - w/2, v26_bar, w, label='2026', color='steelblue', alpha=0.8)
b2 = ax.bar(x + w/2, v18_bar, w, label='2018', color='darkorange', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=15, ha='right', fontsize=9)
ax.set_title('Win/Loss Components')
ax.legend()

# 3. Waterfall decomposition
ax = axes[2]
labels = ['E[pt] 2026', '+T1 win_rate', '+T2 avg_win', '+T3 loss_rate', '+T4 avg_loss', '= 2018']
values = [d26['E_pt'], T1, T2, T3, T4, None]
running = d26['E_pt']
bottoms = []
heights = []
for v in [T1, T2, T3, T4]:
    bottoms.append(min(running, running + v))
    heights.append(abs(v))
    running += v

colors_bar = ['steelblue'] + ['green' if v >= 0 else 'red' for v in [T1,T2,T3,T4]] + ['darkorange']
bar_vals = [d26['E_pt'], T1, T2, T3, T4, d18['E_pt']]
ax.bar(range(len(labels)), bar_vals, color=colors_bar, alpha=0.8, edgecolor='grey', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('CNY')
ax.set_title('Decomposition waterfall')

plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, "test3_profit_decomposition.png")
plt.savefig(outpath, dpi=150)
plt.close()
print(f"\nPlot saved: {outpath}")
print("Test 3 complete.")
