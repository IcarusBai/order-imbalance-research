"""
diag_test2.py - Test 2: Conditional price path after signal
Walk-forward: fit OLS on day d-1, compute signals on day d,
record k-step mid-price change for every triggered signal.
Uses Strategy B features (VOI + OIR + MPB, spread-normalised), q=0.20, k=5.
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

from backtest_engine import fit_ols_model, compute_signals

OUTPUT_DIR = "results_diagnostic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_2026 = "data/market_data.db"
DB_2018 = "data/market_data_2018.db"

COLS = ["InstruID", "TradDay", "ActionDateTime",
        "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",
        "Volume", "Turnover"]

# ── signal construction (identical to notebooks) ─────────────────────────────

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

# ── data loading & feature building ─────────────────────────────────────────

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

    # Use highest-volume contract per day (prev-day) as main contract
    daily_vol = df.groupby(['TradDay','InstruID'])['Volume'].max().unstack('InstruID')
    prev_vol  = daily_vol.shift(1)
    main_map  = prev_vol.dropna(how='all').idxmax(axis=1).dropna()
    df = df[df['TradDay'].isin(main_map.index) &
            (df['InstruID'] == df['TradDay'].map(main_map))].reset_index(drop=True)
    print(f"  {label}: {len(df):,} rows, {df['TradDay'].nunique()} trading days (main contract)")

    # Compute signals per session
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

# ── walk-forward signal collection ──────────────────────────────────────────

def collect_conditional_moves(feats, q=0.20):
    """For every triggered signal, record signed k-step mid-price change."""
    records = []
    days = sorted(feats['TradDay'].unique())
    for i, day in enumerate(days):
        if i == 0:
            continue
        prev = days[i-1]
        tr = feats[feats['TradDay'] == prev]
        coefs = fit_ols_model(tr[FEAT_B].values, tr['y'].values)
        if coefs is None:
            continue
        te = feats[feats['TradDay'] == day]
        sigs = compute_signals(te[FEAT_B].values, coefs)
        y    = te['y'].values

        for sig, move in zip(sigs, y):
            if np.isnan(sig) or np.isnan(move):
                continue
            if sig >= q:
                records.append({'direction': 'buy',  'signed_move':  move})
            elif sig <= -q:
                records.append({'direction': 'sell', 'signed_move': -move})  # flip sign

    return pd.DataFrame(records)


# ── run ───────────────────────────────────────────────────────────────────────

print("Building features for 2026...")
f26 = load_and_build(DB_2026, "2026")
print("Building features for 2018...")
f18 = load_and_build(DB_2018, "2018")

Q = 0.20

print(f"\nCollecting conditional moves (q={Q})...")
moves26 = collect_conditional_moves(f26, q=Q)
moves18 = collect_conditional_moves(f18, q=Q)

print(f"  2026: {len(moves26):,} triggered signals")
print(f"  2018: {len(moves18):,} triggered signals")

def summarise(moves, label):
    m = moves['signed_move']
    wins = m[m > 0]
    loss = m[m < 0]
    wr   = (m > 0).mean()
    return {
        'label':         label,
        'n_signals':     len(m),
        'E_dM':          float(m.mean()),
        'win_rate':      float(wr),
        'loss_rate':     float(1 - wr),
        'E_dM_win':      float(wins.mean()) if len(wins) > 0 else np.nan,
        'E_dM_loss':     float(loss.mean()) if len(loss) > 0 else np.nan,
        'payoff_ratio':  float(wins.mean() / abs(loss.mean())) if (len(wins)>0 and len(loss)>0) else np.nan,
        'median_move':   float(m.median()),
    }

s26 = summarise(moves26, "2026")
s18 = summarise(moves18, "2018")

print()
print("=" * 65)
print(f"TEST 2: Conditional Price Path after Signal  (q={Q}, k={k_B})")
print("=" * 65)
rows = [
    ("Triggered signals",          "n_signals",    "{:>14,.0f}", ""),
    ("E[dMid | signal]",           "E_dM",          "{:>14.5f}", "index pts"),
    ("Win rate P(correct dir)",    "win_rate",      "{:>13.2%}",  ""),
    ("Loss rate",                  "loss_rate",     "{:>13.2%}",  ""),
    ("E[dMid | win]",              "E_dM_win",      "{:>14.5f}", "index pts"),
    ("E[dMid | loss]",             "E_dM_loss",     "{:>14.5f}", "index pts"),
    ("Payoff ratio |win/loss|",    "payoff_ratio",  "{:>14.4f}", ""),
    ("Median signed move",         "median_move",   "{:>14.5f}", "index pts"),
]
print(f"  {'Metric':<35}  {'2026':>12}  {'2018':>12}")
print(f"  {'-'*35}  {'-'*12}  {'-'*12}")
for name, key, fmt, unit in rows:
    v26 = fmt.format(s26[key])
    v18 = fmt.format(s18[key])
    suf = f"  {unit}" if unit else ""
    print(f"  {name:<35}  {v26}  {v18}{suf}")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Test 2: Conditional price path after signal (q={Q}, k={k_B})", fontsize=13)

for ax, moves, label, color in [
    (axes[0], moves26, '2026', 'steelblue'),
    (axes[1], moves18, '2018', 'darkorange'),
]:
    m = moves['signed_move']
    cap = float(np.percentile(np.abs(m), 99))
    ax.hist(m.clip(lower=-cap, upper=cap), bins=80,
            color=color, alpha=0.75, edgecolor='none', density=True)
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(float(m.mean()), color='red', linewidth=1.8, linestyle='--',
               label=f'mean={m.mean():.4f}')
    ax.set_xlabel('Signed k-step dMid (index pts)')
    ax.set_ylabel('Density')
    ax.set_title(f'{label} — win rate={( m>0).mean():.1%}')
    ax.legend(fontsize=9)

plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, "test2_conditional_move.png")
plt.savefig(outpath, dpi=150)
plt.close()
print(f"\nPlot saved: {outpath}")
print("Test 2 complete.")
