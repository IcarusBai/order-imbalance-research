"""
backtest_engine.py — General-purpose backtesting framework (Shen 2015, Strategy A).
Dependencies: numpy, pandas, scipy, matplotlib only.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ── OLS fitting ───────────────────────────────────────────────────────────────

def fit_ols_model(X: np.ndarray, y: np.ndarray):
    """
    Fit OLS via least squares. Returns coefficient array [alpha, beta1, beta2, ...].
    Prepends a column of ones internally (intercept term).
    Returns None if X or y is empty or has fewer than n_features + 2 rows.
    """
    if len(X) == 0 or len(y) == 0:
        return None
    n_features = X.shape[1] if X.ndim > 1 else 1
    if len(X) < n_features + 2:
        return None
    X_aug = np.column_stack([np.ones(len(X)), X])
    coefs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return coefs


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(X: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """
    Compute EFPC (expected future price change) for each tick.
    EFPC = alpha + beta1*x1 + beta2*x2 + ...
    NaN in X propagates to NaN in the output (no explicit loop needed).
    """
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])
    return X_aug @ coefs


# ── Single-session simulation ─────────────────────────────────────────────────

def run_day_simulation(
    signals: np.ndarray,
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    time_seconds: np.ndarray,
    threshold: float,
    tr_cost_open: float,
    tr_cost_close: float,
    contract_multiplier: float,
    trading_hours: dict,
) -> dict:
    """
    Simulate trading for ONE session (morning OR afternoon) of ONE day.

    trading_hours keys: 'open', 'close', 'end' (all ints: seconds since midnight)
      open  — trading allowed from this time
      close — no new opens after this time; closes are still permitted
      end   — force-close all remaining positions at this tick or beyond
    """
    n = len(signals)
    open_t  = trading_hours['open']
    close_t = trading_hours['close']
    end_t   = trading_hours['end']

    trade_pnl_list = []
    total_costs    = 0.0
    trade_volume   = 0
    pnl_series     = np.zeros(n)
    position_series = np.zeros(n, dtype=np.int8)

    position      = 0
    entry_price   = 0.0
    cost_open_val = 0.0
    cum_pnl       = 0.0

    for i in range(n):
        t   = time_seconds[i]
        bid = bid_prices[i]
        ask = ask_prices[i]
        sig = signals[i]

        # Force-close at or after end_t
        if position != 0 and t >= end_t:
            if position == 1:
                close_px = bid
                gross    = (close_px - entry_price) * contract_multiplier
            else:
                close_px = ask
                gross    = (entry_price - close_px) * contract_multiplier
            cost_close  = close_px * contract_multiplier * tr_cost_close
            pnl         = gross - cost_open_val - cost_close
            trade_pnl_list.append(pnl)
            total_costs += cost_close
            trade_volume += 1
            cum_pnl      += pnl
            position      = 0
            entry_price   = 0.0
            cost_open_val = 0.0
            pnl_series[i]      = cum_pnl
            position_series[i] = 0
            continue

        # Skip NaN signals — no position change, just record state
        if np.isnan(sig):
            pnl_series[i]      = cum_pnl
            position_series[i] = position
            continue

        in_open_window = (t >= open_t) and (t <= close_t)

        if position == 0:
            if in_open_window:
                if sig >= threshold:
                    entry_price   = ask
                    cost_open_val = entry_price * contract_multiplier * tr_cost_open
                    total_costs  += cost_open_val
                    position      = 1
                elif sig <= -threshold:
                    entry_price   = bid
                    cost_open_val = entry_price * contract_multiplier * tr_cost_open
                    total_costs  += cost_open_val
                    position      = -1

        elif position == 1:
            if sig <= -threshold:
                close_px   = bid
                cost_close = close_px * contract_multiplier * tr_cost_close
                gross      = (close_px - entry_price) * contract_multiplier
                pnl        = gross - cost_open_val - cost_close
                trade_pnl_list.append(pnl)
                total_costs  += cost_close
                trade_volume += 1
                cum_pnl      += pnl
                position      = 0
                entry_price   = 0.0
                cost_open_val = 0.0
                if in_open_window:
                    entry_price   = bid
                    cost_open_val = entry_price * contract_multiplier * tr_cost_open
                    total_costs  += cost_open_val
                    position      = -1

        elif position == -1:
            if sig >= threshold:
                close_px   = ask
                cost_close = close_px * contract_multiplier * tr_cost_close
                gross      = (entry_price - close_px) * contract_multiplier
                pnl        = gross - cost_open_val - cost_close
                trade_pnl_list.append(pnl)
                total_costs  += cost_close
                trade_volume += 1
                cum_pnl      += pnl
                position      = 0
                entry_price   = 0.0
                cost_open_val = 0.0
                if in_open_window:
                    entry_price   = ask
                    cost_open_val = entry_price * contract_multiplier * tr_cost_open
                    total_costs  += cost_open_val
                    position      = 1

        pnl_series[i]      = cum_pnl
        position_series[i] = position

    return {
        'trade_pnl_list':  trade_pnl_list,
        'trade_costs':     total_costs,
        'trade_volume':    trade_volume,
        'pnl_series':      pnl_series,
        'position_series': position_series,
    }


# ── Full backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    day_col: str,
    session_col: str,
    bid_col: str,
    ask_col: str,
    time_col: str,
    threshold: float,
    tr_cost_open: float,
    tr_cost_close: float,
    contract_multiplier: float,
    trading_hours: dict,
) -> dict:
    """
    Main backtest loop. Rolling window: fit OLS on day d-1, trade on day d.
    Day 1 is skipped (no prior day). Returns dict keyed by TradDay integer.
    """
    sorted_days = sorted(features_df[day_col].unique())
    results = {}

    for i, day in enumerate(sorted_days):
        if i == 0:
            continue

        prev_day = sorted_days[i - 1]

        # Training: all rows from prev_day
        train_df = features_df[features_df[day_col] == prev_day]
        X_train  = train_df[feature_cols].values
        y_train  = train_df[target_col].values

        coefs = fit_ols_model(X_train, y_train)
        if coefs is None:
            continue

        # R² on training day
        X_aug  = np.column_stack([np.ones(len(X_train)), X_train])
        y_pred = X_aug @ coefs
        ss_res = np.sum((y_train - y_pred) ** 2)
        ss_tot = np.sum((y_train - y_train.mean()) ** 2)
        r_sq   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        day_res = {
            'r_squared':            r_sq,
            'coefs':                coefs,
            'daily_pnl':            0.0,
            'daily_trade_volume':   0,
            'daily_trade_pnl_list': [],
        }

        for session in ['morning', 'afternoon']:
            mask      = (features_df[day_col] == day) & (features_df[session_col] == session)
            sess_df   = features_df[mask]
            if len(sess_df) == 0:
                continue

            signals   = compute_signals(sess_df[feature_cols].values, coefs)
            sess_res  = run_day_simulation(
                signals=signals,
                bid_prices=sess_df[bid_col].values,
                ask_prices=sess_df[ask_col].values,
                time_seconds=sess_df[time_col].values,
                threshold=threshold,
                tr_cost_open=tr_cost_open,
                tr_cost_close=tr_cost_close,
                contract_multiplier=contract_multiplier,
                trading_hours=trading_hours[session],
            )
            day_res[session]               = sess_res
            day_res['daily_pnl']          += sum(sess_res['trade_pnl_list'])
            day_res['daily_trade_volume'] += sess_res['trade_volume']
            day_res['daily_trade_pnl_list'].extend(sess_res['trade_pnl_list'])

        results[day] = day_res

    return results


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_performance_metrics(
    backtest_results: dict,
    trading_days: list = None,
    verbose: bool = True,
    output_dir: str = 'results/',
    label: str = '',
) -> dict:
    """
    Compute all performance metrics from backtest_results.
    Saves {label}_performance_summary.txt and {label}_daily_pnl.csv to output_dir.
    Pass output_dir=None to skip file saving.
    """
    if trading_days is not None:
        days = [d for d in sorted(trading_days) if d in backtest_results]
    else:
        days = sorted(backtest_results.keys())

    daily_pnl    = np.array([backtest_results[d]['daily_pnl']          for d in days])
    daily_vol    = np.array([backtest_results[d]['daily_trade_volume']  for d in days])
    daily_r2     = np.array([backtest_results[d]['r_squared']           for d in days])

    n_days       = len(daily_pnl)
    mean_pnl     = float(np.mean(daily_pnl))
    std_pnl      = float(np.std(daily_pnl, ddof=1))
    stderr_pnl   = std_pnl / np.sqrt(n_days) if n_days > 0 else np.nan

    tstat_result = stats.ttest_1samp(daily_pnl, popmean=0, alternative='greater')
    t_stat       = float(tstat_result.statistic)
    p_value      = float(tstat_result.pvalue)

    days_profit  = int(np.sum(daily_pnl > 0))
    days_loss    = int(np.sum(daily_pnl <= 0))

    mean_vol     = float(np.mean(daily_vol))
    avg_r2       = float(np.mean(daily_r2))

    # Per-day Sharpe: mean(trade_pnl_list) / std(trade_pnl_list)
    daily_sharpe = []
    for d in days:
        tpnl = backtest_results[d]['daily_trade_pnl_list']
        if len(tpnl) >= 2:
            s = np.std(tpnl, ddof=1)
            daily_sharpe.append(np.mean(tpnl) / s if s > 0 else np.nan)
        else:
            daily_sharpe.append(np.nan)
    avg_daily_sharpe = float(np.nanmean(daily_sharpe)) if any(~np.isnan(x) for x in daily_sharpe) else np.nan

    ann_sharpe  = float(mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else np.nan

    # Win rate per trade
    all_trades  = [p for d in days for p in backtest_results[d]['daily_trade_pnl_list']]
    if len(all_trades) > 0:
        win_rate = float(np.sum(np.array(all_trades) > 0) / len(all_trades))
    else:
        win_rate = np.nan

    # Max drawdown in cumulative daily PnL curve
    cum_pnl     = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    max_dd      = float(-(cum_pnl - running_max).min()) if n_days > 0 else 0.0

    metrics = {
        'mean_daily_pnl':        mean_pnl,
        'std_daily_pnl':         std_pnl,
        'stderr_daily_pnl':      stderr_pnl,
        't_stat':                t_stat,
        'p_value':               p_value,
        'days_with_profit':      days_profit,
        'days_with_loss':        days_loss,
        'mean_daily_trade_volume': mean_vol,
        'avg_daily_sharpe':      avg_daily_sharpe,
        'annualized_sharpe':     ann_sharpe,
        'avg_r_squared':         avg_r2,
        'win_rate_per_trade':    win_rate,
        'max_drawdown':          max_dd,
        'n_days':                n_days,
        'n_trades':              len(all_trades),
    }

    summary_lines = [
        "Performance Summary",
        "=" * 52,
        f"  Trading days              : {n_days}",
        f"  Total round-trips         : {len(all_trades)}",
        f"  Mean daily PnL (CNY)      : {mean_pnl:>12,.2f}",
        f"  Std daily PnL             : {std_pnl:>12,.2f}",
        f"  Standard error            : {stderr_pnl:>12,.2f}",
        f"  t-statistic               : {t_stat:>12.4f}",
        f"  p-value (one-tailed)      : {p_value:>12.2e}",
        f"  Days with profit          : {days_profit:>5} / {n_days}  ({days_profit/n_days*100:.1f}%)",
        f"  Days with loss            : {days_loss:>5} / {n_days}  ({days_loss/n_days*100:.1f}%)",
        f"  Mean daily trade volume   : {mean_vol:>12.2f}",
        f"  Avg daily Sharpe          : {avg_daily_sharpe:>12.4f}",
        f"  Annualized Sharpe         : {ann_sharpe:>12.4f}",
        f"  Avg R-squared (training)  : {avg_r2:>12.4f}",
        f"  Win rate per trade        : {win_rate*100:>11.2f}%",
        f"  Max drawdown (CNY)        : {max_dd:>12,.2f}",
        "=" * 52,
    ]
    summary_str = "\n".join(summary_lines)

    if verbose:
        print(summary_str)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        prefix = f'{label}_' if label else ''

        with open(os.path.join(output_dir, f'{prefix}performance_summary.txt'), 'w', encoding='utf-8') as f:
            f.write(summary_str + "\n")

        rows = []
        for d in days:
            r = backtest_results[d]
            m_pnl = sum(r['morning']['trade_pnl_list'])   if 'morning'   in r else 0.0
            a_pnl = sum(r['afternoon']['trade_pnl_list']) if 'afternoon' in r else 0.0
            m_vol = r['morning']['trade_volume']           if 'morning'   in r else 0
            a_vol = r['afternoon']['trade_volume']         if 'afternoon' in r else 0
            rows.append({
                'TradDay':           d,
                'daily_pnl':         r['daily_pnl'],
                'morning_pnl':       m_pnl,
                'afternoon_pnl':     a_pnl,
                'daily_trade_volume': r['daily_trade_volume'],
                'morning_volume':    m_vol,
                'afternoon_volume':  a_vol,
                'r_squared':         r['r_squared'],
            })
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, f'{prefix}daily_pnl.csv'), index=False)

    return metrics


# ── Parameter sweep ───────────────────────────────────────────────────────────

def run_parameter_sweep(
    features_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    day_col: str,
    session_col: str,
    bid_col: str,
    ask_col: str,
    time_col: str,
    q_values: list,
    tr_cost_open: float,
    tr_cost_close: float,
    contract_multiplier: float,
    trading_hours: dict,
    trading_days: list = None,
    output_dir: str = 'results/',
    label: str = '',
) -> pd.DataFrame:
    """
    Sweep threshold q. OLS fits and signals are computed once per (day, session)
    and reused across all q values to avoid redundant computation.
    Saves {label}_parameter_sweep.csv to output_dir.
    """
    sorted_days = sorted(features_df[day_col].unique())

    # Precompute OLS coefficients and signals for each (day, session)
    print("Precomputing signals for parameter sweep...")
    cached = {}
    for i, day in enumerate(sorted_days):
        if i == 0:
            continue
        prev_day = sorted_days[i - 1]
        train_df = features_df[features_df[day_col] == prev_day]
        X_train  = train_df[feature_cols].values
        y_train  = train_df[target_col].values
        coefs    = fit_ols_model(X_train, y_train)
        if coefs is None:
            continue

        X_aug  = np.column_stack([np.ones(len(X_train)), X_train])
        y_pred = X_aug @ coefs
        ss_res = np.sum((y_train - y_pred) ** 2)
        ss_tot = np.sum((y_train - y_train.mean()) ** 2)
        r_sq   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        for session in ['morning', 'afternoon']:
            mask    = (features_df[day_col] == day) & (features_df[session_col] == session)
            sess_df = features_df[mask]
            if len(sess_df) == 0:
                continue
            cached[(day, session)] = {
                'signals':   compute_signals(sess_df[feature_cols].values, coefs),
                'bid':       sess_df[bid_col].values,
                'ask':       sess_df[ask_col].values,
                'time':      sess_df[time_col].values,
                'r_squared': r_sq,
                'coefs':     coefs,
            }

    print(f"Signals cached for {len(cached)} (day, session) pairs.")

    rows = []
    for q in q_values:
        q_results = {}
        days_in_cache = sorted({day for (day, _) in cached.keys()})
        for day in days_in_cache:
            day_res = {
                'r_squared':            cached.get((day, 'morning'), cached.get((day, 'afternoon'), {})).get('r_squared', np.nan),
                'coefs':                cached.get((day, 'morning'), cached.get((day, 'afternoon'), {})).get('coefs', None),
                'daily_pnl':            0.0,
                'daily_trade_volume':   0,
                'daily_trade_pnl_list': [],
            }
            for session in ['morning', 'afternoon']:
                key = (day, session)
                if key not in cached:
                    continue
                data     = cached[key]
                sess_res = run_day_simulation(
                    signals=data['signals'],
                    bid_prices=data['bid'],
                    ask_prices=data['ask'],
                    time_seconds=data['time'],
                    threshold=q,
                    tr_cost_open=tr_cost_open,
                    tr_cost_close=tr_cost_close,
                    contract_multiplier=contract_multiplier,
                    trading_hours=trading_hours[session],
                )
                day_res[session]               = sess_res
                day_res['daily_pnl']          += sum(sess_res['trade_pnl_list'])
                day_res['daily_trade_volume'] += sess_res['trade_volume']
                day_res['daily_trade_pnl_list'].extend(sess_res['trade_pnl_list'])
            q_results[day] = day_res

        m = compute_performance_metrics(q_results, trading_days=trading_days, verbose=False, output_dir=None)
        rows.append({
            'q':                   q,
            'mean_daily_pnl':      m['mean_daily_pnl'],
            'stderr':              m['stderr_daily_pnl'],
            't_stat':              m['t_stat'],
            'p_value':             m['p_value'],
            'annualized_sharpe':   m['annualized_sharpe'],
            'avg_daily_sharpe':    m['avg_daily_sharpe'],
            'win_rate_per_trade':  m['win_rate_per_trade'],
            'mean_daily_volume':   m['mean_daily_trade_volume'],
            'days_with_profit':    m['days_with_profit'],
            'days_with_loss':      m['days_with_loss'],
        })
        print(f"  q={q:.3f}  mean_daily_pnl={m['mean_daily_pnl']:>10,.0f}  t={m['t_stat']:.3f}  sharpe={m['annualized_sharpe']:.3f}")

    sweep_df = pd.DataFrame(rows).sort_values('q').reset_index(drop=True)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        prefix = f'{label}_' if label else ''
        sweep_df.to_csv(os.path.join(output_dir, f'{prefix}parameter_sweep.csv'), index=False)
        print(f"Sweep complete. Results saved to {output_dir}{prefix}parameter_sweep.csv")
    else:
        print("Sweep complete.")

    return sweep_df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_results(
    backtest_results: dict,
    sweep_df: pd.DataFrame = None,
    output_dir: str = 'results/',
    label: str = '',
):
    """
    Save: {label}_cumulative_pnl.png, {label}_daily_pnl_hist.png, {label}_sweep_results.png.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f'{label}_' if label else ''

    days     = sorted(backtest_results.keys())
    dpnl     = np.array([backtest_results[d]['daily_pnl'] for d in days])
    cum_pnl  = np.cumsum(dpnl)

    # 1. Cumulative PnL
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(days)), cum_pnl, linewidth=1.5, color='steelblue')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Cumulative Daily PnL')
    ax.set_xlabel('Trading Day Index')
    ax.set_ylabel('Cumulative PnL (CNY)')
    ax.set_xticks(range(0, len(days), max(1, len(days) // 10)))
    ax.set_xticklabels([str(days[j]) for j in range(0, len(days), max(1, len(days) // 10))], rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}cumulative_pnl.png'), dpi=150)
    plt.close()

    # 2. Daily PnL histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dpnl, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(dpnl.mean(), color='red', linewidth=1.5, linestyle='--', label=f'Mean = {dpnl.mean():,.0f}')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_title('Daily PnL Distribution')
    ax.set_xlabel('Daily PnL (CNY)')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}daily_pnl_hist.png'), dpi=150)
    plt.close()

    # 3. Parameter sweep results
    if sweep_df is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sweep_df['q'], sweep_df['mean_daily_pnl'], marker='o', linewidth=1.5, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title('Mean Daily PnL vs Threshold q')
        ax.set_xlabel('Threshold q')
        ax.set_ylabel('Mean Daily PnL (CNY)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}sweep_results.png'), dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")
