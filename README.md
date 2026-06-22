# Order Imbalance Research: VOI-Based Intraday Trading Strategy on CFFEX IF Futures

## 1. Project Overview

This project replicates and extends Shen (2015)'s order imbalance intraday trading strategy on CFFEX CSI 300 Index Futures (IF contracts), using second-level limit order book (LOB) snapshot data stored in a local DuckDB database. Volume Order Imbalance (VOI) — the net change in resting quantity at the best bid and ask between consecutive ticks — is the core predictive signal, fed into a rolling OLS model to generate trade signals. The primary backtest period is 2018 H1; 2026 H1 data serves as a structural comparison to assess how market conditions have changed. Beyond the baseline VOI strategy, the project introduces and tests two depth-of-book signals — `LDistance_diff` (the imbalance in volume-weighted average depth across all five LOB levels) and `CostToTrade` (the round-trip market impact cost of a five-lot order) — as potential enhancements to signal quality and trade filtering.

---

## 2. Quick Start

**Requirements:** Python 3.10+, and the raw CFFEX tick data archives in `raw_data/`.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build databases (run once each)
python scripts/build_database.py --year 2018   # → data/market_data_2018.db
python scripts/build_database.py               # → data/market_data.db  (2026 data)

# 3. Run notebooks in order
#    01_research_2026.ipynb        — baseline VOI strategy on 2026 H1 data
#    02_research_2018.ipynb        — baseline VOI strategy on 2018 H1 data (main results)
#    03_lob_signals_validation.ipynb — LOB depth signal validation (LDistance, CostToTrade)
#    04_strategy_enhanced.ipynb    — B-CTT and LDistance enhancement experiments
#    05_crossday_experiment.ipynb  — 2018 vs 2026 structural comparison + walk-forward OOS
```

> **Note on data:** the raw `.tar.gz` files and compiled `.db` databases are excluded from this repository (see `.gitignore`). The tick data was sourced from CFFEX and is not redistributable.

---

## 3. Data

**Source:** CFFEX IF futures tick data (LOB snapshots at 0.5-second intervals) stored in DuckDB.

| Period | Database date range | Total rows in DB | Main contract rows | Main contract days |
|--------|--------------------|--------------------|--------------------|--------------------|
| 2018 H1 | 20180102 – 20180629 | 23,713,205 | 2,483,302 | 118 |
| 2026 H1 | 20260202 – 20260508 | 30,558,059 | 1,559,260 | 59 |

**Preprocessing pipeline:** (1) Filter to IF contracts only. (2) Restrict to continuous trading sessions (morning: 09:30–11:30; afternoon: 13:00–15:00). (3) Remove crossed-quote rows (bid > ask) and zero-price rows. (4) Select the main contract per day using the previous day's highest cumulative volume — no look-ahead bias is introduced at this step. (5) Derived columns (MidPrice, Spread, session label, VOI, OIR, MPB) are computed within `(TradDay, session)` groups to prevent cross-session contamination.

**Daily trading volume comparison:**

| Dataset | Mean lots/day | Median lots/day | Min | Max | Ratio vs Shen (2015) era (~200,000 lots/day) |
|---------|--------------|-----------------|-----|-----|----------------------------------------------|
| 2018 H1 | 17,133 | 17,420 | 5,468 | 25,264 | 8.6% |
| 2026 H1 | 59,747 | 57,102 | 37,735 | 107,068 | 29.9% |

Even the 2026 sample — larger than 2018 — sits well below the paper's reference period. Both periods represent a structurally thinner market than the environment in which the strategy was originally validated.

---

## 4. Backtest Engine Design

**Session structure.** Each trading day is split into a morning session and an afternoon session, simulated independently. Daily PnL is the sum of the two sessions. Positions do not carry over between sessions or between days; every session begins flat.

**OLS walk-forward.** On each trading day d, the OLS model is trained on all ticks from day d−1 (both sessions combined). The fitted coefficients are then applied to every tick on day d to produce EFPC (Expected Future Price Change) signals. Day 1 is always skipped since it has no prior training day. Cross-day and cross-session contamination is prevented by building feature matrices within `(TradDay, session)` groups before any model fitting.

**Signal and trade trigger.** A long is entered when EFPC ≥ q (buying at the ask); a short is entered when EFPC ≤ −q (selling at the bid). When the opposite signal fires while a position is open, the position is reversed in a single step (long→short or short→long), applied at the prevailing ask or bid respectively.

**Trading window.** New positions can be opened from session open until a `close_t` cutoff. After `close_t`, no new positions may be opened, but existing positions can still be reversed or closed by signal. At `end_t`, any remaining open position is force-closed at the prevailing bid (long) or ask (short).

| Session | Open | `close_t` | `end_t` |
|---------|------|-----------|---------|
| Morning | 09:16 | 11:20 | 11:28 |
| Afternoon | 13:01 | 15:00 | 15:13 |

**Transaction costs.** Two cost regimes are compared throughout. Paper cost replicates the Shen (2015) assumption; current cost reflects the CFFEX fee schedule since 2023, where the same-day close leg is ten times more expensive than the open leg.

| Regime | `tr_cost_open` | `tr_cost_close` |
|--------|---------------|----------------|
| Paper (Shen 2015) | 2.5 × 10⁻⁵ | 2.5 × 10⁻⁵ |
| Current CFFEX | 2.3 × 10⁻⁵ | 2.3 × 10⁻⁴ |

Each leg cost is `price × contract_multiplier (300) × cost_rate`, applied at open and close separately.

**Performance metrics.** The primary metric is annualised Sharpe: `mean(daily_PnL) / std(daily_PnL) × √252`. Daily PnL is in absolute CNY for a one-lot position. Statistical significance is assessed via a one-tailed t-test of daily PnL against zero. Supplementary metrics include win rate per trade (fraction of round-trips with positive PnL), win rate per day (fraction of trading days with positive PnL), total round-trips, and max drawdown on the cumulative PnL curve. `annualized_sharpe` (the cross-day metric) is the figure cited in all comparison tables; `avg_daily_sharpe` (mean of within-day per-trade Sharpe ratios) is a supplementary intraday diagnostic and is not comparable to the annualised figure.

---

## 5. Core Signal: VOI and the OLS Framework

VOI (Volume Order Imbalance) captures the net change in resting order quantity at the best bid and ask between consecutive ticks. When the best bid price rises, the full new bid volume is credited; when it stays flat, only the incremental change is credited. The same logic applies symmetrically to the ask side. The result is a tick-level measure of how aggressively buyers and sellers are updating their resting orders — positive VOI reflects net buying pressure.

**Strategy A** uses only VOI: six features (current-tick VOI plus lags at t−1 through t−5), with forecast horizon k=5 ticks (~2.5 seconds at 0.5-second intervals). **Strategy B** expands the feature set to thirteen predictors: the same six VOI lags plus six OIR (Order Imbalance Ratio — the static bid/ask volume ratio, normalised by the sum) lags and MPB (Mid-Price Basis — the deviation of the average trade price from mid, normalised by spread). OIR adds the current snapshot-level liquidity imbalance; MPB adds information about whether recent trades were aggressive buys or sells. Both strategies use k=5, enabling a direct feature-set comparison at the same forecast horizon.

In-sample OLS diagnostics on the 2018 H1 dataset:

| Metric | Strategy A (k=5) | Strategy B (k=5) |
|--------|------------------|------------------|
| Corr(VOI_t0, y) | 0.1274 | — |
| Avg R² (training days) | 0.0373 | 0.0589 |

The signal is weak but statistically present: VOI alone explains roughly 3.7% of the variance in the 5-tick forward mid-price change. Strategy B's higher R² (0.0589 vs 0.0373) reflects the additional explanatory power of OIR and MPB at the same horizon.

---

## 6. Baseline Backtest Results: VOI Signal Under Paper and Real Costs (2018 H1)

All scenarios below use threshold q = 0.20, forecast horizon k=5 (both strategies), and 117 backtest days.

| Scenario | Annualised Sharpe | Mean daily PnL (CNY) | Win rate (% days > 0) | Total trades | Max drawdown (CNY) |
|----------|------------------|---------------------|----------------------|--------------|--------------------|
| A — Paper cost (2.5 × 10⁻⁵) | −0.229 | −252.61 | 46.2% | 9,751 | 213,401 |
| A — Current cost (2.3 × 10⁻⁴ close) | −14.076 | −20,083.00 | 12.8% | 9,751 | 2,329,867 |
| A — Zero cost | +4.105 | +4,631.77 | 62.4% | 9,751 | 49,921 |
| B — Paper cost (2.5 × 10⁻⁵) | +7.318 | +10,512.49 | 66.7% | 32,380 | 41,449 |
| B — Current cost (2.3 × 10⁻⁴ close) | −38.259 | −55,272.50 | 1.7% | 32,380 | 6,421,962 |
| B — Zero cost | +16.491 | +26,715.75 | 94.0% | 32,380 | 12,125 |

**Key finding.** Strategy B generates a statistically real positive signal under paper costs (t = 4.99, p < 10⁻⁶). Adding OIR and MPB to the VOI model is sufficient to reverse the sign of performance relative to VOI alone. Under paper costs, Strategy A is statistically indistinguishable from zero (Sharpe −0.229, t = −0.156, p = 0.56); Strategy B achieves +7.318. Under current CFFEX transaction costs — where the close-leg fee is ten times the open-leg fee — all profitability is eliminated: Strategy B Sharpe falls from +7.3 to −38.3 and 98% of trading days produce a loss. The cost gap, not signal quality, is the binding constraint throughout this project.

**Parameter sweep findings.** Under paper costs, the optimal threshold for Strategy A is q = 0.350 (Sharpe 2.834), and for Strategy B is q = 0.23 (Sharpe 7.96, PnL +11,294 CNY/day, 23,982 trades). For Strategy B, Sharpe peaks around k = 15–20 ticks (Sharpe ≈ 8.96 at k=20) and declines monotonically thereafter, turning negative at approximately k = 105 ticks. Beyond the optimal horizon, the OLS forecast loses its connection to near-term price changes.

---

## 7. Signal Validation: Two Depth-of-Book Signals (2018 H1)

Given that current transaction costs eliminate VOI-based profitability, the question becomes whether depth-of-book signals using all five LOB levels can provide incremental predictive information or serve as a more discriminating trade filter.

### 6.1 LDistance_diff

`LDistance_diff` is constructed as the volume-weighted mean distance of all ask levels from mid-price minus the equivalent measure on the bid side. A positive value indicates the ask stack is more dispersed (ask resting orders are farther from mid on average than bid resting orders).

Quintile analysis of `LDistance_diff` against future mid-price change (2018 H1, 2,483,302 ticks):

| LDistance_diff Quintile | Mean dMid (k=5) | Mean dMid (k=10) | Mean dMid (k=20) |
|------------------------|-----------------|------------------|------------------|
| Q1 (most bid-dispersed) | −0.147 | −0.159 | −0.160 |
| Q2 | −0.044 | −0.052 | −0.057 |
| Q3 (neutral) | −0.002 | −0.004 | −0.006 |
| Q4 | +0.040 | +0.045 | +0.044 |
| Q5 (most ask-dispersed) | +0.148 | +0.161 | +0.161 |

The relationship is strongly monotone at all three horizons and stable from k=5 to k=20. Notably, the sign is opposite to the original hypothesis: a dispersed ask stack predicts a price rise (~+0.15 index points at k=5), not a fall. The microstructural interpretation is that a dispersed ask side reflects shallow resistance above mid, allowing buyers to push prices up; a dispersed bid side means sellers are far from mid, facilitating downward moves. **Verdict: LDistance_diff is a validated directional signal, but integration requires sign-corrected entry logic.**

### 6.2 CostToTrade

`CostToTrade` is the round-trip cost (as a fraction of mid-price) of simultaneously lifting the ask and hitting the bid for a five-lot market order, computed by walking the book across all five levels. With `TARGET_VOLUME = 5`, the NaN rate is **0.00%** — the five-level book always provides sufficient depth to fill five lots on both sides in the 2018 dataset.

Granger causality test (CostToTrade → realised volatility |ΔMid|, busiest trading day 20180627, 24,141 ticks):

| Lag | F-statistic | p-value |
|-----|------------|---------|
| 1 | 611.85 | 0.000 |
| 2 | 266.63 | 0.000 |
| 3 | 168.04 | 0.000 |
| 5 | 97.32 | 0.000 |
| 10 | 46.97 | 0.000 |

CostToTrade strongly Granger-causes next-tick volatility at all lags 1–10 (all p < 0.001). Both series are stationary by ADF (CostToTrade: ADF = −15.04, p = 0.000; |ΔMid|: ADF = −18.23, p = 0.000), so no differencing was required.

VOI autocorrelation (lag-1 ACF) varies systematically across CostToTrade quartiles:

| CTT Quartile | VOI ACF Lag-1 |
|-------------|--------------|
| Q1 (lowest cost) | 0.1157 |
| Q2 | 0.0742 |
| Q3 | 0.0604 |
| Q4 (highest cost) | 0.0502 |

Higher liquidity cost coincides with lower VOI persistence, consistent with more fragmented order flow and faster quote updating in high-volatility regimes. **Verdict: CostToTrade reliably identifies high-volatility regimes, making it a natural trade-gating variable. However, it does not add directional information.**

---

## 8. Strategy Enhancement Experiments (2018 H1)

### 7.1 B-CTT: CostToTrade as a Dynamic Trading Threshold

**Mechanism.** Within each session, the expanding percentile rank of CostToTrade (computed without look-ahead) is mapped linearly to a dynamic signal threshold q_dynamic ∈ [0.15, 0.25]. High CTT (expensive liquidity) maps to a higher threshold, so only stronger signals trigger trades during expensive periods.

All scenarios below use k=5 and 117 backtest days. The B baseline uses a fixed threshold q=0.15; B-CTT uses q_dynamic ∈ [0.15, 0.25] driven by CostToTrade. "Zero cost" is the frictionless reference (tr_cost=0, matching notebook 07's paper-cost baseline).

| Scenario | Annualised Sharpe | Mean daily PnL (CNY) | Win rate (% days > 0) | Total trades | Max drawdown (CNY) |
|----------|------------------|---------------------|----------------------|--------------|--------------------|
| B baseline — Zero cost (q=0.15) | +18.900 | +33,761.00 | 97.4% (114/117) | 54,731 | 9,061 |
| B-CTT — Zero cost (dynamic q) | +16.807 | +27,770.70 | 96.6% (113/117) | 42,214 | 7,619 |
| B baseline — Current cost (q=0.15) | −54.523 | −105,046.55 | 0.9% (1/117) | 54,731 | 12,200,245 |
| B-CTT — Current cost (dynamic q) | −48.811 | −79,249.83 | 0.9% (1/117) | 42,214 | 9,202,023 |

**Conclusion.** At q=0.15, Strategy B trades very frequently (54,731 round-trips over 117 days, ~468/day). CTT-as-threshold reduces trade count by ~23% (54,731 → 42,214), which under current costs meaningfully reduces daily losses (−105,047 → −79,250 CNY, −25%) and max drawdown (−12.2M → −9.2M CNY, −25%). However, CTT gating does not overcome the fundamental problem: every trade under current costs is deeply unprofitable regardless of when it fires. Under zero cost, both strategies are highly profitable (Sharpe +18.9 and +16.8 respectively), and CTT gating reduces trade count without improving Sharpe — high-CTT periods filter out profitable ticks.

### 7.2 LDistance_diff as an Additional OLS Feature

**What was tried.** `LDistance_diff` was appended as a fourteenth OLS predictor (raw value, no normalisation; and with lags 0–2) alongside the thirteen Strategy B features.

| Strategy | Annualised Sharpe | Mean daily PnL (CNY) | Win rate (% days > 0) | Total trades |
|----------|------------------|---------------------|----------------------|--------------|
| B baseline (13 features) | −2.613 | −2,544.9 | 35.9% | 3,570 |
| B + LDistance (lag 0 only) | −1.962 | −1,705.6 | 43.6% | 7,389 |
| B + LDistance (lags 0–2) | −7.898 | −28,165.3 | 4.3% | 18,355 |

All three under current cost at the same threshold.

Adding LDistance_diff at lag 0 more than doubles trade count (3,570 → 7,389), making the Sharpe comparison unfair: more trades at higher cost is not the same as a better signal. Adding lags 0–2 further explodes trade count to 18,355 and Sharpe collapses to −7.90. This is an open issue: a direct Sharpe comparison requires re-optimising q after the feature is added.

Coefficient sign stability (per-day OLS fits, 117 days): the lag-0 coefficient for `LDistance_diff_raw` is positive on all 117 days (mean 0.2056, std 0.0243), which is consistent with the directional validation in Section 6. The lag-1 coefficient is negative on 116 of 117 days (mean −0.1698), confirming the high autocorrelation of the signal creates an artificial reversal term in the OLS. **Conclusion: sign stability is present, but the trade-count explosion under current costs prevents a fair evaluation at the existing threshold.**

### 7.3 CostToTrade as an Additional OLS Feature

**What was tried.** CostToTrade was added as a fourteenth OLS predictor in two forms: raw (`CTT_raw`) and time-of-day normalised (`CTT_norm`).

| Strategy | Annualised Sharpe | Mean daily PnL (CNY) | Win rate (% days > 0) | Total trades |
|----------|------------------|---------------------|----------------------|--------------|
| B baseline (13 features) | −2.613 | −2,544.9 | 35.9% | 3,570 |
| B + CTT_raw | −3.232 | −2,886.7 | 36.8% | 3,588 |
| B + CTT_norm | −3.360 | −3,082.2 | 32.5% | 3,605 |

Sign distribution of the CTT coefficient across 117 training days: CTT_raw — 58 positive, 59 negative (essentially 50/50, consistent with pure noise); CTT_norm — 77 positive, 40 negative (skewed, but Sharpe still worsens). **Conclusion: CostToTrade carries no stable directional information in the OLS signal layer. Its value lies in the threshold layer (Section 7.1), not in the feature matrix.**

---

## 9. Market Structure Comparison: 2018 H1 vs 2026 H1

The same Strategy B pipeline applied to 2026 H1 data produces uniformly weaker results, even under paper costs. This section documents what the structural comparison reveals.

**Signal decay within the 2026 sample.** Over 59 trading days, the OLS R² and signal standard deviation both decline within the sample itself:

| Half | Avg R² | Signal std |
|------|--------|-----------|
| First 29 days (2026 H1) | 0.0131 | 0.0719 |
| Last 29 days (2026 H1) | 0.0093 | 0.0572 |

For comparison, the 2018 H1 halves: first 58 days R² = 0.0389; last 58 days R² = 0.0358. Both the absolute R² level and the decay rate are worse in 2026.

**LDistance_diff autocorrelation (Lag-1 ACF):**

| Dataset | Lag-1 ACF (pooled) | Lag-1 ACF (per-session avg) |
|---------|-------------------|---------------------------|
| 2018 H1 | 0.744 | — |
| 2026 H1 | 0.620 | 0.628 |

The faster decay in 2026 is consistent with more aggressive algorithmic quoting: depth signals update and revert more quickly, reducing their exploitable persistence.

**B+LD performance in 2026 (q=0.20, k=5, 58 backtest days).**

| Scenario | Annualised Sharpe | Mean daily PnL (CNY) | Win rate (% days > 0) | Total trades |
|----------|------------------|---------------------|----------------------|--------------|
| B — Paper cost | −2.154 | −2,202.3 | 46.6% | 14,116 |
| B+LD — Paper cost | +2.793 | +3,031.9 | 62.1% | 19,709 |
| B — Current cost | −23.002 | −70,170.0 | 0.0% | 14,116 |
| B+LD — Current cost | −23.194 | −91,758.1 | 0.0% | 19,709 |
| B — Zero cost | +12.116 | +14,538.5 | 75.9% | 14,116 |
| B+LD — Zero cost | +16.925 | +26,379.2 | 94.8% | 19,709 |

LDistance_diff improves paper-cost Sharpe in 2026 (−2.15 → +2.79), in contrast to 2018 where it approximately doubled trade count without improving performance. Under current costs, however, the improvement is completely wiped out: B+LD is actually marginally worse (−23.19 vs −23.00), because the higher trade count amplifies the cost drag.

**Walk-forward optimisation (WFO) results.** Three rolling folds, each with a 30-day training window and 9–10 OOS days, covering 20260203–20260508:

| Fold | Training period | OOS period |
|------|----------------|------------|
| 1 | 20260203–20260324 (30 days) | 20260325–20260408 (10 days) |
| 2 | 20260225–20260408 (30 days) | 20260409–20260422 (10 days) |
| 3 | 20260311–20260422 (30 days) | 20260423–20260508 (9 days) |

OOS aggregated results (29 total OOS days):

| Strategy | Annualised Sharpe | Mean daily PnL (CNY) | Win rate (% days > 0) | Total OOS trades |
|----------|------------------|---------------------|----------------------|-----------------|
| B — Paper cost | +0.824 | +509.0 | 51.7% | 636 |
| B+LD — Paper cost | +1.017 | +651.8 | 44.8% | 783 |
| B — Current cost | −8.910 | −4,145.1 | 20.7% | 98 |
| B+LD — Current cost | −7.255 | −2,798.3 | 10.3% | 69 |

B+LD marginally outperforms B under paper cost OOS (Sharpe 1.017 vs 0.824). Both are negative under current cost. **Caveat: 29 OOS days across three folds is too small a sample to draw reliable conclusions; these results should be interpreted as directional, not confirmatory.**

**Market structure narrative.** Lower volume, faster signal decay within the 2026 sample, and reduced persistence of depth signals — all point to a more algorithmically competitive market in 2026 relative to 2018. The VOI-return relationship itself appears non-stationary across years.

---

## 10. Limitations

- **OLS R² declines within each sample period.** In 2018 H1, R² falls from 0.0389 (first 58 days) to 0.0358 (last 58 days). In 2026 H1, R² falls from 0.0131 to 0.0093 within 59 days. The VOI–return relationship is non-stationary even within a single half-year period; a one-day training window cannot adapt quickly enough to this drift.

- **Trading volume is far below the Shen (2015) reference period.** The 2018 H1 sample averages 17,133 lots/day (8.6% of the paper's ~200,000 lots/day era). The 2026 H1 sample averages 59,747 lots/day (29.9% of the paper era). Fewer informative ticks per training day reduce OLS stability and mean that parameter choices calibrated to the paper period may not transfer.

- **Single-day OLS training window.** The model trains on one day and predicts the next. A single trading day may not be representative of the following day's microstructure regime. This limitation is compounded at contract rollover dates, where the incoming front-month contract begins with lower open interest and liquidity; the pipeline does not handle these discontinuities and treats rollover days identically to normal days.

- **Settlement-period volume drift and main-contract mislabelling.** As a contract approaches expiration, trading volume migrates from the near-month contract to the next-month contract — typically within the expiry day itself or one trading day before. The pipeline's main-contract selection rule (`prev_day_vol.idxmax()`) looks back exactly one calendar day and cannot detect intraday volume crossovers, so it systematically lags by one day at each rollover. In the 2018 H1 dataset (6 expiry cycles), this mislabels approximately 6 trading days (~5% of 124 days). On 4 of these 6 days, the mislabeled near-month contract underperforms the next-month contract; the estimated net PnL improvement from a correct rollover rule is +45,287 CNY across the 6 affected days (see Section 11 of notebook 05).

---

## 11. Conclusion

The VOI-based signal is statistically real: under paper cost assumptions that replicate Shen (2015), Strategy B achieves an annualised Sharpe of +7.32 and a highly significant t-statistic of 4.99 (p < 10⁻⁶) over 117 trading days in 2018 H1 (both strategies at k=5 ticks). Strategy A under the same paper costs is statistically indistinguishable from zero (Sharpe −0.229, t = −0.156, p = 0.56), confirming that OIR and MPB — not VOI alone — are the source of the exploitable signal. The signal does not survive current CFFEX transaction costs, where the same-day close-leg fee is ten times the open-leg fee; Strategy B Sharpe collapses to −38.26 and 98% of trading days produce a loss. B-CTT — which uses CostToTrade to dynamically raise the signal threshold during high-liquidity-cost periods — reduces trade count by 16% and modestly reduces max drawdown, but leaves Sharpe essentially unchanged under current costs; the per-trade cost burden is too large for trade filtering alone to overcome. Adding LDistance_diff or CostToTrade to the OLS feature matrix produces no durable improvement: LDistance_diff approximately doubles trade count (amplifying cost drag), and CostToTrade's OLS coefficient alternates sign nearly randomly (58 positive vs 59 negative days), confirming it carries no stable directional information in the signal layer. The 2026 comparison reveals a structurally more competitive market: average daily volume has risen to 59,747 lots/day, but OLS R² is approximately 70% lower than in 2018, signal standard deviation declines within the 59-day period, and LDistance_diff autocorrelation has fallen from 0.744 to 0.620 — all consistent with faster algorithmic quote updating. B+LD achieves a marginally positive OOS paper-cost Sharpe (+1.017 over 29 OOS days in 2026), but this is too small a sample to be conclusive and disappears entirely under current costs. The core constraint throughout this project is not signal quality but transaction costs: the VOI framework produces genuine predictive content, and that content is entirely absorbed by the current CFFEX fee structure before any net return reaches the trader.