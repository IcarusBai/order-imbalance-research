# Strategy Performance Summary: CFFEX IF Main Contract, 2018 H1 (117 Trading Days)

| Strategy | Cost Regime | Threshold q | Horizon k | Open Cost Rate | Close Cost Rate | Mean Daily PnL (CNY) | Ann. Sharpe | Win Rate / Trade | Mean Daily Trades | OLS R² | t-stat | p-value |
|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Zero cost | 0.20 | 20 | 0 | 0 | −1,250 | −0.73 | 36.6% | 225.6 | 0.025 | −0.50 | 0.690 |
| A | Paper cost | 0.20 | 20 | 2.5e-5 | 2.5e-5 | −14,479 | −6.80 | 35.3% | 225.6 | 0.025 | −4.63 | ≈1.00 |
| A | Current cost | 0.20 | 20 | 2.3e-5 | 2.3e-4 | −68,188 | −15.46 | 23.6% | 225.6 | 0.025 | −10.54 | ≈1.00 |
| **A** | **Paper cost (optimal q)** | **0.60** | **20** | **2.5e-5** | **2.5e-5** | **+1,600** | **+2.39** | **42.2%** | **12.9** | **0.025** | **+1.63** | **0.053** |
| B | Zero cost | 0.20 | 5 | 0 | 0 | +26,716 | +16.49 | 46.4% | 276.8 | 0.059 | +11.24 | <0.001 |
| B | Paper cost | 0.20 | 5 | 2.5e-5 | 2.5e-5 | +10,512 | +7.32 | 44.7% | 276.8 | 0.059 | +4.99 | <0.001 |
| B | Current cost | 0.20 | 5 | 2.3e-5 | 2.3e-4 | −55,273 | −38.26 | 27.0% | 276.8 | 0.059 | −26.07 | ≈1.00 |
| **B** | **Paper cost (optimal q)** | **0.23** | **5** | **2.5e-5** | **2.5e-5** | **+11,294** | **+7.96** | **45.4%** | **205.0** | **0.059** | **+5.42** | **<0.001** |

**Notes:**
- **Strategy A:** features are 5 lags of VOI (Volume Order Imbalance); k=20 predicts mid-price change 20 ticks (~10 seconds) ahead.
- **Strategy B:** features are 5 lags each of VOI, OIR, and MPB (all spread-normalised); k=5 predicts mid-price change 5 ticks (~2.5 seconds) ahead.
- **Threshold q:** a trade is entered when the EFPC signal exceeds ±q; higher q means fewer, higher-conviction trades.
- **Optimal q rows:** q chosen by parameter sweep to maximise mean daily PnL under paper cost.
- **Current cost — why unprofitable at any q:** the same-day close-leg fee (2.3e-4) is ~10× the open-leg fee (2.3e-5). Total transaction costs exceed the alpha captured by the signal at all trade frequencies.
- **Paper cost:** CFFEX fee schedule circa 2014 (Shen 2015 era) — symmetric open/close rates.
- **Current cost:** CFFEX IF fee schedule post-2023 — punitive same-day close rate applies to all intraday round-trips.
- **Reference:** Shen (2015), *Order Imbalance Based Strategy in High Frequency Trading*
