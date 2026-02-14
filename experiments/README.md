# Experiments Index: 25 Falsified Strategies

This directory contains standalone simulation scripts for the 25 strategies tested in this study. Each script is self-contained and demonstrates the failure mode of a specific quantitative hypothesis.

## 🏁 The Kill Shot: The Inverse Test
- **[sim_inverse.py](sim_inverse.py)** (S-25): The definitive proof. When both a strategy and its exact mathematical inverse lose money on the same data, the signal contains **zero directional information**.

---

## 🏗️ Structural & Institutional Alpha (Falsified)
- **[sim_cme_gap.py](sim_cme_gap.py)** (S-18): CME Futures weekend gap-fill strategy. Results: -18.22% return. *Verdict: Discovered and front-run by institutional HFTs long ago.*
- **[sim_wick.py](sim_wick.py)** (S-19): Wick microstructure/liquidation rejection. Results: -48.44% return. *Verdict: Buying "forced selling" wicks in a trend-following market is fatal.*
- **[sim_grid.py](sim_grid.py)** (S-16): Geometric grid trading. *Verdict: Small gains wiped out by inevitable trend-extension (ruin).*

## 📈 Higher Timeframe & Persistence (Falsified)
- **[sim_daily.py](sim_daily.py)** (S-20): Daily SMA20 Momentum with Volatility Filter. Results: +36% alpha (2y).
- **[sim_daily_stress.py](sim_daily_stress.py)**: Multi-asset stress test of S-20. *Verdict: Fails on 5y BTC history (-73% alpha). Performance on 2y was a period-specific artifact.*
- **[sim_daily_verify.py](sim_daily_verify.py)**: Walk-forward validation of S-20.
- **[sim_weekly_mr.py](sim_weekly_mr.py)** (S-22): Weekly Bollinger Mean Reversion. *Verdict: Inconsistent across assets; missed massive BTC upside.*

## 🔢 Mathematical & Statistical (Falsified)
- **[sim_hurst.py](sim_hurst.py)** (S-23): Fractal Hurst Exponent regime switching. *Verdict: BTC is too persistently trending (H=0.74) for regime-switching to add value.*
- **[sim_residual.py](sim_residual.py)** (S-24): Synthetic Residual Stat Arb (BTC vs ETH/SPY/GLD). Results: +41% alpha (2y).
- **[sim_residual_verify.py](sim_residual_verify.py)**: 5y stress test of S-24. *Verdict: Fails on 5y history. Alpha collapses to -72%.*
- **[sim_cross_sectional.py](sim_cross_sectional.py)** (S-21): Long/Short momentum across 8 crypto assets. *Verdict: No relative alpha; crypto correlations are too tight.*

## 🤖 Agentic & Sentiment (Falsified)
- **[sim_lighthouse.py](sim_lighthouse.py)** (S-12): Multi-Agent LLM (GPT-4) sentiment & structure consensus. *Verdict: Systematically wrong. LLMs hallucinate signal in pure noise.*
- **[sim_sentiment.py](sim_sentiment.py)** (S-17): Volume/Volatility proxy for social sentiment. *Verdict: Insufficient information to overcome fee drag.*

---


