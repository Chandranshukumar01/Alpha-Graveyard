"""
sim_residual.py — Strategy 24: Synthetic Residual Stat Arb

THE MATH:
- Regress BTC against a basket (ETH, SPY, Gold) using rolling OLS.
- The residual = BTC_actual - BTC_predicted.
- By construction, the residual is stationary (mean-reverting).
- Trade mean-reversion on the residual (Z-score > 2 = short, < -2 = long).

WHY DIFFERENT:
- We tested cointegration (Pairs, LINK/UNI) and it broke.
- Regression residuals are FORCED stationary by construction.
- We're not betting on direction. We're betting on convergence.
- This is how ACTUAL stat arb desks operate.

DATA: BTC, ETH, SPY, GLD from Yahoo Finance (free, 2y daily).
"""

import yfinance as yf
import pandas as pd
import numpy as np

def flatten_cols(df):
    new_cols = []
    for c in df.columns:
        new_cols.append(c[0].lower() if isinstance(c, tuple) else c.lower())
    df.columns = new_cols
    return df

def main():
    # Fetch all assets
    tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SPY": "SPY", "GLD": "GLD"}

    print("Fetching data for BTC, ETH, SPY, GLD (2y daily)...")
    data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        df = flatten_cols(df)
        data[name] = df['close']
        print(f"  {name}: {len(df)} days.")

    # Align
    prices = pd.DataFrame(data).dropna()
    print(f"Aligned: {len(prices)} common trading days.\n")

    # Normalize to returns for regression
    returns = prices.pct_change().dropna()

    # Rolling OLS: BTC ~ ETH + SPY + GLD
    window = 60  # 60-day rolling regression
    residuals = []
    predicted = []

    for i in range(window, len(returns)):
        y = returns['BTC'].iloc[i-window:i].values
        X = returns[['ETH', 'SPY', 'GLD']].iloc[i-window:i].values

        # Add intercept
        X_int = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_int, y, rcond=None)[0]

            # Predict today's BTC return
            x_today = returns[['ETH', 'SPY', 'GLD']].iloc[i].values
            x_today_int = np.concatenate([[1], x_today])
            pred = x_today_int @ beta

            resid = returns['BTC'].iloc[i] - pred
            residuals.append(resid)
            predicted.append(pred)
        except Exception:
            residuals.append(0)
            predicted.append(0)

    # Create residual series
    resid_series = pd.Series(residuals, index=returns.index[window:])

    # Z-score of residual
    z_window = 20
    z_mean = resid_series.rolling(z_window).mean()
    z_std = resid_series.rolling(z_window).std().replace(0, 0.0001)
    z_score = (resid_series - z_mean) / z_std

    # Simulation
    balance = 10000.0
    position = 0  # 1=long BTC, -1=short BTC
    entry_price = 0
    trades = []

    # Get BTC prices aligned with z-scores
    btc_prices = prices['BTC'].loc[z_score.dropna().index]
    z_clean = z_score.dropna()

    # Align indices
    common_idx = btc_prices.index.intersection(z_clean.index)
    btc_prices = btc_prices.loc[common_idx]
    z_clean = z_clean.loc[common_idx]

    bh_start = btc_prices.iloc[0]

    for i in range(1, len(common_idx)):
        z = z_clean.iloc[i]
        price = btc_prices.iloc[i]

        # Exit logic
        if position == 1 and z > -0.5:  # Long, z reverted to mean
            pnl = (price - entry_price) / entry_price - 0.002
            balance *= (1 + pnl)
            trades.append(pnl)
            position = 0
        elif position == -1 and z < 0.5:  # Short, z reverted
            pnl = (entry_price - price) / entry_price - 0.002
            balance *= (1 + pnl)
            trades.append(pnl)
            position = 0

        # Entry logic
        if position == 0:
            if z < -2:  # BTC undervalued vs basket → Long
                position = 1
                entry_price = price
            elif z > 2:  # BTC overvalued vs basket → Short
                position = -1
                entry_price = price

    # Close open
    if position != 0:
        final = btc_prices.iloc[-1]
        if position == 1:
            pnl = (final - entry_price) / entry_price - 0.002
        else:
            pnl = (entry_price - final) / entry_price - 0.002
        balance *= (1 + pnl)
        trades.append(pnl)

    bh_end = btc_prices.iloc[-1]
    bh_ret = (bh_end - bh_start) / bh_start * 100
    strat_ret = (balance - 10000) / 10000 * 100
    winners = [t for t in trades if t > 0]

    print(f"{'='*55}")
    print(f"  Strategy 24: Synthetic Residual Stat Arb")
    print(f"{'='*55}")
    print(f"  Regression: BTC ~ ETH + SPY + GLD (Rolling {window}d)")
    print(f"  Z-Score Window: {z_window} days")
    print(f"  Total Trades: {len(trades)}")
    if trades:
        print(f"  Win Rate: {len(winners)/len(trades)*100:.1f}%")
        print(f"  Avg PnL: {np.mean(trades)*100:.2f}%")
    print(f"  Strategy Return: {strat_ret:+.2f}%")
    print(f"  Buy & Hold (BTC): {bh_ret:+.2f}%")
    print(f"  Alpha: {strat_ret - bh_ret:+.2f}%")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
