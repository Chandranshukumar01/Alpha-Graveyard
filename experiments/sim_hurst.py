"""
sim_hurst.py — Strategy 23: Hurst Exponent Regime Switching

THE MATH:
- Hurst Exponent (H) measures long-term memory in a time series.
  - H > 0.5: Trending (momentum works)
  - H < 0.5: Mean-reverting (fade moves)
  - H = 0.5: Random walk (don't trade)
- We use Rescaled Range (R/S) analysis to estimate H.

THE STRATEGY:
- Rolling 100-bar Hurst on Daily candles.
- If H > 0.6: Buy breakouts (SMA20 crossover).
- If H < 0.4: Sell rallies (RSI > 70 = short, RSI < 30 = long).
- If 0.4 <= H <= 0.6: Do nothing (random walk).

WHY DIFFERENT:
- This is fractal mathematics, not indicator math.
- It detects the regime FIRST, then picks the strategy.
- Previous regime detection (ADX, Vol Switch) used ad-hoc thresholds.
  Hurst has a mathematical basis.
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

def hurst_rs(series, max_lag=20):
    """Calculate Hurst exponent via Rescaled Range (R/S) analysis."""
    n = len(series)
    if n < max_lag * 2:
        return 0.5  # Not enough data, assume random walk

    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split series into chunks of size 'lag'
        chunks = [series[i:i+lag] for i in range(0, n - lag + 1, lag)]
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < lag:
                continue
            mean_c = np.mean(chunk)
            deviations = chunk - mean_c
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_values.append((np.log(lag), np.log(np.mean(rs_chunk))))

    if len(rs_values) < 3:
        return 0.5

    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])

    # Linear regression: log(R/S) = H * log(n) + c
    slope, _ = np.polyfit(x, y, 1)
    return np.clip(slope, 0, 1)

def main():
    print("Fetching BTC-USD Daily (5 years)...")
    df = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
    df = flatten_cols(df)
    print(f"Loaded {len(df)} daily candles.")

    # Features
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.0001)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Rolling Hurst (100-bar window)
    print("Calculating rolling Hurst exponents (this may take a moment)...")
    window = 100
    hurst_vals = [0.5] * window
    returns_arr = df['returns'].values

    for i in range(window, len(df)):
        chunk = returns_arr[i-window:i]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) >= 40:
            h = hurst_rs(chunk, max_lag=20)
        else:
            h = 0.5
        hurst_vals.append(h)

    df['hurst'] = hurst_vals[:len(df)]
    df = df.dropna()

    # Simulation
    balance = 10000.0
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0
    trades = []
    bh_start = df.iloc[0]['close']

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        h = row['hurst']

        # Exit logic first
        if position == 1:  # Long
            # Exit: Price crosses below SMA20 (momentum exit)
            if price < row['sma_20']:
                pnl = (price - entry_price) / entry_price - 0.002
                balance *= (1 + pnl)
                trades.append(pnl)
                position = 0
        elif position == -1:  # Short
            # Exit: RSI < 50 (mean reversion exit)
            if row['rsi'] < 50:
                pnl = (entry_price - price) / entry_price - 0.002
                balance *= (1 + pnl)
                trades.append(pnl)
                position = 0

        # Entry logic
        if position == 0:
            if h > 0.6:
                # TRENDING regime: Buy momentum breakout
                if prev['close'] < prev['sma_20'] and price > row['sma_20']:
                    position = 1
                    entry_price = price
            elif h < 0.4:
                # MEAN-REVERTING regime: Fade extremes
                if row['rsi'] > 70:
                    position = -1
                    entry_price = price
                elif row['rsi'] < 30:
                    position = 1
                    entry_price = price

    # Close open position
    if position != 0:
        final = df.iloc[-1]['close']
        if position == 1:
            pnl = (final - entry_price) / entry_price - 0.002
        else:
            pnl = (entry_price - final) / entry_price - 0.002
        balance *= (1 + pnl)
        trades.append(pnl)

    bh_end = df.iloc[-1]['close']
    bh_ret = (bh_end - bh_start) / bh_start * 100
    strat_ret = (balance - 10000) / 10000 * 100
    winners = [t for t in trades if t > 0]

    print(f"\n{'='*55}")
    print(f"  Strategy 23: Hurst Exponent Regime Switching")
    print(f"{'='*55}")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Avg Hurst: {df['hurst'].mean():.3f}")
    print(f"  % Trending (H>0.6): {(df['hurst']>0.6).mean()*100:.1f}%")
    print(f"  % Mean-Rev (H<0.4): {(df['hurst']<0.4).mean()*100:.1f}%")
    print(f"  % Random Walk: {((df['hurst']>=0.4)&(df['hurst']<=0.6)).mean()*100:.1f}%")
    print(f"  Total Trades: {len(trades)}")
    if trades:
        print(f"  Win Rate: {len(winners)/len(trades)*100:.1f}%")
        print(f"  Avg PnL: {np.mean(trades)*100:.2f}%")
    print(f"  Strategy Return: {strat_ret:+.2f}%")
    print(f"  Buy & Hold Return: {bh_ret:+.2f}%")
    print(f"  Alpha: {strat_ret - bh_ret:+.2f}%")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
