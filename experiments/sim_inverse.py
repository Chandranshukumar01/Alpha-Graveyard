"""
sim_inverse.py — Strategy 25: The Inverse Test (The Final Proof)

THE ARGUMENT:
- If Strategy 20 (SMA20 momentum) LOSES on 5y BTC...
- ...and its INVERSE also loses...
- ...then the signal contains ZERO information.
- It's not "wrong"—it's NOISE.

This is the mathematical proof that EMH holds:
- You can't make money following the signal.
- You can't make money fading the signal.
- Fees eat both sides. QED.

TEST:
1. Strategy 20 (Normal): Long above SMA20, flat below.
2. Strategy 25 (Inverse): SHORT above SMA20, flat below.
3. Strategy 25b (Full Inverse): SHORT above SMA20, LONG below SMA20.
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

def run_sma_strategy(df, label, mode="normal"):
    """
    mode='normal': Long above SMA20, flat below (Strategy 20).
    mode='inverse': Short above SMA20, flat below.
    mode='full_inverse': Short above SMA20, Long below SMA20.
    """
    df = df.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['returns'] = df['close'].pct_change()
    df['vol_10d'] = df['returns'].rolling(10).std()
    df['vol_median'] = df['vol_10d'].rolling(60).median()
    df['high_vol'] = df['vol_10d'] > df['vol_median']
    df = df.dropna()

    balance = 10000.0
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0
    trades = []
    bh_start = df.iloc[0]['close']

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']

        if mode == "normal":
            # LONG when crosses above SMA20
            if position == 0:
                if prev['close'] < prev['sma_20'] and price > row['sma_20'] and row['high_vol']:
                    position = 1
                    entry_price = price
                    balance *= 0.999
            elif position == 1:
                if price < row['sma_20']:
                    pnl = (price - entry_price) / entry_price
                    balance *= (1 + pnl) * 0.999
                    trades.append(pnl)
                    position = 0

        elif mode == "inverse":
            # SHORT when crosses above SMA20 (opposite)
            if position == 0:
                if prev['close'] < prev['sma_20'] and price > row['sma_20'] and row['high_vol']:
                    position = -1
                    entry_price = price
                    balance *= 0.999
            elif position == -1:
                if price < row['sma_20']:
                    pnl = (entry_price - price) / entry_price  # Short profit
                    balance *= (1 + pnl) * 0.999
                    trades.append(pnl)
                    position = 0

        elif mode == "full_inverse":
            # SHORT above SMA20, LONG below
            if position == 0:
                if price > row['sma_20'] and row['high_vol']:
                    position = -1
                    entry_price = price
                    balance *= 0.999
                elif price < row['sma_20'] and row['high_vol']:
                    position = 1
                    entry_price = price
                    balance *= 0.999
            elif position == -1:
                if price < row['sma_20']:
                    pnl = (entry_price - price) / entry_price
                    balance *= (1 + pnl) * 0.999
                    trades.append(pnl)
                    position = 0
            elif position == 1:
                if price > row['sma_20']:
                    pnl = (price - entry_price) / entry_price
                    balance *= (1 + pnl) * 0.999
                    trades.append(pnl)
                    position = 0

    # Close open
    if position != 0:
        final = df.iloc[-1]['close']
        if position == 1:
            pnl = (final - entry_price) / entry_price
        else:
            pnl = (entry_price - final) / entry_price
        balance *= (1 + pnl) * 0.999
        trades.append(pnl)

    bh_end = df.iloc[-1]['close']
    bh_ret = (bh_end - bh_start) / bh_start * 100
    strat_ret = (balance - 10000) / 10000 * 100
    winners = [t for t in trades if t > 0]

    print(f"  [{label}]")
    print(f"    Trades: {len(trades)} | Win: {len(winners)}/{len(trades)}")
    print(f"    Strategy: {strat_ret:+.2f}% | B&H: {bh_ret:+.2f}% | Alpha: {strat_ret-bh_ret:+.2f}%")
    return strat_ret

def main():
    print("Fetching BTC-USD Daily (5 years)...")
    df = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
    df = flatten_cols(df)
    print(f"Loaded {len(df)} days.\n")

    print("=" * 55)
    print("  THE INVERSE TEST: Does Fading Our Signal Work?")
    print("=" * 55)

    print("\n--- Strategy 20 (Normal: Long above SMA20) ---")
    normal = run_sma_strategy(df, "S-20 Normal", mode="normal")

    print("\n--- Strategy 25 (Inverse: Short above SMA20) ---")
    inverse = run_sma_strategy(df, "S-25 Inverse", mode="inverse")

    print("\n--- Strategy 25b (Full Inverse: Short above, Long below) ---")
    full_inv = run_sma_strategy(df, "S-25b Full Inverse", mode="full_inverse")

    print(f"\n{'='*55}")
    print(f"  THE VERDICT")
    print(f"{'='*55}")

    if normal < 0 and inverse < 0:
        print("  BOTH Normal AND Inverse LOSE.")
        print("  → The signal has ZERO information content.")
        print("  → Fees destroy both sides.")
        print("  → The market is EFFICIENT. QED.")
    elif inverse > 0:
        print("  The INVERSE works!")
        print("  → The original signal is systematically wrong.")
        print("  → There IS information—you just need to reverse it.")
    else:
        print("  Normal wins, Inverse loses.")
        print("  → Standard momentum. Signal has directional value.")

    print(f"\n  Normal:  {normal:+.2f}%")
    print(f"  Inverse: {inverse:+.2f}%")
    print(f"  Full:    {full_inv:+.2f}%")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
