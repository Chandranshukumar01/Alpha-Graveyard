"""
sim_daily_verify.py - Walk-Forward Verification of Strategy 20

Strategy 20 showed +41.53% vs +4.99% Buy & Hold.
This is suspicious. Let's verify it isn't curve-fitting.

Tests:
1. Split data: First half (In-Sample) vs Second half (Out-of-Sample).
2. No Vol Filter: Does the edge survive without the filter?
3. Random Entry Benchmark: Does random daily timing beat this?
"""

import yfinance as yf
import pandas as pd
import numpy as np

def run_strategy(df, label, use_vol_filter=True):
    """Run SMA20 momentum on a given dataframe slice."""
    df = df.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['returns'] = df['close'].pct_change()
    df['vol_10d'] = df['returns'].rolling(10).std()
    df['vol_median'] = df['vol_10d'].rolling(60).median()
    df['high_vol'] = df['vol_10d'] > df['vol_median']
    df = df.dropna()
    
    if len(df) < 30:
        print(f"  [{label}] Not enough data ({len(df)} rows).")
        return None
    
    balance = 10000.0
    position = 0
    entry_price = 0
    trades = []
    bh_start = df.iloc[0]['close']
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        
        if position == 0:
            vol_ok = row['high_vol'] if use_vol_filter else True
            if prev['close'] < prev['sma_20'] and price > row['sma_20'] and vol_ok:
                position = 1
                entry_price = price
                balance *= (1 - 0.001)
                trades.append({'type': 'BUY', 'price': price})
        
        elif position == 1:
            if price < row['sma_20']:
                pnl = (price - entry_price) / entry_price
                balance *= (1 + pnl) * (1 - 0.001)
                trades.append({'type': 'SELL', 'pnl': pnl})
                position = 0
    
    # Close open position
    if position == 1:
        final_price = df.iloc[-1]['close']
        pnl = (final_price - entry_price) / entry_price
        balance *= (1 + pnl) * (1 - 0.001)
        trades.append({'type': 'SELL', 'pnl': pnl})
    
    bh_end = df.iloc[-1]['close']
    bh_return = (bh_end - bh_start) / bh_start * 100
    strat_return = (balance - 10000) / 10000 * 100
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winners = [t for t in sell_trades if t['pnl'] > 0]
    
    print(f"  [{label}]")
    print(f"    Trades: {len(sell_trades)}")
    if sell_trades:
        print(f"    Win Rate: {len(winners)/len(sell_trades)*100:.1f}%")
    print(f"    Strategy: {strat_return:+.2f}%")
    print(f"    Buy&Hold: {bh_return:+.2f}%")
    print(f"    Alpha:    {strat_return - bh_return:+.2f}%")
    
    return strat_return, bh_return

def random_benchmark(df, n_sims=1000):
    """Random entry/exit benchmark. How often does random beat B&H?"""
    df = df.copy()
    df = df.dropna()
    prices = df['close'].values
    bh = (prices[-1] - prices[0]) / prices[0] * 100
    
    beats = 0
    for _ in range(n_sims):
        bal = 10000.0
        pos = 0
        ep = 0
        for i in range(len(prices)):
            if pos == 0 and np.random.random() < 0.05:  # 5% chance enter
                pos = 1
                ep = prices[i]
                bal *= 0.999
            elif pos == 1 and np.random.random() < 0.1:  # 10% chance exit
                pnl = (prices[i] - ep) / ep
                bal *= (1 + pnl) * 0.999
                pos = 0
        if pos == 1:
            pnl = (prices[-1] - ep) / ep
            bal *= (1 + pnl) * 0.999
        ret = (bal - 10000) / 10000 * 100
        if ret > bh:
            beats += 1
    
    print(f"  [Random Benchmark] {beats}/{n_sims} random strategies beat B&H ({beats/n_sims*100:.1f}%)")

def main():
    print("Fetching Daily BTC-USD (2 years)...")
    df = yf.download("BTC-USD", period="2y", interval="1d", progress=False)
    
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            new_cols.append(c[0].lower())
        else:
            new_cols.append(c.lower())
    df.columns = new_cols
    
    print(f"Total: {len(df)} candles.\n")
    
    mid = len(df) // 2
    first_half = df.iloc[:mid]
    second_half = df.iloc[mid:]
    
    print("=" * 50)
    print("  WALK-FORWARD VERIFICATION")
    print("=" * 50)
    
    print("\n--- Test 1: Full Period (With Vol Filter) ---")
    run_strategy(df, "Full+VolFilter", use_vol_filter=True)
    
    print("\n--- Test 2: Full Period (NO Vol Filter) ---")
    run_strategy(df, "Full-NoFilter", use_vol_filter=False)
    
    print("\n--- Test 3: In-Sample (First Half, With Filter) ---")
    run_strategy(first_half, "IS+Filter", use_vol_filter=True)
    
    print("\n--- Test 4: Out-of-Sample (Second Half, With Filter) ---")
    run_strategy(second_half, "OOS+Filter", use_vol_filter=True)
    
    print("\n--- Test 5: Out-of-Sample (NO Filter) ---")
    run_strategy(second_half, "OOS-NoFilter", use_vol_filter=False)
    
    print("\n--- Test 6: Random Entry Benchmark (1000 sims) ---")
    random_benchmark(df)
    
    print("\n" + "=" * 50)
    print("  VERDICT")
    print("=" * 50)

if __name__ == "__main__":
    main()
