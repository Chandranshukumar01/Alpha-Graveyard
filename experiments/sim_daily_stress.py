"""
sim_daily_stress.py - Stress Test Strategy 20 Across Assets & History

Tests:
1. BTC 5-year daily (longer history)
2. ETH 5-year daily (different asset)
3. SOL 2-year daily (newer, higher-vol asset)
4. SPY 5-year daily (equity benchmark - does momentum work everywhere?)
"""

import yfinance as yf
import pandas as pd
import numpy as np

def flatten_cols(df):
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            new_cols.append(c[0].lower())
        else:
            new_cols.append(c.lower())
    df.columns = new_cols
    return df

def run_momentum(df, label):
    """SMA20 crossover + vol filter on daily data."""
    df = df.copy()
    df = flatten_cols(df)
    
    df['sma_20'] = df['close'].rolling(20).mean()
    df['returns'] = df['close'].pct_change()
    df['vol_10d'] = df['returns'].rolling(10).std()
    df['vol_median'] = df['vol_10d'].rolling(60).median()
    df['high_vol'] = df['vol_10d'] > df['vol_median']
    df = df.dropna()
    
    if len(df) < 80:
        print(f"  [{label}] Insufficient data ({len(df)} rows). Skipping.")
        return
    
    balance = 10000.0
    position = 0
    entry_price = 0
    trade_count = 0
    winners = 0
    bh_start = df.iloc[0]['close']
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        
        if position == 0:
            if prev['close'] < prev['sma_20'] and price > row['sma_20'] and row['high_vol']:
                position = 1
                entry_price = price
                balance *= 0.999
        elif position == 1:
            if price < row['sma_20']:
                pnl = (price - entry_price) / entry_price
                balance *= (1 + pnl) * 0.999
                trade_count += 1
                if pnl > 0: winners += 1
                position = 0
    
    if position == 1:
        final = df.iloc[-1]['close']
        pnl = (final - entry_price) / entry_price
        balance *= (1 + pnl) * 0.999
        trade_count += 1
        if pnl > 0: winners += 1
    
    bh_end = df.iloc[-1]['close']
    bh_ret = (bh_end - bh_start) / bh_start * 100
    strat_ret = (balance - 10000) / 10000 * 100
    wr = (winners / trade_count * 100) if trade_count > 0 else 0
    
    print(f"  [{label}] ({len(df)} days)")
    print(f"    Trades: {trade_count} | Win Rate: {wr:.0f}%")
    print(f"    Strategy: {strat_ret:+.2f}% | B&H: {bh_ret:+.2f}% | Alpha: {strat_ret-bh_ret:+.2f}%")
    return strat_ret, bh_ret

def main():
    assets = {
        "BTC-USD (5y)": ("BTC-USD", "5y"),
        "ETH-USD (5y)": ("ETH-USD", "5y"),
        "SOL-USD (2y)": ("SOL-USD", "2y"),
        "SPY (5y)":     ("SPY", "5y"),
    }
    
    print("=" * 55)
    print("  STRESS TEST: Strategy 20 Across Assets & History")
    print("=" * 55)
    
    for label, (ticker, period) in assets.items():
        print(f"\nFetching {ticker} ({period})...")
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if len(df) == 0:
                print(f"  No data for {ticker}.")
                continue
            run_momentum(df, label)
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
    
    print("\n" + "=" * 55)
    print("  CONCLUSION")
    print("=" * 55)

if __name__ == "__main__":
    main()
