"""
sim_daily.py - Strategy 20: Daily Timeframe Momentum

THE CORE ARGUMENT:
- Every 1H strategy lost because fees (0.1%) ate the edge (0.05-0.2% per trade).
- On DAILY candles, average move is 2-5%. Fee drag becomes ~2-5% of move, not 50-100%.
- Fewer trades = less friction.

Strategy:
- Simple 20-day SMA crossover (the most basic momentum).
- Vol filter: Only trade when 10d vol > median (avoid choppy regimes).
- Hold for days/weeks, not hours.

Data: 2 years of Daily BTC-USD from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np

def run():
    print("Fetching Daily BTC-USD (2 years)...")
    df = yf.download("BTC-USD", period="2y", interval="1d", progress=False)
    
    # Flatten MultiIndex columns
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            new_cols.append(c[0].lower())
        else:
            new_cols.append(c.lower())
    df.columns = new_cols
    
    print(f"Loaded {len(df)} daily candles.")
    
    # --- Features ---
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['returns'] = df['close'].pct_change()
    df['vol_10d'] = df['returns'].rolling(10).std()
    df['vol_median'] = df['vol_10d'].rolling(60).median()
    df['high_vol'] = df['vol_10d'] > df['vol_median']
    
    df = df.dropna()
    
    # --- Simulation ---
    balance = 10000.0
    position = 0  # 1=Long, 0=Flat
    entry_price = 0
    trades = []
    
    # Buy & Hold benchmark
    bh_start = df.iloc[0]['close']
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        
        # LONG signal: Price crosses above SMA20 + High Vol regime
        if position == 0:
            if prev['close'] < prev['sma_20'] and price > row['sma_20'] and row['high_vol']:
                position = 1
                entry_price = price
                balance *= (1 - 0.001)  # 0.1% fee
                trades.append({'type': 'BUY', 'price': price, 'date': df.index[i]})
        
        # EXIT signal: Price crosses below SMA20
        elif position == 1:
            if price < row['sma_20']:
                pnl = (price - entry_price) / entry_price
                balance *= (1 + pnl)
                balance *= (1 - 0.001)  # 0.1% fee
                trades.append({'type': 'SELL', 'price': price, 'pnl': pnl, 'date': df.index[i]})
                position = 0
    
    # Close open position
    if position == 1:
        final_price = df.iloc[-1]['close']
        pnl = (final_price - entry_price) / entry_price
        balance *= (1 + pnl)
        balance *= (1 - 0.001)
        trades.append({'type': 'SELL', 'price': final_price, 'pnl': pnl, 'date': df.index[-1]})
    
    # Buy & Hold
    bh_end = df.iloc[-1]['close']
    bh_return = (bh_end - bh_start) / bh_start * 100
    
    # Strategy
    strat_return = (balance - 10000) / 10000 * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winners = [t for t in sell_trades if t['pnl'] > 0]
    
    print("\n" + "=" * 50)
    print("  Strategy 20: Daily Momentum (SMA20 + Vol Filter)")
    print("=" * 50)
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Total Trades: {len(sell_trades)}")
    if sell_trades:
        print(f"  Win Rate: {len(winners)/len(sell_trades)*100:.1f}%")
        print(f"  Avg PnL/Trade: {np.mean([t['pnl'] for t in sell_trades])*100:.2f}%")
    print(f"  Strategy Return: {strat_return:+.2f}%")
    print(f"  Buy & Hold Return: {bh_return:+.2f}%")
    print(f"  Alpha (vs B&H): {strat_return - bh_return:+.2f}%")
    print("=" * 50)
    
    if strat_return > bh_return:
        print("  RESULT: Strategy OUTPERFORMED Buy & Hold.")
    else:
        print("  RESULT: Strategy UNDERPERFORMED Buy & Hold.")

if __name__ == "__main__":
    run()
