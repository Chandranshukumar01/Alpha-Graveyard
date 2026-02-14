"""
sim_wick.py - Strategy 19: Wick Microstructure ("The Pinbar")

Hypothesis:
- A candle with a long lower wick relative to its body indicates buyer aggression / seller exhaustion.
- "Lower Wick / Total Range > 0.6" = Strong Support Test.
- User Context: "Long wicks = rejection. Body/wick ratios = aggression."

Signal:
- LONG if:
  1. Lower Wick Ratio > 0.6
  2. Volume > 20-period SMA (Validation)
  3. Close near 20-period Low (Buying the dip)
- EXIT:
  - 4 hours later (Time-based exit as per user request "1-4H hold")
  - Or Stop Loss 2%

Data:
- 5 Years of 1H BTC/USDT (btc_1h.csv)
"""

import pandas as pd
import numpy as np

import yfinance as yf

def run_simulation():
    # Load Data (Yahoo Finance for free 2y data)
    print("Fetching BTC-USD from Yahoo Finance...")
    try:
        df = yf.download("BTC-USD", period="2y", interval="1h", progress=False)
        if len(df) == 0:
            print("No data fetched.")
            return

        # Flatten MultiIndex columns if necessary
        # yfinance often returns columns like ('Close', 'BTC-USD')
        # We want just 'close'
        
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                new_cols.append(c[0].lower())
            else:
                new_cols.append(c.lower())
        df.columns = new_cols
        
        # If 'adj close' exists alongside 'close', keep 'close' or prefer 'adj close'?
        # crypto adjusted close is usually same as close.
        # Ensure we have ohlcv at least
        
        print(f"Loaded {len(df)} candles.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # --- Feature Engineering ---
    
    # 1. Candle Microstructure
    df['total_range'] = df['high'] - df['low']
    # Avoid div by zero
    df['total_range'] = df['total_range'].replace(0, 0.00001)
    
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['lower_wick_ratio'] = df['lower_wick'] / df['total_range']
    
    # 2. Volume Verification
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    # 3. Trend Context (Buying the Dip)
    df['low_20'] = df['low'].rolling(20).min()
    
    # Signal Logic
    df['signal'] = 0
    
    # LONG Condition
    # Wick > 0.6 AND Vol > MA AND Low is near 20-period low (within 1%)
    cond_wick = df['lower_wick_ratio'] > 0.6
    cond_vol = df['volume'] > df['vol_ma']
    cond_dip = df['low'] <= df['low_20'] * 1.01
    
    df.loc[cond_wick & cond_vol & cond_dip, 'signal'] = 1
    
    # --- Backtest Loop ---
    balance = 10000.0
    position = 0
    entry_price = 0
    entry_idx = 0
    trades = []
    
    # Iterate
    # Vectorized check for signal counts first
    print(f"Potential Signals: {df['signal'].sum()}")
    
    for i in range(20, len(df)):
        row = df.iloc[i]
        curr_price = row['close']
        ts = df.index[i]
        
        # Check Exit first
        if position == 1:
            # Time-based exit (4 hours)
            if i - entry_idx >= 4:
                pnl = (curr_price - entry_price) / entry_price
                pnl -= 0.002 # Fees
                balance *= (1 + pnl)
                trades.append(pnl)
                position = 0
                continue
                
            # Stop Loss (2%)
            if curr_price < entry_price * 0.98:
                pnl = (curr_price - entry_price) / entry_price
                pnl -= 0.002
                balance *= (1 + pnl)
                trades.append(pnl)
                position = 0
                continue
        
        # Check Entry
        if position == 0 and row['signal'] == 1:
            position = 1
            entry_price = curr_price
            entry_idx = i
            # Fee paid on exit simplification or here
            # We pay entry fee technically
            
    # Final Stats
    if not trades:
        print("No trades triggered.")
        return
        
    cum_ret = (balance - 10000) / 10000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
    
    print("\n" + "="*40)
    print("Strategy 19: Wick Microstructure Results")
    print("="*40)
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Cumulative Return: {cum_ret:.2f}%")
    print(f"Avg PnL: {np.mean(trades)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_simulation()
