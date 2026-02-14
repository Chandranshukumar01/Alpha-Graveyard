"""
sim_cme_gap.py - Strategy 18: CME Futures Gap Fill

Hypothesis:
- CME Bitcoin Futures (BTC=F) close Friday 16:00 CST and open Sunday 17:00 CST.
- Spot Bitcoin trades 24/7.
- A "Gap" appears if Spot moves significantly during the weekend.
- Theory: The market tends to "fill the gap" (price returns to Friday's close) when CME re-opens.

Data:
- CME Futures (BTC=F) from Yahoo Finance (free).
- Spot BTC (BTC/USDT) from Binance (free).

Logic:
1. Identify CME Close (Friday) and CME Open (Sunday).
2. Calculate Gap % = (Sunday Open - Friday Close) / Friday Close.
3. If Gap > 1.0% (significant):
   - Gap UP: Short Spot (Target = Friday Close).
   - Gap DOWN: Long Spot (Target = Friday Close).
4. Hold max 48 hours or until filled.
"""

import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
from datetime import timedelta

def fetch_data():
    print("Fetching CME Futures data (BTC=F)...")
    try:
        cme = yf.download("BTC=F", period="2y", interval="1h", progress=False)
        if len(cme) == 0:
            print("ERROR: No data fetched from Yahoo Finance.")
            return None, None
            
        cme['weekday'] = cme.index.dayofweek
        cme['hour'] = cme.index.hour
        return cme, None
    except Exception as e:
        print(f"Error fetching CME data: {e}")
        return None, None

def run_simulation():
    # Fetch CME Data
    cme, _ = fetch_data()
    if cme is None: return

    print(f"Loaded {len(cme)} CME candles.")
    
    # Identify Gaps
    # Friday Close is usually the last candle before a break > 24h
    # Or simplified: Look for timestamps where delta > 1 day
    
    cme['dt'] = cme.index
    cme['dt_delta'] = cme['dt'].diff()
    
    # Gaps are where time difference > 24 hours (weekend)
    gaps = cme[cme['dt_delta'] > timedelta(hours=24)]
    
    print(f"Found {len(gaps)} Weekend Gaps in 2 years.")
    
    trades = []
    balance = 10000.0
    
    for i in range(len(gaps)):
        sunday_candle = gaps.iloc[i]
        sunday_idx = gaps.index[i]
        
        # Find the Friday candle (previous row in original dataframe)
        # We need the index integer location
        
        # Using searchsorted / get_loc might be complex with irregular index
        # Let's iterate raw logic for simplicity or use 'shift'
        pass

    # SIMPLIFIED LOGIC for Robustness
    # 1. Iterate through CME dataframe
    # 2. If time_diff > 30 hours (Weekend)
    # 3. Gap = Open (Sun) - Close (Fri)
    # 4. Check if Gap was filled in subsequent Spot data (we don't have Spot loaded yet)
    
    # To properly Sim, we need Spot data covering the week following the gap.
    # We will fetch Spot BTC 1h for the same period.
    
    print("\nFetching Spot BTC/USDT for verification...")
    exchange = ccxt.binance()
    # CCXT fetch_ohlcv limit is small 1000. 
    # For 2 years we need loop. 
    # For SIMPLICITY validation: Let's use Spot data from yfinance too (BTC-USD)
    spot = yf.download("BTC-USD", period="2y", interval="1h", progress=False)
    print(f"Loaded {len(spot)} Spot candles.")
    
    gap_stats = []
    
    for idx, row in gaps.iterrows():
        # Sunday Open Time
        sun_time = idx
        sun_open_price = float(row['Open'].iloc[0]) if isinstance(row['Open'], pd.Series) else float(row['Open'])
        
        # Friday Close Time (approx 48h prior)
        # Get the row before this gap in the original CME df
        # We can find it by finding the location of 'idx'
        loc = cme.index.get_loc(idx)
        if loc == 0: continue
        
        fri_row = cme.iloc[loc-1]
        fri_close_price = float(fri_row['Close'].iloc[0]) if isinstance(fri_row['Close'], pd.Series) else float(fri_row['Close'])
        
        gap_pct = (sun_open_price - fri_close_price) / fri_close_price
        
        # Trade Logic
        # Gap > 1.5% ?
        if abs(gap_pct) < 0.015:
            continue
            
        direction = -1 if gap_pct > 0 else 1 # Fade the gap
        target = fri_close_price
        
        # Look forward in SPOT data for fill
        # Start looking from sun_time
        future_spot = spot[spot.index >= sun_time].head(48) # 48 hour limit
        
        filled = False
        pnl = 0
        
        for f_idx, f_row in future_spot.iterrows():
            high = float(f_row['High'].iloc[0]) if isinstance(f_row['High'], pd.Series) else float(f_row['High'])
            low = float(f_row['Low'].iloc[0]) if isinstance(f_row['Low'], pd.Series) else float(f_row['Low'])
            close = float(f_row['Close'].iloc[0]) if isinstance(f_row['Close'], pd.Series) else float(f_row['Close'])
            
            # Did we hit target?
            hit_target = False
            if direction == 1: # Long, Target is higher (Fri Close)
                if high >= target: hit_target = True
            else: # Short, Target is lower
                if low <= target: hit_target = True
                
            if hit_target:
                filled = True
                # PnL = Gap Size - Fees
                # Roughly: we captured the move from Sun Open to Fri Close
                gross_pnl = abs(sun_open_price - target) / sun_open_price
                pnl = gross_pnl - 0.002 # Fees
                break
                
        if not filled:
            # Closed at 48h mark
            exit_price = float(future_spot.iloc[-1]['Close'].iloc[0]) if isinstance(future_spot.iloc[-1]['Close'], pd.Series) else float(future_spot.iloc[-1]['Close'])
            if direction == 1:
                pnl = (exit_price - sun_open_price) / sun_open_price
            else:
                pnl = (sun_open_price - exit_price) / sun_open_price
            pnl -= 0.002
            
        trades.append(pnl)
        gap_stats.append({
            'date': sun_time,
            'gap_pct': gap_pct,
            'filled': filled,
            'pnl': pnl
        })

    # Results
    if not trades:
        print("No trades found (Gaps < 1.5%).")
        return

    df_res = pd.DataFrame(gap_stats)
    win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
    cum_pnl = sum(trades) * 100
    
    print("\n" + "="*40)
    print("Strategy 18: CME Gap Fill Results")
    print("="*40)
    print(f"Total Gaps Traded: {len(trades)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Cumulative Return: {cum_pnl:.2f}% (Linear)")
    print(f"Avg PnL per Trade: {np.mean(trades)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    try:
        run_simulation()
    except ImportError:
        print("Please install yfinance: pip install yfinance")
    except Exception as e:
        print(f"Crash: {e}")
