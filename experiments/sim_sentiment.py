"""
sim_sentiment.py - Strategy 17: Sentiment Reversion (OI + Funding)

Hypothesis:
- High Open Interest (OI) = Leverage in the system.
- High Positive Funding = Longs are paying Shorts (Greed).
- High Negative Funding = Shorts are paying Longs (Fear).

Logic:
- When OI is High (Z-Score > 2) AND Asset is "Crowded":
  - Crowded Long (Funding > 0.01%): SHORT (Expect Long Squeeze)
  - Crowded Short (Funding < -0.01%): LONG (Expect Short Squeeze)

Data:
- Fetches real 1h data from Binance Futures (Limit ~500-1000 hours).
"""

import ccxt
import pandas as pd
import numpy as np
import time

def fetch_data(symbol='BTC/USDT'):
    print(f"Fetching Data for {symbol}...")
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    
    # 1. Fetch OHLCV (Price)
    # We need matching timestamps
    limit = 480 # 20 days
    ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # 2. Fetch Open Interest
    
    end_time = exchange.milliseconds()
    start_time = end_time - (limit * 60 * 60 * 1000) 
    
    try:
        # Binance sometimes needs exact start time alignments or smaller windows
        oi_data = exchange.fetch_open_interest_history(symbol, '1h', limit=limit, since=start_time)
        df_oi = pd.DataFrame(oi_data)
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
        df_oi.set_index('timestamp', inplace=True)
        # Rename column to avoid confusion
        df_oi = df_oi[['openInterestValue']] # Use Value in USD usually better
        df_oi.columns = ['oi_usd']
    except Exception as e:
        print(f"Error fetching OI: {e}")
        return None

    # 3. Fetch Funding Rates
    try:
        funding_data = exchange.fetch_funding_rate_history(symbol, since=start_time, limit=limit)
        df_funding = pd.DataFrame(funding_data)
        df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'], unit='ms')
        df_funding.set_index('timestamp', inplace=True)
        df_funding = df_funding[['fundingRate']]
    except Exception as e:
        print(f"Error fetching Funding: {e}")
        return None
    
    # Merge
    # Funding is 8h usually, we need to forward fill it for 1h candles
    # OI is 1h usually.
    
    full_df = df.join(df_oi).join(df_funding)
    
    # Forward fill funding rates (persist until changed)
    full_df['fundingRate'] = full_df['fundingRate'].ffill()
    full_df = full_df.dropna()
    
    return full_df

def run_strategy():
    df = fetch_data()
    if df is None or len(df) < 50:
        print("Not enough data.")
        return

    print(f"Data Loaded: {len(df)} candles.")
    
    # --- Feature Engineering ---
    
    # 1. OI Z-Score
    window = 24 # Adjusted for short history
    df['oi_mean'] = df['oi_usd'].rolling(window).mean()
    df['oi_std'] = df['oi_usd'].rolling(window).std()
    df['oi_z'] = (df['oi_usd'] - df['oi_mean']) / df['oi_std']
    
    print("\n--- Data Stats ---")
    print(f"OI Z-Score Range: {df['oi_z'].min():.2f} to {df['oi_z'].max():.2f}")
    print(f"Funding Range: {df['fundingRate'].min()*100:.4f}% to {df['fundingRate'].max()*100:.4f}%")
    print("-" * 20)
    
    # 2. Funding Regime
    # High Positive = > 0.01% (Baseline)
    # High Negative = < -0.01%
    
    # --- Simulation ---
    balance = 10000.0
    position = 0 # 1=Long, -1=Short, 0=Flat
    entry_price = 0
    
    trades = []
    equity_curve = []
    
    print("Running Simulation...")
    
    for i in range(window, len(df)):
        row = df.iloc[i]
        curr_price = row['close']
        ts = df.index[i]
        
        # Signals
        oi_z = row['oi_z']
        funding = row['fundingRate']
        
        signal = 0
        
        # MEAN REVERSION SETUP
        # Crowd is Long (High OI + High Funding) -> We Short
        if oi_z > 1.5 and funding > 0.0001: # 0.01%
            signal = -1
            
        # Crowd is Short (High OI + Low Funding) -> We Long
        elif oi_z > 1.5 and funding < -0.0001:
            signal = 1
            
        # EXIT CONDITIONS
        # Exit if OI cools down (Z < 0.5) OR Funding normalizes
        exit_signal = False
        if position != 0:
            if abs(oi_z) < 0.5:
                exit_signal = True
            
            # Stop Loss (2%)
            if position == 1 and curr_price < entry_price * 0.98: exit_signal = True
            if position == -1 and curr_price > entry_price * 1.02: exit_signal = True
            
        # EXECUTION
        if exit_signal and position != 0:
            pnl = 0
            if position == 1:
                pnl = (curr_price - entry_price) / entry_price
            elif position == -1:
                pnl = (entry_price - curr_price) / entry_price
                
            balance *= (1 + pnl - 0.002) # 0.1% fee x2
            position = 0
            trades.append({'ts': ts, 'type': 'EXIT', 'pnl': pnl})
            
        if position == 0 and signal != 0:
            position = signal
            entry_price = curr_price
            balance *= (1 - 0.001) # Entry fee
            trades.append({'ts': ts, 'type': 'LONG' if signal==1 else 'SHORT', 'price': curr_price})
            
        equity_curve.append(balance)
        
    # Stats
    final_balance = equity_curve[-1] if equity_curve else 10000
    ret = (final_balance - 10000) / 10000 * 100
    
    print("\n" + "="*40)
    print("Strategy 17: Sentiment Reversion Results")
    print("="*40)
    print(f"Data Range: {df.index[0]} to {df.index[-1]}")
    print(f"Final Balance: ${final_balance:.2f} ({ret:+.2f}%)")
    print(f"Total Trades: {len([t for t in trades if t['type']=='EXIT'])}")
    
    if len(trades) > 0:
        winning_trades = [t for t in trades if t['type']=='EXIT' and t['pnl'] > 0]
        win_rate = len(winning_trades) / len([t for t in trades if t['type']=='EXIT']) * 100
        print(f"Win Rate: {win_rate:.1f}%")
        
    print("="*40)
    
    # Save chart data if needed
    # df.to_csv('sentiment_results.csv')

if __name__ == "__main__":
    run_strategy()
