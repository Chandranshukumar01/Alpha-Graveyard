"""
sim_lighthouse.py - Project Lighthouse Simulation

Runs the Council of Agents on historical data (2025).
"""

import pandas as pd
import numpy as np
import sqlite3
import time
from alpha_graveyard.strategies.lighthouse import Council

def run_simulation(db_path='btc_pipeline.db'):
    # Load Data
    print("Loading Data...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM candles ORDER BY timestamp", conn)
    conn.close()
    
    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Initialize Council
    council = Council()
    
    # Simulation State
    position = 0 # 0, 1, -1
    entry_price = 0.0
    capital = 10000.0 # Initial Capital
    equity_curve = [capital]
    trades = []
    
    # Start Index (Ensure enough history for Captain)
    # Require ~300 days = 7200 hours
    start_idx = 7200
    if len(df) <= start_idx:
        print("Not enough data for simulation (Need > 7200 candles).")
        return

    print(f"Starting Simulation on {len(df)-start_idx} candles...")
    start_time = time.time()
    
    for i in range(start_idx, len(df)):
        # Progress
        if (i - start_idx) % 1000 == 0:
            print(f"  Processing candle {i}/{len(df)}... (Equity: ${capital:.2f})")
            
        current_price = df['close'].iloc[i]
        current_ts = df.index[i]
        
        # Ask the Council
        # Vote: 1 (Long), -1 (Short), 0 (Neutral)
        vote, reason = council.convene(df, i)
        
        # Execution Logic
        # Fee/Slippage: 0.1% per trade (0.001)
        cost = 0.001
        
        if position == 0:
            if vote == 1:
                # Enter Long
                position = 1
                entry_price = current_price
                capital *= (1 - cost) # Entry Cost
                print(f"[{current_ts}] LONG ENTRY @ {current_price:.2f} | {reason}")
                trades.append({'type': 'LONG_ENTRY', 'price': current_price, 'ts': current_ts})
                
            elif vote == -1:
                # Enter Short
                position = -1
                entry_price = current_price
                capital *= (1 - cost) # Entry Cost
                print(f"[{current_ts}] SHORT ENTRY @ {current_price:.2f} | {reason}")
                trades.append({'type': 'SHORT_ENTRY', 'price': current_price, 'ts': current_ts})
                
        elif position == 1:
            # Exit Long if Vote changes (to Neutral or Short)
            if vote != 1:
                exit_price = current_price
                pnl_pct = (exit_price - entry_price) / entry_price
                capital *= (1 + pnl_pct)
                capital *= (1 - cost) # Exit Cost
                position = 0
                print(f"[{current_ts}] LONG EXIT @ {exit_price:.2f} | PnL: {pnl_pct*100:.2f}% | {reason}")
                trades.append({'type': 'LONG_EXIT', 'price': exit_price, 'ts': current_ts, 'pnl': pnl_pct})
                
        elif position == -1:
            # Exit Short if Vote changes (to Neutral or Long)
            if vote != -1:
                exit_price = current_price
                pnl_pct = (entry_price - exit_price) / entry_price
                capital *= (1 + pnl_pct)
                capital *= (1 - cost) # Exit Cost
                position = 0
                print(f"[{current_ts}] SHORT EXIT @ {exit_price:.2f} | PnL: {pnl_pct*100:.2f}% | {reason}")
                trades.append({'type': 'SHORT_EXIT', 'price': exit_price, 'ts': current_ts, 'pnl': pnl_pct})
                
        equity_curve.append(capital)

    elapsed = time.time() - start_time
    print(f"\nSimulation Complete in {elapsed:.1f}s")
    print("=" * 50)
    print(f"Final Capital: ${capital:,.2f}")
    if capital > 10000:
        print("✅ PROJECT LIGHTHOUSE SUCCESS!")
    else:
        print("❌ PROJECT LIGHTHOUSE FAILURE")
    print("=" * 50)

if __name__ == "__main__":
    run_simulation()
