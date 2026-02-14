"""
sim_grid.py - Strategy 16: Neutral Grid Trading (The Martingale)

Concept:
- Place a grid of BUY/SELL orders in a fixed range.
- Buy Low, Sell High.
- Profit from "noise" (volatility).

The Trap:
- If price leaves the grid (trend), you are left holding a "bag" (if down)
  or sold out early (if up).
- "Picking up pennies in front of a steamroller."
"""

import pandas as pd
import numpy as np
import sqlite3
import time

def run_grid_simulation(db_path='btc_pipeline.db'):
    print("Loading Data...")
    conn = sqlite3.connect(db_path)
    # Get 2024-2025 data
    df = pd.read_sql("SELECT * FROM candles WHERE timestamp >= '2024-01-01' ORDER BY timestamp", conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print(f"Data Loaded: {len(df)} candles (2024-2025)")
    
    # Grid Parameters
    initial_captial = 10000.0
    
    # SPLIT CAPITAL: 50% USD, 50% BTC
    usd_balance = initial_captial / 2
    btc_balance = (initial_captial / 2) / df['close'].iloc[0]
    
    print(f"Initial Split: ${usd_balance:.2f} USD + {btc_balance:.4f} BTC (@ ${df['close'].iloc[0]:.2f})")
    
    # Grid Settings
    grid_center = df['close'].iloc[0]
    grid_range_pct = 0.20 # +/- 20% range
    n_grids = 40          # 1% spacing (0.20 * 2 / 40 = 0.01)
    grid_step_pct = 0.01
    
    # Fees
    fee_pct = 0.001 # 0.1% Taker
    
    equity_curve = []
    trades = []
    
    print(f"Starting Grid Simulation...")
    print(f"Grid Center: ${grid_center:.2f} | Step: 1%")
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        prev_price = df['close'].iloc[i-1]
        ts = df.index[i]
        
        # Calculate Grid Levels
        # Level 0 = Center
        # +1 = Center * 1.01
        # -1 = Center * 0.99
        
        curr_level_idx = int(np.log(current_price / grid_center) / np.log(1 + grid_step_pct))
        prev_level_idx = int(np.log(prev_price / grid_center) / np.log(1 + grid_step_pct))
        
        if curr_level_idx != prev_level_idx:
            # Crossed a level
            
            # PRICE WENT DOWN (e.g. Level 0 -> -1) -> BUY
            if curr_level_idx < prev_level_idx:
                # Buy $100 worth
                order_amt = 100.0
                if usd_balance >= order_amt:
                    qty = order_amt / current_price
                    usd_balance -= order_amt * (1 + fee_pct)
                    btc_balance += qty
                    trades.append({'side': 'BUY', 'price': current_price, 'ts': ts})
            
            # PRICE WENT UP (e.g. Level 0 -> 1) -> SELL
            elif curr_level_idx > prev_level_idx:
                # Sell $100 worth
                order_amt = 100.0
                qty = order_amt / current_price
                if btc_balance >= qty:
                    usd_balance += order_amt * (1 - fee_pct)
                    btc_balance -= qty
                    trades.append({'side': 'SELL', 'price': current_price, 'ts': ts})
        
        # Mark to Market Equity
        equity = usd_balance + (btc_balance * current_price)
        equity_curve.append(equity)
        
        # Grid Reset (Active Management)
        # If price moves > 25% from center, reset center to current price
        # This realizes the "Impermanent Loss"
        if abs(current_price - grid_center) / grid_center > 0.25:
             # print(f"[{ts}] Grid Reset! Price=${current_price:.0f} (Was ${grid_center:.0f})")
             grid_center = current_price

    # Final Stats
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_captial) / initial_captial * 100
    
    # Buy & Hold Comparison
    bh_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
    # Actually B&H of the 50/50 portfolio
    initial_btc_value = (initial_captial / 2)
    initial_usd = (initial_captial / 2)
    final_bh = initial_usd + (initial_btc_value * (1 + bh_return/100))
    bh_adj_return = (final_bh - initial_captial) / initial_captial * 100
    
    print("\n" + "=" * 50)
    print("GRID TRADING RESULTS (2024-2025)")
    print("=" * 50)
    print(f"Initial Capital: ${initial_captial:,.2f}")
    print(f"Final Equity:    ${final_equity:,.2f}")
    print(f"Total Return:    {total_return:+.2f}%")
    print(f"Buy/Hold (50/50):{bh_adj_return:+.2f}%")
    print(f"Total Trades:    {len(trades)}")
    
    # Calculate Max Drawdown
    peak = initial_captial
    max_dd = 0
    for e in equity_curve:
        if e > peak: peak = e
        dd = (peak - e) / peak
        if dd > max_dd: max_dd = dd
        
    print(f"Max Drawdown:    {max_dd*100:.2f}%")
    print("=" * 50)
    
    if total_return < 0:
        print("❌ FAILED: The steamroller crushed the pennies.")
    else:
        print("⚠️ SURVIVED: You got lucky (for now).")

if __name__ == "__main__":
    run_grid_simulation()
