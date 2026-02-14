"""
sim_weekly_mr.py - Strategy 22: Weekly Mean Reversion (Bollinger Bands)

Hypothesis:
- 1H Bollinger Bands failed (Strategy 3).
- On WEEKLY candles, mean reversion should work better:
  - Fewer false signals.
  - Larger moves = fees negligible.
  - Weekly extremes represent genuine overextension, not noise.

Logic:
- Buy when Weekly Close < Lower Bollinger Band (2 std, 20-week lookback).
- Sell when Weekly Close > Middle Band (20-week SMA).
- Stop Loss: 15% (weekly risk is higher).
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

def main():
    print("Fetching BTC-USD Weekly (5 years)...")
    df = yf.download("BTC-USD", period="5y", interval="1wk", progress=False)
    df = flatten_cols(df)
    print(f"Loaded {len(df)} weekly candles.")
    
    # Bollinger Bands (20-week, 2 std)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['std_20'] = df['close'].rolling(20).std()
    df['upper_bb'] = df['sma_20'] + 2 * df['std_20']
    df['lower_bb'] = df['sma_20'] - 2 * df['std_20']
    df = df.dropna()
    
    # Simulation
    balance = 10000.0
    position = 0
    entry_price = 0
    trades = []
    bh_start = df.iloc[0]['close']
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        
        # BUY: Close < Lower BB
        if position == 0 and price < row['lower_bb']:
            position = 1
            entry_price = price
            balance *= 0.999
            trades.append({'type': 'BUY', 'price': price, 'date': df.index[i]})
        
        # SELL: Close > SMA20 (mean reversion target)
        elif position == 1:
            # Stop Loss 15%
            if price < entry_price * 0.85:
                pnl = (price - entry_price) / entry_price
                balance *= (1 + pnl) * 0.999
                trades.append({'type': 'SELL', 'pnl': pnl, 'reason': 'STOP'})
                position = 0
            elif price > row['sma_20']:
                pnl = (price - entry_price) / entry_price
                balance *= (1 + pnl) * 0.999
                trades.append({'type': 'SELL', 'pnl': pnl, 'reason': 'TARGET'})
                position = 0
    
    # Close open position
    if position == 1:
        final = df.iloc[-1]['close']
        pnl = (final - entry_price) / entry_price
        balance *= (1 + pnl) * 0.999
        trades.append({'type': 'SELL', 'pnl': pnl, 'reason': 'EOF'})
    
    bh_end = df.iloc[-1]['close']
    bh_ret = (bh_end - bh_start) / bh_start * 100
    strat_ret = (balance - 10000) / 10000 * 100
    
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winners = [t for t in sell_trades if t['pnl'] > 0]
    stops = [t for t in sell_trades if t.get('reason') == 'STOP']
    
    print(f"\n{'='*55}")
    print(f"  Strategy 22: Weekly Mean Reversion (Bollinger)")
    print(f"{'='*55}")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Total Trades: {len(sell_trades)}")
    if sell_trades:
        print(f"  Win Rate: {len(winners)/len(sell_trades)*100:.1f}%")
        print(f"  Avg PnL: {np.mean([t['pnl'] for t in sell_trades])*100:.2f}%")
        print(f"  Stop Losses Hit: {len(stops)}")
    print(f"  Strategy Return: {strat_ret:+.2f}%")
    print(f"  Buy & Hold Return: {bh_ret:+.2f}%")
    print(f"  Alpha: {strat_ret - bh_ret:+.2f}%")
    print(f"{'='*55}")
    
    # Also test ETH
    print(f"\n--- ETH-USD Weekly (5y) ---")
    df_eth = yf.download("ETH-USD", period="5y", interval="1wk", progress=False)
    df_eth = flatten_cols(df_eth)
    
    df_eth['sma_20'] = df_eth['close'].rolling(20).mean()
    df_eth['std_20'] = df_eth['close'].rolling(20).std()
    df_eth['lower_bb'] = df_eth['sma_20'] - 2 * df_eth['std_20']
    df_eth = df_eth.dropna()
    
    bal_eth = 10000.0
    pos_eth = 0
    ep_eth = 0
    trades_eth = []
    bh_s_eth = df_eth.iloc[0]['close']
    
    for i in range(1, len(df_eth)):
        row = df_eth.iloc[i]
        price = row['close']
        
        if pos_eth == 0 and price < row['lower_bb']:
            pos_eth = 1
            ep_eth = price
            bal_eth *= 0.999
        elif pos_eth == 1:
            if price < ep_eth * 0.85:
                pnl = (price - ep_eth) / ep_eth
                bal_eth *= (1 + pnl) * 0.999
                trades_eth.append(pnl)
                pos_eth = 0
            elif price > row['sma_20']:
                pnl = (price - ep_eth) / ep_eth
                bal_eth *= (1 + pnl) * 0.999
                trades_eth.append(pnl)
                pos_eth = 0
    
    if pos_eth == 1:
        pnl = (df_eth.iloc[-1]['close'] - ep_eth) / ep_eth
        bal_eth *= (1 + pnl) * 0.999
        trades_eth.append(pnl)
    
    bh_e_eth = df_eth.iloc[-1]['close']
    eth_strat = (bal_eth - 10000) / 10000 * 100
    eth_bh = (bh_e_eth - bh_s_eth) / bh_s_eth * 100
    
    print(f"  ETH Trades: {len(trades_eth)}")
    if trades_eth:
        print(f"  ETH Win Rate: {sum(1 for t in trades_eth if t > 0)/len(trades_eth)*100:.1f}%")
    print(f"  ETH Strategy: {eth_strat:+.2f}%")
    print(f"  ETH B&H: {eth_bh:+.2f}%")
    print(f"  ETH Alpha: {eth_strat - eth_bh:+.2f}%")

if __name__ == "__main__":
    main()
