"""
sim_cross_sectional.py - Strategy 21: Cross-Sectional Momentum

Hypothesis:
- All our strategies were univariate (BTC predicts BTC).
- Cross-sectional = rank assets by momentum, long top 3, short bottom 3.
- This is RELATIVE alpha (BTC vs ETH vs SOL), not absolute direction.

Implementation:
- Universe: BTC, ETH, SOL, ADA, AVAX, LINK, DOT, UNI (8 assets)
- Every week: Rank by 20-day return
- Long top 3. Short bottom 3.
- Rebalance weekly.
- Market-neutral (equal $ long/short).
"""

import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "UNI-USD"]

def flatten_cols(df, ticker):
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            new_cols.append(c[0].lower())
        else:
            new_cols.append(c.lower())
    df.columns = new_cols
    return df

def main():
    print("=" * 55)
    print("  Strategy 21: Cross-Sectional Momentum (Long/Short)")
    print("=" * 55)
    
    # Fetch all assets
    print("\nFetching data for 8 crypto assets (2y daily)...")
    closes = {}
    for t in TICKERS:
        try:
            df = yf.download(t, period="2y", interval="1d", progress=False)
            df = flatten_cols(df, t)
            if len(df) > 100:
                closes[t] = df['close']
                print(f"  {t}: {len(df)} days loaded.")
            else:
                print(f"  {t}: Insufficient data. Skipped.")
        except Exception as e:
            print(f"  {t}: Error - {e}")
    
    if len(closes) < 6:
        print("Need at least 6 assets. Aborting.")
        return
    
    # Align all series to common dates
    price_df = pd.DataFrame(closes)
    price_df = price_df.dropna()
    print(f"\nAligned {len(price_df)} common trading days across {len(closes)} assets.")
    
    # Calculate returns
    ret_df = price_df.pct_change()
    
    # Weekly rebalancing (every 5 trading days)
    rebal_freq = 5
    lookback = 20  # 20-day momentum
    
    portfolio_returns = []
    fee_per_trade = 0.001  # 0.1%
    
    for i in range(lookback, len(price_df) - rebal_freq, rebal_freq):
        # Calculate 20-day momentum for each asset
        momentum = {}
        for col in price_df.columns:
            start_price = price_df[col].iloc[i - lookback]
            end_price = price_df[col].iloc[i]
            if start_price > 0:
                momentum[col] = (end_price - start_price) / start_price
        
        if len(momentum) < 6:
            continue
        
        # Rank by momentum
        ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        longs = [r[0] for r in ranked[:3]]   # Top 3 (winners)
        shorts = [r[0] for r in ranked[-3:]]  # Bottom 3 (losers)
        
        # Calculate return over next week
        week_ret_long = 0
        week_ret_short = 0
        
        for col in longs:
            start = price_df[col].iloc[i]
            end = price_df[col].iloc[min(i + rebal_freq, len(price_df) - 1)]
            week_ret_long += (end - start) / start
        
        for col in shorts:
            start = price_df[col].iloc[i]
            end = price_df[col].iloc[min(i + rebal_freq, len(price_df) - 1)]
            week_ret_short += (start - end) / start  # Short profit
        
        # Average across positions (equal weight)
        avg_long = week_ret_long / 3
        avg_short = week_ret_short / 3
        
        # Net return = Long + Short (market neutral) - fees (6 trades: 3 open + 3 close)
        net_ret = (avg_long + avg_short) / 2 - (6 * fee_per_trade / 6)  # Simplified
        portfolio_returns.append(net_ret)
    
    if not portfolio_returns:
        print("No rebalancing periods. Aborting.")
        return
    
    # Results
    cum_ret = 1.0
    for r in portfolio_returns:
        cum_ret *= (1 + r)
    cum_ret = (cum_ret - 1) * 100
    
    wins = sum(1 for r in portfolio_returns if r > 0)
    
    # B&H benchmark (equal-weight all 8)
    bh_start = price_df.iloc[lookback].mean()
    bh_end = price_df.iloc[-1].mean()
    bh_ret = (bh_end - bh_start) / bh_start * 100
    
    print(f"\n{'='*55}")
    print(f"  RESULTS: Cross-Sectional Momentum")
    print(f"{'='*55}")
    print(f"  Rebalance Periods: {len(portfolio_returns)}")
    print(f"  Win Rate: {wins/len(portfolio_returns)*100:.1f}%")
    print(f"  Cum Return: {cum_ret:+.2f}%")
    print(f"  Equal-Weight B&H: {bh_ret:+.2f}%")
    print(f"  Alpha: {cum_ret - bh_ret:+.2f}%")
    print(f"  Avg Weekly Return: {np.mean(portfolio_returns)*100:.3f}%")
    print(f"  Sharpe (Weekly): {np.mean(portfolio_returns)/np.std(portfolio_returns)*np.sqrt(52):.2f}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
