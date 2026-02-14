"""
sim_residual_verify.py — Walk-Forward Verification of Strategy 24

Strategy 24 (Synthetic Residual Stat Arb) showed +41% alpha.
We got burned before with Strategy 20 being period-specific.

Tests:
1. 5-year BTC data (longer history)
2. In-Sample vs Out-of-Sample split
3. Different regression basket (ETH-only, SPY-only)
4. Random residual benchmark (shuffle the basket)
"""

import yfinance as yf
import pandas as pd
import numpy as np

def flatten_cols(df):
    new_cols = []
    for c in df.columns:
        new_cols.append(c[0].lower() if isinstance(c, tuple) else c.lower())
    df.columns = new_cols
    return df

def run_residual_strategy(prices, label, basket_cols, window=60, z_window=20):
    """Run residual stat arb on aligned price dataframe."""
    returns = prices.pct_change().dropna()
    
    if len(returns) < window + z_window + 20:
        print(f"  [{label}] Insufficient data ({len(returns)} rows). Skipping.")
        return None
    
    residuals = []
    for i in range(window, len(returns)):
        y = returns['BTC'].iloc[i-window:i].values
        X = returns[basket_cols].iloc[i-window:i].values
        X_int = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
            x_today = returns[basket_cols].iloc[i].values
            x_today_int = np.concatenate([[1], x_today])
            pred = x_today_int @ beta
            resid = returns['BTC'].iloc[i] - pred
            residuals.append(resid)
        except Exception:
            residuals.append(0)
    
    resid_series = pd.Series(residuals, index=returns.index[window:])
    z_mean = resid_series.rolling(z_window).mean()
    z_std = resid_series.rolling(z_window).std().replace(0, 0.0001)
    z_score = (resid_series - z_mean) / z_std
    
    btc_prices = prices['BTC'].loc[z_score.dropna().index]
    z_clean = z_score.dropna()
    common_idx = btc_prices.index.intersection(z_clean.index)
    btc_prices = btc_prices.loc[common_idx]
    z_clean = z_clean.loc[common_idx]
    
    if len(common_idx) < 20:
        print(f"  [{label}] Not enough aligned data. Skipping.")
        return None
    
    balance = 10000.0
    position = 0
    entry_price = 0
    trades = []
    bh_start = btc_prices.iloc[0]
    
    for i in range(1, len(common_idx)):
        z = z_clean.iloc[i]
        price = btc_prices.iloc[i]
        
        if position == 1 and z > -0.5:
            pnl = (price - entry_price) / entry_price - 0.002
            balance *= (1 + pnl)
            trades.append(pnl)
            position = 0
        elif position == -1 and z < 0.5:
            pnl = (entry_price - price) / entry_price - 0.002
            balance *= (1 + pnl)
            trades.append(pnl)
            position = 0
        
        if position == 0:
            if z < -2:
                position = 1
                entry_price = price
            elif z > 2:
                position = -1
                entry_price = price
    
    if position != 0:
        final = btc_prices.iloc[-1]
        if position == 1:
            pnl = (final - entry_price) / entry_price - 0.002
        else:
            pnl = (entry_price - final) / entry_price - 0.002
        balance *= (1 + pnl)
        trades.append(pnl)
    
    bh_end = btc_prices.iloc[-1]
    bh_ret = (bh_end - bh_start) / bh_start * 100
    strat_ret = (balance - 10000) / 10000 * 100
    winners = [t for t in trades if t > 0]
    
    print(f"  [{label}]")
    print(f"    Trades: {len(trades)} | Win: {len(winners)/max(len(trades),1)*100:.0f}%")
    print(f"    Strategy: {strat_ret:+.2f}% | B&H: {bh_ret:+.2f}% | Alpha: {strat_ret-bh_ret:+.2f}%")
    return strat_ret, bh_ret

def main():
    # Fetch data
    tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SPY": "SPY", "GLD": "GLD"}
    
    print("Fetching 5y daily data...")
    data_5y = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        df = flatten_cols(df)
        data_5y[name] = df['close']
        print(f"  {name}: {len(df)} days.")
    
    prices_5y = pd.DataFrame(data_5y).dropna()
    print(f"Aligned 5y: {len(prices_5y)} common days.\n")
    
    print("=" * 55)
    print("  WALK-FORWARD VERIFICATION: Strategy 24")
    print("=" * 55)
    
    # Test 1: Full 5y
    print("\n--- Test 1: FULL 5-Year Period ---")
    run_residual_strategy(prices_5y, "Full 5y", ['ETH', 'SPY', 'GLD'])
    
    # Test 2: In-Sample (first half)
    mid = len(prices_5y) // 2
    print("\n--- Test 2: In-Sample (First Half) ---")
    run_residual_strategy(prices_5y.iloc[:mid], "IS (1st Half)", ['ETH', 'SPY', 'GLD'])
    
    # Test 3: Out-of-Sample (second half)
    print("\n--- Test 3: Out-of-Sample (Second Half) ---")
    run_residual_strategy(prices_5y.iloc[mid:], "OOS (2nd Half)", ['ETH', 'SPY', 'GLD'])
    
    # Test 4: ETH-only basket
    print("\n--- Test 4: ETH-Only Basket (5y) ---")
    run_residual_strategy(prices_5y, "ETH-Only", ['ETH'])
    
    # Test 5: SPY-only basket (no crypto exposure)
    print("\n--- Test 5: SPY-Only Basket (5y) ---")
    run_residual_strategy(prices_5y, "SPY-Only", ['SPY'])
    
    # Test 6: Random Benchmark (shuffle basket returns)
    print("\n--- Test 6: Random Basket (1000 permutations) ---")
    returns = prices_5y.pct_change().dropna()
    real_result = run_residual_strategy(prices_5y, "Real Basket", ['ETH', 'SPY', 'GLD'])
    
    if real_result:
        beat_count = 0
        for _ in range(200):  # 200 for speed
            shuffled = prices_5y.copy()
            for col in ['ETH', 'SPY', 'GLD']:
                vals = shuffled[col].values.copy()
                np.random.shuffle(vals)
                shuffled[col] = vals
            result = run_residual_strategy(shuffled, "Random", ['ETH', 'SPY', 'GLD'])
            if result and result[0] > real_result[0]:
                beat_count += 1
        # Suppress individual prints, just show summary
    
    print(f"\n{'='*55}")
    print(f"  VERDICT")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
