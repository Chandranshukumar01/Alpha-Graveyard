"""
Clean OOS Test: SOL S-20 Strategy
Same test for SOL to verify if any alt-coin edge exists
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

def run_s20_strict_oos(ticker="SOL-USD", train_start="2020-01-01", train_end="2022-12-31", 
                       test_start="2023-01-01", test_end="2024-12-31", cost=0.002):
    
    print(f"Fetching {ticker}...")
    df_full = yf.download(ticker, start=train_start, end=test_end, progress=False)
    df_full = flatten_cols(df_full)
    
    df_train = df_full[(df_full.index >= train_start) & (df_full.index <= train_end)].copy()
    df_test = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)].copy()
    
    print(f"  Train: {len(df_train)} days | Test: {len(df_test)} days")
    
    # Training
    df_train['sma_20'] = df_train['close'].rolling(20).mean()
    df_train['returns'] = df_train['close'].pct_change()
    df_train['vol_10d'] = df_train['returns'].rolling(10).std()
    df_train['vol_median'] = df_train['vol_10d'].rolling(60).median()
    vol_threshold = df_train['vol_median'].median()
    print(f"  Vol threshold: {vol_threshold:.4f}")
    
    # Testing
    df_test = df_test.copy()
    df_test['sma_20'] = df_test['close'].rolling(20).mean()
    df_test['returns'] = df_test['close'].pct_change()
    df_test['vol_10d'] = df_test['returns'].rolling(10).std()
    df_test['vol_median'] = df_test['vol_10d'].rolling(60).median()
    df_test['high_vol'] = df_test['vol_10d'] > vol_threshold
    df_test = df_test.dropna()
    
    # Simulate
    balance = 10000.0
    position = 0
    entry_price = 0
    trades = []
    
    for i in range(1, len(df_test)):
        row = df_test.iloc[i]
        prev = df_test.iloc[i-1]
        price = row['close']
        
        if position == 0:
            if prev['close'] < prev['sma_20'] and price > row['sma_20'] and row['high_vol']:
                position = 1
                entry_price = price
                balance *= (1 - cost)
        elif position == 1:
            if price < row['sma_20']:
                pnl = (price - entry_price) / entry_price
                balance *= (1 + pnl) * (1 - cost)
                trades.append(pnl - 2*cost)
                position = 0
    
    if position == 1:
        final = df_test.iloc[-1]['close']
        pnl = (final - entry_price) / entry_price
        balance *= (1 + pnl) * (1 - cost)
        trades.append(pnl - 2*cost)
    
    start_price = df_test.iloc[0]['close']
    end_price = df_test.iloc[-1]['close']
    bh_return = (end_price - start_price) / start_price * 100
    strat_return = (balance - 10000) / 10000 * 100
    alpha = strat_return - bh_return
    
    winners = [t for t in trades if t > 0]
    
    print(f"  Trades: {len(trades)} | Win Rate: {len(winners)/len(trades)*100 if trades else 0:.1f}%")
    print(f"  Strategy: {strat_return:+.2f}% | B&H: {bh_return:+.2f}% | Alpha: {alpha:+.2f}%")
    
    return alpha

def main():
    print("=" * 60)
    print("CLEAN OOS TEST: SOL S-20 Strategy")
    print("=" * 60)
    
    alphas = []
    for trial in range(1, 4):
        print(f"\n--- Trial {trial} ---")
        alpha = run_s20_strict_oos()
        alphas.append(alpha)
    
    print("\n" + "=" * 60)
    print("SUMMARY - 3 TRIALS")
    print("=" * 60)
    print(f"Alpha Range: {min(alphas):+.2f}% to {max(alphas):+.2f}%")
    print(f"Mean Alpha: {np.mean(alphas):+.2f}% ± {np.std(alphas):.2f}%")
    
    if np.mean(alphas) > 5:
        print("\n✅ SOL shows possible edge")
    else:
        print("\n❌ SOL also fails")
        print("   Confirmed: No asset-specific alpha")

if __name__ == "__main__":
    main()
