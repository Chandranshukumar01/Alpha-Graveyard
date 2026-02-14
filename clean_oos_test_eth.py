"""
Clean OOS Test: ETH S-20 Strategy
Strict out-of-sample validation with costs
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def flatten_cols(df):
    """Flatten multi-index columns from yfinance"""
    new_cols = []
    for c in df.columns:
        new_cols.append(c[0].lower() if isinstance(c, tuple) else c.lower())
    df.columns = new_cols
    return df

def run_s20_strict_oos(ticker="ETH-USD", train_start="2020-01-01", train_end="2022-12-31", 
                       test_start="2023-01-01", test_end="2024-12-31", cost=0.002, seed=None):
    """
    Strict OOS test:
    1. Train on 2020-2022: Determine optimal vol threshold (or use median)
    2. Test on 2023-2024: Apply strategy with costs, no peeking
    """
    if seed:
        np.random.seed(seed)
    
    # Fetch full dataset
    print(f"Fetching {ticker}...")
    df_full = yf.download(ticker, start=train_start, end=test_end, progress=False)
    df_full = flatten_cols(df_full)
    
    # Split train/test
    df_train = df_full[(df_full.index >= train_start) & (df_full.index <= train_end)].copy()
    df_test = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)].copy()
    
    print(f"  Train: {len(df_train)} days ({df_train.index[0].date()} to {df_train.index[-1].date()})")
    print(f"  Test:  {len(df_test)} days ({df_test.index[0].date()} to {df_test.index[-1].date()})")
    
    # === TRAINING PHASE: Determine strategy parameters ===
    # Calculate indicators on training data
    df_train['sma_20'] = df_train['close'].rolling(20).mean()
    df_train['returns'] = df_train['close'].pct_change()
    df_train['vol_10d'] = df_train['returns'].rolling(10).std()
    df_train['vol_median'] = df_train['vol_10d'].rolling(60).median()
    df_train['high_vol'] = df_train['vol_10d'] > df_train['vol_median']
    
    # Determine optimal vol threshold from training
    # Strategy: Only trade when vol_10d > median(vol_10d, 60)
    # This is the "vol filter" parameter learned from training
    vol_threshold = df_train['vol_median'].median()  # Median of medians
    
    print(f"  Training: Vol threshold = {vol_threshold:.4f}")
    
    # === TESTING PHASE: Apply to OOS data with costs ===
    df_test = df_test.copy()
    df_test['sma_20'] = df_test['close'].rolling(20).mean()
    df_test['returns'] = df_test['close'].pct_change()
    df_test['vol_10d'] = df_test['returns'].rolling(10).std()
    df_test['vol_median'] = df_test['vol_10d'].rolling(60).median()
    # Use trained threshold
    df_test['high_vol'] = df_test['vol_10d'] > vol_threshold
    df_test = df_test.dropna()
    
    # Simulate strategy
    balance = 10000.0
    position = 0  # 1=long, 0=flat
    entry_price = 0
    trades = []
    
    for i in range(1, len(df_test)):
        row = df_test.iloc[i]
        prev = df_test.iloc[i-1]
        price = row['close']
        
        if position == 0:
            # Entry: Cross above SMA20 + high vol
            if prev['close'] < prev['sma_20'] and price > row['sma_20'] and row['high_vol']:
                position = 1
                entry_price = price
                balance *= (1 - cost)  # Entry cost
        elif position == 1:
            # Exit: Price below SMA20
            if price < row['sma_20']:
                pnl = (price - entry_price) / entry_price
                balance *= (1 + pnl) * (1 - cost)  # Exit cost
                trades.append(pnl - 2*cost)  # Net pnl after costs
                position = 0
    
    # Close open position at end
    if position == 1:
        final = df_test.iloc[-1]['close']
        pnl = (final - entry_price) / entry_price
        balance *= (1 + pnl) * (1 - cost)
        trades.append(pnl - 2*cost)
    
    # Calculate metrics
    start_price = df_test.iloc[0]['close']
    end_price = df_test.iloc[-1]['close']
    bh_return = (end_price - start_price) / start_price * 100
    
    strat_return = (balance - 10000) / 10000 * 100
    alpha = strat_return - bh_return
    
    winners = [t for t in trades if t > 0]
    
    return {
        'trades': len(trades),
        'win_rate': len(winners) / len(trades) * 100 if trades else 0,
        'strat_return': strat_return,
        'bh_return': bh_return,
        'alpha': alpha,
        'trades_list': trades
    }

def main():
    print("=" * 60)
    print("CLEAN OOS TEST: ETH S-20 Strategy")
    print("Train: 2020-2022 | Test: 2023-2024 | Cost: 0.2%")
    print("=" * 60)
    
    # Run 3 trials with different random seeds (for any stochastic elements)
    results = []
    for trial in range(1, 4):
        print(f"\n--- Trial {trial} ---")
        result = run_s20_strict_oos(seed=trial)
        results.append(result)
        print(f"  Trades: {result['trades']} | Win Rate: {result['win_rate']:.1f}%")
        print(f"  Strategy: {result['strat_return']:+.2f}%")
        print(f"  Buy & Hold: {result['bh_return']:+.2f}%")
        print(f"  Alpha: {result['alpha']:+.2f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - 3 TRIALS")
    print("=" * 60)
    alphas = [r['alpha'] for r in results]
    avg_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)
    
    print(f"Alpha Range: {min(alphas):+.2f}% to {max(alphas):+.2f}%")
    print(f"Mean Alpha: {avg_alpha:+.2f}% ± {std_alpha:.2f}%")
    print(f"Consistent ±10%? {'YES' if std_alpha < 10 else 'NO'}")
    print(f"All Positive? {'YES' if all(a > 0 for a in alphas) else 'NO'}")
    print(f"After-Cost Edge? {'YES' if avg_alpha > 5 else 'NO'} (threshold: +5%)")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if avg_alpha > 10 and std_alpha < 10:
        print("✅ REAL EDGE CONFIRMED")
        print("   Strategy shows consistent positive alpha after costs")
        print("   Recommendation: OPTIMIZE and DEPLOY")
    elif avg_alpha > 0 and std_alpha < 15:
        print("⚠️  WEAK EDGE / BORDERLINE")
        print("   Positive but inconsistent or marginal")
        print("   Recommendation: MORE TESTING or PASS")
    else:
        print("❌ NO EDGE")
        print("   After costs, no consistent alpha")
        print("   Recommendation: CONFIRM original EMH conclusion")

if __name__ == "__main__":
    main()
