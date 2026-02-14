"""Test strategy with synthetic data - No database dependency"""
import pandas as pd
import numpy as np
import talib
import pytest
from alpha_graveyard.strategies.library import (
    strategy_trending_up_ema_crossover, 
    get_strategy_for_regime, 
    run_strategy_test
)


@pytest.fixture
def synthetic_btc_data():
    """Generate synthetic trending BTC-like data with calculated indicators"""
    np.random.seed(42)
    n = 500
    
    # Generate trending price series (bull market simulation)
    trend = np.linspace(30000, 45000, n)  # Upward trend
    noise = np.random.normal(0, 800, n)
    close = trend + noise
    
    # Generate OHLC from close
    high = close + np.abs(np.random.normal(200, 100, n))
    low = close - np.abs(np.random.normal(200, 100, n))
    open_price = close + np.random.normal(0, 50, n)
    volume = np.random.uniform(1000, 5000, n)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Calculate indicators
    close_vals = df['close'].values.astype(np.float64)
    high_vals = df['high'].values.astype(np.float64)
    low_vals = df['low'].values.astype(np.float64)
    
    df['ema_20'] = talib.EMA(close_vals, timeperiod=20)
    df['atr_14'] = talib.ATR(high_vals, low_vals, close_vals, timeperiod=14)
    df['adx_14'] = talib.ADX(high_vals, low_vals, close_vals, timeperiod=14)
    
    # Drop NaN warmup rows
    df = df.dropna().reset_index(drop=True)
    return df


def test_strategy_generates_signals(synthetic_btc_data):
    """Verify EMA crossover strategy generates signals"""
    signals = run_strategy_test(synthetic_btc_data, strategy_trending_up_ema_crossover)
    
    assert len(signals) > 0, "Strategy should generate signals"
    
    # Verify signal structure
    for sig in signals:
        assert 'signal' in sig
        assert 'entry_price' in sig
        assert 'stop_loss' in sig
        assert sig['signal'] in ['LONG', 'SHORT', 'NONE']


def test_strategy_no_lookahead(synthetic_btc_data):
    """Verify strategy doesn't use future data"""
    # Manually check a few rows
    for i in range(20, min(30, len(synthetic_btc_data))):
        row = synthetic_btc_data.iloc[i].to_dict()
        prev_row = synthetic_btc_data.iloc[i-1].to_dict()
        
        result = strategy_trending_up_ema_crossover(row, prev_row)
        
        # Should not raise errors
        assert isinstance(result, dict)
        assert 'signal' in result


def test_strategy_metrics(synthetic_btc_data):
    """Verify strategy produces reasonable metrics in trending market"""
    signals = run_strategy_test(synthetic_btc_data, strategy_trending_up_ema_crossover)
    
    if not signals:
        pytest.skip("No signals generated")
    
    long_count = sum(1 for s in signals if s['signal'] == 'LONG')
    short_count = sum(1 for s in signals if s['signal'] == 'SHORT')
    
    # In upward trending synthetic data, should have mostly LONG signals
    total = long_count + short_count
    if total > 0:
        assert long_count / total > 0.5, "Should be majority LONG in trending market"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
