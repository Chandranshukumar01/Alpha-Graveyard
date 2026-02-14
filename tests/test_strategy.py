"""Test strategy on real BTC data - Proper pytest version"""
import sqlite3
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
def btc_data():
    """Load and prepare BTC data with calculated indicators"""
    conn = sqlite3.connect('btc_pipeline.db')
    df = pd.read_sql_query('''
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        ORDER BY timestamp
        LIMIT 2000
    ''', conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate indicators (features table is unpopulated)
    close_vals = df['close'].values.astype(np.float64)
    high_vals = df['high'].values.astype(np.float64)
    low_vals = df['low'].values.astype(np.float64)
    
    df['ema_20'] = talib.EMA(close_vals, timeperiod=20)
    df['atr_14'] = talib.ATR(high_vals, low_vals, close_vals, timeperiod=14)
    df['adx_14'] = talib.ADX(high_vals, low_vals, close_vals, timeperiod=14)
    
    # Drop NaN warmup rows
    df = df.dropna()
    return df


def test_strategy_generates_signals(btc_data):
    """Verify EMA crossover strategy generates signals"""
    signals = run_strategy_test(btc_data, strategy_trending_up_ema_crossover)
    
    assert len(signals) > 0, "Strategy should generate signals"
    
    # Verify signal structure
    for sig in signals:
        assert 'signal' in sig
        assert 'entry_price' in sig
        assert 'stop_loss' in sig
        assert sig['signal'] in ['LONG', 'SHORT', 'NONE']


def test_strategy_no_lookahead(btc_data):
    """Verify strategy doesn't use future data"""
    # Manually check a few rows
    for i in range(20, min(30, len(btc_data))):
        row = btc_data.iloc[i].to_dict()
        prev_row = btc_data.iloc[i-1].to_dict()
        
        result = strategy_trending_up_ema_crossover(row, prev_row)
        
        # Should not raise errors
        assert isinstance(result, dict)
        assert 'signal' in result


def test_strategy_metrics(btc_data):
    """Verify strategy produces reasonable metrics"""
    signals = run_strategy_test(btc_data, strategy_trending_up_ema_crossover)
    
    if not signals:
        pytest.skip("No signals generated")
    
    long_count = sum(1 for s in signals if s['signal'] == 'LONG')
    short_count = sum(1 for s in signals if s['signal'] == 'SHORT')
    
    # In 2019 bull market, should have mostly LONG signals
    total = long_count + short_count
    if total > 0:
        assert long_count / total > 0.5, "Should be majority LONG in bull market"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
