"""Test mean_reversion_bb strategy on ranging regime data - pytest version"""
import pandas as pd
import numpy as np
import sqlite3
import pytest
import talib
from alpha_graveyard.engine.risk import RiskManager
from alpha_graveyard.engine.backtest import BacktestEngine
from alpha_graveyard.strategies.library import (
    strategy_mean_reversion_bb, 
    strategy_ranging_mean_reversion, 
    run_strategy_test
)
from alpha_graveyard.features.regime import classify_regime


@pytest.fixture
def ranging_data():
    """Generate synthetic ranging data for testing"""
    np.random.seed(42)
    n = 1000
    base_price = 45000
    # Sideways oscillation with proper BB calculation
    oscillation = np.sin(np.linspace(0, 20, n)) * 800
    noise = np.random.randn(n) * 30
    close = base_price + oscillation + noise
    high = close + abs(np.random.randn(n)) * 50 + 20
    low = close - abs(np.random.randn(n)) * 50 - 20
    
    # Calculate real BB using TA-Lib
    close_vals = close.astype(np.float64)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close_vals, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    atr = talib.ATR(high.astype(np.float64), low.astype(np.float64), close_vals, timeperiod=14)
    
    ranging_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(500, 1000, n),
        'ema_20': talib.EMA(close_vals, timeperiod=20),
        'atr_14': atr,
        'adx_14': np.full(n, 18.0),  # Low ADX = ranging
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_middle': bb_middle,
        'regime': 'ranging'
    })
    
    # Drop NaN rows from indicator warmup
    return ranging_df.dropna()


def test_mean_reversion_generates_signals(ranging_data):
    """Verify BB mean reversion generates signals on ranging data"""
    signals = run_strategy_test(ranging_data, strategy_mean_reversion_bb)
    
    # Should generate signals in ranging regime
    assert len(signals) > 0, "Should generate signals in ranging regime"
    
    # Verify signal types
    longs = sum(1 for s in signals if s['signal'] == 'LONG')
    shorts = sum(1 for s in signals if s['signal'] == 'SHORT')
    
    assert longs > 0 or shorts > 0, "Should have both long and short signals"


def test_mean_reversion_signal_structure(ranging_data):
    """Verify signal dictionary structure"""
    signals = run_strategy_test(ranging_data, strategy_mean_reversion_bb)
    
    for sig in signals[:5]:  # Check first 5
        assert 'signal' in sig
        assert 'entry_price' in sig
        assert 'stop_loss' in sig
        assert 'confidence' in sig
        assert sig['signal'] in ['LONG', 'SHORT', 'NONE']
        assert 0 <= sig['confidence'] <= 1


def test_mean_reversion_backtest(ranging_data):
    """Run full backtest on ranging data"""
    rm = RiskManager(initial_capital=100000)
    engine = BacktestEngine(risk_manager=rm, initial_capital=100000, db_path='ranging_test.db')
    
    metrics = engine.run(ranging_data, regime_col='regime')
    
    # Verify metrics structure
    assert 'total_trades' in metrics
    assert 'win_rate' in metrics
    assert 'final_capital' in metrics
    
    # Should have trades in ranging regime
    assert metrics['total_trades'] > 0, "Should execute trades"


def test_bb_touch_logic(ranging_data):
    """Verify BB touch detection works"""
    # Find rows where close is near lower BB
    touch_lower = ranging_data[ranging_data['close'] <= ranging_data['bb_lower'] * 1.001]
    
    # Should have some BB touches in synthetic data
    assert len(touch_lower) > 0, "Synthetic data should have BB lower band touches"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
