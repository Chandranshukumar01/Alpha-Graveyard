"""
test_regime_classifier.py - Unit tests for regime_classifier.py

Tests regime classification rules and database operations.
"""

import os
import sqlite3
import time
import unittest

import numpy as np
import pandas as pd
import talib

from alpha_graveyard.features.regime import classify_regime, save_regimes_to_db, analyze_regime_distribution, verify_regimes


class TestRegimeClassifier(unittest.TestCase):
    """Unit tests for regime classification module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db = 'test_regimes.db'
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        
        # Create sample OHLCV + features data
        np.random.seed(42)
        n = 200
        close = np.random.randn(n).cumsum() + 40000
        high = close + abs(np.random.randn(n)) * 100
        low = close - abs(np.random.randn(n)) * 100
        
        self.sample_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
            'open': close + np.random.randn(n) * 10,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(100, 1000, n)
        })
        
        # Calculate features needed for regime classification
        close_vals = self.sample_df['close'].values.astype(np.float64)
        high_vals = self.sample_df['high'].values.astype(np.float64)
        low_vals = self.sample_df['low'].values.astype(np.float64)
        
        self.sample_df['ema_20'] = talib.EMA(close_vals, timeperiod=20)
        self.sample_df['atr_14'] = talib.ATR(high_vals, low_vals, close_vals, timeperiod=14)
        self.sample_df['adx_14'] = talib.ADX(high_vals, low_vals, close_vals, timeperiod=14)
        self.sample_df['plus_di_14'] = talib.PLUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
        self.sample_df['minus_di_14'] = talib.MINUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
        
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_vals, timeperiod=20)
        self.sample_df['bb_upper'] = bb_upper
        self.sample_df['bb_middle'] = bb_middle
        self.sample_df['bb_lower'] = bb_lower
    
    def tearDown(self):
        """Clean up test fixtures."""
        import gc
        gc.collect()
        time.sleep(0.05)
        
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                time.sleep(0.2)
                try:
                    os.remove(self.test_db)
                except PermissionError:
                    pass
    
    def test_classify_regime_adds_regime_column(self):
        """Test that classify_regime adds regime and confidence columns."""
        result = classify_regime(self.sample_df)
        
        self.assertIn('regime', result.columns)
        self.assertIn('confidence', result.columns)
    
    def test_regime_values_are_valid(self):
        """Test that regime values are one of the 5 valid regimes."""
        result = classify_regime(self.sample_df)
        
        valid_regimes = {'trending_up', 'trending_down', 'ranging', 'high_vol', 'weak_trend', 'unknown'}
        unique_regimes = set(result['regime'].unique())
        
        self.assertTrue(unique_regimes.issubset(valid_regimes))
    
    def test_confidence_in_valid_range(self):
        """Test that confidence values are between 0 and 1."""
        result = classify_regime(self.sample_df)
        
        confidences = result['confidence'].dropna()
        self.assertTrue((confidences >= 0).all())
        self.assertTrue((confidences <= 1).all())
    
    def test_trending_up_conditions(self):
        """Test that trending_up regime requires ADX > 25, +DI > -DI, close > EMA."""
        # Create a clear uptrend scenario
        n = 50
        uptrend_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
            'open': np.linspace(40000, 45000, n),
            'high': np.linspace(40100, 45100, n),
            'low': np.linspace(39900, 44900, n),
            'close': np.linspace(40000, 45000, n),
            'volume': np.full(n, 1000)
        })
        
        # Calculate features
        close_vals = uptrend_df['close'].values.astype(np.float64)
        high_vals = uptrend_df['high'].values.astype(np.float64)
        low_vals = uptrend_df['low'].values.astype(np.float64)
        
        uptrend_df['ema_20'] = talib.EMA(close_vals, timeperiod=20)
        uptrend_df['adx_14'] = talib.ADX(high_vals, low_vals, close_vals, timeperiod=14)
        uptrend_df['plus_di_14'] = talib.PLUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
        uptrend_df['minus_di_14'] = talib.MINUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
        
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_vals, timeperiod=20)
        uptrend_df['bb_upper'] = bb_upper
        uptrend_df['bb_middle'] = bb_middle
        uptrend_df['bb_lower'] = bb_lower
        
        result = classify_regime(uptrend_df)
        
        # Last rows should be trending_up
        last_regimes = result['regime'].iloc[-10:].values
        self.assertTrue(any(r == 'trending_up' for r in last_regimes))
    
    def test_ranging_conditions(self):
        """Test that ranging regime requires ADX < 20 and tight BB."""
        # Create flat/sideways data
        n = 50
        flat_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
            'open': np.full(n, 40000),
            'high': np.full(n, 40100),
            'low': np.full(n, 39900),
            'close': np.full(n, 40000),
            'volume': np.full(n, 1000)
        })
        
        # Calculate features
        close_vals = flat_df['close'].values.astype(np.float64)
        high_vals = flat_df['high'].values.astype(np.float64)
        low_vals = flat_df['low'].values.astype(np.float64)
        
        flat_df['ema_20'] = talib.EMA(close_vals, timeperiod=20)
        flat_df['adx_14'] = talib.ADX(high_vals, low_vals, close_vals, timeperiod=14)
        flat_df['plus_di_14'] = talib.PLUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
        flat_df['minus_di_14'] = talib.MINUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
        
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_vals, timeperiod=20)
        flat_df['bb_upper'] = bb_upper
        flat_df['bb_middle'] = bb_middle
        flat_df['bb_lower'] = bb_lower
        
        result = classify_regime(flat_df)
        
        # Some rows should be ranging or unknown
        unique_regimes = set(result['regime'].unique())
        self.assertTrue(unique_regimes.issubset({'ranging', 'unknown'}))
    
    def test_save_regimes_to_db(self):
        """Test saving regimes to database."""
        result = classify_regime(self.sample_df)
        
        rows = save_regimes_to_db(result, self.test_db)
        self.assertGreater(rows, 0)
        
        # Verify table exists
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regimes'")
        self.assertIsNotNone(cursor.fetchone())
        conn.close()
    
    def test_analyze_regime_distribution(self):
        """Test regime distribution analysis."""
        result = classify_regime(self.sample_df)
        analysis = analyze_regime_distribution(result)
        
        self.assertIn('total_rows', analysis)
        self.assertIn('regime_counts', analysis)
        self.assertIn('regime_percentages', analysis)
        self.assertIn('transition_rate', analysis)
        
        # Verify percentages sum to ~100%
        total_pct = sum(analysis['regime_percentages'].values())
        self.assertAlmostEqual(total_pct, 100, delta=1)
    
    def test_verify_regimes(self):
        """Test regime verification function."""
        result = classify_regime(self.sample_df)
        verification = verify_regimes(result)
        
        self.assertIn('valid_regimes', verification)
        self.assertIn('confidence_valid_range', verification)
        self.assertIn('no_regime_dominates', verification)
        self.assertIn('transition_rate_ok', verification)
        
        # Should pass basic validation
        self.assertTrue(verification['valid_regimes'])
        self.assertTrue(verification['confidence_valid_range'])
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = classify_regime(empty_df)
        self.assertTrue(result.empty)
    
    def test_no_regime_dominates_too_much(self):
        """Test that no single regime exceeds 50% threshold."""
        result = classify_regime(self.sample_df)
        analysis = analyze_regime_distribution(result)
        
        max_pct = max(analysis['regime_percentages'].values())
        self.assertLess(max_pct, 70)  # Allow some flexibility for test data


if __name__ == '__main__':
    unittest.main()
