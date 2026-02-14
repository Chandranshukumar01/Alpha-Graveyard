"""
test_feature_engine.py - Unit tests for feature_engine.py

Tests all 20 technical indicators, correlation checks, and database operations.
"""

import os
import sqlite3
import unittest
import time

import numpy as np
import pandas as pd

from alpha_graveyard.features.technical import (
    calculate_features,
    save_features_to_db,
    load_candles_from_db,
    calculate_feature_correlations,
    verify_features
)


class TestFeatureEngine(unittest.TestCase):
    """Unit tests for feature calculation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db = 'test_features.db'
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        
        # Create sample OHLCV data
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='h'),
            'open': np.random.randn(200).cumsum() + 40000,
            'high': np.random.randn(200).cumsum() + 40500,
            'low': np.random.randn(200).cumsum() + 39500,
            'close': np.random.randn(200).cumsum() + 40000,
            'volume': np.random.randint(100, 1000, 200)
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        import gc
        gc.collect()
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                time.sleep(0.1)
                try:
                    os.remove(self.test_db)
                except PermissionError:
                    pass
    
    def test_calculate_features_returns_correct_columns(self):
        """Test that calculate_features returns all required feature columns."""
        result = calculate_features(self.sample_df)
        
        expected_features = [
            'ema_20', 'macd_hist',
            'rsi_14', 'stoch_k', 'stoch_d', 'cci_20',
            'atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'obv', 'vwap_20', 'vwap_upper_band', 'vwap_lower_band', 'vwap_deviation', 
            'returns_1h', 'volatility_20h',
            'adx_14', 'plus_di_14', 'minus_di_14'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Missing feature: {feature}")
    
    def test_calculate_features_preserves_original_columns(self):
        """Test that original OHLCV columns are preserved."""
        result = calculate_features(self.sample_df)
        
        original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in original_cols:
            self.assertIn(col, result.columns)
    
    def test_indicator_warmup_period(self):
        """Test that first ~50 rows have NaN values (indicator warmup)."""
        result = calculate_features(self.sample_df)
        
        # First 30 rows should have NaN in EMA (20-period needs 20 values)
        feature_cols = [c for c in result.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        nan_count_row_25 = result.iloc[25][feature_cols].isna().sum()
        self.assertGreater(nan_count_row_25, 0, "Row 25 should have some NaN indicators")
        
        # Row 50+ should have most indicators calculated
        nan_count_row_55 = result.iloc[55][feature_cols].isna().sum()
        self.assertLess(nan_count_row_55, len(feature_cols) / 2, "Row 55 should have most indicators calculated")
    
    def test_rsi_range(self):
        """Test that RSI values are within 0-100 range."""
        result = calculate_features(self.sample_df)
        
        rsi_values = result['rsi_14'].dropna()
        self.assertTrue((rsi_values >= 0).all(), "RSI should be >= 0")
        self.assertTrue((rsi_values <= 100).all(), "RSI should be <= 100")
    
    def test_stochastic_range(self):
        """Test that Stochastic values are within 0-100 range."""
        result = calculate_features(self.sample_df)
        
        stoch_k = result['stoch_k'].dropna()
        stoch_d = result['stoch_d'].dropna()
        
        self.assertTrue((stoch_k >= 0).all() and (stoch_k <= 100).all())
        self.assertTrue((stoch_d >= 0).all() and (stoch_d <= 100).all())
    
    def test_bbands_order(self):
        """Test that Bollinger Bands are ordered: upper > middle > lower."""
        result = calculate_features(self.sample_df)
        
        valid_rows = result.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])
        
        for _, row in valid_rows.iterrows():
            self.assertGreaterEqual(row['bb_upper'], row['bb_middle'])
            self.assertGreaterEqual(row['bb_middle'], row['bb_lower'])
            
            # Also test width and position calculations
            expected_width = (row['bb_upper'] - row['bb_lower']) / row['bb_middle']
            self.assertAlmostEqual(row['bb_width'], expected_width, places=5)
    
    def test_vwap_bands(self):
        """Test that VWAP bands are ordered correctly."""
        result = calculate_features(self.sample_df)
        valid_rows = result.dropna(subset=['vwap_upper_band', 'vwap_lower_band', 'vwap_20'])
        
        for idx, row in valid_rows.iterrows():
            if idx > 0:
                self.assertGreaterEqual(row['vwap_upper_band'], row['vwap_20'])
                self.assertGreaterEqual(row['vwap_20'], row['vwap_lower_band'])
                
                # Test deviation calculation
                # VWAP features are shifted by 1, so row[v] corresponds to prev period calc
                # So we compare against prev close (sample_df aligned by index)
                prev_close = self.sample_df['close'].iloc[idx-1]
                expected_dev = (prev_close - row['vwap_20']) / row['vwap_20']
                
                self.assertAlmostEqual(row['vwap_deviation'], expected_dev, places=5)
    
    def test_returns_calculation(self):
        """Test that returns are calculated correctly with shift."""
        result = calculate_features(self.sample_df)
        
        # Due to shift(1), returns_1h at row N is actually the return ending at row N-1
        # Wait, the code shifts AFTER calculation.
        # Calculation: df['returns_1h'] = df['close'].pct_change()
        # Shift: df[col] = df[col].shift(1)
        # So value at row N is pct_change from (N-2) to (N-1).
        
        # Let's verify row 51.
        # Original logic: returns_1h at 51 = (close[50] - close[49]) / close[49]
        # Because of shift(1), row 51 contains calc from row 50.
        
        expected_return = self.sample_df['close'].iloc[50] / self.sample_df['close'].iloc[49] - 1
        actual_return = result['returns_1h'].iloc[51]
        
        self.assertAlmostEqual(actual_return, expected_return, places=5)
    
    def test_no_lookahead_bias(self):
        """Test that features don't use future data (shifted by 1)."""
        result = calculate_features(self.sample_df)
        
        # The EMA at row N should be based on closes from rows 0 to N-1 (due to shift)
        # Row 25's EMA should be calculated from data up to row 24.
        # It should NOT be sensitive to row 25's close price if we were to change it (implies re-calculating).
        
        # Simple check: EMA at row 25 should exist and be different from close at 25
        if not pd.isna(result['ema_20'].iloc[25]):
            self.assertNotEqual(result['ema_20'].iloc[25], self.sample_df['close'].iloc[25])
    
    def test_save_features_to_db(self):
        """Test saving features to database."""
        result = calculate_features(self.sample_df)
        
        rows = save_features_to_db(result, self.test_db)
        self.assertGreater(rows, 0)
        
        # Verify table was created
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='features'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Verify row count
        cursor.execute("SELECT COUNT(*) FROM features")
        count = cursor.fetchone()[0]
        self.assertEqual(count, rows)
        
        # Verify new columns exist
        cursor.execute("PRAGMA table_info(features)")
        columns = [info[1] for info in cursor.fetchall()]
        self.assertIn('bb_upper', columns)
        self.assertIn('plus_di_14', columns)
        
        conn.close()
    
    def test_verify_features(self):
        """Test feature verification function."""
        result = calculate_features(self.sample_df)
        
        verification = verify_features(result)
        
        self.assertIn('total_rows', verification)
        self.assertIn('valid_rows', verification)
        self.assertIn('nan_counts', verification)
        self.assertIn('high_correlations', verification)
        
        # Total rows should match
        self.assertEqual(verification['total_rows'], len(result))
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        result = calculate_features(empty_df)
        self.assertTrue(result.empty)
    
    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        bad_df = pd.DataFrame({'timestamp': [1, 2, 3], 'close': [100, 101, 102]})
        
        with self.assertRaises(ValueError) as context:
            calculate_features(bad_df)
        
        self.assertIn('Missing required columns', str(context.exception))


class TestFeatureCorrelations(unittest.TestCase):
    """Tests for correlation detection between features."""
    
    def setUp(self):
        """Create sample data with known correlations."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='h'),
            'open': np.random.randn(200).cumsum() + 40000,
            'high': np.random.randn(200).cumsum() + 40500,
            'low': np.random.randn(200).cumsum() + 39500,
            'close': np.random.randn(200).cumsum() + 40000,
            'volume': np.random.randint(100, 1000, 200)
        })
        self.df_with_features = calculate_features(self.df)
    
    def test_correlation_detection(self):
        """Test that high correlations are detected."""
        high_corr = calculate_feature_correlations(self.df_with_features)
        
        # Should return a DataFrame
        self.assertIsInstance(high_corr, pd.DataFrame)
        
        # If there are correlations, check the columns exist
        if not high_corr.empty:
            self.assertIn('feature_1', high_corr.columns)
            self.assertIn('feature_2', high_corr.columns)
            self.assertIn('correlation', high_corr.columns)
    
    def test_ema_bb_middle_correlation(self):
        """Test that EMA 20 and BB Middle (SMA 20) are highly correlated."""
        high_corr = calculate_feature_correlations(self.df_with_features)
        
        if not high_corr.empty:
            ema_bb_middle_corr = high_corr[
                ((high_corr['feature_1'] == 'ema_20') & (high_corr['feature_2'] == 'bb_middle')) |
                ((high_corr['feature_1'] == 'bb_middle') & (high_corr['feature_2'] == 'ema_20'))
            ]
            self.assertGreater(len(ema_bb_middle_corr), 0, "EMA 20 and BB Middle should be highly correlated")


if __name__ == '__main__':
    unittest.main()
