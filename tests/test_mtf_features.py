
import unittest
import pandas as pd
import numpy as np
from alpha_graveyard.features import technical as feature_engine
import os
import sqlite3

class TestMTFFeatures(unittest.TestCase):
    
    def setUp(self):
        # Create a large enough dataset for MTF calculations
        # 50 days * 24h = 1200 hours needed for daily EMA-50
        # Let's generate 2000 hours (~83 days)
        periods = 2000
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='h')
        
        # Consistent random data
        np.random.seed(42)
        close = np.cumsum(np.random.randn(periods)) + 10000
        high = close + np.abs(np.random.randn(periods))
        low = close - np.abs(np.random.randn(periods))
        open_p = close + np.random.randn(periods) * 0.5
        volume = np.random.randint(100, 10000, periods)
        
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        self.db_path = 'test_mtf_features.db'
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_mtf_columns_exist_and_populated(self):
        """Test that MTF columns are created and have valid data."""
        df_features = feature_engine.calculate_features(self.df)
        
        # Check 4h columns
        self.assertIn('ema_50_4h', df_features.columns)
        self.assertIn('rsi_14_4h', df_features.columns)
        
        # Check 1d columns
        self.assertIn('ema_50_1d', df_features.columns)
        self.assertIn('atr_14_1d', df_features.columns)
        
        # Check for non-null values (tail should be valid)
        # Daily EMA-50 needs 50 days (1200 hrs) + shift
        # So last rows should be valid
        self.assertFalse(pd.isna(df_features['ema_50_1d'].iloc[-1]))
        self.assertFalse(pd.isna(df_features['ema_50_4h'].iloc[-1]))
        
        # Verify shifting/ffill logic:
        # 4h features should remain constant for 4 hours
        # e.g. rows 100, 101, 102, 103 might share same feature values
        # Let's check a random slice
        
        # Pandas resample aligns to 00, 04, 08...
        # Row at 100 is 2023-01-05 04:00:00
        # 4h candle 04:00-08:00 closes at 08:00.
        # But we shift features.
        # So row 04:00 should have features from 00:00-04:00 candle?
        # No, row 04:00 is "open". It's the START of the 04:00 bar.
        # It should know about 00:00-04:00 candle features.
        # So 04:00, 05:00, 06:00, 07:00 should all have same 4h features (from 00:00-04:00).
        
        # Let's check rows at index 100, 101, 102, 103
        vals = df_features['ema_50_4h'].iloc[100:104].values
        # Note: EMA might be NaN at row 100 if not enough data, but with 2000 rows it should be fine?
        # 100 rows = 4 days ? No, 100 hours.
        # EMA-50-4h needs 50 * 4h = 200 hours.
        # So row 100 is still NaN.
        
        # Let's check rows at index 1000 (well into dataset)
        vals = df_features['ema_50_4h'].iloc[1000:1004].values
        self.assertEqual(len(set(vals)), 1, "4h feature should be constant for 4 hours (ffilled)")
        
        # And next block should be different
        next_val = df_features['ema_50_4h'].iloc[1004]
        # It might be same if price didn't change enough, unlikely with random float
        self.assertNotEqual(vals[0], next_val)

    def test_save_features_to_db_mtf(self):
        """Test DB saving with new columns."""
        df_features = feature_engine.calculate_features(self.df)
        feature_engine.save_features_to_db(df_features, self.db_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check schema
        cursor.execute("PRAGMA table_info(features)")
        columns = {row[1] for row in cursor.fetchall()}
        
        self.assertIn('ema_50_4h', columns)
        self.assertIn('ema_200_1d', columns)
        
        # Check data
        cursor.execute("SELECT ema_50_4h FROM features LIMIT 1 OFFSET 1900")
        val = cursor.fetchone()[0]
        self.assertIsNotNone(val)
        
        conn.close()

if __name__ == '__main__':
    unittest.main()
