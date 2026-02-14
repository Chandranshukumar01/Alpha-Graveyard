"""
test_data_ingest.py - Unit tests for data_ingest.py

Tests fetch_ohlcv(), save_to_db(), get_last_timestamp(), and verify_data().
Uses small data samples to avoid long test times.
"""

import os
import sqlite3
import unittest
from datetime import datetime, timezone

import pytest
import pandas as pd

from alpha_graveyard.engine.data import fetch_ohlcv, save_to_db, get_last_timestamp, verify_data

# Skip integration tests in CI (Binance API blocked from GitHub Actions)
IN_CI = os.environ.get('CI') == 'true'


class TestDataIngest(unittest.TestCase):
    """Unit tests for data ingestion module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db = 'test_btc_pipeline.db'
        # Clean up any existing test database
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Force close any open SQLite connections before deleting
        import gc
        gc.collect()
        
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                # If still locked, try again after short delay
                import time
                time.sleep(0.1)
                try:
                    os.remove(self.test_db)
                except PermissionError:
                    pass  # Best effort cleanup
    
    def test_get_last_timestamp_empty_db(self):
        result = get_last_timestamp(self.test_db)
        self.assertIsNone(result)
    
    def test_save_to_db_creates_table(self):
        """Test that save_to_db creates the candles table."""
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 00:00:00+00:00', '2024-01-01 01:00:00+00:00']),
            'open': [40000.0, 40100.0],
            'high': [40500.0, 40600.0],
            'low': [39500.0, 39600.0],
            'close': [40200.0, 40300.0],
            'volume': [100.0, 150.0]
        })
        
        rows = save_to_db(df, self.test_db)
        self.assertEqual(rows, 2)
        
        # Verify table was created
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='candles'")
        self.assertIsNotNone(cursor.fetchone())
        conn.close()
    
    def test_save_to_db_prevents_duplicates(self):
        """Test that save_to_db prevents duplicate timestamps."""
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 00:00:00+00:00']),
            'open': [40000.0],
            'high': [40500.0],
            'low': [39500.0],
            'close': [40200.0],
            'volume': [100.0]
        })
        
        # First save
        rows1 = save_to_db(df, self.test_db)
        self.assertEqual(rows1, 1)
        
        # Second save (duplicate)
        rows2 = save_to_db(df, self.test_db)
        self.assertEqual(rows2, 0)  # Should skip duplicate
        
        # Verify only 1 row in DB
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candles")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()
    
    def test_get_last_timestamp_returns_correct_value(self):
        """Test that get_last_timestamp returns the most recent timestamp."""
        # Create sample data with multiple rows
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 00:00:00+00:00',
                '2024-01-01 01:00:00+00:00',
                '2024-01-01 02:00:00+00:00'
            ]),
            'open': [40000.0, 40100.0, 40200.0],
            'high': [40500.0, 40600.0, 40700.0],
            'low': [39500.0, 39600.0, 39700.0],
            'close': [40200.0, 40300.0, 40400.0],
            'volume': [100.0, 150.0, 200.0]
        })
        
        save_to_db(df, self.test_db)
        
        # Get last timestamp
        last_ts = get_last_timestamp(self.test_db)
        self.assertIsNotNone(last_ts)
        
        expected = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(last_ts, expected)
    
    def test_verify_data_returns_correct_stats(self):
        """Test that verify_data returns correct statistics."""
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 00:00:00+00:00',
                '2024-01-01 01:00:00+00:00',
                '2024-01-01 02:00:00+00:00'
            ]),
            'open': [40000.0, 40100.0, 40200.0],
            'high': [40500.0, 40600.0, 40700.0],
            'low': [39500.0, 39600.0, 39700.0],
            'close': [40200.0, 40300.0, 40400.0],
            'volume': [100.0, 150.0, 200.0]
        })
        
        save_to_db(df, self.test_db)
        
        stats = verify_data(self.test_db)
        
        self.assertEqual(stats['row_count'], 3)
        self.assertEqual(stats['gaps_over_2h'], 0)  # No gaps in hourly data
    
    def test_verify_data_detects_gaps(self):
        """Test that verify_data detects gaps > 2 hours."""
        # Create data with a gap
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 00:00:00+00:00',
                '2024-01-01 01:00:00+00:00',
                '2024-01-01 05:00:00+00:00',  # 4 hour gap
                '2024-01-01 06:00:00+00:00'
            ]),
            'open': [40000.0, 40100.0, 40200.0, 40300.0],
            'high': [40500.0, 40600.0, 40700.0, 40800.0],
            'low': [39500.0, 39600.0, 39700.0, 39800.0],
            'close': [40200.0, 40300.0, 40400.0, 40500.0],
            'volume': [100.0, 150.0, 200.0, 250.0]
        })
        
        save_to_db(df, self.test_db)
        
        stats = verify_data(self.test_db)
        
        self.assertEqual(stats['gaps_over_2h'], 1)
        self.assertEqual(len(stats['gap_details']), 1)
    
    def test_save_to_db_empty_dataframe(self):
        """Test that save_to_db handles empty DataFrame gracefully."""
        df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        rows = save_to_db(df, self.test_db)
        self.assertEqual(rows, 0)


class TestDataIngestIntegration(unittest.TestCase):
    """Integration tests that actually fetch data from exchange."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db = 'test_integration.db'
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Force close any open SQLite connections before deleting
        import gc
        gc.collect()
        
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                # If still locked, try again after short delay
                import time
                time.sleep(0.1)
                try:
                    os.remove(self.test_db)
                except PermissionError:
                    pass  # Best effort cleanup
    
    @unittest.skipIf(IN_CI, "Integration test - Binance API blocked in CI")
    def test_fetch_small_sample(self):
        """Fetch small sample of real data (1 day) to verify CCXT works."""
        # Fetch just 1 day of data
        df = fetch_ohlcv(
            symbol='BTC/USDT',
            timeframe='1h',
            since='2024-01-01',
            db_path=self.test_db,
            limit=100  # Small limit for quick test
        )
        
        # Should have approximately 24 rows (1 day of hourly data)
        # But might be less depending on current date
        self.assertGreater(len(df), 0)
        
        # Verify columns
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(df.columns), expected_cols)
        
        # Verify data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))
        
        print(f"Fetched {len(df)} rows of real data")
    
    @unittest.skipIf(IN_CI, "Integration test - Binance API blocked in CI")
    def test_fetch_and_save_integration(self):
        """Test full fetch and save cycle with resume."""
        # First fetch - small amount
        df1 = fetch_ohlcv(
            symbol='BTC/USDT',
            timeframe='1h',
            since='2024-01-01',
            db_path=self.test_db,
            limit=50
        )
        
        rows1 = save_to_db(df1, self.test_db)
        self.assertGreater(rows1, 0)
        
        # Second fetch - should resume from last timestamp
        df2 = fetch_ohlcv(
            symbol='BTC/USDT',
            timeframe='1h',
            since='2024-01-01',  # This should be ignored due to resume
            db_path=self.test_db,
            limit=50
        )
        
        rows2 = save_to_db(df2, self.test_db)
        
        # Verify total rows in DB
        stats = verify_data(self.test_db)
        print(f"Total rows after resume: {stats['row_count']}")


if __name__ == '__main__':
    unittest.main()
