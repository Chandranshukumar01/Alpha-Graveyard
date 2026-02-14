"""
data_ingest.py - Cryptocurrency OHLCV Data Ingestion Module

Fetches historical candlestick data from Binance using CCXT.
Supports resume capability, duplicate prevention, and rate limit handling.
"""

import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd


def fetch_ohlcv(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    since: str = '2019-01-01',
    exchange_id: str = 'binance',
    limit: int = 1000,
    max_retries: int = 3,
    db_path: str = 'btc_pipeline.db'
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from exchange using CCXT.
    
    Automatically resumes from last fetched timestamp if database exists.
    Handles rate limits with exponential backoff.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h', '1d')
        since: Start date string 'YYYY-MM-DD' (only used if no existing data)
        exchange_id: CCXT exchange identifier
        limit: Number of candles per request (max 1000 for most exchanges)
        max_retries: Max retry attempts for failed requests
        db_path: Path to SQLite database for resume capability
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,  # CCXT handles rate limits automatically
        'options': {
            'defaultType': 'spot',  # Spot market (not futures)
        }
    })
    
    # Check for existing data to resume from
    last_timestamp = get_last_timestamp(db_path)
    
    if last_timestamp:
        # Resume from last fetched candle + 1 hour
        since_ms = int(last_timestamp.timestamp() * 1000) + (60 * 60 * 1000)
        print(f"Resuming from {last_timestamp} ({since_ms})")
    else:
        # Start from beginning
        since_dt = datetime.strptime(since, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        since_ms = int(since_dt.timestamp() * 1000)
        print(f"Starting fresh from {since} ({since_ms})")
    
    all_candles = []
    fetch_count = 0
    
    while True:
        for attempt in range(max_retries):
            try:
                # Fetch candles
                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=limit
                )
                
                fetch_count += 1
                
                if not candles:
                    print(f"No more candles returned. Total fetches: {fetch_count}")
                    break
                
                all_candles.extend(candles)
                
                # Update since_ms to last candle + 1ms for next batch
                last_candle_time = candles[-1][0]
                since_ms = last_candle_time + 1
                
                # Progress update
                if fetch_count % 10 == 0:
                    last_dt = datetime.fromtimestamp(last_candle_time / 1000, tz=timezone.utc)
                    print(f"Fetched {len(all_candles)} candles, up to {last_dt}")
                
                # Rate limiting is handled by CCXT, but add small buffer
                time.sleep(exchange.rateLimit / 1000)
                
                break  # Success, exit retry loop
                
            except ccxt.NetworkError as e:
                print(f"Network error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
                    
            except ccxt.ExchangeError as e:
                print(f"Exchange error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        else:
            # All retries exhausted
            continue
        
        # Check if we got fewer candles than limit (end of data)
        if len(candles) < limit:
            print(f"Got {len(candles)} candles (< limit {limit}), stopping.")
            break
    
    # Convert to DataFrame
    if not all_candles:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    df = df.drop('timestamp_ms', axis=1)
    
    # Reorder columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"Total candles fetched: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def save_to_db(df: pd.DataFrame, db_path: str = 'btc_pipeline.db') -> int:
    """
    Save OHLCV DataFrame to SQLite database.
    
    Uses INSERT OR IGNORE to prevent duplicates based on timestamp primary key.
    Creates table if it doesn't exist.
    
    Args:
        df: DataFrame with columns timestamp, open, high, low, close, volume
        db_path: Path to SQLite database
        
    Returns:
        Number of rows inserted
    """
    if df.empty:
        print("No data to save")
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            timestamp DATETIME PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    """)
    
    # Insert data with duplicate prevention
    rows_inserted = 0
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO candles (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row['timestamp'].isoformat(),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume'])
        ))
        if cursor.rowcount > 0:
            rows_inserted += 1
    
    conn.commit()
    conn.close()
    
    print(f"Rows inserted: {rows_inserted} (skipped {len(df) - rows_inserted} duplicates)")
    return rows_inserted


def get_last_timestamp(db_path: str = 'btc_pipeline.db') -> Optional[datetime]:
    """
    Get the timestamp of the most recent candle in the database.
    
    Used to resume fetching without duplicates.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        datetime of last candle, or None if table doesn't exist or is empty
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='candles'
        """)
        if not cursor.fetchone():
            return None
        
        # Get last timestamp
        cursor.execute("SELECT MAX(timestamp) FROM candles")
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            # Parse ISO format timestamp
            return datetime.fromisoformat(result[0])
        return None
        
    except sqlite3.Error as e:
        print(f"Database error reading last timestamp: {e}")
        return None


def verify_data(db_path: str = 'btc_pipeline.db') -> dict:
    """
    Verify data integrity and return statistics.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with row count, date range, and gap analysis
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Basic stats
    cursor.execute("SELECT COUNT(*) FROM candles")
    row_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM candles")
    min_ts, max_ts = cursor.fetchone()
    
    # Check for large gaps (> 2 hours)
    cursor.execute("""
        SELECT timestamp,
               LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
        FROM candles
    """)
    rows = cursor.fetchall()
    
    gaps = []
    for ts, prev_ts in rows:
        if prev_ts:
            current = datetime.fromisoformat(ts)
            previous = datetime.fromisoformat(prev_ts)
            gap_hours = (current - previous).total_seconds() / 3600
            if gap_hours > 2:
                gaps.append((previous, current, gap_hours))
    
    conn.close()
    
    return {
        'row_count': row_count,
        'min_timestamp': min_ts,
        'max_timestamp': max_ts,
        'gaps_over_2h': len(gaps),
        'gap_details': gaps[:10]  # First 10 gaps
    }


if __name__ == '__main__':
    # Main execution: fetch and save data
    print("=" * 60)
    print("BTC/USDT 1h Data Ingestion")
    print("=" * 60)
    
    # Fetch data
    df = fetch_ohlcv(
        symbol='BTC/USDT',
        timeframe='1h',
        since='2019-01-01',
        db_path='btc_pipeline.db'
    )
    
    # Save to database
    rows_inserted = save_to_db(df, db_path='btc_pipeline.db')
    
    # Verify
    stats = verify_data(db_path='btc_pipeline.db')
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Total rows in DB: {stats['row_count']}")
    print(f"Date range: {stats['min_timestamp']} to {stats['max_timestamp']}")
    print(f"Gaps > 2 hours: {stats['gaps_over_2h']}")
    if stats['gap_details']:
        print("First few gaps:")
        for prev, curr, hours in stats['gap_details'][:5]:
            print(f"  {prev} -> {curr}: {hours:.1f}h")
    print("=" * 60)
