"""
feature_engine.py - Technical Indicator Calculation Module

Calculates 20 technical indicators from OHLCV candlestick data.
Uses TA-Lib for standard indicators, Pandas for derived features.
All calculations use only past data (no lookahead bias).
"""

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
import talib



def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1h OHLCV data to higher timeframe."""
    df_copy = df.copy()
    if 'timestamp' in df_copy.columns:
        df_copy = df_copy.set_index('timestamp')
    elif not isinstance(df_copy.index, pd.DatetimeIndex):
         # If no timestamp column and no datetime index, try to rely on index if it's datetime
         pass
    
    resampled = df_copy.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    return resampled

def calculate_mtf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 4h and 1d features and merge back to 1h DataFrame."""
    if df.empty: return df
    
    # Work with timestamp index
    work_df = df.copy()
    original_idx_is_dt = isinstance(df.index, pd.DatetimeIndex)
    
    if 'timestamp' in work_df.columns:
        work_df = work_df.set_index('timestamp')
    elif not original_idx_is_dt:
        if 'timestamp' in df.columns: work_df = df.set_index('timestamp')
        else: return df # Cannot proceed without timestamp
            
    # 4H Features
    try:
        df_4h = resample_to_timeframe(work_df, '4h')
        if len(df_4h) > 50: # Need enough data for EMA
            close_4h = df_4h['close'].values.astype(np.float64)
            high_4h = df_4h['high'].values.astype(np.float64)
            low_4h = df_4h['low'].values.astype(np.float64)
            
            df_4h['ema_50_4h'] = talib.EMA(close_4h, timeperiod=50)
            df_4h['ema_200_4h'] = talib.EMA(close_4h, timeperiod=200)
            df_4h['rsi_14_4h'] = talib.RSI(close_4h, timeperiod=14)
            df_4h['atr_14_4h'] = talib.ATR(high_4h, low_4h, close_4h, timeperiod=14)
            
            # Shift 4h features by 1 to prevent lookahead
            features_4h = ['ema_50_4h', 'ema_200_4h', 'rsi_14_4h', 'atr_14_4h']
            df_4h_shifted = df_4h[features_4h].shift(1)
            
            # Reindex to 1h and ffill
            # Using reindex with method='ffill' propagates the shifted value forward
            df_4h_aligned = df_4h_shifted.reindex(work_df.index, method='ffill')
            
            for col in features_4h:
                work_df[col] = df_4h_aligned[col]
    except Exception as e:
        print(f"Error calculating 4h features: {e}")
        
    # 1D Features
    try:
        df_1d = resample_to_timeframe(work_df, '1d')
        if len(df_1d) > 50: 
            close_1d = df_1d['close'].values.astype(np.float64)
            high_1d = df_1d['high'].values.astype(np.float64)
            low_1d = df_1d['low'].values.astype(np.float64)
            
            df_1d['ema_50_1d'] = talib.EMA(close_1d, timeperiod=50)
            df_1d['ema_200_1d'] = talib.EMA(close_1d, timeperiod=200)
            df_1d['rsi_14_1d'] = talib.RSI(close_1d, timeperiod=14)
            df_1d['atr_14_1d'] = talib.ATR(high_1d, low_1d, close_1d, timeperiod=14)
            
            # Shift 1d features by 1
            features_1d = ['ema_50_1d', 'ema_200_1d', 'rsi_14_1d', 'atr_14_1d']
            df_1d_shifted = df_1d[features_1d].shift(1)
            
            df_1d_aligned = df_1d_shifted.reindex(work_df.index, method='ffill')
            
            for col in features_1d:
                work_df[col] = df_1d_aligned[col]
    except Exception as e:
        print(f"Error calculating 1d features: {e}")
        
    if not original_idx_is_dt:
        work_df = work_df.reset_index()
        
    return work_df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20 technical indicators from OHLCV data.
    
    Input DataFrame columns: timestamp, open, high, low, close, volume
    Output DataFrame adds 20 feature columns (all shifted to prevent lookahead).
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with original columns + 20 indicator columns
    """
    if df.empty:
        return df
    
    # Ensure required columns exist
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Sort by timestamp to ensure correct calculation order
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Extract numpy arrays for TA-Lib (faster than pandas series)
    open_vals = df['open'].values.astype(np.float64)
    high_vals = df['high'].values.astype(np.float64)
    low_vals = df['low'].values.astype(np.float64)
    close_vals = df['close'].values.astype(np.float64)
    volume_vals = df['volume'].values.astype(np.float64)
    
    # --- TREND INDICATORS ---
    
    # 1. EMA 20 (keep this, drop SMA as they're highly correlated)
    df['ema_20'] = talib.EMA(close_vals, timeperiod=20)
    
    # 2. MACD - keep only histogram (macd_hist), drop macd and macd_signal (correlated)
    macd, macd_signal, macd_hist = talib.MACD(
        close_vals, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['macd_hist'] = macd_hist
    
    # --- MOMENTUM INDICATORS ---
    
    # 3. RSI 14
    df['rsi_14'] = talib.RSI(close_vals, timeperiod=14)
    
    # 4-5. Stochastic (14, 3, 3) - keep both as they measure different things
    stoch_k, stoch_d = talib.STOCH(
        high_vals, low_vals, close_vals,
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    # 6. CCI 20
    df['cci_20'] = talib.CCI(high_vals, low_vals, close_vals, timeperiod=20)
    
    # --- VOLATILITY INDICATORS ---
    
    # 7. ATR 14
    df['atr_14'] = talib.ATR(high_vals, low_vals, close_vals, timeperiod=14)
    
    # 8. Bollinger Band Width (instead of upper/middle/lower which are correlated)
    # Calculate width as % of middle band
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        close_vals, timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (close_vals - bb_lower) / (bb_upper - bb_lower)
    
    # --- VOLUME INDICATORS ---
    
    # 9. OBV
    df['obv'] = talib.OBV(close_vals, volume_vals)
    
    # 10. VWAP 20 (manual calculation - shifted to prevent lookahead)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_volume = typical_price * df['volume']
    df['vwap_20'] = tp_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # 10b. VWAP Bands (VWAP +/- 1 std dev of typical price)
    # Estimate VWAP std dev using rolling std of typical price
    tp_std = typical_price.rolling(window=20).std()
    df['vwap_upper_band'] = df['vwap_20'] + (1.0 * tp_std)
    df['vwap_lower_band'] = df['vwap_20'] - (1.0 * tp_std)
    
    # 10c. VWAP Deviation (percentage distance from VWAP)
    df['vwap_deviation'] = (df['close'] - df['vwap_20']) / df['vwap_20']
    
    # --- DERIVED FEATURES (Pandas) ---
    
    # 11. Returns 1h (prevent lookahead with shift)
    df['returns_1h'] = df['close'].pct_change()
    
    # 12. Volatility 20h (rolling std of 1h returns)
    df['volatility_20h'] = df['returns_1h'].rolling(window=20).std()
    
    # --- ADDITIONAL TREND INDICATORS ---
    
    # 13. ADX 14 (trend strength indicator)
    # 13. Directional Movement (ADX, +DI, -DI)
    df['adx_14'] = talib.ADX(high_vals, low_vals, close_vals, timeperiod=14)
    df['plus_di_14'] = talib.PLUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
    df['minus_di_14'] = talib.MINUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
    
    # 14. Multi-Timeframe Features
    df = calculate_mtf_features(df)
    
    # --- CLEANUP (Shift & Drop) ---
    
    # List of features to shift (shift 1 to prevent lookahead bias)
    # MTF features are ALREADY shifted inside calculate_mtf_features and should not be shifted again.
    features_to_shift = [
        'ema_20', 'macd_hist', 'rsi_14', 'stoch_k', 'stoch_d', 'cci_20',
        'atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'obv', 'vwap_20', 'vwap_upper_band', 'vwap_lower_band', 'vwap_deviation', 'returns_1h',
        'volatility_20h', 'adx_14', 'plus_di_14', 'minus_di_14'
    ]
    
    for col in features_to_shift:
        if col in df.columns:
            df[col] = df[col].shift(1)
    
    # Drop rows with insufficient data (first 50 rows have NaN indicators)
    # Keep them but let downstream handle NaN filtering
    
    return df


def save_features_to_db(df: pd.DataFrame, db_path: str = 'btc_pipeline.db') -> int:
    """
    Save calculated features to SQLite database.
    
    Args:
        df: DataFrame with feature columns
        db_path: Path to SQLite database
        
    Returns:
        Number of rows inserted
    """
    if df.empty:
        return 0
    
    # Select only feature columns + timestamp for storage
    feature_cols = [
        'timestamp', 'ema_20', 'macd_hist',
        'rsi_14', 'stoch_k', 'stoch_d', 'cci_20',
        'atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'obv', 'vwap_20', 'vwap_upper_band', 'vwap_lower_band', 'vwap_deviation',
        'returns_1h', 'volatility_20h',
        'adx_14', 'plus_di_14', 'minus_di_14',
        'ema_50_4h', 'ema_200_4h', 'rsi_14_4h', 'atr_14_4h',
        'ema_50_1d', 'ema_200_1d', 'rsi_14_1d', 'atr_14_1d'
    ]
    
    # Only include columns that exist in the DataFrame
    available_cols = [c for c in feature_cols if c in df.columns]
    features_df = df[available_cols].copy()
    
    # Convert timestamp to string for SQLite
    if 'timestamp' in features_df.columns:
        features_df['timestamp'] = features_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with schema matching actual calculated features
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            timestamp DATETIME PRIMARY KEY,
            ema_20 REAL, macd_hist REAL,
            rsi_14 REAL, stoch_k REAL, stoch_d REAL, cci_20 REAL,
            atr_14 REAL, bb_upper REAL, bb_middle REAL, bb_lower REAL,
            bb_width REAL, bb_position REAL,
            obv REAL, vwap_20 REAL, vwap_upper_band REAL, vwap_lower_band REAL, vwap_deviation REAL,
            returns_1h REAL, volatility_20h REAL,
            adx_14 REAL, plus_di_14 REAL, minus_di_14 REAL,
            ema_50_4h REAL, ema_200_4h REAL, rsi_14_4h REAL, atr_14_4h REAL,
            ema_50_1d REAL, ema_200_1d REAL, rsi_14_1d REAL, atr_14_1d REAL
        )
    """)
    
    # Bulk insert with duplicate prevention
    records = features_df.to_records(index=False).tolist()
    
    # Build INSERT OR IGNORE query dynamically based on available columns
    col_names = ', '.join(available_cols)
    placeholders = ', '.join(['?' for _ in available_cols])
    
    cursor.executemany(f"""
        INSERT OR IGNORE INTO features ({col_names})
        VALUES ({placeholders})
    """, records)
    
    rows_inserted = cursor.rowcount
    conn.commit()
    conn.close()
    
    return rows_inserted


def load_candles_from_db(db_path: str = 'btc_pipeline.db', limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load OHLCV data from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        limit: Maximum rows to load (for testing)
        
    Returns:
        DataFrame with OHLCV columns
    """
    conn = sqlite3.connect(db_path)
    
    query = "SELECT * FROM candles ORDER BY timestamp"
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def calculate_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for features.
    
    Returns DataFrame with correlation pairs > 0.95 flagged.
    """
    feature_cols = [
        'ema_20', 'macd_hist',
        'rsi_14', 'stoch_k', 'stoch_d', 'cci_20',
        'atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'obv', 'vwap_20', 'vwap_upper_band', 'vwap_lower_band', 'vwap_deviation',
        'returns_1h', 'volatility_20h',
        'adx_14', 'plus_di_14', 'minus_di_14',
        'ema_50_4h', 'ema_200_4h', 'rsi_14_4h', 'atr_14_4h',
        'ema_50_1d', 'ema_200_1d', 'rsi_14_1d', 'atr_14_1d'
    ]
    
    # Only include columns that exist and have non-NaN values
    available_cols = [c for c in feature_cols if c in df.columns]
    features_df = df[available_cols].dropna()
    
    if features_df.empty:
        return pd.DataFrame()
    
    corr_matrix = features_df.corr()
    
    # Find pairs with correlation > 0.95
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.95:
                high_corr.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    return pd.DataFrame(high_corr)


def verify_features(df: pd.DataFrame) -> dict:
    """
    Verify feature quality and return statistics.
    
    Returns dict with NaN counts, valid ranges, correlation flags.
    """
    feature_cols = [
        'ema_20', 'macd_hist',
        'rsi_14', 'stoch_k', 'stoch_d', 'cci_20',
        'atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'obv', 'vwap_20', 'vwap_upper_band', 'vwap_lower_band', 'vwap_deviation',
        'returns_1h', 'volatility_20h',
        'adx_14', 'plus_di_14', 'minus_di_14',
        'ema_50_4h', 'ema_200_4h', 'rsi_14_4h', 'atr_14_4h',
        'ema_50_1d', 'ema_200_1d', 'rsi_14_1d', 'atr_14_1d'
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    
    # Count NaN values per column
    nan_counts = df[available_cols].isna().sum().to_dict()
    
    # Count rows with all NaN (indicator warmup period)
    total_nan_rows = df[available_cols].isna().all(axis=1).sum()
    
    # Rows with valid data (after warmup)
    valid_rows = len(df) - df[available_cols].isna().any(axis=1).sum()
    
    # Check for high correlations
    high_corr = calculate_feature_correlations(df)
    
    return {
        'total_rows': len(df),
        'valid_rows': valid_rows,
        'nan_rows_all_features': total_nan_rows,
        'nan_counts': nan_counts,
        'high_correlations': len(high_corr),
        'correlation_details': high_corr.to_dict('records') if not high_corr.empty else []
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Feature Engine - Calculate Technical Indicators")
    print("=" * 60)
    
    # Load sample data (or all data if available)
    try:
        df = load_candles_from_db(limit=1000)  # Start with 1000 for testing
        print(f"Loaded {len(df)} rows of candle data")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create sample data for testing
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'open': np.random.randn(100).cumsum() + 40000,
            'high': np.random.randn(100).cumsum() + 40500,
            'low': np.random.randn(100).cumsum() + 39500,
            'close': np.random.randn(100).cumsum() + 40000,
            'volume': np.random.randint(100, 1000, 100)
        })
        print(f"Using sample data: {len(df)} rows")
    
    # Calculate features
    print("\nCalculating 20 technical indicators...")
    df_features = calculate_features(df)
    
    # Verify quality
    print("\nVerifying feature quality...")
    verification = verify_features(df_features)
    print(f"Total rows: {verification['total_rows']}")
    print(f"Rows with valid features: {verification['valid_rows']}")
    print(f"High correlation pairs (>0.95): {verification['high_correlations']}")
    
    # Save to database
    print("\nSaving features to database...")
    rows = save_features_to_db(df_features)
    print(f"Saved {rows} rows to features table")
    
    print("=" * 60)
    print("Feature calculation complete")
    print("=" * 60)
