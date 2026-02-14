"""
regime_classifier.py - Market Regime Classification Module

Classifies each timestamp into one of 5 market regimes based on technical indicators.
Uses only past data (no lookahead bias).
"""

import sqlite3
import json
from typing import Optional

import numpy as np
import pandas as pd
import talib


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify market regime for each row based on indicator values.
    
    Regimes:
    - trending_up: ADX > 25, +DI > -DI, close > EMA(20)
    - trending_down: ADX > 25, -DI > +DI, close < EMA(20)
    - ranging: ADX < 20, BB width < 5%
    - high_vol: ATR(14) > 2× 100-period ATR
    - unknown: None of above
    
    Args:
        df: DataFrame with OHLCV + features (must include ADX, DI, BB, ATR, EMA)
        
    Returns:
        DataFrame with added 'regime' and 'confidence' columns
    """
    if df.empty:
        df['regime'] = []
        df['confidence'] = []
        return df
    
    # Ensure required columns exist
    required = ['high', 'low', 'close', 'ema_20', 'atr_14', 'adx_14', 'plus_di_14']
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Calculate missing indicators
        close_vals = df['close'].values.astype(np.float64)
        high_vals = df['high'].values.astype(np.float64)
        low_vals = df['low'].values.astype(np.float64)
        
        if 'ema_20' not in df.columns:
            df['ema_20'] = talib.EMA(close_vals, timeperiod=20)
        if 'atr_14' not in df.columns:
            df['atr_14'] = talib.ATR(high_vals, low_vals, close_vals, timeperiod=14)
        if 'adx_14' not in df.columns:
            df['adx_14'] = talib.ADX(high_vals, low_vals, close_vals, timeperiod=14)
        if 'plus_di_14' not in df.columns:
            df['plus_di_14'] = talib.PLUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
    
    # Calculate minus DI if not present
    if 'minus_di_14' not in df.columns:
        high_vals = df['high'].values.astype(np.float64)
        low_vals = df['low'].values.astype(np.float64)
        close_vals = df['close'].values.astype(np.float64)
        df['minus_di_14'] = talib.MINUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
    
    # Calculate Bollinger Band width if BB columns exist
    if all(c in df.columns for c in ['bb_upper', 'bb_lower', 'bb_middle']):
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    else:
        # Calculate BB width manually
        close_vals = df['close'].values.astype(np.float64)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_vals, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # Calculate 100-period ATR for high_vol detection
    high_vals = df['high'].values.astype(np.float64)
    low_vals = df['low'].values.astype(np.float64)
    close_vals = df['close'].values.astype(np.float64)
    df['atr_100'] = talib.ATR(high_vals, low_vals, close_vals, timeperiod=100)
    
    # Initialize regime columns
    regimes = []
    confidences = []
    scores_list = []
    
    for idx, row in df.iterrows():
        regime, confidence, scores = _classify_single_row(row)
        regimes.append(regime)
        confidences.append(confidence)
        scores_list.append(json.dumps(scores))
    
    df['regime'] = regimes
    df['confidence'] = confidences
    df['regime_json'] = scores_list
    
    return df


def _classify_single_row(row: pd.Series) -> tuple:
    """
    Classify a single row into a regime.
    
    Returns:
        (regime_name, confidence_score)
    """
    # Check for valid data (atr_100 is optional)
    required_fields = ['adx_14', 'plus_di_14', 'minus_di_14', 'close', 'ema_20', 
                      'atr_14', 'bb_width']
    
    for field in required_fields:
        if field not in row or pd.isna(row[field]):
            return ('unknown', 0.0, {})
    
    adx = row['adx_14']
    plus_di = row['plus_di_14']
    minus_di = row['minus_di_14']
    close = row['close']
    ema_20 = row['ema_20']
    atr_14 = row['atr_14']
    atr_14 = row['atr_14']
    atr_100 = row.get('atr_100', np.nan)
    bb_width = row['bb_width']
    
    # Score each regime possibility (0-1)
    scores = {}
    
    # 1. Trending Up (ADX > 20, relaxed from 25)
    if adx > 20 and plus_di > minus_di and close > ema_20:
        adx_strength = min((adx - 20) / 50, 1.0)
        di_diff = abs(plus_di - minus_di) / 100
        scores['trending_up'] = 0.5 + 0.25 * adx_strength + 0.25 * di_diff
    
    # 2. Trending Down (ADX > 20)
    if adx > 20 and minus_di > plus_di and close < ema_20:
        adx_strength = min((adx - 20) / 50, 1.0)
        di_diff = abs(minus_di - plus_di) / 100
        scores['trending_down'] = 0.5 + 0.25 * adx_strength + 0.25 * di_diff
    
    # 3. Weak Trend (ADX 15-25, directional but not strong)
    if 15 <= adx <= 25:
        if plus_di > minus_di and close > ema_20:
            scores['weak_trend'] = 0.5 + 0.3 * ((25 - adx) / 10) + 0.2 * (abs(plus_di - minus_di) / 100)
        elif minus_di > plus_di and close < ema_20:
            scores['weak_trend'] = 0.5 + 0.3 * ((25 - adx) / 10) + 0.2 * (abs(minus_di - plus_di) / 100)
        else:
            scores['weak_trend'] = 0.4 + 0.3 * ((25 - adx) / 10)
    
    # 4. Ranging (consolidation) - ADX < 15 (relaxed from 20)
    if adx < 15 and bb_width < 0.05:
        adx_score = (15 - adx) / 15
        bb_score = max(0, (0.05 - bb_width) / 0.05)
        scores['ranging'] = 0.5 + 0.25 * adx_score + 0.25 * bb_score
    
    # 5. High Volatility
    is_high_vol_atr = False
    if not pd.isna(atr_100) and atr_100 > 0:
        is_high_vol_atr = atr_14 > 2 * atr_100
        vol_ratio = min(atr_14 / (2 * atr_100), 3) / 3
    else:
        vol_ratio = 0.0
        
    if is_high_vol_atr or bb_width > 0.10:
        bb_vol_score = min(bb_width / 0.10, 1.0)
        scores['high_vol'] = 0.5 + 0.3 * vol_ratio + 0.2 * bb_vol_score
    
    # Select best regime
    if not scores:
        return ('unknown', 0.0, {})
    
    best_regime = max(scores, key=scores.get)
    best_score = scores[best_regime]
    
    return (best_regime, round(best_score, 4), scores)


def save_regimes_to_db(df: pd.DataFrame, db_path: str = 'btc_pipeline.db') -> int:
    """
    Save regime classifications to database.
    
    Args:
        df: DataFrame with regime columns
        db_path: Path to SQLite database
        
    Returns:
        Number of rows inserted
    """
    if df.empty or 'regime' not in df.columns:
        return 0
    
    # Select relevant columns
    regime_cols = ['timestamp', 'regime', 'adx_14', 'plus_di_14', 'minus_di_14', 
                   'bb_width', 'confidence', 'regime_json']
    available_cols = [c for c in regime_cols if c in df.columns]
    
    if 'confidence' not in available_cols:
        available_cols.append('confidence')
        df['confidence'] = 0.0
    
    # Map column names to match database schema
    column_mapping = {
        'adx_14': 'adx',
        'plus_di_14': 'plus_di',
        'minus_di_14': 'minus_di'
    }
    
    regimes_df = df[available_cols].copy()
    
    # Rename columns to match database schema
    regimes_df.rename(columns=column_mapping, inplace=True)
    
    # Convert timestamp to string
    if 'timestamp' in regimes_df.columns:
        regimes_df['timestamp'] = regimes_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regimes (
            timestamp DATETIME PRIMARY KEY,
            regime TEXT CHECK(regime IN ('trending_up', 'trending_down', 'ranging', 'high_vol', 'weak_trend', 'unknown')),
            adx REAL,
            plus_di REAL,
            minus_di REAL,
            bb_width REAL,
            confidence REAL,
            regime_json TEXT
        )
    """)
    
    # Bulk insert with explicit column mapping
    db_records = []
    for _, row in regimes_df.iterrows():
        record = (
            row['timestamp'],
            row['regime'],
            row.get('adx', None),      # Use renamed column
            row.get('plus_di', None),  # Use renamed column  
            row.get('minus_di', None), # Use renamed column
            row.get('bb_width', None),
            row.get('confidence', 0.0),
            row.get('regime_json', '{}')
        )
        db_records.append(record)
    
    cursor.executemany("""
        INSERT OR IGNORE INTO regimes (timestamp, regime, adx, plus_di, minus_di, bb_width, confidence, regime_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, db_records)
    
    rows_inserted = cursor.rowcount
    conn.commit()
    conn.close()
    
    return rows_inserted


def load_features_for_regime(db_path: str = 'btc_pipeline.db', limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load features from database for regime classification.
    
    Args:
        db_path: Path to SQLite database
        limit: Maximum rows to load
        
    Returns:
        DataFrame with features
    """
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT c.*, f.ema_20, f.atr_14, f.adx_14, f.plus_di_14,
               f.bb_upper, f.bb_middle, f.bb_lower
        FROM candles c
        LEFT JOIN features f ON c.timestamp = f.timestamp
        ORDER BY c.timestamp
    """
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def analyze_regime_distribution(df: pd.DataFrame) -> dict:
    """
    Analyze distribution of regimes and transition statistics.
    
    Returns:
        Dict with regime counts, percentages, transition rate
    """
    if 'regime' not in df.columns or df.empty:
        return {}
    
    total = len(df)
    regime_counts = df['regime'].value_counts().to_dict()
    regime_pcts = {k: round(v / total * 100, 2) for k, v in regime_counts.items()}
    
    # Calculate transition rate (regime changes / total rows)
    regime_changes = (df['regime'] != df['regime'].shift(1)).sum()
    transition_rate = regime_changes / total
    
    # Average confidence by regime
    if 'confidence' in df.columns:
        avg_confidence = df.groupby('regime')['confidence'].mean().to_dict()
    else:
        avg_confidence = {}
    
    return {
        'total_rows': total,
        'regime_counts': regime_counts,
        'regime_percentages': regime_pcts,
        'transition_rate': round(transition_rate, 4),
        'avg_confidence': avg_confidence
    }


def verify_regimes(df: pd.DataFrame) -> dict:
    """
    Verify regime classification quality.
    
    Checks:
    - Regimes are valid strings
    - Confidence is in [0, 1]
    - Transition rate < 5%
    - No regime dominates > 50%
    
    Returns:
        Verification report dict
    """
    if 'regime' not in df.columns:
        return {'error': 'No regime column in DataFrame'}
    
    valid_regimes = {'trending_up', 'trending_down', 'ranging', 'high_vol', 'weak_trend', 'unknown'}
    
    # Check valid regimes
    invalid_regimes = set(df['regime'].dropna().unique()) - valid_regimes
    
    # Check confidence range
    if 'confidence' in df.columns:
        conf_valid = df['confidence'].dropna().apply(lambda x: 0 <= x <= 1).all()
    else:
        conf_valid = False
    
    # Analyze distribution
    analysis = analyze_regime_distribution(df)
    
    # Check no regime dominates
    max_pct = max(analysis.get('regime_percentages', {}).values()) if analysis.get('regime_percentages') else 0
    
    # Check transition rate
    trans_rate = analysis.get('transition_rate', 1.0)
    
    return {
        'valid_regimes': len(invalid_regimes) == 0,
        'invalid_regimes_found': list(invalid_regimes),
        'confidence_valid_range': conf_valid,
        'no_regime_dominates': max_pct < 50,
        'max_regime_percentage': max_pct,
        'transition_rate_ok': trans_rate < 0.05,
        'transition_rate': trans_rate,
        'distribution': analysis.get('regime_percentages', {})
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Regime Classifier")
    print("=" * 60)
    
    # Load features (or create sample data)
    try:
        df = load_features_for_regime(limit=1000)
        print(f"Loaded {len(df)} rows from database")
    except Exception as e:
        print(f"Database error: {e}")
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='h'),
            'open': np.random.randn(200).cumsum() + 40000,
            'high': np.random.randn(200).cumsum() + 40500,
            'low': np.random.randn(200).cumsum() + 39500,
            'close': np.random.randn(200).cumsum() + 40000,
            'volume': np.random.randint(100, 1000, 200)
        })
        
        # Calculate basic features needed for regime
        close_vals = df['close'].values.astype(np.float64)
        high_vals = df['high'].values.astype(np.float64)
        low_vals = df['low'].values.astype(np.float64)
        
        df['ema_20'] = talib.EMA(close_vals, timeperiod=20)
        df['atr_14'] = talib.ATR(high_vals, low_vals, close_vals, timeperiod=14)
        df['adx_14'] = talib.ADX(high_vals, low_vals, close_vals, timeperiod=14)
        df['plus_di_14'] = talib.PLUS_DI(high_vals, low_vals, close_vals, timeperiod=14)
        
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_vals, timeperiod=20)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        print(f"Using sample data: {len(df)} rows")
    
    # Classify regimes
    print("\nClassifying regimes...")
    df_classified = classify_regime(df)
    
    # Analyze
    analysis = analyze_regime_distribution(df_classified)
    
    print("\nRegime Distribution:")
    for regime, pct in analysis['regime_percentages'].items():
        count = analysis['regime_counts'].get(regime, 0)
        print(f"  {regime}: {count} ({pct}%)")
    
    print(f"\nTransition rate: {analysis['transition_rate']:.2%}")
    
    # Verify
    verification = verify_regimes(df_classified)
    print(f"\nVerification:")
    print(f"  Valid regimes: {verification['valid_regimes']}")
    print(f"  Confidence valid: {verification['confidence_valid_range']}")
    print(f"  No regime dominates: {verification['no_regime_dominates']}")
    print(f"  Transition rate OK: {verification['transition_rate_ok']}")
    
    # Save to database
    print("\nSaving to database...")
    rows = save_regimes_to_db(df_classified)
    print(f"Saved {rows} rows")
    
    print("=" * 60)
    print("Regime classification complete")
    print("=" * 60)
