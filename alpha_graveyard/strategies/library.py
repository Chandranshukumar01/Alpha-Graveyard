"""
strategy_library.py - Modular Trading Strategies

Each strategy function takes:
    - row: current timestamp data with features (dict-like)
    - prev_row: previous timestamp data (dict-like)
    
Returns dict with:
    - signal: 'LONG', 'SHORT', or 'NONE'
    - entry_price: float (or None if no signal)
    - stop_loss: float (or None if no signal)
    - confidence: float 0.0-1.0

All strategies use only past data (row and prev_row) - no lookahead.
"""

from typing import Dict, Optional, Any


def strategy_trending_up_ema_crossover(row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    EMA Crossover strategy for trending_up regime.
    
    Long signal: Close crosses above EMA20 (prev close <= prev ema, current close > current ema)
    Stop loss: EMA20 - 1×ATR14
    Confidence: 0.8 if ADX > 30, else 0.6
    
    Args:
        row: Current timestamp data with features
        prev_row: Previous timestamp data
        
    Returns:
        dict with signal, entry_price, stop_loss, confidence
    """
    # Check required fields exist
    required = ['close', 'ema_20', 'atr_14', 'adx_14']
    if not all(k in row for k in required) or not all(k in prev_row for k in required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    # Check for valid (non-NaN) values
    if any(row[k] is None or (isinstance(row[k], float) and row[k] != row[k]) for k in required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    if any(prev_row[k] is None or (isinstance(prev_row[k], float) and prev_row[k] != prev_row[k]) for k in required[:2]):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    current_close = float(row['close'])
    current_ema = float(row['ema_20'])
    prev_close = float(prev_row['close'])
    prev_ema = float(prev_row['ema_20'])
    atr = float(row['atr_14'])
    adx = float(row['adx_14'])
    
    # EMA Crossover signal: previous close <= previous EMA, current close > current EMA
    # This indicates price crossing above the EMA (bullish)
    crossover_up = prev_close <= prev_ema and current_close > current_ema
    
    if crossover_up:
        entry_price = current_close
        stop_loss = current_ema - (2 * atr)  # 2 ATR stop for sizing
        
        # Confidence based on trend strength (ADX)
        if adx > 30:
            confidence = 0.8
        elif adx > 25:
            confidence = 0.7
        else:
            confidence = 0.6
        
        return {
            'signal': 'LONG',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'confidence': round(confidence, 2)
        }
    
    return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}


def strategy_mean_reversion_bb(row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mean reversion strategy using Bollinger Bands.
    
    Long signal: Price touches lower BB (crosses from above to below/within)
    Short signal: Price touches upper BB (crosses from below to above/within)
    Stop loss: Middle BB (mean reversion target becomes stop if wrong)
    Take profit: Opposite band (upper for longs, lower for shorts)
    Confidence: Based on BB width (wider bands = higher confidence in mean reversion)
    
    Args:
        row: Current timestamp data with features
        prev_row: Previous timestamp data
        
    Returns:
        dict with signal, entry_price, stop_loss, confidence
    """
    required = ['close', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
    prev_required = ['close', 'bb_upper', 'bb_lower']
    
    if not all(k in row for k in required) or not all(k in prev_row for k in prev_required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    # Check for valid values
    if any(row[k] is None or (isinstance(row[k], float) and row[k] != row[k]) for k in required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    if any(prev_row[k] is None or (isinstance(prev_row[k], float) and prev_row[k] != prev_row[k]) for k in prev_required[:2]):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    close = float(row['close'])
    bb_upper = float(row['bb_upper'])
    bb_lower = float(row['bb_lower'])
    bb_middle = float(row['bb_middle'])
    atr = float(row['atr_14'])
    
    prev_close = float(prev_row['close'])
    prev_bb_upper = float(prev_row['bb_upper'])
    prev_bb_lower = float(prev_row['bb_lower'])
    
    # Calculate BB width for confidence
    band_width = bb_upper - bb_lower
    band_width_pct = band_width / bb_middle if bb_middle > 0 else 0
    
    # Long signal: Close crosses from above lower BB to at/below lower BB (touch)
    long_touch = close <= bb_lower and prev_close > prev_bb_lower
    
    # Short signal: Close crosses from below upper BB to at/above upper BB (touch)
    short_touch = close >= bb_upper and prev_close < prev_bb_upper
    
    if long_touch:
        entry_price = close
        stop_loss = bb_middle  # Stop at middle if mean reversion fails
        take_profit = bb_upper  # TP at upper band (full mean reversion)
        
        # Confidence: Wider bands = more room for mean reversion = higher confidence
        # Normalize: 2% band width = 0.5 confidence, 6% = 1.0 confidence
        confidence = 0.5 + 0.5 * min(band_width_pct / 0.04, 1.0)
        
        return {
            'signal': 'LONG',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'confidence': round(confidence, 2)
        }
    
    if short_touch:
        entry_price = close
        stop_loss = bb_middle  # Stop at middle
        take_profit = bb_lower  # TP at lower band
        
        # Confidence based on band width
        confidence = 0.5 + 0.5 * min(band_width_pct / 0.04, 1.0)
        
        return {
            'signal': 'SHORT',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'confidence': round(confidence, 2)
        }
    
    return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}


def strategy_ranging_mean_reversion(row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mean reversion strategy for ranging regime using Bollinger Bands.
    
    Long signal: Close touches lower BB (close <= lower + 0.5% buffer)
    Short signal: Close touches upper BB (close >= upper - 0.5% buffer)
    Stop loss: 0.5×ATR14 beyond the band (lower - 0.5×ATR for longs)
    Confidence: Based on distance from middle band (closer to band = higher confidence)
    
    Args:
        row: Current timestamp data with features
        prev_row: Previous timestamp data
        
    Returns:
        dict with signal, entry_price, stop_loss, confidence
    """
    required = ['close', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
    if not all(k in row for k in required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    # Check for valid values
    if any(row[k] is None or (isinstance(row[k], float) and row[k] != row[k]) for k in required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    close = float(row['close'])
    bb_upper = float(row['bb_upper'])
    bb_lower = float(row['bb_lower'])
    bb_middle = float(row['bb_middle'])
    atr = float(row['atr_14'])
    
    # Buffer for touching bands (0.5% of band width)
    band_width = bb_upper - bb_lower
    touch_buffer = band_width * 0.005  # 0.5% buffer
    
    # Long signal: Close at or below lower band (mean reversion - expect bounce up)
    long_signal = close <= (bb_lower + touch_buffer)
    
    # Short signal: Close at or above upper band (mean reversion - expect bounce down)
    short_signal = close >= (bb_upper - touch_buffer)
    
    if long_signal:
        entry_price = close
        stop_loss = bb_lower - (0.5 * atr)  # Stop below lower band
        
        # Confidence: Distance from middle (1.0 at lower, 0.5 at middle)
        distance_from_middle = abs(close - bb_middle) / (bb_middle - bb_lower)
        confidence = 0.5 + 0.5 * min(distance_from_middle, 1.0)
        
        return {
            'signal': 'LONG',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'confidence': round(confidence, 2)
        }
    
    if short_signal:
        entry_price = close
        stop_loss = bb_upper + (0.5 * atr)  # Stop above upper band
        
        # Confidence: Distance from middle
        distance_from_middle = abs(close - bb_middle) / (bb_upper - bb_middle)
        confidence = 0.5 + 0.5 * min(distance_from_middle, 1.0)
        
        return {
            'signal': 'SHORT',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'confidence': round(confidence, 2)
        }
    
    return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}


def strategy_breakout_volatility(row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Breakout strategy for high volatility regime.
    
    Long signal: Price breaks above recent high (rolling 24-bar max)
    Stop loss: Entry - 3×ATR (wider stop for volatility)
    Take profit: Entry + 2×ATR (risk/reward ~0.67)
    Confidence: Based on volume and volatility expansion
    
    Args:
        row: Current timestamp data with features
        prev_row: Previous timestamp data
        
    Returns:
        dict with signal, entry_price, stop_loss, confidence
    """
    required = ['close', 'high', 'low', 'atr_14', 'volume']
    if not all(k in row for k in required) or not all(k in prev_row for k in ['high', 'volume']):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    # Check for valid values
    if any(row[k] is None or (isinstance(row[k], float) and row[k] != row[k]) for k in required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    if any(prev_row[k] is None or (isinstance(prev_row[k], float) and prev_row[k] != prev_row[k]) for k in ['high', 'volume']):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    current_close = float(row['close'])
    current_high = float(row['high'])
    current_low = float(row['low'])
    current_volume = float(row['volume'])
    prev_high = float(prev_row['high'])
    prev_volume = float(prev_row['volume'])
    atr = float(row['atr_14'])
    
    # Simple breakout: current high > previous high with minimum threshold
    breakout_threshold = 0.002  # 0.2% minimum breakout
    breakout_up = current_high > prev_high * (1 + breakout_threshold)
    
    # Volume confirmation: current volume > previous volume (optional)
    volume_ok = current_volume > prev_volume * 0.8  # At least 80% of previous volume
    
    if breakout_up and volume_ok:
        entry_price = current_close
        stop_loss = entry_price - (3 * atr)  # Wider stop for high volatility
        take_profit = entry_price + (2 * atr)  # Risk/reward ~0.67
        
        # Confidence: base 0.6, boosted by volume and ATR size
        confidence = 0.6
        if current_volume > prev_volume * 1.2:
            confidence += 0.1  # Volume spike increases confidence
        
        # Higher ATR (relative to price) indicates strong volatility, good for breakouts
        atr_pct = atr / entry_price
        if atr_pct > 0.02:  # >2% daily volatility
            confidence += 0.1
        
        confidence = min(confidence, 0.9)  # Cap at 0.9
        
        return {
            'signal': 'LONG',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'confidence': round(confidence, 2)
        }
    
    return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}


def strategy_weak_trend_hold(row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conservative strategy for weak trend regime - mostly hold/no trade.
    
    Only enters if there's a strong momentum signal (RSI extreme).
    
    Args:
        row: Current timestamp data with features
        prev_row: Previous timestamp data
        
    Returns:
        dict with signal, entry_price, stop_loss, confidence
    """
    required = ['close', 'rsi_14', 'ema_20', 'atr_14']
    if not all(k in row for k in required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    if any(row[k] is None or (isinstance(row[k], float) and row[k] != row[k]) for k in required):
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
    
    rsi = float(row['rsi_14'])
    close = float(row['close'])
    ema = float(row['ema_20'])
    atr = float(row['atr_14'])
    
    # RSI oversold bounce (RSI < 30 and price near EMA support)
    oversold_bounce = rsi < 30 and close > ema - (0.5 * atr)
    
    if oversold_bounce:
        entry_price = close
        stop_loss = ema - (1.5 * atr)
        
        return {
            'signal': 'LONG',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'confidence': 0.5  # Low confidence in weak trend
        }
    
    return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}


def strategy_vwap_reversion(row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    VWAP Mean Reversion Strategy:
    - Long if price touches/crosses lower VWAP band (revert to mean)
    - Short if price touches/crosses upper VWAP band
    - Target: VWAP line (implied take profit)
    - Stop: 1.5 * ATR (or band width based)
    """
    required = ['close', 'vwap_upper_band', 'vwap_lower_band', 'vwap_20', 'atr_14']
    if not all(k in row for k in required):
        # Fallback to empty if columns missing
        return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}
        
    if any(row[k] is None or (isinstance(row[k], float) and row[k] != row[k]) for k in required):
         return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}

    current_price = float(row['close'])
    upper_band = float(row['vwap_upper_band'])
    lower_band = float(row['vwap_lower_band'])
    vwap = float(row['vwap_20'])
    atr = float(row['atr_14'])
    
    # Optional RSI confirmation
    rsi = float(row.get('rsi_14', 50))
    
    # Reversion Long: Price < Lower Band
    if current_price < lower_band:
        entry_price = current_price
        stop_loss = current_price - (1.5 * atr)
        confidence = 0.7
        
        if rsi < 30:
            confidence = 0.85
            
        return {
            'signal': 'LONG',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'confidence': confidence
        }
        
    # Reversion Short: Price > Upper Band
    elif current_price > upper_band:
        entry_price = current_price
        stop_loss = current_price + (1.5 * atr)
        confidence = 0.7
        
        if rsi > 70:
            confidence = 0.85
            
        return {
            'signal': 'SHORT',
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'confidence': confidence
        }

    return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}



def strategy_unknown_cash(row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Default strategy for unknown regime - stay in cash, no trades.
    
    Args:
        row: Current timestamp data with features
        prev_row: Previous timestamp data
        
    Returns:
        dict with signal: 'NONE'
    """
    return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}


# Strategy registry mapping regimes to strategies
STRATEGY_REGISTRY = {
    'trending_up': strategy_trending_up_ema_crossover,
    'trending_down': strategy_mean_reversion_bb,  # Short at upper band
    'weak_trend': strategy_weak_trend_hold,
    'ranging': strategy_mean_reversion_bb,
    'ranging_vwap': strategy_vwap_reversion,  # Alternative ranging strategy
    'high_vol': strategy_breakout_volatility,
    'funding_arb': None,  # Lazy-loaded from funding_rate_arb module
    'unknown': strategy_unknown_cash
}


def get_strategy_for_regime(regime: str):
    """
    Get the appropriate strategy function for a given regime.
    
    Args:
        regime: Regime name string
        
    Returns:
        Strategy function or default cash strategy
    """
    func = STRATEGY_REGISTRY.get(regime, strategy_unknown_cash)
    # Lazy-load funding_arb to avoid circular imports
    if func is None and regime == 'funding_arb':
        from alpha_graveyard.strategies.funding_arb import strategy_funding_arb
        STRATEGY_REGISTRY['funding_arb'] = strategy_funding_arb
        return strategy_funding_arb
    return func if func is not None else strategy_unknown_cash


def run_strategy_test(df, strategy_func, verbose=False):
    """
    Test a strategy on historical data.
    
    Args:
        df: DataFrame with OHLCV + features
        strategy_func: Strategy function to test
        verbose: Print detailed output
        
    Returns:
        List of signal dictionaries
    """
    signals = []
    
    for i in range(1, len(df)):
        row = df.iloc[i].to_dict()
        prev_row = df.iloc[i-1].to_dict()
        
        result = strategy_func(row, prev_row)
        
        if result['signal'] != 'NONE':
            signals.append({
                'timestamp': row.get('timestamp'),
                'index': i,
                **result
            })
            
            if verbose:
                print(f"Signal at {row.get('timestamp')}: {result['signal']} "
                      f"@ {result['entry_price']}, stop: {result['stop_loss']}")
    
    return signals


if __name__ == '__main__':
    import sqlite3
    import pandas as pd
    
    print("=" * 60)
    print("Strategy Library Test")
    print("=" * 60)
    
    # Load real data
    try:
        conn = sqlite3.connect('btc_pipeline.db')
        query = """
            SELECT c.*, f.ema_20, f.atr_14, f.adx_14, f.rsi_14, 
                   f.bb_upper, f.bb_lower, f.bb_middle
            FROM candles c
            LEFT JOIN features f ON c.timestamp = f.timestamp
            ORDER BY c.timestamp
            LIMIT 1000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} rows of real BTC data")
        
    except Exception as e:
        print(f"Database error: {e}")
        exit(1)
    
    # Test EMA Crossover strategy
    print("\n" + "=" * 60)
    print("Testing EMA Crossover Strategy (trending_up)")
    print("=" * 60)
    
    signals = run_strategy_test(df, strategy_trending_up_ema_crossover, verbose=False)
    
    print(f"Total signals generated: {len(signals)}")
    
    if signals:
        long_count = sum(1 for s in signals if s['signal'] == 'LONG')
        short_count = sum(1 for s in signals if s['signal'] == 'SHORT')
        
        print(f"  LONG: {long_count}")
        print(f"  SHORT: {short_count}")
        
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
        print(f"  Average confidence: {avg_confidence:.2f}")
        
        print("\nFirst 3 signals:")
        for sig in signals[:3]:
            print(f"  {sig['timestamp']}: {sig['signal']} @ {sig['entry_price']}, "
                  f"stop: {sig['stop_loss']}, conf: {sig['confidence']}")
    
    print("=" * 60)
    print("Strategy test complete")
    print("=" * 60)
