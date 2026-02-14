"""
Features subpackage: Feature engineering and market regime classification.

Modules:
    technical: 20 technical indicators from OHLCV (RSI, MACD, BB, ATR, etc.)
    regime: Market regime classifier (trending, ranging, high-vol, weak-trend)
"""

from alpha_graveyard.features.technical import calculate_features
from alpha_graveyard.features.regime import classify_regime
