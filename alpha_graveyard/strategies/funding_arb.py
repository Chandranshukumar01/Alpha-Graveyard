"""
funding_rate_arb.py - Funding Rate Arbitrage Strategy

Exploits extreme funding rates on perpetual futures.
- Extremely positive funding (>0.03%): SHORT perp (pay less) + implied LONG spot.
- Extremely negative funding (<-0.03%): LONG perp (get paid) + implied SHORT spot.

This is a standalone, market-neutral strategy that is NOT regime-dependent.
It is registered under 'funding_arb' in the strategy registry.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class FundingRateArb:
    """
    Funding rate arbitrage signal generator.
    
    In backtest mode, generates synthetic funding rate data based on
    BTC price momentum (positive momentum -> positive funding, etc.).
    In live mode, would fetch from Binance via ccxt.
    """
    
    POSITIVE_THRESHOLD = 0.0003   # 0.03%
    NEGATIVE_THRESHOLD = -0.0003  # -0.03%
    
    def __init__(self, mode: str = 'backtest'):
        """
        Args:
            mode: 'backtest' (synthetic) or 'live' (ccxt fetch).
        """
        self.mode = mode
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def precompute_funding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-compute funding rate for every row.
        
        Adds columns:
          - funding_rate    : 8-hourly funding rate (decimal, e.g. 0.0001 = 0.01%)
          - funding_signal  : 'SHORT_PERP', 'LONG_PERP', or 'NONE'
        
        Args:
            df: DataFrame with 'timestamp' and 'close'.
            
        Returns:
            DataFrame with added funding columns.
        """
        df = df.copy()
        
        if self.mode == 'backtest':
            df = self._generate_synthetic_funding(df)
        else:
            df['funding_rate'] = 0.0001  # Placeholder neutral rate
        
        # Classify signal
        df['funding_signal'] = df['funding_rate'].apply(self._classify_signal)
        
        return df
    
    def get_funding_signal(
        self, row: Dict[str, Any], prev_row: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trading signal based on funding rate.
        
        Compatible with strategy_library function signature.
        
        Args:
            row: Current candle data (must have 'funding_rate' and 'close').
            prev_row: Previous candle data.
            
        Returns:
            Signal dict: {signal, entry_price, stop_loss, confidence, strategy}
        """
        funding_rate = row.get('funding_rate', 0.0)
        close = row.get('close', 0.0)
        atr = row.get('atr_14', close * 0.01)  # Fallback 1%
        
        if funding_rate > self.POSITIVE_THRESHOLD:
            # High positive funding -> SHORT perp, collect funding
            magnitude = min((funding_rate - self.POSITIVE_THRESHOLD) / 0.001, 1.0)
            return {
                'signal': 'SHORT',
                'entry_price': close,
                'stop_loss': close + 2 * atr,
                'take_profit': close - 1.5 * atr,
                'confidence': round(0.5 + 0.4 * magnitude, 2),
                'strategy': 'funding_arb'
            }
        
        elif funding_rate < self.NEGATIVE_THRESHOLD:
            # High negative funding -> LONG perp, collect funding
            magnitude = min((self.NEGATIVE_THRESHOLD - funding_rate) / 0.001, 1.0)
            return {
                'signal': 'LONG',
                'entry_price': close,
                'stop_loss': close - 2 * atr,
                'take_profit': close + 1.5 * atr,
                'confidence': round(0.5 + 0.4 * magnitude, 2),
                'strategy': 'funding_arb'
            }
        
        return {
            'signal': 'NONE',
            'entry_price': None,
            'stop_loss': None,
            'confidence': 0.0,
            'strategy': 'funding_arb'
        }
    
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    
    def _generate_synthetic_funding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic funding rates correlated with momentum.
        
        Methodology:
          - Compute 8h rolling return as a momentum proxy.
          - Base funding = momentum * scale + noise.
          - Clamp to realistic range [-0.1%, +0.1%].
          - Funding resets every 8 hours (matching Binance schedule).
        """
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 8h momentum
        momentum_8h = df['close'].pct_change(8).fillna(0)
        
        np.random.seed(123)
        noise = np.random.normal(0, 0.0001, len(df))
        
        # Scale: 1% 8h return -> ~0.01% funding
        raw_funding = momentum_8h * 0.01 + noise
        
        # Clamp to realistic range
        df['funding_rate'] = raw_funding.clip(-0.001, 0.001)
        
        # Quantize to 8h intervals (00:00, 08:00, 16:00 UTC)
        df['_hour'] = df['timestamp'].dt.hour
        df['_funding_period'] = (df['_hour'] // 8) * 8
        
        # Group by date + period and use last funding value for all rows in that period
        df['_date'] = df['timestamp'].dt.date
        period_funding = df.groupby(['_date', '_funding_period'])['funding_rate'].transform('last')
        df['funding_rate'] = period_funding
        
        df.drop(columns=['_hour', '_funding_period', '_date'], inplace=True)
        
        return df
    
    def _classify_signal(self, rate: float) -> str:
        """Classify funding rate into signal category."""
        if pd.isna(rate):
            return 'NONE'
        if rate > self.POSITIVE_THRESHOLD:
            return 'SHORT_PERP'
        elif rate < self.NEGATIVE_THRESHOLD:
            return 'LONG_PERP'
        return 'NONE'


# ---------------------------------------------------------------------------
# Strategy function for STRATEGY_REGISTRY integration
# ---------------------------------------------------------------------------

_arb_instance = FundingRateArb(mode='backtest')

def strategy_funding_arb(row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper matching the standard strategy function signature."""
    return _arb_instance.get_funding_signal(row, prev_row)
