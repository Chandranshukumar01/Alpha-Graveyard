"""
etf_flow_signal.py - ETF Flow Signal Module

Provides ETF flow-based bias for BTC trading.
- Live mode: Fetches daily ETF flow data from a public API.
- Backtest mode: Generates synthetic flow data correlated to BTC price changes.

The flow bias is used as a confidence multiplier in the trading pipeline:
  - 'inflow'  -> bullish bias (confidence × 1.2)
  - 'outflow' -> bearish bias (confidence × 0.7 if conflicting)
  - 'neutral' -> no adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class ETFFlowSignal:
    """
    ETF inflow/outflow signal generator.
    
    In backtest mode, generates synthetic flow data that is loosely
    correlated with BTC daily returns (simulating real-world behaviour
    where large inflows tend to precede or accompany price rises).
    """
    
    def __init__(self, mode: str = 'backtest', lookback: int = 5):
        """
        Args:
            mode: 'backtest' (synthetic) or 'live' (API fetch).
            lookback: Number of days to compute flow trend over.
        """
        self.mode = mode
        self.lookback = lookback
        self._flow_cache: Optional[pd.Series] = None
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def precompute_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-compute daily ETF flow values for every row in the DataFrame.
        
        Adds columns:
          - etf_flow       : synthetic daily flow (USD millions)
          - etf_flow_5d    : rolling 5-day sum of flows
          - etf_flow_bias  : 'inflow' / 'outflow' / 'neutral'
        
        Args:
            df: DataFrame with 'timestamp' and 'close' columns (1h data).
            
        Returns:
            DataFrame with added flow columns.
        """
        df = df.copy()
        
        if self.mode == 'backtest':
            df = self._generate_synthetic_flows(df)
        else:
            # Live mode placeholder – would call an API here
            df['etf_flow'] = 0.0
        
        # Rolling sum of flows over lookback period (in daily terms)
        # Since data is hourly, 1 day = 24 rows
        window = self.lookback * 24
        df['etf_flow_5d'] = df['etf_flow'].rolling(window=window, min_periods=1).sum()
        
        # Classify bias
        df['etf_flow_bias'] = df['etf_flow_5d'].apply(self._classify_bias)
        
        return df
    
    def get_flow_bias(self, row: Dict[str, Any]) -> str:
        """Get flow bias for a single row (must have been precomputed)."""
        return row.get('etf_flow_bias', 'neutral')
    
    def apply_confidence_modifier(
        self, signal_dict: Dict[str, Any], flow_bias: str
    ) -> Dict[str, Any]:
        """
        Adjust signal confidence based on ETF flow alignment.
        
        Rules:
          - LONG + inflow  -> confidence × 1.2  (aligned)
          - SHORT + outflow -> confidence × 1.2  (aligned)
          - LONG + outflow  -> confidence × 0.7  (conflicting)
          - SHORT + inflow  -> confidence × 0.7  (conflicting)
          - neutral         -> no change
          
        Returns a NEW dict (original is not mutated).
        """
        out = dict(signal_dict)
        if out.get('signal', 'NONE') == 'NONE' or flow_bias == 'neutral':
            return out
        
        conf = out.get('confidence', 0.5)
        
        aligned = (
            (out['signal'] == 'LONG' and flow_bias == 'inflow') or
            (out['signal'] == 'SHORT' and flow_bias == 'outflow')
        )
        
        if aligned:
            conf = min(conf * 1.2, 1.0)
        else:
            conf = max(conf * 0.7, 0.0)
        
        out['confidence'] = round(conf, 4)
        return out
    
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    
    def _generate_synthetic_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic ETF flows correlated with daily BTC returns.
        
        Methodology:
          1. Compute daily returns from 'close'.
          2. Base flow = daily_return × scale + noise.
             Positive return day -> likely inflow.
             Negative return day -> likely outflow.
          3. Assign same daily flow to all intra-day (hourly) rows.
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Daily returns
        df['_date'] = df['timestamp'].dt.date
        daily_close = df.groupby('_date')['close'].last()
        daily_returns = daily_close.pct_change().fillna(0)
        
        # Synthetic flow: scale * return + noise
        np.random.seed(42)  # Reproducible for backtesting
        scale = 500  # Millions USD
        noise = np.random.normal(0, 50, len(daily_returns))
        
        daily_flow = (daily_returns.values * scale + noise)
        flow_series = pd.Series(daily_flow, index=daily_returns.index, name='etf_flow')
        
        # Map back to hourly data
        df['etf_flow'] = df['_date'].map(flow_series)
        df.drop(columns=['_date'], inplace=True)
        
        return df
    
    def _classify_bias(self, flow_5d: float) -> str:
        """Classify 5-day cumulative flow into bias category."""
        if pd.isna(flow_5d):
            return 'neutral'
        if flow_5d > 100:   # Net inflow > $100M over 5 days
            return 'inflow'
        elif flow_5d < -100:  # Net outflow
            return 'outflow'
        return 'neutral'
