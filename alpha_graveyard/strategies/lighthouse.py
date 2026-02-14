"""
lighthouse_core.py - Project Lighthouse: The Council of Agents

Implements a Multi-Agent system where specialized agents vote on trading decisions.
"""

import pandas as pd
import numpy as np

class Agent:
    def __init__(self, name):
        self.name = name
        
    def analyze(self, df_1h, current_idx):
        """
        Analyzes the market state at `current_idx` (integer index of df_1h).
        Returns:
            signal: 1 (Long), -1 (Short), 0 (Neutral/Hold)
            reason: String explanation
        """
        raise NotImplementedError

class Captain(Agent):
    """
    The Captain (Macro Trend).
    Perspective: Daily Chart.
    Tools: 200-Day SMA, 50-Day SMA (Golden Cross/Death Cross).
    Philosophy: "Never sail against the tide."
    """
    def __init__(self):
        super().__init__("Captain (Macro)")

    def analyze(self, df_1h, current_idx):
        # We need enough history for Daily MAs.
        # 1 Day = 24 hours. 200 Days = 4800 hours.
        if current_idx < 4800:
            return 0, "Not enough data for Macro view"
            
        # Optimization: We don't resample everything every step.
        # We just look at the 1h moving averages which approximate the daily ones effectively
        # or we pre-calculate them.
        # For simplicity/speed in this prototype, we'll use 1h EMAs scaled up.
        # EMA(200 Daily) ~= EMA(4800 Hourly) roughly, but better to be precise.
        
        # Actually, let's look at the resampled daily data up to this point.
        # Current time
        current_ts = df_1h.index[current_idx]
        
        # Get data window (last 300 days)
        # Slicing via timestamp is safer
        start_ts = current_ts - pd.Timedelta(days=300)
        
        # Create a view (not copy if possible)
        window = df_1h.loc[start_ts:current_ts]
        
        if len(window) < 24*200:
             return 0, "Insufficient history"

        # Resample to Daily
        daily = window.resample('D').agg({'close': 'last'}).dropna()
        
        if len(daily) < 200:
            return 0, "Not enough daily candles"
            
        close = daily['close'].iloc[-1]
        sma_200 = daily['close'].rolling(200).mean().iloc[-1]
        sma_50 = daily['close'].rolling(50).mean().iloc[-1]
        
        # Logic
        if close > sma_200:
            if sma_50 > sma_200:
                return 1, "Bull Market (Price > SMA200 & Golden Cross)"
            else:
                return 1, "Bullish (Price > SMA200)"
        else:
            if sma_50 < sma_200:
                return -1, "Bear Market (Price < SMA200 & Death Cross)"
            else:
                return -1, "Bearish (Price < SMA200)"

class Navigator(Agent):
    """
    The Navigator (Micro Entry).
    Perspective: 4h/1h Chart.
    Tools: RSI(14), EMA(20).
    Philosophy: "Find a clean entry."
    """
    def __init__(self):
        super().__init__("Navigator (Micro)")
        
    def analyze(self, df_1h, current_idx):
        if current_idx < 50:
            return 0, "Calibrating"
            
        # Look at last 100 hours
        window = df_1h.iloc[current_idx-100 : current_idx+1]
        
        current_close = window['close'].iloc[-1]
        
        # Calculate RSI (14)
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Entry Logic
        # Pullback in Uptrend: RSI < 40
        # Rally in Downtrend: RSI > 60
        
        if current_rsi < 35:
            return 1, f"Oversold (RSI {current_rsi:.1f})"
        elif current_rsi > 65:
            return -1, f"Overbought (RSI {current_rsi:.1f})"
        else:
            return 0, f"No Signal (RSI {current_rsi:.1f})"

class Watchman(Agent):
    """
    The Watchman (Risk/Volatility).
    Perspective: Volatility.
    Tools: ATR, Bollinger Band Width.
    Philosophy: "Don't sail into a storm."
    """
    def __init__(self):
        super().__init__("Watchman (Risk)")
        
    def analyze(self, df_1h, current_idx):
        if current_idx < 24:
            return 0, "Calibrating"
            
        window = df_1h.iloc[current_idx-50 : current_idx+1]
        
        # Calculate Volatility (ATR-like or Standard Dev)
        # BB Width
        rolling_std = window['close'].rolling(20).std()
        rolling_mean = window['close'].rolling(20).mean()
        upper = rolling_mean + (2 * rolling_std)
        lower = rolling_mean - (2 * rolling_std)
        bb_width = (upper - lower) / rolling_mean
        
        current_width = bb_width.iloc[-1]
        avg_width = bb_width.mean()
        
        # Identification of "Storm" (High Volatility Expansion)
        # If width is 2x the average, it's too volatile.
        if current_width > (avg_width * 2.0):
            return 0, f"STORM WARNING: High Volatility ({current_width:.3f})"
            
        # Identification of "Dead Calm" (Squeeze)
        # if current_width < (avg_width * 0.5):
        #    return 0, "Dead Calm (Squeeze)" 
            
        return 1, "Clear Skies" # 1 means "Safe to Proceed", not "Buy"

class Council:
    def __init__(self):
        self.captain = Captain()
        self.navigator = Navigator()
        self.watchman = Watchman()
        
    def convene(self, df_1h, current_idx):
        # 1. Watchman first (Safety)
        safe, w_reason = self.watchman.analyze(df_1h, current_idx)
        if safe == 0:
            return 0, f"Blocked by Watchman: {w_reason}"
            
        # 2. Captain (Trend)
        trend, c_reason = self.captain.analyze(df_1h, current_idx)
        if trend == 0:
            return 0, f"Captain Neutral: {c_reason}"
            
        # 3. Navigator (Entry)
        signal, n_reason = self.navigator.analyze(df_1h, current_idx)
        
        # Voting Logic
        # We need Alignment: Captain says Long (1) AND Navigator says Long (1) or Oversold
        # Actually Navigator says "Oversold" (1) which matches Captain Long (1).
        
        if trend == 1 and signal == 1:
            return 1, "LONG | Crew Agreed (Trend Up + Dip)"
        
        if trend == -1 and signal == -1:
            return -1, "SHORT | Crew Agreed (Trend Down + Rally)"
            
        return 0, f"Disagreement: Capt={trend} vs Nav={signal}"
