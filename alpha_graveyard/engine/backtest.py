"""
backtest_engine.py - Strategy Backtesting Engine

Integrates RiskManager with strategy execution on historical data.
Uses regime-based strategy selection.
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

import pandas as pd
import numpy as np

from alpha_graveyard.engine.risk import RiskManager
from alpha_graveyard.strategies.library import get_strategy_for_regime, STRATEGY_REGISTRY
from alpha_graveyard.strategies.ensemble import WeightedEnsembleStrategy
from alpha_graveyard.strategies.etf_flow import ETFFlowSignal
from alpha_graveyard.strategies.funding_arb import FundingRateArb


class BacktestEngine:
    """
    Backtest engine that simulates trading with real historical data.
    
    Integrates:
    - RiskManager for position sizing and risk limits
    - Regime-based strategy selection
    - Trade execution and P&L calculation
    - Results tracking and reporting
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        initial_capital: float = 100000.0,
        db_path: str = 'backtest_results.db',
        slippage_pct: float = 0.001,
        fee_pct: float = 0.001
    ):
        """
        Initialize backtest engine.
        
        Args:
            risk_manager: RiskManager instance for position sizing
            initial_capital: Starting capital for simulation
            db_path: Path to SQLite database for results
        """
        self.rm = risk_manager
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.db_path = db_path
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct
        
        # Tracking
        self.trades: List[Dict] = []
        self.daily_pnls: List[float] = []
        self.equity_curve: List[tuple] = []  # (timestamp, capital)
        
        # Current position
        self.position: Optional[Dict] = None
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize results database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                regime TEXT,
                strategy TEXT,
                signal TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                shares INTEGER,
                pnl REAL,
                pnl_pct REAL,
                confidence REAL,
                drawdown_at_entry REAL,
                daily_pnl_before REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                timestamp TEXT PRIMARY KEY,
                capital REAL,
                drawdown_pct REAL,
                daily_pnl REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def run(
        self,
        df: pd.DataFrame,
        regime_col: str = 'regime',
        use_ensemble: bool = False,
        use_etf_flow: bool = False,
        use_funding_arb: bool = False
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV, features, and regime columns
            regime_col: Column name for regime classification
            use_ensemble: If True, use WeightedEnsembleStrategy
            use_etf_flow: If True, use ETFFlowSignal confidence modifier
            use_funding_arb: If True, trade FundingRateArb opportunities
            
        Returns:
            Dictionary with performance metrics
        """
        # Precompute supplementary signals if needed
        self.etf_signal = None
        if use_etf_flow:
            self.etf_signal = ETFFlowSignal(mode='backtest')
            df = self.etf_signal.precompute_flows(df)
            
        self.funding_arb = None
        if use_funding_arb:
            self.funding_arb = FundingRateArb(mode='backtest')
            df = self.funding_arb.precompute_funding(df)

        self.total_trades = 0
        print(f"Starting backtest with {len(df)} rows")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Ensure required columns exist
        required_ohlcv = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_ohlcv if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.trades = []
        self.daily_pnls = []
        self.equity_curve = []
        self.position = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.peak_capital = self.initial_capital
        
        current_day = None
        
        # Iterate through data (start from index 1 to have prev_row)
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            ts = current_row['timestamp']
            if hasattr(ts, 'date'):
                day = ts.date()
            else:
                day = ts
            
            if current_day is None:
                current_day = day
            elif day != current_day:
                self.rm.reset_daily()
                self.daily_pnls = []
                current_day = day
            
            # Get current regime
            regime = current_row.get(regime_col, 'unknown')
            
            # Generate signal
            if use_ensemble:
                # Ensemble mode: combine signals from multiple strategies
                if not hasattr(self, '_ensemble'):
                    self._ensemble = WeightedEnsembleStrategy()
                signal_dict = self._ensemble.get_signal(
                    current_row.to_dict(),
                    prev_row.to_dict()
                )
            else:
                # Standard mode: single strategy per regime
                strategy_func = get_strategy_for_regime(regime)
                signal_dict = strategy_func(
                    current_row.to_dict(),
                    prev_row.to_dict()
                )
            
            # 1. Apply ETF Flow Confidence Modifier
            if use_etf_flow and self.etf_signal and signal_dict['signal'] != 'NONE':
                flow_bias = self.etf_signal.get_flow_bias(current_row)
                signal_dict = self.etf_signal.apply_confidence_modifier(signal_dict, flow_bias)
                
            # 2. Check Funding Arb Signal (Parallel Opportunity)
            if use_funding_arb and self.funding_arb:
                arb_signal = self.funding_arb.get_funding_signal(
                    current_row.to_dict(),
                    prev_row.to_dict()
                )
                if arb_signal['signal'] != 'NONE':
                    # Prioritize arb signal if main signal is weak or NONE
                    if signal_dict['signal'] == 'NONE' or arb_signal['confidence'] > signal_dict.get('confidence', 0):
                         signal_dict = arb_signal

            # Process signal
            if signal_dict['signal'] != 'NONE':
                self._process_signal(
                    timestamp=current_row['timestamp'],
                    regime=regime,
                    signal_dict=signal_dict,
                    current_row=current_row,
                    current_idx=i
                )
            
            # Check existing position for exit
            if self.position:
                self._check_position_exit(current_row, i)
            
            # Record equity
            self._update_equity(current_row['timestamp'])
        
        # Close any open position at end
        if self.position:
            final_row = df.iloc[-1]
            self._close_position(
                timestamp=final_row['timestamp'],
                exit_price=final_row['close'],
                reason='end_of_data'
            )
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Save results
        self._save_results()
        
        print(f"\nBacktest complete!")
        print(f"Final capital: ${self.current_capital:,.2f}")
        print(f"Total return: {metrics['total_return_pct']:.2f}%")
        print(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        return metrics
    
    def _process_signal(
        self,
        timestamp: datetime,
        regime: str,
        signal_dict: Dict,
        current_row: pd.Series,
        current_idx: int
    ) -> None:
        """Process a trading signal."""
        # Skip if already in position
        if self.position:
            print(f"    DEBUG: Already in position, skipping signal")
            return
        
        # Check risk limits
        daily_pnl = sum(self.daily_pnls) if self.daily_pnls else 0.0
        
        can_trade = self.rm.can_trade(self.current_capital, daily_pnl)
        if not can_trade:
            print(f"    DEBUG: RiskManager blocked trade. Capital: ${self.current_capital:.2f}, Daily PnL: ${daily_pnl:.2f}")
            return
        
        # Get position size from RiskManager
        entry_price = signal_dict['entry_price']
        stop_loss = signal_dict['stop_loss']
        atr = current_row.get('atr_14', abs(entry_price - stop_loss))
        
        shares = self.rm.position_size(entry_price, stop_loss, atr)
        
        if shares <= 0:
            print(f"    DEBUG: Position size <= 0. Entry: ${entry_price:.2f}, Stop: ${stop_loss:.2f}, ATR: ${atr:.2f}, Shares: {shares}")
            return
        
        # Open position
        self.position = {
            'timestamp': timestamp,
            'regime': regime,
            'strategy': signal_dict.get('strategy', 'unknown'),
            'signal': signal_dict['signal'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': signal_dict.get('take_profit'),
            'atr': atr,
            'shares': shares,
            'confidence': signal_dict.get('confidence', 0.5),
            'drawdown_at_entry': self._get_current_drawdown(),
            'daily_pnl_before': daily_pnl,
            'entry_idx': current_idx
        }
        
        print(f"  {timestamp}: {signal_dict['signal']} {shares} shares @ ${entry_price:.2f} "
              f"(stop: ${stop_loss:.2f}, regime: {regime})")
    
    def _check_position_exit(self, current_row: pd.Series, current_idx: int) -> None:
        """Check if current position should be closed."""
        if not self.position:
            return
        
        position = self.position
        
        # Minimum holding period: 4 bars (avoid immediate close)
        bars_held = current_idx - position.get('entry_idx', current_idx)
        if bars_held < 4:
            return  # Don't check exit yet
        
        # Calculate take profit target
        # Prefer strategy-provided take_profit when available (e.g., BB opposite band)
        if position.get('take_profit') is not None:
            take_profit = float(position['take_profit'])
        else:
            take_profit = position['entry_price'] + (2 * position['atr']) if position['signal'] == 'LONG' else position['entry_price'] - (2 * position['atr'])
        
        exit_triggered = False
        exit_price = None
        reason = ''
        
        # Check take profit first (for longs: high >= target)
        if position['signal'] == 'LONG':
            if current_row['high'] >= take_profit:
                exit_triggered = True
                exit_price = max(current_row['open'], take_profit)  # Gap up protection
                reason = 'take_profit'
        else:  # SHORT
            if current_row['low'] <= take_profit:
                exit_triggered = True
                exit_price = min(current_row['open'], take_profit)  # Gap down protection
                reason = 'take_profit'
        
        # Check stop loss (if not already exited)
        if not exit_triggered:
            if position['signal'] == 'LONG':
                if current_row['low'] <= position['stop_loss']:
                    exit_triggered = True
                    exit_price = min(current_row['open'], position['stop_loss'])  # Gap down
                    reason = 'stop_loss'
            else:  # SHORT
                if current_row['high'] >= position['stop_loss']:
                    exit_triggered = True
                    exit_price = max(current_row['open'], position['stop_loss'])  # Gap up
                    reason = 'stop_loss'
        
        # Time-based exit: max hold 24 bars (1 day)
        if not exit_triggered and bars_held >= 24:
            exit_triggered = True
            exit_price = current_row['close']
            reason = 'time_exit'
        
        if exit_triggered:
            self._close_position(
                timestamp=current_row['timestamp'],
                exit_price=exit_price,
                reason=reason,
                bars_held=bars_held
            )
    
    def _close_position(
        self,
        timestamp: datetime,
        exit_price: float,
        reason: str,
        bars_held: int = 0
    ) -> None:
        """Close current position and record P&L."""
        if not self.position:
            return
        
        position = self.position
        
        # Calculate P&L
        if position['signal'] == 'LONG':
            raw_pnl = position['shares'] * (exit_price - position['entry_price'])
        else:  # SHORT
            raw_pnl = position['shares'] * (position['entry_price'] - exit_price)
        
        # Deduct realistic friction: slippage + fees on entry AND exit
        notional = position['shares'] * position['entry_price']
        friction = notional * (self.slippage_pct + self.fee_pct) * 2  # both legs
        pnl = raw_pnl - friction
        
        pnl_pct = (pnl / self.current_capital) * 100
        
        # Update capital
        self.current_capital += pnl
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'regime': position['regime'],
            'strategy': position['strategy'],
            'signal': position['signal'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'stop_loss': position['stop_loss'],
            'shares': position['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'confidence': position['confidence'],
            'drawdown_at_entry': position['drawdown_at_entry'],
            'daily_pnl_before': position['daily_pnl_before'],
            'exit_reason': reason,
            'bars_held': bars_held
        }
        
        self.trades.append(trade)
        self.daily_pnls.append(pnl)
        
        # Update statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Record with RiskManager
        self.rm.record_trade(pnl)
        
        print(f"  -> Close @ ${exit_price:.2f}, P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) [{reason}, held {bars_held} bars]")
        
        # Clear position
        self.position = None
    
    def _update_equity(self, timestamp: datetime) -> None:
        """Update equity curve and drawdown."""
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Calculate drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Record equity point
        daily_pnl = sum(self.daily_pnls) if self.daily_pnls else 0.0
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'capital': self.current_capital,
            'drawdown_pct': drawdown * 100,
            'daily_pnl': daily_pnl
        })
    
    def _get_current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        if self.peak_capital > 0:
            return (self.peak_capital - self.current_capital) / self.peak_capital
        return 0.0
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_return_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'final_capital': self.current_capital,
                'regime_stats': {}
            }
        
        # Total return
        total_return_pct = ((self.current_capital - self.initial_capital) 
                           / self.initial_capital) * 100
        
        # Win rate
        win_rate = (self.winning_trades / self.total_trades) * 100
        
        # Sharpe ratio (simplified - assume risk-free rate = 0)
        if len(self.daily_pnls) > 1:
            returns = np.array(self.daily_pnls) / self.initial_capital
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)  # Hourly to annual
        else:
            sharpe = 0.0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Regime breakdown
        regime_stats = {}
        for trade in self.trades:
            regime = trade['regime']
            if regime not in regime_stats:
                regime_stats[regime] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            
            regime_stats[regime]['trades'] += 1
            if trade['pnl'] > 0:
                regime_stats[regime]['wins'] += 1
            regime_stats[regime]['pnl'] += trade['pnl']
        
        return {
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'profit_factor': profit_factor,
            'regime_stats': regime_stats,
            'final_capital': self.current_capital
        }
    
    def _save_results(self) -> None:
        """Save backtest results to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save trades
        for trade in self.trades:
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, regime, strategy, signal, entry_price, exit_price,
                    stop_loss, shares, pnl, pnl_pct, confidence,
                    drawdown_at_entry, daily_pnl_before
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(trade['timestamp']),
                trade['regime'],
                trade['strategy'],
                trade['signal'],
                trade['entry_price'],
                trade['exit_price'],
                trade['stop_loss'],
                trade['shares'],
                trade['pnl'],
                trade['pnl_pct'],
                trade['confidence'],
                trade['drawdown_at_entry'],
                trade['daily_pnl_before']
            ))
        
        # Save equity curve
        for point in self.equity_curve:
            cursor.execute("""
                INSERT OR REPLACE INTO equity_curve (timestamp, capital, drawdown_pct, daily_pnl)
                VALUES (?, ?, ?, ?)
            """, (
                str(point['timestamp']),
                point['capital'],
                point['drawdown_pct'],
                point['daily_pnl']
            ))
        
        conn.commit()
        conn.close()


def run_backtest_full():
    """Run full backtest on BTC data."""
    print("=" * 70)
    print("BACKTEST ENGINE - Full Run")
    print("=" * 70)
    
    # Load data
    print("\nLoading data from database...")
    conn = sqlite3.connect('btc_pipeline.db')
    
    query = """
        SELECT c.timestamp, c.open, c.high, c.low, c.close, c.volume,
               r.regime, f.ema_20, f.atr_14, f.adx_14, f.rsi_14,
               f.bb_upper, f.bb_lower, f.bb_middle
        FROM candles c
        LEFT JOIN regimes r ON c.timestamp = r.timestamp
        LEFT JOIN features f ON c.timestamp = f.timestamp
        WHERE r.regime IS NOT NULL
          AND f.ema_20 IS NOT NULL
        ORDER BY c.timestamp
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} rows with regimes and features")
    
    # Initialize RiskManager
    rm = RiskManager(
        initial_capital=100000,
        max_daily_risk_pct=0.01,
        max_drawdown_pct=0.10,
        db_path='risk_log_backtest.db'
    )
    
    # Initialize backtest engine
    engine = BacktestEngine(
        risk_manager=rm,
        initial_capital=100000,
        db_path='backtest_results.db'
    )
    
    # Run backtest
    metrics = engine.run(df, regime_col='regime')
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Total Return:        {metrics['total_return_pct']:+.2f}%")
    print(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate:            {metrics['win_rate']:.1f}%")
    print(f"Total Trades:        {metrics['total_trades']}")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    
    if metrics['regime_stats']:
        print("\n" + "=" * 70)
        print("PERFORMANCE BY REGIME")
        print("=" * 70)
        for regime, stats in metrics['regime_stats'].items():
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            print(f"{regime:15s}: {stats['trades']:3d} trades, {win_rate:5.1f}% win, ${stats['pnl']:+,.2f}")
    
    print("=" * 70)
    
    return metrics


if __name__ == '__main__':
    metrics = run_backtest_full()
