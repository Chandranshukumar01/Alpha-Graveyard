"""
Engine subpackage: Backtesting infrastructure.

Modules:
    backtest: Event-driven backtesting engine with regime-based strategy selection
    risk: Risk management kernel (Kelly Criterion, vol targeting, drawdown breaks)
    data: OHLCV data ingestion from exchanges via CCXT
"""

from alpha_graveyard.engine.backtest import BacktestEngine
from alpha_graveyard.engine.risk import RiskManager
