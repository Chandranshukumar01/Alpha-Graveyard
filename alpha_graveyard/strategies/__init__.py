"""
Strategies subpackage: All 25 falsified trading strategies.

Each strategy follows a common interface:
    Input: row (current OHLCV + features), prev_row (previous bar)
    Output: dict with signal ('LONG'/'SHORT'/'NONE'), entry_price, stop_loss, confidence

Modules:
    library: Core strategies S-01 to S-08 (EMA crossover, Bollinger, VWAP, etc.)
    ensemble: Weighted ensemble strategy (S-08)
    signals: ETF flow signal (S-04) and funding rate arbitrage (S-05)
    lighthouse: Multi-agent LLM agentic strategy (S-12)

See experiments/ for standalone simulation scripts (S-16 through S-25).
"""

from alpha_graveyard.strategies.library import (
    STRATEGY_REGISTRY,
    get_strategy_for_regime,
)
