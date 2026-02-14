import sqlite3
from datetime import datetime
from typing import Optional, Tuple


class RiskManager:
    """
    Risk management kernel for trading systems.
    
    Hard constraints (non-negotiable):
    - MAX_DAILY_RISK = 1% of capital per day
    - MAX_DRAWDOWN = 10% total loss = full stop
    - MAX_SINGLE_TRADE_RISK = 0.5% per position (max 2 trades per day)
    """
    
    MAX_DAILY_RISK = 0.01
    MAX_DRAWDOWN = 0.10
    MAX_SINGLE_TRADE_RISK = 0.005
    ATR_MULTIPLIER = 1.0
    
    def __init__(
        self, 
        initial_capital: float, 
        max_daily_risk_pct: float = 0.01,
        max_drawdown_pct: float = 0.10,
        db_path: str = "risk_log.db"
    ):
        """
        Initialize RiskManager with hard limits.
        
        Args:
            initial_capital: Starting capital amount
            max_daily_risk_pct: Max daily risk as decimal (default 1%)
            max_drawdown_pct: Max drawdown before trading stops (default 10%)
            db_path: Path to SQLite database for logging
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_daily_risk_pct = max_daily_risk_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.db_path = db_path
        
        # Track daily and total state
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database with required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                decision TEXT CHECK(decision IN ('ALLOWED', 'BLOCKED_DAILY_LIMIT', 'BLOCKED_DRAWDOWN')),
                capital_before REAL,
                capital_after REAL,
                drawdown_pct REAL,
                position_size_requested REAL,
                position_size_allowed REAL,
                reason TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _log_decision(
        self,
        decision: str,
        capital_before: float,
        capital_after: float,
        position_size_requested: float = 0.0,
        position_size_allowed: float = 0.0,
        reason: str = ""
    ) -> None:
        """Log risk decision to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO risk_decisions 
            (decision, capital_before, capital_after, drawdown_pct, position_size_requested, position_size_allowed, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            decision,
            capital_before,
            capital_after,
            self.current_drawdown,
            position_size_requested,
            position_size_allowed,
            reason
        ))
        
        conn.commit()
        conn.close()
    
    def can_trade(self, portfolio_value: float, todays_pnl: float) -> bool:
        """
        Determine if trading is allowed today.
        
        Blocks trading if:
        - Daily loss limit hit (1% of capital)
        - Drawdown exceeded (10% of peak capital)
        
        Args:
            portfolio_value: Current total portfolio value
            todays_pnl: Today's realized P&L
            
        Returns:
            True if trading allowed, False otherwise
        """
        self.current_capital = portfolio_value
        self.daily_pnl = todays_pnl
        
        # Update peak capital and drawdown
        if portfolio_value > self.peak_capital:
            self.peak_capital = portfolio_value
        
        self.current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
        
        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown_pct:
            self._log_decision(
                'BLOCKED_DRAWDOWN',
                portfolio_value,
                portfolio_value,
                reason=f"Drawdown limit exceeded: {self.current_drawdown:.4f} >= {self.max_drawdown_pct}"
            )
            return False
        
        # Check daily risk limit
        daily_risk_used = abs(todays_pnl) / self.initial_capital
        if daily_risk_used >= self.max_daily_risk_pct:
            self._log_decision(
                'BLOCKED_DAILY_LIMIT',
                portfolio_value,
                portfolio_value,
                reason=f"Daily risk limit reached: {daily_risk_used:.4f} >= {self.max_daily_risk_pct}"
            )
            return False
        
        # Check max trades per day (implied by 0.5% per trade * 2 = 1% daily)
        if self.daily_trades_count >= 2:
            self._log_decision(
                'BLOCKED_DAILY_LIMIT',
                portfolio_value,
                portfolio_value,
                reason=f"Max daily trades reached: {self.daily_trades_count}"
            )
            return False
        
        self._log_decision(
            'ALLOWED',
            portfolio_value,
            portfolio_value,
            reason="Trading approved"
        )
        return True
    
    def position_size(
        self, 
        entry_price: float, 
        stop_loss_price: float, 
        atr: float
    ) -> int:
        """
        Calculate position size using Kelly-inspired volatility-adjusted sizing.
        
        Formula: risk_amount / (ATR * multiplier)
        - risk_amount = capital * 0.5% (max single trade risk)
        - ATR multiplier = 2.0 (wider stops in high volatility = smaller size)
        
        Args:
            entry_price: Planned entry price
            stop_loss_price: Stop loss price level
            atr: Average True Range (volatility measure)
            
        Returns:
            Number of shares/contracts to trade (integer)
        """
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")
        
        if atr <= 0:
            # Handle zero/negative volatility edge case
            # Use minimum volatility of 1% of entry price
            atr = entry_price * 0.01
        
        # Risk amount: 0.5% of current capital
        risk_amount = self.current_capital * self.MAX_SINGLE_TRADE_RISK
        
        # Calculate volatility-adjusted risk
        # price_risk = distance to stop loss
        # atr_adjustment = ATR * multiplier (default 2.0 for wider stops in volatility)
        price_risk = abs(entry_price - stop_loss_price)
        atr_adjustment = atr * 2.0  # ATR multiplier from class constant
        volatility_adjusted_risk = price_risk + atr_adjustment
        
        if volatility_adjusted_risk <= 0:
            return 0
        
        # Calculate shares
        shares = int(risk_amount / volatility_adjusted_risk)
        
        return shares
    
    def record_trade(self, pnl: float) -> None:
        """
        Record trade result and update internal state.
        
        Args:
            pnl: Profit/loss from the trade (positive for profit, negative for loss)
        """
        capital_before = self.current_capital
        self.current_capital += pnl
        self.daily_pnl += pnl
        self.daily_trades_count += 1
        
        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Log the trade
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO risk_decisions 
            (decision, capital_before, capital_after, drawdown_pct, position_size_requested, position_size_allowed, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'ALLOWED',
            capital_before,
            self.current_capital,
            self.current_drawdown,
            0.0,
            0.0,
            f"Trade recorded: P&L=${pnl:.2f}, daily_pnl=${self.daily_pnl:.2f}, trades_today={self.daily_trades_count}"
        ))
        conn.commit()
        conn.close()
    
    def reset_daily(self) -> None:
        """Reset daily counters (call at start of new trading day)."""
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
    
    def get_stats(self) -> dict:
        """Return current risk statistics."""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown_pct': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_trades_count': self.daily_trades_count,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_daily_risk_pct': self.max_daily_risk_pct
        }
