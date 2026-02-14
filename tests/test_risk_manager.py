import unittest
import sqlite3
import os
from alpha_graveyard.engine.risk import RiskManager


class TestRiskManager(unittest.TestCase):
    """Unit tests for RiskManager class covering edge cases and constraints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db = "test_risk_log.db"
        # Clean up any existing test database
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_init_valid_capital(self):
        """Test initialization with valid capital."""
        rm = RiskManager(100000, db_path=self.test_db)
        self.assertEqual(rm.initial_capital, 100000)
        self.assertEqual(rm.current_capital, 100000)
        self.assertEqual(rm.peak_capital, 100000)
    
    def test_init_zero_capital_raises_error(self):
        """Test that zero capital raises ValueError."""
        with self.assertRaises(ValueError) as context:
            RiskManager(0, db_path=self.test_db)
        self.assertIn("Initial capital must be positive", str(context.exception))
    
    def test_init_negative_capital_raises_error(self):
        """Test that negative capital raises ValueError."""
        with self.assertRaises(ValueError) as context:
            RiskManager(-1000, db_path=self.test_db)
        self.assertIn("Initial capital must be positive", str(context.exception))
    
    def test_can_trade_allows_normal_conditions(self):
        """Test that trading is allowed under normal conditions."""
        rm = RiskManager(100000, db_path=self.test_db)
        result = rm.can_trade(100000, 0)
        self.assertTrue(result)
    
    def test_can_trade_blocks_at_max_drawdown(self):
        """Test that trading is blocked when drawdown exceeds 10%."""
        rm = RiskManager(100000, db_path=self.test_db)
        # Simulate 10% drawdown
        result = rm.can_trade(90000, 0)
        self.assertFalse(result)
    
    def test_can_trade_blocks_near_max_drawdown(self):
        """Test blocking when drawdown is exactly at limit."""
        rm = RiskManager(100000, db_path=self.test_db)
        # Capital at exactly 90% of peak = 10% drawdown
        result = rm.can_trade(90000, 0)
        self.assertFalse(result)
    
    def test_can_trade_blocks_daily_limit(self):
        """Test blocking when daily risk limit reached."""
        rm = RiskManager(100000, db_path=self.test_db)
        # 1% daily limit = $1000 loss
        result = rm.can_trade(99000, -1000)
        self.assertFalse(result)
    
    def test_can_trade_blocks_max_trades_per_day(self):
        """Test blocking after 2 trades (0.5% each = 1% daily)."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.daily_trades_count = 2
        result = rm.can_trade(100000, 0)
        self.assertFalse(result)
    
    def test_position_size_basic_calculation(self):
        """Test basic position size calculation."""
        rm = RiskManager(100000, db_path=self.test_db)
        # Entry $100, stop $98, ATR $2
        # Risk amount = $100000 * 0.005 = $500
        # Vol risk = |100-98| + (2*2) = 2 + 4 = $6
        # Shares = $500 / $6 = 83
        shares = rm.position_size(100, 98, 2)
        self.assertEqual(shares, 83)
    
    def test_position_size_zero_atr_uses_minimum(self):
        """Test that zero ATR uses minimum volatility buffer."""
        rm = RiskManager(100000, db_path=self.test_db)
        # Zero ATR should use 1% of entry price as minimum
        shares = rm.position_size(100, 98, 0)
        # Risk = $500, vol_risk = 2 + (1*2) = 4, shares = 125
        self.assertGreater(shares, 0)
    
    def test_position_size_negative_atr_uses_minimum(self):
        """Test that negative ATR uses minimum volatility buffer."""
        rm = RiskManager(100000, db_path=self.test_db)
        shares = rm.position_size(100, 98, -1)
        self.assertGreater(shares, 0)
    
    def test_position_size_zero_entry_raises_error(self):
        """Test that zero entry price raises ValueError."""
        rm = RiskManager(100000, db_path=self.test_db)
        with self.assertRaises(ValueError) as context:
            rm.position_size(0, 98, 2)
        self.assertIn("Entry price must be positive", str(context.exception))
    
    def test_position_size_negative_entry_raises_error(self):
        """Test that negative entry price raises ValueError."""
        rm = RiskManager(100000, db_path=self.test_db)
        with self.assertRaises(ValueError) as context:
            rm.position_size(-100, 98, 2)
        self.assertIn("Entry price must be positive", str(context.exception))
    
    def test_position_size_high_volatility_reduces_size(self):
        """Test that high ATR reduces position size."""
        rm = RiskManager(100000, db_path=self.test_db)
        
        # Low volatility
        low_vol_shares = rm.position_size(100, 98, 1)
        
        # High volatility
        high_vol_shares = rm.position_size(100, 98, 10)
        
        # High volatility should result in fewer shares
        self.assertLess(high_vol_shares, low_vol_shares)
    
    def test_record_trade_updates_capital(self):
        """Test that recording a trade updates capital correctly."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.record_trade(1000)
        self.assertEqual(rm.current_capital, 101000)
    
    def test_record_trade_updates_daily_pnl(self):
        """Test that recording a trade updates daily P&L."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.record_trade(1000)
        self.assertEqual(rm.daily_pnl, 1000)
    
    def test_record_trade_updates_trade_count(self):
        """Test that recording a trade increments daily trade count."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.record_trade(1000)
        self.assertEqual(rm.daily_trades_count, 1)
    
    def test_record_trade_updates_drawdown(self):
        """Test that recording a loss updates drawdown correctly."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.record_trade(-5000)
        self.assertEqual(rm.current_drawdown, 0.05)  # 5%
    
    def test_record_trade_updates_peak(self):
        """Test that recording a profit updates peak capital."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.record_trade(5000)
        self.assertEqual(rm.peak_capital, 105000)
    
    def test_reset_daily_clears_counters(self):
        """Test that reset_daily clears daily counters."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.record_trade(1000)
        rm.record_trade(500)
        rm.reset_daily()
        self.assertEqual(rm.daily_pnl, 0)
        self.assertEqual(rm.daily_trades_count, 0)
    
    def test_get_stats_returns_correct_values(self):
        """Test that get_stats returns expected dictionary."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.record_trade(-5000)
        stats = rm.get_stats()
        
        self.assertEqual(stats['initial_capital'], 100000)
        self.assertEqual(stats['current_capital'], 95000)
        self.assertEqual(stats['peak_capital'], 100000)
        self.assertEqual(stats['current_drawdown_pct'], 0.05)
        self.assertEqual(stats['daily_pnl'], -5000)
        self.assertEqual(stats['daily_trades_count'], 1)
    
    def test_database_created_with_schema(self):
        """Test that database is created with correct schema."""
        rm = RiskManager(100000, db_path=self.test_db)
        
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_decisions'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check schema
        cursor.execute("PRAGMA table_info(risk_decisions)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {'id', 'timestamp', 'decision', 'capital_before', 'capital_after', 
                   'drawdown_pct', 'position_size_requested', 'position_size_allowed', 'reason'}
        self.assertEqual(columns, expected)
        
        conn.close()
    
    def test_can_trade_logs_to_database(self):
        """Test that can_trade logs decisions to database."""
        rm = RiskManager(100000, db_path=self.test_db)
        rm.can_trade(100000, 0)
        
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM risk_decisions")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertGreater(count, 0)
    
    def test_max_drawdown_circuit_breaker(self):
        """Test the 10% max drawdown hard constraint."""
        rm = RiskManager(100000, db_path=self.test_db)
        
        # Reset daily to isolate drawdown test from daily limit
        rm.reset_daily()
        
        # Trade 1: Small loss (0.5% daily, well under 1% limit)
        rm.record_trade(-500)
        self.assertTrue(rm.can_trade(99500, -500))
        rm.reset_daily()  # Reset for next trade
        
        # Trade 2: Another small loss, cumulative drawdown now ~1%
        rm.record_trade(-500)
        self.assertTrue(rm.can_trade(99000, -500))
        rm.reset_daily()
        
        # Continue until near drawdown limit
        # Each trade loses 500, need to get to 10% drawdown (10000 loss)
        for _ in range(16):  # 16 more trades = 8000 additional loss
            rm.reset_daily()
            rm.record_trade(-500)
        
        # Now at ~9000 total loss, 9% drawdown
        self.assertTrue(rm.can_trade(91000, -500))
        rm.reset_daily()
        
        # One more loss to push over 10% drawdown
        rm.record_trade(-2000)  # Total loss now 11000, 11% drawdown
        
        # Should be blocked by drawdown
        result = rm.can_trade(89000, -2000)
        self.assertFalse(result)
    
    def test_daily_risk_circuit_breaker(self):
        """Test the 1% daily risk hard constraint."""
        rm = RiskManager(100000, db_path=self.test_db)
        
        # First trade: 0.4% loss ($400, well under 1% = $1000)
        rm.record_trade(-400)
        self.assertTrue(rm.can_trade(99600, -400))
        
        # Record second trade to reach trade count limit
        rm.record_trade(-400)
        
        # Test that 1.2% daily loss would be blocked by daily limit (before trade count check)
        # Note: can_trade checks daily risk first, then trade count
        result = rm.can_trade(98800, -1200)  # 1.2% daily loss
        self.assertFalse(result)  # Blocked by 1% daily risk limit
        
        # Verify 0.8% is under daily limit but still blocked by trade count = 2
        result_under_limit = rm.can_trade(99200, -800)  # 0.8% daily loss
        # At this point daily_trades_count = 2, so should be blocked by count, not risk
        self.assertFalse(result_under_limit)  # Blocked by 2-trade limit
    
    def test_two_trade_daily_limit(self):
        """Test that max 2 trades per day is enforced."""
        rm = RiskManager(100000, db_path=self.test_db)
        
        # Two profitable trades
        rm.record_trade(500)
        rm.record_trade(500)
        
        # Third trade should be blocked by count limit
        result = rm.can_trade(101000, 1000)
        self.assertFalse(result)
    
    def test_circuit_breaker_triggered(self):
        """Test that circuit breaker triggers at >10% drawdown."""
        rm = RiskManager(initial_capital=100000, db_path=self.test_db)
        
        # Force 11% drawdown by manipulating internal state
        rm.peak_capital = 100000
        rm.current_capital = 89000
        rm.daily_pnl = 0
        rm.daily_trades_count = 0
        
        # Verify drawdown calculation
        expected_drawdown = (100000 - 89000) / 100000  # 0.11
        self.assertAlmostEqual(expected_drawdown, 0.11)
        
        # Circuit breaker should block trading
        result = rm.can_trade(89000, 0)
        self.assertFalse(result)
        
        # Verify the drawdown was recorded
        self.assertAlmostEqual(rm.current_drawdown, 0.11, places=4)


if __name__ == '__main__':
    unittest.main()
