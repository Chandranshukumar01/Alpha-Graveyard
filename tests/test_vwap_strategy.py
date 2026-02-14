
import unittest
from alpha_graveyard.strategies.library import strategy_vwap_reversion

class TestStrategyVWAP(unittest.TestCase):
    
    def test_vwap_long_signal(self):
        """Test LONG signal when price < vwap_lower_band"""
        row = {
            'close': 40000.0,
            'vwap_lower_band': 40100.0,  # Price below lower band
            'vwap_upper_band': 41000.0,
            'vwap_20': 40500.0,
            'atr_14': 200.0,
            'rsi_14': 40.0
        }
        prev_row = {} # Not used by this strategy currently
        
        result = strategy_vwap_reversion(row, prev_row)
        
        self.assertEqual(result['signal'], 'LONG')
        self.assertEqual(result['entry_price'], 40000.0)
        self.assertEqual(result['stop_loss'], 40000.0 - (1.5 * 200.0))
        self.assertEqual(result['confidence'], 0.7)

    def test_vwap_long_signal_high_confidence(self):
        """Test LONG signal with high confidence (RSI < 30)"""
        row = {
            'close': 40000.0,
            'vwap_lower_band': 40100.0,
            'vwap_upper_band': 41000.0,
            'vwap_20': 40500.0,
            'atr_14': 200.0,
            'rsi_14': 25.0  # Oversold
        }
        prev_row = {}
        
        result = strategy_vwap_reversion(row, prev_row)
        
        self.assertEqual(result['signal'], 'LONG')
        self.assertEqual(result['confidence'], 0.85)

    def test_vwap_short_signal(self):
        """Test SHORT signal when price > vwap_upper_band"""
        row = {
            'close': 42000.0,
            'vwap_lower_band': 40000.0,
            'vwap_upper_band': 41500.0,  # Price above upper band
            'vwap_20': 40750.0,
            'atr_14': 200.0,
            'rsi_14': 60.0
        }
        prev_row = {}
        
        result = strategy_vwap_reversion(row, prev_row)
        
        self.assertEqual(result['signal'], 'SHORT')
        self.assertEqual(result['entry_price'], 42000.0)
        self.assertEqual(result['stop_loss'], 42000.0 + (1.5 * 200.0))
        self.assertEqual(result['confidence'], 0.7)

    def test_vwap_short_signal_high_confidence(self):
        """Test SHORT signal with high confidence (RSI > 70)"""
        row = {
            'close': 42000.0,
            'vwap_lower_band': 40000.0,
            'vwap_upper_band': 41500.0,
            'vwap_20': 40750.0,
            'atr_14': 200.0,
            'rsi_14': 75.0  # Overbought
        }
        prev_row = {}
        
        result = strategy_vwap_reversion(row, prev_row)
        
        self.assertEqual(result['signal'], 'SHORT')
        self.assertEqual(result['confidence'], 0.85)

    def test_no_signal_inside_bands(self):
        """Test NO signal when price is inside bands"""
        row = {
            'close': 40500.0,
            'vwap_lower_band': 40000.0,
            'vwap_upper_band': 41000.0,
            'vwap_20': 40500.0,
            'atr_14': 200.0,
            'rsi_14': 50.0
        }
        prev_row = {}
        
        result = strategy_vwap_reversion(row, prev_row)
        
        self.assertEqual(result['signal'], 'NONE')

    def test_missing_data_returns_none(self):
        """Test graceful handling of missing data"""
        row = {
            'close': 40000.0,
            # Missing vwap bands
        }
        prev_row = {}
        
        result = strategy_vwap_reversion(row, prev_row)
        self.assertEqual(result['signal'], 'NONE')

if __name__ == '__main__':
    unittest.main()
