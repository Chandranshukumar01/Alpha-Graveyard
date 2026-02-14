
import unittest
import json
from alpha_graveyard.strategies.ensemble import WeightedEnsembleStrategy

# Mock strategy functions
def mock_sub_strategy_long(row, prev_row):
    return {'signal': 'LONG', 'entry_price': 100.0, 'stop_loss': 90.0, 'confidence': 0.8}

def mock_sub_strategy_short(row, prev_row):
    return {'signal': 'SHORT', 'entry_price': 100.0, 'stop_loss': 110.0, 'confidence': 0.8}

def mock_sub_strategy_none(row, prev_row):
    return {'signal': 'NONE', 'entry_price': None, 'stop_loss': None, 'confidence': 0.0}

class TestEnsembleStrategy(unittest.TestCase):
    
    def setUp(self):
        # Initialize with standard thresholds
        self.ensemble = WeightedEnsembleStrategy(probability_threshold=0.15, signal_threshold=0.4)
        
        # Override strategies with mocks to isolate ensemble logic
        self.ensemble.strategies = {
            'trending_up': mock_sub_strategy_long,
            'trending_down': mock_sub_strategy_short,
            'ranging': mock_sub_strategy_none,
            'high_vol': mock_sub_strategy_short
        }

    def test_strong_long_consensus(self):
        """Test simple case where one regime dominates with strong signal."""
        # Scenario: 80% Trending Up, 20% Ranging
        probs = {'trending_up': 0.8, 'ranging': 0.2}
        row = {'regime_json': json.dumps(probs)}
        
        result = self.ensemble.get_signal(row, {})
        
        # Calculation:
        # Trending Up: 0.8 (prob) * 0.8 (conf) * 1 (dir) = 0.64
        # Ranging: 0.2 (prob) * 0 (conf) * 0 (dir) = 0.0
        # Total Weighted Score: 0.64
        # Threshold: 0.4
        # Result: LONG
        
        self.assertEqual(result['signal'], 'LONG')
        self.assertEqual(result['entry_price'], 100.0)
        self.assertEqual(result['stop_loss'], 90.0)
        
        # Confidence: 0.64 / (0.8 + 0.2) = 0.64
        self.assertAlmostEqual(result['confidence'], 0.64)

    def test_conflicting_signals_cancellation(self):
        """Test opposing signals canceling each other out."""
        # Scenario: 45% Up, 45% Down -> Uncertainty
        probs = {'trending_up': 0.45, 'trending_down': 0.45}
        row = {'regime_json': json.dumps(probs)}
        
        result = self.ensemble.get_signal(row, {})
        
        # Calculation:
        # Up: 0.45 * 0.8 * 1 = 0.36
        # Down: 0.45 * 0.8 * -1 = -0.36
        # Net: 0.0
        
        self.assertEqual(result['signal'], 'NONE')
        self.assertEqual(result['confidence'], 0.0)

    def test_weak_signals_below_threshold(self):
        """Test weak probability regime not triggering signal."""
        # Scenario: 30% Up (below signal threshold contribution), 70% Ranging
        probs = {'trending_up': 0.3, 'ranging': 0.7}
        row = {'regime_json': json.dumps(probs)}
        
        result = self.ensemble.get_signal(row, {})
        
        # Calculation:
        # Up: 0.3 * 0.8 * 1 = 0.24
        # Ranging: 0.7 * 0 * 0 = 0
        # Net: 0.24 ( < 0.4 Threshold)
        
        self.assertEqual(result['signal'], 'NONE')

    def test_mixed_short_signal(self):
        """Test multiple regimes agreeing on Short."""
        # Scenario: 50% Trending Down, 30% High Vol (Short)
        probs = {'trending_down': 0.5, 'high_vol': 0.3}
        row = {'regime_json': json.dumps(probs)}
        
        result = self.ensemble.get_signal(row, {})
        
        # Calculation:
        # Down: 0.5 * 0.8 * -1 = -0.40
        # HighVol: 0.3 * 0.8 * -1 = -0.24
        # Net: -0.64 ( < -0.4 ) -> SHORT
        
        self.assertEqual(result['signal'], 'SHORT')
        self.assertEqual(result['entry_price'], 100.0)
        
        # Confidence: |-0.64| / (0.5 + 0.3) = 0.8
        self.assertAlmostEqual(result['confidence'], 0.8)

    def test_missing_json_returns_none(self):
        """Test handling of missing/invalid JSON."""
        row = {}
        result = self.ensemble.get_signal(row, {})
        self.assertEqual(result['signal'], 'NONE')
        
        row = {'regime_json': 'invalid'}
        result = self.ensemble.get_signal(row, {})
        self.assertEqual(result['signal'], 'NONE')

if __name__ == '__main__':
    unittest.main()
