"""Tests for ETF Flow Signal module."""

import unittest
import pandas as pd
import numpy as np
from alpha_graveyard.strategies.etf_flow import ETFFlowSignal


class TestETFFlowSignal(unittest.TestCase):

    def setUp(self):
        periods = 500
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=periods, freq='h'),
            'open': np.linspace(100, 200, periods),
            'high': np.linspace(101, 201, periods),
            'low': np.linspace(99, 199, periods),
            'close': np.linspace(100, 200, periods),
            'volume': np.random.randint(100, 1000, periods)
        })
        self.etf = ETFFlowSignal(mode='backtest', lookback=5)

    def test_precompute_adds_columns(self):
        """Test that precompute adds all expected columns."""
        df_result = self.etf.precompute_flows(self.df)

        self.assertIn('etf_flow', df_result.columns)
        self.assertIn('etf_flow_5d', df_result.columns)
        self.assertIn('etf_flow_bias', df_result.columns)

    def test_bias_values_are_valid(self):
        """Test that bias only contains valid categories."""
        df_result = self.etf.precompute_flows(self.df)
        valid = {'inflow', 'outflow', 'neutral'}
        unique = set(df_result['etf_flow_bias'].unique())
        self.assertTrue(unique.issubset(valid), f"Invalid biases: {unique - valid}")

    def test_confidence_modifier_aligned_long(self):
        """LONG + inflow should boost confidence."""
        signal = {'signal': 'LONG', 'confidence': 0.7}
        modified = self.etf.apply_confidence_modifier(signal, 'inflow')
        self.assertGreater(modified['confidence'], 0.7)

    def test_confidence_modifier_conflicting(self):
        """LONG + outflow should reduce confidence."""
        signal = {'signal': 'LONG', 'confidence': 0.7}
        modified = self.etf.apply_confidence_modifier(signal, 'outflow')
        self.assertLess(modified['confidence'], 0.7)

    def test_confidence_modifier_neutral(self):
        """Neutral bias should not change confidence."""
        signal = {'signal': 'LONG', 'confidence': 0.7}
        modified = self.etf.apply_confidence_modifier(signal, 'neutral')
        self.assertEqual(modified['confidence'], 0.7)

    def test_confidence_modifier_none_signal(self):
        """NONE signal should pass through unchanged."""
        signal = {'signal': 'NONE', 'confidence': 0.0}
        modified = self.etf.apply_confidence_modifier(signal, 'inflow')
        self.assertEqual(modified['confidence'], 0.0)


if __name__ == '__main__':
    unittest.main()
