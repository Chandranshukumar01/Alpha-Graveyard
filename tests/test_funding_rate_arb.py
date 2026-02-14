"""Tests for Funding Rate Arbitrage module."""

import unittest
import pandas as pd
import numpy as np
from alpha_graveyard.strategies.funding_arb import FundingRateArb, strategy_funding_arb


class TestFundingRateArb(unittest.TestCase):

    def setUp(self):
        periods = 200
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=periods, freq='h'),
            'close': np.linspace(40000, 42000, periods),
            'high': np.linspace(40100, 42100, periods),
            'low': np.linspace(39900, 41900, periods),
            'volume': np.random.randint(100, 1000, periods)
        })
        self.arb = FundingRateArb(mode='backtest')

    def test_precompute_adds_columns(self):
        """Funding rate and signal columns should be added."""
        df_result = self.arb.precompute_funding(self.df)
        self.assertIn('funding_rate', df_result.columns)
        self.assertIn('funding_signal', df_result.columns)

    def test_funding_rate_range(self):
        """Funding rates should be within realistic bounds."""
        df_result = self.arb.precompute_funding(self.df)
        self.assertTrue((df_result['funding_rate'] >= -0.001).all())
        self.assertTrue((df_result['funding_rate'] <= 0.001).all())

    def test_signal_values(self):
        """Funding signals should only be valid categories."""
        df_result = self.arb.precompute_funding(self.df)
        valid = {'SHORT_PERP', 'LONG_PERP', 'NONE'}
        unique = set(df_result['funding_signal'].unique())
        self.assertTrue(unique.issubset(valid), f"Invalid: {unique - valid}")

    def test_high_positive_funding_shorts(self):
        """Extreme positive funding should give SHORT signal."""
        row = {'funding_rate': 0.0005, 'close': 40000, 'atr_14': 200}
        result = self.arb.get_funding_signal(row, {})
        self.assertEqual(result['signal'], 'SHORT')
        self.assertGreater(result['confidence'], 0.5)

    def test_high_negative_funding_longs(self):
        """Extreme negative funding should give LONG signal."""
        row = {'funding_rate': -0.0005, 'close': 40000, 'atr_14': 200}
        result = self.arb.get_funding_signal(row, {})
        self.assertEqual(result['signal'], 'LONG')
        self.assertGreater(result['confidence'], 0.5)

    def test_neutral_funding_none(self):
        """Normal funding rate should give NONE signal."""
        row = {'funding_rate': 0.0001, 'close': 40000, 'atr_14': 200}
        result = self.arb.get_funding_signal(row, {})
        self.assertEqual(result['signal'], 'NONE')

    def test_strategy_wrapper_compatibility(self):
        """strategy_funding_arb wrapper should match standard interface."""
        row = {'funding_rate': 0.0005, 'close': 40000, 'atr_14': 200}
        result = strategy_funding_arb(row, {})
        self.assertIn('signal', result)
        self.assertIn('entry_price', result)
        self.assertIn('stop_loss', result)
        self.assertEqual(result['strategy'], 'funding_arb')

    def test_registry_lazy_load(self):
        """Funding arb should be loadable from strategy registry."""
        from alpha_graveyard.strategies.library import get_strategy_for_regime
        func = get_strategy_for_regime('funding_arb')
        result = func({'funding_rate': 0.0005, 'close': 40000, 'atr_14': 200}, {})
        self.assertEqual(result['signal'], 'SHORT')


if __name__ == '__main__':
    unittest.main()
