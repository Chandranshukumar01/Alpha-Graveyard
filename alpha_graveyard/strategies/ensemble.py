
import json
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from alpha_graveyard.strategies import library as strategy_library

class WeightedEnsembleStrategy:
    """
    Combines signals from multiple strategies based on regime probability.
    
    Logic:
    1. Parse regime probabilities from `regime_json`.
    2. Run strategies for all potential regimes (score > threshold).
    3. Weight each strategy's signal by its regime probability.
    4. Sum weighted signals to get a net directional score.
    5. If net score exceeds threshold, trigger trade.
    """
    
    def __init__(self, probability_threshold: float = 0.15, signal_threshold: float = 0.4):
        """
        Args:
            probability_threshold: Minimum regime probability to consider a strategy.
            signal_threshold: Minimum net weighted score to trigger a trade.
        """
        self.prob_threshold = probability_threshold
        self.signal_threshold = signal_threshold
        
        # Mapping from regime name to strategy function
        # We use the registry from strategy_library
        self.strategies = strategy_library.STRATEGY_REGISTRY

    def get_signal(self, row: Dict[str, Any], prev_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate ensemble signal for a single row.
        
        Args:
            row: Current candle data + features
            prev_row: Previous candle data
            
        Returns:
            Signal dictionary (signal, entry, stop, confidence)
        """
        regime_json = row.get('regime_json', '{}')
        try:
            scores = json.loads(regime_json)
        except:
            # Fallback if no json or invalid
            return {'signal': 'NONE', 'confidence': 0.0}
            
        if not scores:
            return {'signal': 'NONE', 'confidence': 0.0}
            
        # 1. Identify active strategies
        active_strategies = []
        total_weight = 0.0
        
        for regime, score in scores.items():
            if score >= self.prob_threshold:
                strategy_func = self.strategies.get(regime)
                if strategy_func:
                    active_strategies.append((regime, score, strategy_func))
                    total_weight += score
        
        if not active_strategies:
            return {'signal': 'NONE', 'confidence': 0.0}
            
        # 2. Collect signals
        weighted_score_sum = 0.0
        signals = []
        
        for regime, score, func in active_strategies:
            res = func(row, prev_row)
            signal_dir = 0
            if res['signal'] == 'LONG':
                signal_dir = 1
            elif res['signal'] == 'SHORT':
                signal_dir = -1
            elif res['signal'] == 'NONE':
                continue # Contribution is 0
                
            confidence = res.get('confidence', 0.5)
            
            # Weighted contribution: Regime Prob * Strategy Confidence * Direction
            contribution = score * confidence * signal_dir
            weighted_score_sum += contribution
            
            if signal_dir != 0:
                signals.append({
                    'regime': regime,
                    'weight': score,
                    'direction': signal_dir,
                    'entry': res.get('entry_price'),
                    'stop': res.get('stop_loss'),
                    'confidence': confidence
                })

        # 3. Decision Logic
        final_signal = 'NONE'
        if weighted_score_sum > self.signal_threshold:
            final_signal = 'LONG'
        elif weighted_score_sum < -self.signal_threshold:
            final_signal = 'SHORT'
            
        if final_signal == 'NONE':
            return {'signal': 'NONE', 'confidence': 0.0}
            
        # 4. Aggregation (Entry/Stop)
        # Filter signals matching the final direction
        relevant_signals = [s for s in signals if (s['direction'] == 1 and final_signal == 'LONG') or 
                           (s['direction'] == -1 and final_signal == 'SHORT')]
        
        if not relevant_signals:
            return {'signal': 'NONE', 'confidence': 0.0}
            
        # Weighted average for Entry and Stop
        # Weight by (Regime Score * Confidence)
        
        total_rel_weight = sum(s['weight'] * s['confidence'] for s in relevant_signals)
        
        if total_rel_weight == 0:
            avg_entry = relevant_signals[0]['entry']
            avg_stop = relevant_signals[0]['stop']
        else:
            avg_entry = sum(s['entry'] * s['weight'] * s['confidence'] for s in relevant_signals) / total_rel_weight
            avg_stop = sum(s['stop'] * s['weight'] * s['confidence'] for s in relevant_signals) / total_rel_weight
            
        # Normalize confidence (roughly)
        # Net Score / Total Regime Weight (of active strategies that fired)
        # Or just use abs(weighted_score_sum) clipped to 1.0?
        # Let's use proportional confidence
        final_confidence = min(abs(weighted_score_sum) / max(total_weight, 0.01), 1.0)
        
        return {
            'signal': final_signal,
            'entry_price': round(avg_entry, 2),
            'stop_loss': round(avg_stop, 2),
            'confidence': round(final_confidence, 2),
            'metadata': {
                'weighted_score': round(weighted_score_sum, 3),
                'active_regimes': [s['regime'] for s in relevant_signals]
            }
        }

def run_ensemble_on_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run ensemble strategy on entire DataFrame."""
    ensemble = WeightedEnsembleStrategy()
    results = []
    
    prev_row = {}
    for idx, row in df.iterrows():
        res = ensemble.get_signal(row.to_dict(), prev_row)
        results.append(res)
        prev_row = row.to_dict()
        
    # Convert list of dicts to DataFrame columns or return as Struct
    signals_df = pd.DataFrame(results)
    return signals_df
