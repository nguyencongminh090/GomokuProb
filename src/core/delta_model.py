"""
GomoProb V2: Winrate Delta Model

Implements the Exponential distribution model for move quality analysis.
Δ_i = W_best - W_play ~ Exponential(λ)

Core concepts:
- λ_human: Expected rate for legitimate players (larger variance, more mistakes)
- λ_cheater: Expected rate for cheaters (smaller variance, near-optimal play)
- AccuracyIndex = 1/λ: Lower = closer to engine-optimal
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import statistics


@dataclass
class MoveAnalysis:
    """Analysis data for a single move."""
    move_index: int
    move_notation: str
    best_winrate: float      # W_best
    played_winrate: float    # W_play
    delta: float             # Δ = W_best - W_play (always >= 0)
    is_forced: bool = False  # True if only one reasonable option
    position_complexity: float = 1.0  # Weight for importance
    

@dataclass
class GameAnalysis:
    """Aggregated analysis for an entire game."""
    moves: List[MoveAnalysis] = field(default_factory=list)
    
    # Computed Statistics
    lambda_mle: float = 0.0          # MLE estimate of λ
    accuracy_index: float = 0.0      # 1/λ
    mean_delta: float = 0.0
    std_delta: float = 0.0
    near_optimal_ratio: float = 0.0  # Proportion with Δ < ε
    
    # V2 Metrics
    weighted_mean_delta: float = 0.0  # Weighted by position complexity


class DeltaModel:
    """
    Implements the Winrate Delta Model from the anti-cheating paper.
    
    The model assumes:
    - For humans: Δ ~ Exponential(λ_human), where λ_human is small (large variance)
    - For cheaters: Δ ~ Exponential(λ_cheater), where λ_cheater is large (small variance)
    
    We estimate λ using Maximum Likelihood Estimation (MLE).
    For Exponential distribution: λ_MLE = 1 / mean(Δ)
    """
    
    def __init__(self, near_optimal_threshold: float = 0.02):
        """
        Args:
            near_optimal_threshold: Δ below this is considered "near-optimal".
                                   Default 0.02 = 2% winrate difference.
        """
        self.near_optimal_threshold = near_optimal_threshold
    
    def calculate_delta(self, best_wr: float, played_wr: float) -> float:
        """
        Calculate winrate delta.
        
        Δ = W_best - W_play
        Always >= 0 by definition (played can't be better than best in hindsight).
        """
        delta = best_wr - played_wr
        return max(0.0, delta)  # Ensure non-negative
    
    def estimate_lambda_mle(self, deltas: List[float]) -> float:
        """
        Estimate λ using Maximum Likelihood Estimation.
        
        For Exponential(λ):
            L(λ) = Π λ * exp(-λ * Δ_i)
            log L = n*log(λ) - λ * Σ Δ_i
            d/dλ = n/λ - Σ Δ_i = 0
            λ_MLE = n / Σ Δ_i = 1 / mean(Δ)
        
        Returns:
            λ estimate (higher = smaller deltas = more suspicious)
        """
        if not deltas:
            return 0.0
        
        # Filter out zero deltas (perfect moves) to avoid division issues
        # In practice, we add a small epsilon to handle Δ=0
        epsilon = 0.001  # 0.1% winrate
        adjusted_deltas = [max(d, epsilon) for d in deltas]
        
        mean_delta = statistics.mean(adjusted_deltas)
        
        if mean_delta <= 0:
            return float('inf')  # Perfect play (extremely suspicious)
        
        lambda_mle = 1.0 / mean_delta
        return lambda_mle
    
    def analyze_game(self, moves: List[MoveAnalysis]) -> GameAnalysis:
        """
        Perform full game analysis.
        
        Returns GameAnalysis with all statistics computed.
        """
        result = GameAnalysis(moves=moves)
        
        if not moves:
            return result
        
        # Extract deltas
        deltas = [m.delta for m in moves]
        
        # Basic statistics
        result.mean_delta = statistics.mean(deltas) if deltas else 0.0
        result.std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        
        # MLE λ
        result.lambda_mle = self.estimate_lambda_mle(deltas)
        result.accuracy_index = 1.0 / result.lambda_mle if result.lambda_mle > 0 else float('inf')
        
        # Near-optimal ratio
        near_optimal_count = sum(1 for d in deltas if d < self.near_optimal_threshold)
        result.near_optimal_ratio = near_optimal_count / len(deltas) if deltas else 0.0
        
        # Weighted mean delta (by position complexity)
        total_weight = sum(m.position_complexity for m in moves)
        if total_weight > 0:
            weighted_sum = sum(m.delta * m.position_complexity for m in moves)
            result.weighted_mean_delta = weighted_sum / total_weight
        else:
            result.weighted_mean_delta = result.mean_delta
        
        return result
    
    def log_likelihood_human(self, deltas: List[float], lambda_human: float = 5.0) -> float:
        """
        Calculate log-likelihood under Human model.
        
        Args:
            deltas: List of Δ values
            lambda_human: Expected λ for humans (default 5.0 = avg 20% winrate loss)
        
        Returns:
            Log-likelihood value
        """
        if not deltas:
            return 0.0
        
        epsilon = 0.001
        log_lik = 0.0
        for d in deltas:
            d_adj = max(d, epsilon)
            # log(λ * exp(-λd)) = log(λ) - λd
            log_lik += math.log(lambda_human) - lambda_human * d_adj
        
        return log_lik
    
    def log_likelihood_cheater(self, deltas: List[float], lambda_cheater: float = 50.0) -> float:
        """
        Calculate log-likelihood under Cheater model.
        
        Args:
            deltas: List of Δ values
            lambda_cheater: Expected λ for cheaters (default 50.0 = avg 2% winrate loss)
        
        Returns:
            Log-likelihood value
        """
        if not deltas:
            return 0.0
        
        epsilon = 0.001
        log_lik = 0.0
        for d in deltas:
            d_adj = max(d, epsilon)
            log_lik += math.log(lambda_cheater) - lambda_cheater * d_adj
        
        return log_lik
