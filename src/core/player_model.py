"""
GomoProb V2: Bayesian Player Model

Implements the Bayesian framework for anti-cheating detection.
P(Cheat | D) ∝ P(D | Cheat) × P(Cheat)

Core concepts:
- Prior P(Cheat): Base rate of cheating in the population
- Likelihood P(D | Cheat): How likely is this data given cheating
- Posterior P(Cheat | D): Updated probability after seeing data
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.core.delta_model import DeltaModel, MoveAnalysis, GameAnalysis


@dataclass
class PlayerClassification:
    """Final classification result for a player/game."""
    p_cheat: float               # Posterior P(Cheat | D)
    p_human: float               # Posterior P(Human | D) = 1 - P(Cheat)
    classification: str          # "Human", "Suspicious", "Cheater"
    confidence: float            # How confident (based on data amount)
    
    # Supporting data
    lambda_estimated: float      # MLE λ from delta model
    log_likelihood_human: float
    log_likelihood_cheater: float
    
    # Thresholds used
    threshold_suspicious: float
    threshold_cheater: float


class BayesianPlayerModel:
    """
    Bayesian framework for player classification.
    
    Uses Bayes' theorem:
        P(Cheat | D) = P(D | Cheat) × P(Cheat) / P(D)
        
    Where:
        P(D) = P(D | Cheat) × P(Cheat) + P(D | Human) × P(Human)
    
    The likelihoods are computed from the Exponential delta model.
    """
    
    def __init__(
        self,
        prior_cheat: float = 0.01,           # 1% base rate
        lambda_human: float = 5.0,            # Expected λ for humans
        lambda_cheater: float = 50.0,         # Expected λ for cheaters
        threshold_suspicious: float = 0.3,    # P(Cheat) > this → Suspicious
        threshold_cheater: float = 0.7        # P(Cheat) > this → Cheater
    ):
        """
        Args:
            prior_cheat: Prior probability of cheating (before seeing data)
            lambda_human: Expected Exponential rate for humans (lower = more variance)
            lambda_cheater: Expected Exponential rate for cheaters (higher = less variance)
            threshold_suspicious: P(Cheat) threshold for "Suspicious" label
            threshold_cheater: P(Cheat) threshold for "Cheater" label
        """
        self.prior_cheat = prior_cheat
        self.prior_human = 1.0 - prior_cheat
        self.lambda_human = lambda_human
        self.lambda_cheater = lambda_cheater
        self.threshold_suspicious = threshold_suspicious
        self.threshold_cheater = threshold_cheater
        
        self.delta_model = DeltaModel()
    
    def compute_posterior(self, deltas: List[float]) -> Tuple[float, float, float, float]:
        """
        Compute posterior P(Cheat | D) using Bayes' theorem.
        
        Args:
            deltas: List of winrate delta values for the game
            
        Returns:
            Tuple of (p_cheat, p_human, log_lik_human, log_lik_cheater)
        """
        if not deltas:
            # No data → stick with prior
            return self.prior_cheat, self.prior_human, 0.0, 0.0
        
        # Compute log-likelihoods
        log_lik_human = self.delta_model.log_likelihood_human(deltas, self.lambda_human)
        log_lik_cheater = self.delta_model.log_likelihood_cheater(deltas, self.lambda_cheater)
        
        # Convert to likelihoods (use log-sum-exp for numerical stability)
        # P(D | Human) ∝ exp(log_lik_human)
        # P(D | Cheat) ∝ exp(log_lik_cheater)
        
        # Bayes' rule in log space:
        # log P(Cheat | D) ∝ log_lik_cheater + log(prior_cheat)
        # log P(Human | D) ∝ log_lik_human + log(prior_human)
        
        log_post_cheat = log_lik_cheater + math.log(self.prior_cheat)
        log_post_human = log_lik_human + math.log(self.prior_human)
        
        # Normalize using log-sum-exp
        max_log = max(log_post_cheat, log_post_human)
        
        # exp(log_post_x - max) for numerical stability
        exp_cheat = math.exp(log_post_cheat - max_log)
        exp_human = math.exp(log_post_human - max_log)
        
        total = exp_cheat + exp_human
        
        p_cheat = exp_cheat / total
        p_human = exp_human / total
        
        return p_cheat, p_human, log_lik_human, log_lik_cheater
    
    def classify_game(self, game_analysis: GameAnalysis) -> PlayerClassification:
        """
        Classify a game based on its analysis.
        
        Args:
            game_analysis: GameAnalysis object from DeltaModel
            
        Returns:
            PlayerClassification with final verdict
        """
        deltas = [m.delta for m in game_analysis.moves]
        
        p_cheat, p_human, log_lik_human, log_lik_cheater = self.compute_posterior(deltas)
        
        # Determine classification
        if p_cheat >= self.threshold_cheater:
            classification = "Cheater"
        elif p_cheat >= self.threshold_suspicious:
            classification = "Suspicious"
        else:
            classification = "Human"
        
        # Confidence based on data amount and posterior certainty
        n_moves = len(deltas)
        data_confidence = min(1.0, n_moves / 30)  # Full confidence at 30+ moves
        posterior_certainty = abs(p_cheat - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%
        confidence = data_confidence * posterior_certainty
        
        return PlayerClassification(
            p_cheat=p_cheat,
            p_human=p_human,
            classification=classification,
            confidence=confidence,
            lambda_estimated=game_analysis.lambda_mle,
            log_likelihood_human=log_lik_human,
            log_likelihood_cheater=log_lik_cheater,
            threshold_suspicious=self.threshold_suspicious,
            threshold_cheater=self.threshold_cheater
        )
    
    def update_online(
        self, 
        current_posterior: float, 
        new_delta: float, 
        complexity: float = 1.0
    ) -> float:
        """
        Online update of P(Cheat) after observing a new move.
        
        Uses sequential Bayesian update with complexity weighting:
            P(Cheat | D_new) ∝ P(D_new | Cheat) × P(Cheat | D_old)
            
        The update is weighted by complexity:
        - High complexity (1.0): Full update weight
        - Low complexity (0.0): Minimal update (trivial positions don't count)
        
        Args:
            current_posterior: Current P(Cheat | D_old)
            new_delta: New Δ value observed
            complexity: Position complexity 0.0-1.0 (from calculate_complexity)
            
        Returns:
            Updated P(Cheat | D_new)
        """
        epsilon = 0.001
        delta = max(new_delta, epsilon)
        
        # For trivial positions (complexity < 0.2), skip update entirely
        if complexity < 0.2:
            return current_posterior
        
        # Single observation likelihoods
        lik_human = self.lambda_human * math.exp(-self.lambda_human * delta)
        lik_cheater = self.lambda_cheater * math.exp(-self.lambda_cheater * delta)
        
        # Use current posterior as new prior
        prior_cheat = current_posterior
        prior_human = 1.0 - current_posterior
        
        # Bayes update
        numerator = lik_cheater * prior_cheat
        denominator = lik_cheater * prior_cheat + lik_human * prior_human
        
        if denominator == 0:
            return current_posterior
        
        new_posterior = numerator / denominator
        
        # Weight the update by complexity
        # High complexity (1.0) → use new_posterior fully
        # Low complexity (0.3) → mostly keep old posterior
        # Formula: interpolate between old and new based on complexity
        weight = min(1.0, complexity * 1.25)  # Scale: 0.8 complexity = full weight
        
        weighted_posterior = (1.0 - weight) * current_posterior + weight * new_posterior
        
        return weighted_posterior
