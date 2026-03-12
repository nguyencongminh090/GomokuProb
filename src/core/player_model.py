"""
GomoProb V3: Bayesian Player Model

Implements the Bayesian framework for anti-cheating detection.
P(Cheat | D) ∝ P(D | Cheat) × P(Cheat)

V3 Changes:
- Uses MixtureModel for human likelihood (2-component Exponential)
- Uses tempered likelihood for complexity weighting (replaces linear interpolation)
- Online update is the single authoritative pathway
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.core.delta_model import DeltaModel, MoveAnalysis, GameAnalysis
from src.core.mixture_model import (
    MixtureModel,
    log_likelihood_exponential,
    log_likelihood_exponential_tempered,
)


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
    
    V3 uses:
    - MixtureModel for P(D | Human): 2-component Exponential captures both
      careful play and occasional blunders
    - Single Exponential for P(D | Cheater): cheaters consistently play near-optimal
    - Tempered likelihood for complexity weighting instead of linear interpolation
    """
    
    def __init__(
        self,
        prior_cheat: float = 0.01,           # 1% base rate
        lambda_human: float = 5.0,            # LEGACY: kept for batch compatibility
        lambda_cheater: float = 50.0,         # Expected λ for cheaters
        threshold_suspicious: float = 0.3,    # P(Cheat) > this → Suspicious
        threshold_cheater: float = 0.7,       # P(Cheat) > this → Cheater
        # V3: Mixture parameters for human model
        mixture_pi: float = 0.75,             # Weight of "good play" component
        mixture_lambda_good: float = 20.0,    # Rate for good plays
        mixture_lambda_blunder: float = 3.0   # Rate for blunders
    ):
        self.prior_cheat = prior_cheat
        self.prior_human = 1.0 - prior_cheat
        self.lambda_human = lambda_human  # Legacy batch parameter
        self.lambda_cheater = lambda_cheater
        self.threshold_suspicious = threshold_suspicious
        self.threshold_cheater = threshold_cheater
        
        self.delta_model = DeltaModel()
        
        # V3: Mixture model for human likelihood
        self.mixture_model = MixtureModel(
            pi=mixture_pi,
            lambda_good=mixture_lambda_good,
            lambda_blunder=mixture_lambda_blunder
        )
    
    def compute_posterior(self, deltas: List[float]) -> Tuple[float, float, float, float]:
        """
        Compute posterior P(Cheat | D) using Bayes' theorem.
        
        V3: Uses mixture model for human likelihood.
        
        Args:
            deltas: List of winrate delta values for the game
            
        Returns:
            Tuple of (p_cheat, p_human, log_lik_human, log_lik_cheater)
        """
        if not deltas:
            return self.prior_cheat, self.prior_human, 0.0, 0.0
        
        # V3: Use mixture model for human likelihood
        log_lik_human = self.mixture_model.log_likelihood(deltas)
        
        # Cheater: single Exponential (unchanged)
        log_lik_cheater = self.delta_model.log_likelihood_cheater(
            deltas, self.lambda_cheater
        )
        
        # Bayes' rule in log space
        log_post_cheat = log_lik_cheater + math.log(self.prior_cheat)
        log_post_human = log_lik_human + math.log(self.prior_human)
        
        # Normalize using log-sum-exp
        max_log = max(log_post_cheat, log_post_human)
        exp_cheat = math.exp(log_post_cheat - max_log)
        exp_human = math.exp(log_post_human - max_log)
        total = exp_cheat + exp_human
        
        p_cheat = exp_cheat / total
        p_human = exp_human / total
        
        return p_cheat, p_human, log_lik_human, log_lik_cheater
    
    def classify_game(self, game_analysis: GameAnalysis) -> PlayerClassification:
        """
        Classify a game based on its analysis.
        
        Note: In V3, this uses the mixture model for batch classification.
        The online posterior (from update_online) is the preferred result
        since it respects per-move complexity tempering.
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
        data_confidence = min(1.0, n_moves / 30)
        posterior_certainty = abs(p_cheat - 0.5) * 2
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
        
        V3: Uses TEMPERED LIKELIHOOD instead of linear interpolation.
        
        The tempering exponent α = complexity means:
        - High complexity (1.0): Full evidence weight → P(D|H)^1.0
        - Low complexity (0.1): Fractional evidence → P(D|H)^0.1
        - Zero complexity (0.0): No evidence → posterior unchanged
        
        This is mathematically proper: tempering scales the log-likelihood,
        making low-complexity observations contribute proportionally less
        evidence without breaking Bayesian coherence.
        
        Args:
            current_posterior: Current P(Cheat | D_old)
            new_delta: New Δ value observed
            complexity: Position complexity 0.0-1.0 (used as tempering exponent)
            
        Returns:
            Updated P(Cheat | D_new)
        """
        # α = 0 means no update (position is trivial, no information)
        if complexity <= 0.0:
            return current_posterior
        
        # Tempered log-likelihoods
        # Human: mixture model tempered
        log_lik_human = self.mixture_model.log_likelihood_tempered(
            new_delta, alpha=complexity
        )
        # Cheater: single Exponential tempered
        log_lik_cheater = log_likelihood_exponential_tempered(
            new_delta, self.lambda_cheater, alpha=complexity
        )
        
        # Use current posterior as prior (sequential Bayesian update)
        # Handle edge cases where posterior is exactly 0 or 1
        prior_cheat = max(1e-15, min(1.0 - 1e-15, current_posterior))
        prior_human = 1.0 - prior_cheat
        
        # Bayes in log space
        log_post_cheat = log_lik_cheater + math.log(prior_cheat)
        log_post_human = log_lik_human + math.log(prior_human)
        
        # Normalize via log-sum-exp
        max_log = max(log_post_cheat, log_post_human)
        exp_cheat = math.exp(log_post_cheat - max_log)
        exp_human = math.exp(log_post_human - max_log)
        total = exp_cheat + exp_human
        
        if total == 0:
            return current_posterior
        
        new_posterior = exp_cheat / total
        return new_posterior
