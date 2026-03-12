"""
GomoProb V2/V4: Winrate Delta Model

Implements the distributional family for move quality analysis.
Δ_i = W_best - W_play

Supported distributions (Paper Section 3.2):
- Exponential(λ): Baseline, single parameter
- Gamma(α, β): Shape-rate generalization, unimodal for α > 1
- Weibull(k, λ): Tail flexibility, k < 1 heavy-tailed, k > 1 light-tailed

Model selection via AIC/BIC (Paper Section 3.2.4).

Core concepts:
- λ_human: Expected rate for legitimate players (larger variance, more mistakes)
- λ_cheater: Expected rate for cheaters (smaller variance, near-optimal play)
- AccuracyIndex = 1/λ: Lower = closer to engine-optimal
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
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


class SoftmaxChoiceModel:
    """
    Models player move selection using softmax (Boltzmann) distribution.
    
    P(choosing move_i | candidates, τ) = exp(W_i / τ) / Σ exp(W_j / τ)
    
    Where:
    - W_i = winrate of move i
    - τ = temperature parameter
    - Low τ → always picks best move (engine-like, suspicious)
    - High τ → more uniform/random choice (human-like)
    
    The temperature τ can be estimated via MLE from observed games.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def softmax_probs(winrates: List[float], temperature: float) -> List[float]:
        """
        Compute softmax probabilities for candidate moves.
        
        Args:
            winrates: Winrates of each candidate move
            temperature: τ parameter (> 0)
            
        Returns:
            List of probabilities (sum to 1.0)
        """
        if not winrates or temperature <= 0:
            n = len(winrates) if winrates else 1
            return [1.0 / n] * n
        
        # Scale by temperature
        scaled = [w / temperature for w in winrates]
        
        # log-sum-exp for stability
        max_val = max(scaled)
        exps = [math.exp(s - max_val) for s in scaled]
        total = sum(exps)
        
        return [e / total for e in exps]
    
    @staticmethod
    def log_prob_of_choice(
        winrates: List[float],
        chosen_index: int,
        temperature: float
    ) -> float:
        """
        Log-probability that the player chose the move at chosen_index.
        
        log P(choice | τ) = W_chosen/τ - log(Σ exp(W_j/τ))
        
        Args:
            winrates: Winrates of all candidates
            chosen_index: Index of the actually played move (0-indexed)
            temperature: τ parameter
            
        Returns:
            Log-probability (always ≤ 0)
        """
        if not winrates or chosen_index < 0 or chosen_index >= len(winrates):
            return 0.0
        if temperature <= 0:
            return 0.0
        
        scaled = [w / temperature for w in winrates]
        max_val = max(scaled)
        
        # log P = scaled[chosen] - log(Σ exp(scaled))
        # = scaled[chosen] - max_val - log(Σ exp(scaled - max_val))
        log_sum_exp = max_val + math.log(
            sum(math.exp(s - max_val) for s in scaled)
        )
        
        return scaled[chosen_index] - log_sum_exp
    
    @staticmethod
    def estimate_temperature(
        observations: List[Tuple[List[float], int]],
        temp_range: Tuple[float, float] = (0.005, 0.5),
        steps: int = 50
    ) -> float:
        """
        Estimate optimal temperature via grid search MLE.
        
        Finds τ that maximizes Σ log P(choice_i | candidates_i, τ).
        
        Args:
            observations: List of (candidate_winrates, chosen_index) tuples
            temp_range: (min_τ, max_τ) search range
            steps: Number of grid search steps
            
        Returns:
            Estimated τ (lower = more engine-like)
        """
        if not observations:
            return 0.1  # Default
        
        best_tau = temp_range[0]
        best_ll = float('-inf')
        
        tau_min, tau_max = temp_range
        
        for i in range(steps):
            tau = tau_min + (tau_max - tau_min) * i / (steps - 1)
            
            total_ll = 0.0
            for winrates, chosen_idx in observations:
                total_ll += SoftmaxChoiceModel.log_prob_of_choice(
                    winrates, chosen_idx, tau
                )
            
            if total_ll > best_ll:
                best_ll = total_ll
                best_tau = tau
        
        return best_tau

    @staticmethod
    def temperature_score(tau: float) -> float:
        """
        Compute temperature score S_τ = 1/τ̂ (Paper Eq. tau_score).

        High S_τ (low temperature) indicates engine-like selection behavior.

        Args:
            tau: Estimated temperature (> 0)

        Returns:
            Temperature score (higher = more suspicious)
        """
        if tau <= 0:
            return float('inf')
        return 1.0 / tau


# =============================================================================
# V4: Gamma Distribution Model (Paper Section 3.2.2)
# =============================================================================

@dataclass
class FitResult:
    """Result of fitting a distribution to data."""
    distribution: str   # "exponential", "gamma", "weibull"
    params: dict        # Distribution parameters
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float     # KS test p-value


class GammaModel:
    """
    Gamma distribution model for winrate deltas (Paper Section 3.2.2).

    f(Δ | α, β) = β^α · Δ^(α-1) · exp(-βΔ) / Γ(α)

    When α=1, reduces to Exponential(β).
    For α > 1, unimodal with mode at (α-1)/β.
    """

    @staticmethod
    def log_pdf(delta: float, alpha: float, beta: float) -> float:
        """Log-PDF of Gamma(α, β) at delta."""
        delta = max(delta, 1e-10)
        return (
            alpha * math.log(beta)
            + (alpha - 1) * math.log(delta)
            - beta * delta
            - math.lgamma(alpha)
        )

    @staticmethod
    def pdf(delta: float, alpha: float, beta: float) -> float:
        """PDF of Gamma(α, β) at delta."""
        return math.exp(GammaModel.log_pdf(delta, alpha, beta))

    @staticmethod
    def log_likelihood(deltas: List[float], alpha: float, beta: float) -> float:
        """Total log-likelihood of data under Gamma(α, β)."""
        if not deltas:
            return 0.0
        return sum(GammaModel.log_pdf(max(d, 1e-4), alpha, beta) for d in deltas)

    @staticmethod
    def fit_mle(deltas: List[float]) -> Tuple[float, float]:
        """
        Fit Gamma(α, β) via MLE using scipy (Paper Eq. gamma_mle).

        Returns:
            (alpha, beta) tuple
        """
        from scipy.stats import gamma as gamma_dist
        adjusted = [max(d, 1e-4) for d in deltas]
        a, _, scale = gamma_dist.fit(adjusted, floc=0)
        beta = 1.0 / scale  # scipy uses scale = 1/β
        return (a, beta)


class WeibullModel:
    """
    Weibull distribution model for winrate deltas (Paper Section 3.2.3).

    f(Δ | k, λ) = (k/λ) · (Δ/λ)^(k-1) · exp(-(Δ/λ)^k)

    k=1: Exponential(1/λ)
    k<1: Heavier tail (human blunders)
    k>1: Lighter tail (cheater-like)
    """

    @staticmethod
    def log_pdf(delta: float, k: float, lam: float) -> float:
        """Log-PDF of Weibull(k, λ) at delta."""
        delta = max(delta, 1e-10)
        return (
            math.log(k) - math.log(lam)
            + (k - 1) * (math.log(delta) - math.log(lam))
            - (delta / lam) ** k
        )

    @staticmethod
    def pdf(delta: float, k: float, lam: float) -> float:
        """PDF of Weibull(k, λ) at delta."""
        return math.exp(WeibullModel.log_pdf(delta, k, lam))

    @staticmethod
    def log_likelihood(deltas: List[float], k: float, lam: float) -> float:
        """Total log-likelihood of data under Weibull(k, λ)."""
        if not deltas:
            return 0.0
        return sum(WeibullModel.log_pdf(max(d, 1e-4), k, lam) for d in deltas)

    @staticmethod
    def fit_mle(deltas: List[float]) -> Tuple[float, float]:
        """
        Fit Weibull(k, λ) via MLE using scipy.

        Returns:
            (k, lambda) tuple
        """
        from scipy.stats import weibull_min
        adjusted = [max(d, 1e-4) for d in deltas]
        c, _, scale = weibull_min.fit(adjusted, floc=0)
        return (c, scale)


# =============================================================================
# V4: Model Selection (Paper Section 3.2.4)
# =============================================================================

class ModelSelector:
    """
    Select between Exponential, Gamma, and Weibull using AIC/BIC
    (Paper Eqs. aic, bic) and Kolmogorov-Smirnov goodness-of-fit test.
    """

    @staticmethod
    def compute_aic(log_lik: float, n_params: int) -> float:
        """AIC = 2k - 2·ln(L̂) (Paper Eq. aic)."""
        return 2 * n_params - 2 * log_lik

    @staticmethod
    def compute_bic(log_lik: float, n_params: int, n_samples: int) -> float:
        """BIC = k·ln(n) - 2·ln(L̂) (Paper Eq. bic)."""
        if n_samples <= 0:
            return float('inf')
        return n_params * math.log(n_samples) - 2 * log_lik

    @staticmethod
    def fit_and_compare(deltas: List[float]) -> List[FitResult]:
        """
        Fit Exponential, Gamma, and Weibull to data, returning results
        sorted from best to worst by BIC.

        Args:
            deltas: List of winrate delta values (≥ 0)

        Returns:
            List of FitResult sorted by BIC (best first)
        """
        from scipy.stats import (
            expon as expon_dist,
            gamma as gamma_dist,
            weibull_min,
            kstest,
        )

        adjusted = [max(d, 1e-4) for d in deltas]
        n = len(adjusted)
        if n < 3:
            return []

        results = []

        # --- Exponential ---
        try:
            mean_d = statistics.mean(adjusted)
            lam = 1.0 / mean_d
            ll = sum(math.log(lam) - lam * d for d in adjusted)
            aic = ModelSelector.compute_aic(ll, 1)
            bic = ModelSelector.compute_bic(ll, 1, n)
            ks_stat, ks_p = kstest(adjusted, 'expon', args=(0, mean_d))
            results.append(FitResult(
                distribution="exponential",
                params={"lambda": lam},
                log_likelihood=ll, aic=aic, bic=bic,
                ks_statistic=ks_stat, ks_pvalue=ks_p,
            ))
        except Exception:
            pass

        # --- Gamma ---
        try:
            a, _, scale = gamma_dist.fit(adjusted, floc=0)
            beta = 1.0 / scale
            ll = GammaModel.log_likelihood(adjusted, a, beta)
            aic = ModelSelector.compute_aic(ll, 2)
            bic = ModelSelector.compute_bic(ll, 2, n)
            ks_stat, ks_p = kstest(adjusted, 'gamma', args=(a, 0, scale))
            results.append(FitResult(
                distribution="gamma",
                params={"alpha": a, "beta": beta},
                log_likelihood=ll, aic=aic, bic=bic,
                ks_statistic=ks_stat, ks_pvalue=ks_p,
            ))
        except Exception:
            pass

        # --- Weibull ---
        try:
            c, _, scale = weibull_min.fit(adjusted, floc=0)
            ll = WeibullModel.log_likelihood(adjusted, c, scale)
            aic = ModelSelector.compute_aic(ll, 2)
            bic = ModelSelector.compute_bic(ll, 2, n)
            ks_stat, ks_p = kstest(adjusted, 'weibull_min', args=(c, 0, scale))
            results.append(FitResult(
                distribution="weibull",
                params={"k": c, "lambda": scale},
                log_likelihood=ll, aic=aic, bic=bic,
                ks_statistic=ks_stat, ks_pvalue=ks_p,
            ))
        except Exception:
            pass

        # Sort by BIC (lower is better)
        results.sort(key=lambda r: r.bic)
        return results

    @staticmethod
    def best_model(deltas: List[float]) -> Optional[FitResult]:
        """Return the best model by BIC, or None if fitting fails."""
        results = ModelSelector.fit_and_compare(deltas)
        return results[0] if results else None

