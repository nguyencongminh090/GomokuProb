"""
GomoProb V3: Mixture Model for Human Likelihood

Replaces the single Exponential with a 2-component Exponential mixture:
    P(Δ | Human) = π · λ_good · exp(-λ_good · Δ) + (1-π) · λ_blunder · exp(-λ_blunder · Δ)

This captures the bimodal nature of human play:
- Component 1 (good plays): Most moves have small Δ (careful, near-optimal)
- Component 2 (blunders): Occasional moves have large Δ (mistakes, miscalculations)

A single Exponential cannot model both behaviors simultaneously.
"""

import math
from typing import List, Tuple


class MixtureModel:
    """
    2-component Exponential mixture for modeling human move quality.

    Parameters:
        pi: Mixing weight for the "good play" component (0 to 1)
        lambda_good: Rate for good plays (higher = tighter around 0)
        lambda_blunder: Rate for blunders (lower = heavier tail)
    """

    def __init__(
        self,
        pi: float = 0.75,
        lambda_good: float = 20.0,
        lambda_blunder: float = 3.0
    ):
        if not (0.0 < pi < 1.0):
            raise ValueError(f"pi must be in (0, 1), got {pi}")
        if lambda_good <= 0 or lambda_blunder <= 0:
            raise ValueError("lambda values must be positive")

        self.pi = pi
        self.lambda_good = lambda_good
        self.lambda_blunder = lambda_blunder

    def pdf(self, delta: float) -> float:
        """
        Mixture PDF at a single delta value.

        P(Δ) = π · λ_g · exp(-λ_g · Δ) + (1-π) · λ_b · exp(-λ_b · Δ)
        """
        delta = max(delta, 1e-10)
        comp_good = self.pi * self.lambda_good * math.exp(-self.lambda_good * delta)
        comp_blunder = (1.0 - self.pi) * self.lambda_blunder * math.exp(-self.lambda_blunder * delta)
        return comp_good + comp_blunder

    def log_pdf(self, delta: float) -> float:
        """
        Log of mixture PDF at a single delta value.

        Uses log-sum-exp for numerical stability:
        log(π·f₁ + (1-π)·f₂) = log_sum_exp(log(π·f₁), log((1-π)·f₂))
        """
        delta = max(delta, 1e-4)  # epsilon floor

        # Log of each weighted component
        log_comp_good = (
            math.log(self.pi)
            + math.log(self.lambda_good)
            - self.lambda_good * delta
        )
        log_comp_blunder = (
            math.log(1.0 - self.pi)
            + math.log(self.lambda_blunder)
            - self.lambda_blunder * delta
        )

        # log-sum-exp
        max_log = max(log_comp_good, log_comp_blunder)
        result = max_log + math.log(
            math.exp(log_comp_good - max_log)
            + math.exp(log_comp_blunder - max_log)
        )
        return result

    def log_likelihood(self, deltas: List[float]) -> float:
        """
        Total log-likelihood of a sequence of deltas under the mixture model.

        log L = Σ log P(Δ_i)
        """
        if not deltas:
            return 0.0

        total = 0.0
        for d in deltas:
            total += self.log_pdf(d)
        return total

    def log_likelihood_tempered(self, delta: float, alpha: float) -> float:
        """
        Tempered log-likelihood for a single observation.

        log P(Δ)^α = α · log P(Δ)

        Used for complexity-weighted Bayesian updates:
        α = complexity means low-complexity positions contribute
        fractional evidence instead of being skipped entirely.

        Args:
            delta: Winrate loss value
            alpha: Tempering exponent (0 to 1), typically = position complexity

        Returns:
            Tempered log-likelihood
        """
        return alpha * self.log_pdf(delta)

    def fit_em(
        self,
        deltas: List[float],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Tuple[float, float, float]:
        """
        Fit mixture parameters via Expectation-Maximization (Paper Section 3.3).

        E-step: Compute responsibilities γ_{i,1} and γ_{i,2}  (Paper Eq. e_step)
        M-step: Update π, λ_good, λ_blunder                   (Paper Eqs. m_step_pi, m_step_lambda)

        This modifies self.pi, self.lambda_good, self.lambda_blunder in-place
        and returns the fitted parameters.

        Args:
            deltas: Observed winrate delta values
            max_iter: Maximum EM iterations
            tol: Convergence tolerance for log-likelihood change

        Returns:
            Tuple of (pi, lambda_good, lambda_blunder) after convergence
        """
        if len(deltas) < 5:
            return (self.pi, self.lambda_good, self.lambda_blunder)

        adjusted = [max(d, 1e-4) for d in deltas]
        n = len(adjusted)

        # Initialize from current parameters
        pi = self.pi
        lam_g = self.lambda_good
        lam_b = self.lambda_blunder

        prev_ll = float('-inf')

        for iteration in range(max_iter):
            # --- E-step: compute responsibilities ---
            gamma_good = []
            for d in adjusted:
                log_comp_g = math.log(pi) + math.log(lam_g) - lam_g * d
                log_comp_b = math.log(1 - pi) + math.log(lam_b) - lam_b * d
                max_log = max(log_comp_g, log_comp_b)
                # Normalize in log space
                resp_g = math.exp(log_comp_g - max_log)
                resp_b = math.exp(log_comp_b - max_log)
                total = resp_g + resp_b
                gamma_good.append(resp_g / total if total > 0 else 0.5)

            # --- M-step: update parameters ---
            sum_gamma_g = sum(gamma_good)
            sum_gamma_b = n - sum_gamma_g

            # Prevent degenerate components
            if sum_gamma_g < 1e-6 or sum_gamma_b < 1e-6:
                break

            pi = sum_gamma_g / n

            weighted_sum_g = sum(g * d for g, d in zip(gamma_good, adjusted))
            weighted_sum_b = sum((1 - g) * d for g, d in zip(gamma_good, adjusted))

            lam_g = sum_gamma_g / weighted_sum_g if weighted_sum_g > 0 else lam_g
            lam_b = sum_gamma_b / weighted_sum_b if weighted_sum_b > 0 else lam_b

            # Ensure lambda_good > lambda_blunder (good component is tighter)
            if lam_g < lam_b:
                lam_g, lam_b = lam_b, lam_g
                pi = 1.0 - pi

            # Compute log-likelihood for convergence check
            ll = 0.0
            for d in adjusted:
                comp_g = pi * lam_g * math.exp(-lam_g * d)
                comp_b = (1 - pi) * lam_b * math.exp(-lam_b * d)
                ll += math.log(max(comp_g + comp_b, 1e-300))

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        # Clamp pi to valid range
        pi = max(0.01, min(0.99, pi))

        # Update self
        self.pi = pi
        self.lambda_good = lam_g
        self.lambda_blunder = lam_b

        return (pi, lam_g, lam_b)


def log_likelihood_exponential(delta: float, lambda_val: float) -> float:
    """
    Log-likelihood of a single delta under a simple Exponential(λ).

    log(λ · exp(-λΔ)) = log(λ) - λΔ

    Used for the cheater hypothesis (single Exponential is appropriate
    since cheaters consistently play near-optimal).
    """
    delta = max(delta, 1e-4)
    return math.log(lambda_val) - lambda_val * delta


def log_likelihood_exponential_tempered(
    delta: float, lambda_val: float, alpha: float
) -> float:
    """
    Tempered log-likelihood for cheater model.

    α · [log(λ) - λΔ]
    """
    delta = max(delta, 1e-4)
    return alpha * (math.log(lambda_val) - lambda_val * delta)
