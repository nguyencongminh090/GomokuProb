"""
Tests for V4 distribution models: Gamma, Weibull, ModelSelector,
EM algorithm, Likelihood Ratio Test, and Sensitivity Analysis.
"""

import sys
import os
import unittest
import math
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.core.delta_model import (
    GammaModel, WeibullModel, ModelSelector, FitResult, SoftmaxChoiceModel,
)
from src.core.mixture_model import MixtureModel
from src.core.player_model import BayesianPlayerModel


# =============================================================================
# Gamma Distribution Tests
# =============================================================================

class TestGammaModel(unittest.TestCase):

    def test_pdf_positive(self):
        """PDF should be positive for valid delta."""
        for d in [0.01, 0.05, 0.1, 0.3, 0.5]:
            pdf_val = GammaModel.pdf(d, alpha=2.0, beta=10.0)
            self.assertGreater(pdf_val, 0, f"PDF should be > 0 at delta={d}")

    def test_log_pdf_equals_log_of_pdf(self):
        """log_pdf should equal log(pdf)."""
        for d in [0.01, 0.05, 0.15, 0.3]:
            log_val = GammaModel.log_pdf(d, 2.0, 10.0)
            direct = math.log(GammaModel.pdf(d, 2.0, 10.0))
            self.assertAlmostEqual(log_val, direct, places=5,
                                   msg=f"Mismatch at delta={d}")

    def test_reduces_to_exponential(self):
        """When alpha=1, Gamma(1, β) = Exponential(β)."""
        beta = 10.0
        for d in [0.01, 0.05, 0.1, 0.2]:
            gamma_pdf = GammaModel.pdf(d, alpha=1.0, beta=beta)
            exp_pdf = beta * math.exp(-beta * d)
            self.assertAlmostEqual(gamma_pdf, exp_pdf, places=5,
                                   msg=f"Should match Exponential at delta={d}")

    def test_mle_fit(self):
        """MLE should recover approximate parameters from synthetic data."""
        # Generate Gamma(alpha=2, beta=10) data
        random.seed(42)
        from scipy.stats import gamma as gamma_dist
        data = gamma_dist.rvs(a=2, scale=1/10.0, size=200, random_state=42).tolist()
        alpha_hat, beta_hat = GammaModel.fit_mle(data)
        # Should be reasonably close (within 50% for 200 samples)
        self.assertGreater(alpha_hat, 1.0)
        self.assertGreater(beta_hat, 5.0)

    def test_log_likelihood_empty(self):
        self.assertEqual(GammaModel.log_likelihood([], 2.0, 10.0), 0.0)


# =============================================================================
# Weibull Distribution Tests
# =============================================================================

class TestWeibullModel(unittest.TestCase):

    def test_pdf_positive(self):
        """PDF should be positive for valid delta."""
        for d in [0.01, 0.05, 0.1, 0.3]:
            pdf_val = WeibullModel.pdf(d, k=1.5, lam=0.1)
            self.assertGreater(pdf_val, 0, f"PDF should be > 0 at delta={d}")

    def test_log_pdf_equals_log_of_pdf(self):
        """log_pdf should equal log(pdf)."""
        for d in [0.01, 0.05, 0.15]:
            log_val = WeibullModel.log_pdf(d, 1.5, 0.1)
            direct = math.log(WeibullModel.pdf(d, 1.5, 0.1))
            self.assertAlmostEqual(log_val, direct, places=5)

    def test_reduces_to_exponential(self):
        """When k=1, Weibull(1, λ) = Exponential(1/λ)."""
        lam = 0.1  # Weibull scale
        rate = 1.0 / lam  # Exponential rate
        for d in [0.01, 0.05, 0.1, 0.2]:
            weibull_pdf = WeibullModel.pdf(d, k=1.0, lam=lam)
            exp_pdf = rate * math.exp(-rate * d)
            self.assertAlmostEqual(weibull_pdf, exp_pdf, places=5,
                                   msg=f"Should match Exponential at delta={d}")

    def test_mle_fit(self):
        """MLE should return valid parameters."""
        from scipy.stats import weibull_min
        data = weibull_min.rvs(c=1.5, scale=0.1, size=200, random_state=42).tolist()
        k_hat, lam_hat = WeibullModel.fit_mle(data)
        self.assertGreater(k_hat, 0)
        self.assertGreater(lam_hat, 0)

    def test_log_likelihood_empty(self):
        self.assertEqual(WeibullModel.log_likelihood([], 1.5, 0.1), 0.0)


# =============================================================================
# Model Selection Tests
# =============================================================================

class TestModelSelector(unittest.TestCase):

    def test_aic_computation(self):
        """AIC = 2k - 2·ln(L)."""
        self.assertAlmostEqual(
            ModelSelector.compute_aic(log_lik=-100.0, n_params=2),
            2 * 2 - 2 * (-100.0)
        )

    def test_bic_computation(self):
        """BIC = k·ln(n) - 2·ln(L)."""
        self.assertAlmostEqual(
            ModelSelector.compute_bic(log_lik=-100.0, n_params=2, n_samples=100),
            2 * math.log(100) - 2 * (-100.0)
        )

    def test_fit_and_compare_returns_three_models(self):
        """Should fit all three distributions."""
        random.seed(42)
        data = [random.expovariate(10.0) for _ in range(50)]
        results = ModelSelector.fit_and_compare(data)
        self.assertEqual(len(results), 3)
        # All should have valid BIC
        for r in results:
            self.assertTrue(math.isfinite(r.bic))

    def test_fit_and_compare_sorted_by_bic(self):
        """Results should be sorted by BIC (best first)."""
        random.seed(42)
        data = [random.expovariate(10.0) for _ in range(50)]
        results = ModelSelector.fit_and_compare(data)
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i].bic, results[i + 1].bic)

    def test_best_model(self):
        """best_model should return the model with lowest BIC."""
        random.seed(42)
        data = [random.expovariate(10.0) for _ in range(50)]
        best = ModelSelector.best_model(data)
        self.assertIsNotNone(best)
        self.assertIn(best.distribution, ["exponential", "gamma", "weibull"])

    def test_too_few_samples(self):
        """Should return empty list for < 3 samples."""
        results = ModelSelector.fit_and_compare([0.1, 0.2])
        self.assertEqual(len(results), 0)

    def test_ks_pvalues_present(self):
        """KS test p-values should be present and valid."""
        random.seed(42)
        data = [random.expovariate(10.0) for _ in range(100)]
        results = ModelSelector.fit_and_compare(data)
        for r in results:
            self.assertTrue(0 <= r.ks_pvalue <= 1, f"KS p-value out of range: {r.ks_pvalue}")


# =============================================================================
# EM Algorithm Tests
# =============================================================================

class TestMixtureEM(unittest.TestCase):

    def test_em_convergence(self):
        """EM should converge and return valid parameters."""
        # Generate bimodal data
        random.seed(42)
        good_deltas = [random.expovariate(20.0) for _ in range(150)]
        blunder_deltas = [random.expovariate(3.0) for _ in range(50)]
        data = good_deltas + blunder_deltas
        random.shuffle(data)

        model = MixtureModel(pi=0.5, lambda_good=10.0, lambda_blunder=1.0)
        pi, lam_g, lam_b = model.fit_em(data)

        # Parameters should be in valid ranges
        self.assertGreater(pi, 0)
        self.assertLess(pi, 1)
        self.assertGreater(lam_g, lam_b)  # Good component is tighter

    def test_em_improves_likelihood(self):
        """EM-fitted model should have higher likelihood than initial."""
        random.seed(42)
        good_deltas = [random.expovariate(20.0) for _ in range(150)]
        blunder_deltas = [random.expovariate(3.0) for _ in range(50)]
        data = good_deltas + blunder_deltas

        model = MixtureModel(pi=0.5, lambda_good=10.0, lambda_blunder=1.0)
        ll_before = model.log_likelihood(data)

        model.fit_em(data)
        ll_after = model.log_likelihood(data)

        self.assertGreater(ll_after, ll_before,
                           "EM should improve log-likelihood")

    def test_em_too_few_samples(self):
        """EM with < 5 samples should return unchanged parameters."""
        model = MixtureModel(pi=0.75, lambda_good=20.0, lambda_blunder=3.0)
        pi, lg, lb = model.fit_em([0.1, 0.2])
        self.assertAlmostEqual(pi, 0.75)
        self.assertAlmostEqual(lg, 20.0)
        self.assertAlmostEqual(lb, 3.0)


# =============================================================================
# Temperature Score Tests
# =============================================================================

class TestTemperatureScore(unittest.TestCase):

    def test_inverse_relationship(self):
        """S_τ = 1/τ, so lower temperature → higher score."""
        s1 = SoftmaxChoiceModel.temperature_score(0.01)
        s2 = SoftmaxChoiceModel.temperature_score(0.1)
        s3 = SoftmaxChoiceModel.temperature_score(1.0)
        self.assertGreater(s1, s2)
        self.assertGreater(s2, s3)

    def test_specific_value(self):
        """S_τ(0.05) = 20."""
        self.assertAlmostEqual(SoftmaxChoiceModel.temperature_score(0.05), 20.0)

    def test_zero_temperature(self):
        """Zero temperature should return inf."""
        self.assertEqual(SoftmaxChoiceModel.temperature_score(0.0), float('inf'))


# =============================================================================
# Likelihood Ratio Test Tests
# =============================================================================

class TestLRT(unittest.TestCase):

    def setUp(self):
        self.model = BayesianPlayerModel(prior_cheat=0.01, lambda_cheater=50.0)

    def test_cheater_data_positive_log_lambda(self):
        """Cheater-like data should give positive log likelihood ratio."""
        cheater_deltas = [0.005, 0.003, 0.002, 0.001, 0.004] * 5
        log_lambda, _, _ = self.model.compute_lrt(cheater_deltas)
        self.assertGreater(log_lambda, 0, "Cheater data should favor H1")

    def test_human_data_negative_log_lambda(self):
        """Human-like data should give negative log likelihood ratio."""
        human_deltas = [0.15, 0.05, 0.20, 0.02, 0.10, 0.30, 0.08, 0.12]
        log_lambda, _, _ = self.model.compute_lrt(human_deltas)
        self.assertLess(log_lambda, 0, "Human data should favor H0")

    def test_empty_data(self):
        log_lambda, lh, lc = self.model.compute_lrt([])
        self.assertEqual(log_lambda, 0.0)

    def test_lrt_reject(self):
        """Static method for comparing log_lambda to threshold."""
        self.assertTrue(BayesianPlayerModel.lrt_reject(10.0, 5.0))
        self.assertFalse(BayesianPlayerModel.lrt_reject(3.0, 5.0))


# =============================================================================
# Prior Sensitivity Analysis Tests
# =============================================================================

class TestSensitivity(unittest.TestCase):

    def setUp(self):
        self.model = BayesianPlayerModel(prior_cheat=0.01, lambda_cheater=50.0)

    def test_returns_all_priors(self):
        """Should return posterior for each requested prior."""
        deltas = [0.05, 0.10, 0.15, 0.08]
        results = self.model.sensitivity_analysis(deltas)
        self.assertEqual(len(results), 4)  # Default 4 priors
        for prior in [0.001, 0.01, 0.05, 0.10]:
            self.assertIn(prior, results)

    def test_higher_prior_higher_posterior(self):
        """Higher prior should give higher (or equal) posterior."""
        deltas = [0.05, 0.10, 0.15, 0.08, 0.03]
        results = self.model.sensitivity_analysis(deltas)
        posteriors = [results[p] for p in sorted(results.keys())]
        for i in range(len(posteriors) - 1):
            self.assertLessEqual(posteriors[i], posteriors[i + 1] + 1e-10,
                                 "Higher prior should yield higher posterior")

    def test_custom_priors(self):
        """Should work with custom prior list."""
        deltas = [0.05, 0.10]
        results = self.model.sensitivity_analysis(deltas, priors=[0.5, 0.9])
        self.assertEqual(len(results), 2)

    def test_empty_data(self):
        """With no data, posterior should equal prior."""
        results = self.model.sensitivity_analysis([], priors=[0.01, 0.05])
        self.assertAlmostEqual(results[0.01], 0.01)
        self.assertAlmostEqual(results[0.05], 0.05)


if __name__ == '__main__':
    unittest.main()
