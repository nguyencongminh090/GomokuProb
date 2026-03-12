"""
Tests for the Mixture Model (V3).
"""

import sys
import os
import unittest
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.core.mixture_model import (
    MixtureModel,
    log_likelihood_exponential,
    log_likelihood_exponential_tempered,
)


class TestMixtureModel(unittest.TestCase):

    def setUp(self):
        self.model = MixtureModel(
            pi=0.75,
            lambda_good=20.0,
            lambda_blunder=3.0
        )

    def test_pdf_positive(self):
        """PDF should always be positive for positive delta."""
        for d in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]:
            pdf_val = self.model.pdf(d)
            self.assertGreater(pdf_val, 0, f"PDF should be > 0 at delta={d}")

    def test_pdf_decreasing(self):
        """PDF should generally decrease (not strictly, but at extremes)."""
        self.assertGreater(self.model.pdf(0.01), self.model.pdf(0.5))

    def test_log_pdf_finite(self):
        """Log-PDF should be finite for reasonable values."""
        for d in [0.001, 0.01, 0.1, 0.5]:
            val = self.model.log_pdf(d)
            self.assertTrue(math.isfinite(val), f"log_pdf({d}) = {val} is not finite")

    def test_log_pdf_equals_log_of_pdf(self):
        """log_pdf should equal log(pdf) within tolerance."""
        for d in [0.01, 0.05, 0.15, 0.3]:
            log_val = self.model.log_pdf(d)
            direct = math.log(self.model.pdf(d))
            self.assertAlmostEqual(log_val, direct, places=3,
                                   msg=f"Mismatch at delta={d}")

    def test_mixture_better_for_bimodal(self):
        """
        Mixture should fit bimodal data (small deltas + occasional blunders)
        better than a single Exponential with λ=11.
        """
        # Bimodal: mostly good, some blunders
        deltas = [0.02, 0.01, 0.03, 0.015, 0.02, 0.01, 0.25, 0.02, 0.015, 0.3]

        mixture_ll = self.model.log_likelihood(deltas)

        # Single exponential with typical human lambda
        single_ll = sum(log_likelihood_exponential(d, 11.0) for d in deltas)

        self.assertGreater(mixture_ll, single_ll,
                           "Mixture should have higher likelihood for bimodal data")

    def test_tempered_scales_linearly(self):
        """Tempered log-likelihood should equal alpha * log_pdf."""
        delta = 0.05
        alpha = 0.3
        tempered = self.model.log_likelihood_tempered(delta, alpha)
        expected = alpha * self.model.log_pdf(delta)
        self.assertAlmostEqual(tempered, expected, places=10)

    def test_tempered_zero_alpha(self):
        """Alpha=0 should give zero tempered likelihood."""
        self.assertAlmostEqual(
            self.model.log_likelihood_tempered(0.1, alpha=0.0), 0.0
        )

    def test_log_likelihood_empty(self):
        self.assertEqual(self.model.log_likelihood([]), 0.0)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            MixtureModel(pi=1.5)
        with self.assertRaises(ValueError):
            MixtureModel(pi=0.5, lambda_good=-1.0)


class TestExponentialHelpers(unittest.TestCase):

    def test_exponential_log_likelihood(self):
        """log(λ·exp(-λΔ)) = log(λ) - λΔ."""
        lam = 50.0
        delta = 0.02
        expected = math.log(lam) - lam * delta
        self.assertAlmostEqual(
            log_likelihood_exponential(delta, lam), expected, places=5
        )

    def test_tempered_exponential(self):
        lam = 50.0
        delta = 0.02
        alpha = 0.5
        expected = alpha * (math.log(lam) - lam * delta)
        self.assertAlmostEqual(
            log_likelihood_exponential_tempered(delta, lam, alpha),
            expected, places=5
        )


if __name__ == '__main__':
    unittest.main()
