"""
Tests for V2/V3 math modules.
Replaces old V1 ProbabilityModel tests.
"""

import sys
import os
import unittest
import math

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.core.delta_model import DeltaModel, MoveAnalysis
from src.core.complexity import calculate_complexity, calculate_accuracy
from src.core.player_model import BayesianPlayerModel


class TestDeltaModel(unittest.TestCase):

    def setUp(self):
        self.model = DeltaModel(near_optimal_threshold=0.02)

    def test_delta_non_negative(self):
        """Delta must always be >= 0."""
        self.assertEqual(self.model.calculate_delta(0.8, 0.8), 0.0)
        self.assertEqual(self.model.calculate_delta(0.8, 0.9), 0.0)  # Clamped
        self.assertAlmostEqual(self.model.calculate_delta(0.8, 0.5), 0.3)

    def test_lambda_mle_basic(self):
        """λ_MLE = 1 / mean(Δ)."""
        deltas = [0.1, 0.1, 0.1]  # mean = 0.1
        lam = self.model.estimate_lambda_mle(deltas)
        self.assertAlmostEqual(lam, 10.0, places=1)

    def test_lambda_mle_zero_deltas(self):
        """All zeros should use epsilon floor, not crash."""
        deltas = [0.0, 0.0, 0.0]
        lam = self.model.estimate_lambda_mle(deltas)
        # Should be 1/epsilon = 1000
        self.assertAlmostEqual(lam, 1000.0, places=0)

    def test_lambda_mle_empty(self):
        self.assertEqual(self.model.estimate_lambda_mle([]), 0.0)

    def test_analyze_game(self):
        moves = [
            MoveAnalysis(0, "h8", 0.8, 0.75, 0.05),
            MoveAnalysis(1, "i9", 0.7, 0.7, 0.0),
            MoveAnalysis(2, "j10", 0.6, 0.4, 0.2),
        ]
        result = self.model.analyze_game(moves)
        self.assertAlmostEqual(result.mean_delta, (0.05 + 0.0 + 0.2) / 3, places=5)
        self.assertEqual(len(result.moves), 3)
        # Near-optimal: only delta=0.0 < 0.02
        self.assertAlmostEqual(result.near_optimal_ratio, 1 / 3, places=3)


class TestComplexity(unittest.TestCase):

    def test_trivial_won_position(self):
        """Trivially won: all candidates > 95%, low variance → Low complexity."""
        candidates = [0.98, 0.97, 0.96, 0.95, 0.95]
        result = calculate_complexity(candidates, prev_winrate=0.95)
        # Low complexity (candidate spread is small but above trivial threshold)
        self.assertLessEqual(result.complexity, 0.25)

    def test_complex_balanced(self):
        """Balanced position with big spread: should be complex."""
        candidates = [0.7, 0.5, 0.3, 0.2, 0.1]
        result = calculate_complexity(candidates, prev_winrate=0.5)
        self.assertGreater(result.complexity, 0.4)

    def test_empty_candidates(self):
        result = calculate_complexity([], prev_winrate=0.5)
        self.assertAlmostEqual(result.complexity, 0.5)


class TestAccuracy(unittest.TestCase):

    def test_near_optimal(self):
        """Delta <= 0.02 should give accuracy 1.0."""
        self.assertEqual(calculate_accuracy(0.0, 0.5), 1.0)
        self.assertEqual(calculate_accuracy(0.02, 0.5), 1.0)

    def test_blunder(self):
        """Large delta should give low accuracy."""
        acc = calculate_accuracy(0.3, 0.5)
        self.assertLess(acc, 0.3)

    def test_complexity_makes_lenient(self):
        """Higher complexity should give higher accuracy for same delta."""
        acc_easy = calculate_accuracy(0.1, complexity=0.2)
        acc_hard = calculate_accuracy(0.1, complexity=0.8)
        self.assertGreater(acc_hard, acc_easy)


class TestBayesianPlayerModel(unittest.TestCase):

    def setUp(self):
        self.model = BayesianPlayerModel(
            prior_cheat=0.01,
            lambda_cheater=50.0,
            threshold_suspicious=0.3,
            threshold_cheater=0.7,
        )

    def test_no_data_returns_prior(self):
        p_cheat, p_human, _, _ = self.model.compute_posterior([])
        self.assertAlmostEqual(p_cheat, 0.01)

    def test_cheater_data_increases_p_cheat(self):
        """Very small deltas should increase P(Cheat)."""
        cheater_deltas = [0.005, 0.003, 0.002, 0.001, 0.004] * 5
        p_cheat, _, _, _ = self.model.compute_posterior(cheater_deltas)
        self.assertGreater(p_cheat, 0.5)

    def test_human_data_keeps_low_p_cheat(self):
        """Large varied deltas should keep P(Cheat) low."""
        human_deltas = [0.15, 0.05, 0.20, 0.02, 0.10, 0.30, 0.08, 0.12]
        p_cheat, _, _, _ = self.model.compute_posterior(human_deltas)
        self.assertLess(p_cheat, 0.3)

    def test_online_update_monotonicity(self):
        """Small deltas should consistently increase P(Cheat)."""
        posterior = 0.01
        for _ in range(10):
            posterior = self.model.update_online(posterior, 0.005, complexity=0.8)
        self.assertGreater(posterior, 0.01)  # Should have increased

    def test_online_update_zero_complexity(self):
        """Zero complexity should not change posterior."""
        posterior = 0.5
        new = self.model.update_online(posterior, 0.001, complexity=0.0)
        self.assertAlmostEqual(new, posterior)

    def test_tempered_vs_full_update(self):
        """Low complexity should produce smaller update than high complexity."""
        posterior = 0.1
        # Same delta, different complexity
        low_cx = self.model.update_online(posterior, 0.005, complexity=0.2)
        high_cx = self.model.update_online(posterior, 0.005, complexity=1.0)
        # Both should move in same direction, high_cx should move more
        self.assertGreater(abs(high_cx - posterior), abs(low_cx - posterior))


if __name__ == '__main__':
    unittest.main()
