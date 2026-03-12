"""
Tests for the Softmax Choice Model (V3).
"""

import sys
import os
import unittest
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.core.delta_model import SoftmaxChoiceModel


class TestSoftmaxProbs(unittest.TestCase):

    def test_sums_to_one(self):
        """Softmax probabilities must sum to 1."""
        winrates = [0.8, 0.6, 0.4, 0.2]
        for temperature in [0.01, 0.05, 0.1, 0.5, 1.0]:
            probs = SoftmaxChoiceModel.softmax_probs(winrates, temperature)
            self.assertAlmostEqual(sum(probs), 1.0, places=10,
                                   msg=f"Failed at τ={temperature}")

    def test_low_temperature_picks_best(self):
        """Very low τ should concentrate almost all probability on best move."""
        winrates = [0.8, 0.6, 0.4, 0.2]
        probs = SoftmaxChoiceModel.softmax_probs(winrates, temperature=0.005)
        self.assertGreater(probs[0], 0.99)

    def test_high_temperature_more_uniform(self):
        """Very high τ should make probabilities more uniform."""
        winrates = [0.8, 0.6, 0.4, 0.2]
        probs = SoftmaxChoiceModel.softmax_probs(winrates, temperature=100.0)
        # Should be close to uniform (0.25 each)
        for p in probs:
            self.assertAlmostEqual(p, 0.25, places=1)

    def test_empty_winrates(self):
        probs = SoftmaxChoiceModel.softmax_probs([], 0.1)
        self.assertEqual(probs, [1.0])

    def test_invalid_temperature(self):
        probs = SoftmaxChoiceModel.softmax_probs([0.5, 0.3], temperature=0)
        self.assertAlmostEqual(sum(probs), 1.0)


class TestLogProbOfChoice(unittest.TestCase):

    def test_always_negative_or_zero(self):
        """Log probability must be <= 0."""
        winrates = [0.8, 0.6, 0.4]
        for idx in range(3):
            lp = SoftmaxChoiceModel.log_prob_of_choice(winrates, idx, 0.1)
            self.assertLessEqual(lp, 0.0001)  # Small tolerance

    def test_best_move_highest_log_prob(self):
        """Picking the best move should give highest log-prob."""
        winrates = [0.8, 0.6, 0.4]
        lp_best = SoftmaxChoiceModel.log_prob_of_choice(winrates, 0, 0.1)
        lp_worst = SoftmaxChoiceModel.log_prob_of_choice(winrates, 2, 0.1)
        self.assertGreater(lp_best, lp_worst)

    def test_consistent_with_softmax(self):
        """Log-prob should equal log of softmax probability."""
        winrates = [0.7, 0.5, 0.3]
        temperature = 0.1
        probs = SoftmaxChoiceModel.softmax_probs(winrates, temperature)
        for idx in range(3):
            lp = SoftmaxChoiceModel.log_prob_of_choice(winrates, idx, temperature)
            expected = math.log(probs[idx])
            self.assertAlmostEqual(lp, expected, places=5)

    def test_invalid_index(self):
        self.assertEqual(
            SoftmaxChoiceModel.log_prob_of_choice([0.5], 5, 0.1), 0.0
        )
        self.assertEqual(
            SoftmaxChoiceModel.log_prob_of_choice([0.5], -1, 0.1), 0.0
        )


class TestTemperatureEstimation(unittest.TestCase):

    def test_engine_like_play(self):
        """Always picking best move should give low τ."""
        # Simulate cheater: always picks index 0 (best)
        observations = [
            ([0.9, 0.6, 0.4, 0.2], 0),
            ([0.85, 0.5, 0.3, 0.1], 0),
            ([0.7, 0.6, 0.5, 0.4], 0),
            ([0.95, 0.8, 0.6, 0.3], 0),
            ([0.8, 0.4, 0.2, 0.1], 0),
        ]
        tau = SoftmaxChoiceModel.estimate_temperature(observations)
        self.assertLess(tau, 0.15, f"Engine-like play should have low τ, got {tau}")

    def test_random_play(self):
        """Picking non-best moves frequently should give higher τ."""
        # Simulate human: picks various moves, not always best
        observations = [
            ([0.8, 0.7, 0.6, 0.5], 1),  # Picked 2nd best
            ([0.9, 0.8, 0.7, 0.6], 2),  # Picked 3rd
            ([0.7, 0.6, 0.5, 0.4], 0),  # Picked best
            ([0.85, 0.8, 0.75, 0.7], 3), # Picked worst
            ([0.6, 0.5, 0.4, 0.3], 1),  # Picked 2nd
        ]
        tau = SoftmaxChoiceModel.estimate_temperature(observations)
        # Should be higher than engine-like play
        self.assertGreater(tau, 0.05)

    def test_empty_observations(self):
        tau = SoftmaxChoiceModel.estimate_temperature([])
        self.assertEqual(tau, 0.1)  # Default


if __name__ == '__main__':
    unittest.main()
