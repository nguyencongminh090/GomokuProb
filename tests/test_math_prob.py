"""
Unit tests for the Probability Model.
Verifies correct behavior of Softmax and Weighted Accuracy calculations.
"""

import unittest
from src.core.math_prob import ProbabilityModel, MoveEvaluation

class TestProbabilityModel(unittest.TestCase):
    
    def setUp(self):
        self.model = ProbabilityModel(sensitivity_k=10.0)
        
    def test_perfect_move(self):
        """Test that a perfect move yields 1.0 accuracy."""
        acc = self.model.calculate_accuracy(0.8, 0.8)
        self.assertAlmostEqual(acc, 1.0)
        
    def test_minor_mistake(self):
        """Test a small mistake (5% WR loss)."""
        # exp(-10 * 0.05) = exp(-0.5) ≈ 0.606
        acc = self.model.calculate_accuracy(0.75, 0.8)
        self.assertAlmostEqual(acc, 0.6065306597, places=5)
        
    def test_blunder(self):
        """Test a huge blunder (50% WR loss)."""
        # exp(-10 * 0.5) = exp(-5) ≈ 0.0067
        acc = self.model.calculate_accuracy(0.3, 0.8)
        self.assertAlmostEqual(acc, 0.006737947, places=5)

    def test_criticality_calculation(self):
        """Test calculation of criticality and weighted score."""
        played = MoveEvaluation("h8", 0.9)
        candidates = [
            MoveEvaluation("h8", 0.9, is_best=True),
            MoveEvaluation("h9", 0.4), # Huge drop -> High critical
            MoveEvaluation("k9", 0.1)
        ]
        
        result = self.model.analyze_move(played, candidates)
        
        self.assertAlmostEqual(result.best_winrate, 0.9)
        self.assertAlmostEqual(result.criticality, 0.5) # 0.9 - 0.4
        self.assertAlmostEqual(result.accuracy_score, 1.0)
        self.assertAlmostEqual(result.weighted_score, 0.5) # 1.0 * 0.5

if __name__ == '__main__':
    unittest.main()
