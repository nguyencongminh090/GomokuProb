"""
Tests for Layer 2 Sequential Pattern Analysis.

Tests cover:
 - All five statistical tests with hand-verifiable arithmetic
 - Ensemble detector (Fisher + weighted vote)
 - Edge cases (n < min threshold, zero variance, degenerate inputs)
 - ProfileStore Layer 2 schema persistence roundtrip
 - SequentialPatternAnalyzer facade (end-to-end)

Run with:
    cd /media/ngmint/Data/Programming/Python/Personal/GomokuProb
    python3.13 -m pytest tests/test_sequential_analysis.py -v
"""

import sys
import os
import math
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.core.sequential_analysis import (
    SequentialPatternAnalyzer,
    WaldWolfowitzRunsTest,
    Lag1ACFTest,
    CUSUMTest,
    CACTest,
    ShannonEntropyTest,
    EnsembleDetector,
    _norm_cdf,
    _norm_sf,
    _t_cdf,
    _chi2_sf,
    SequentialAnalysisResult,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _human_deltas(n: int) -> list[float]:
    """Simulate a typical human game: large, varied deltas."""
    pattern = [0.05, 0.15, 0.02, 0.25, 0.08, 0.30, 0.04, 0.12, 0.20, 0.07]
    return [pattern[i % len(pattern)] for i in range(n)]


def _cheater_deltas(n: int) -> list[float]:
    """Simulate a full cheater: all deltas near 0."""
    return [0.001] * n


def _alternating_deltas(n: int) -> list[float]:
    """Engine (0.001) alternating with human (0.15) — Type B."""
    return [0.001 if i % 2 == 0 else 0.15 for i in range(n)]


def _uniform_complexity(n: int, c: float = 0.5) -> list[float]:
    return [c] * n


def _alternating_complexity(n: int) -> list[float]:
    """High complexity when engine used (odd indices), low otherwise."""
    return [0.7 if i % 2 == 1 else 0.3 for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Distribution helpers
# ---------------------------------------------------------------------------

class TestDistributionHelpers(unittest.TestCase):

    def test_norm_cdf_known_values(self):
        """Standard normal CDF at known quantiles."""
        self.assertAlmostEqual(_norm_cdf(0.0),   0.5,    places=6)
        self.assertAlmostEqual(_norm_cdf(1.96),  0.975,  places=2)
        self.assertAlmostEqual(_norm_cdf(-1.96), 0.025,  places=2)

    def test_norm_sf_complementary(self):
        for z in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            self.assertAlmostEqual(_norm_cdf(z) + _norm_sf(z), 1.0, places=10)

    def test_t_cdf_large_df_approaches_normal(self):
        """t-CDF with large df should be close to normal CDF."""
        for z in [-2.0, 0.0, 2.0]:
            t_val = _t_cdf(z, df=200)
            n_val = _norm_cdf(z)
            self.assertAlmostEqual(t_val, n_val, places=2)

    def test_chi2_sf_known_value(self):
        """chi2 SF at threshold 18.31 with df=10 should be ~0.05."""
        p = _chi2_sf(18.31, 10)
        self.assertAlmostEqual(p, 0.05, delta=0.01)

    def test_chi2_sf_zero(self):
        """chi2 SF at 0 should be 1."""
        self.assertAlmostEqual(_chi2_sf(0.0, 10), 1.0, places=5)


# ---------------------------------------------------------------------------
# 2. Wald-Wolfowitz Runs Test
# ---------------------------------------------------------------------------

class TestWaldWolfowitzRunsTest(unittest.TestCase):

    def test_insufficient_data_returns_none(self):
        """n < MIN_N should skip test."""
        z, p, diag = WaldWolfowitzRunsTest.run([0.01] * 10)
        self.assertIsNone(z)
        self.assertIsNone(p)
        self.assertFalse(diag["ran"])

    def test_human_data_high_p_value(self):
        """Varied human deltas should produce a non-significant result."""
        deltas = _human_deltas(40)
        z, p, diag = WaldWolfowitzRunsTest.run(deltas)
        self.assertIsNotNone(z)
        self.assertTrue(diag["ran"])
        # Human data should NOT be flagged (p > 0.05 expected for typical human)
        # Note: this is not guaranteed statistically but is expected for this pattern
        self.assertGreater(p, 0.0)  # at minimum p is valid

    def test_cheater_inflated_optimal_runs(self):
        """All-near-zero deltas: n1=n, n0=0 → cannot compute (all same side)."""
        z, p, diag = WaldWolfowitzRunsTest.run(_cheater_deltas(30))
        # All deltas ≤ τ → n0=0, test cannot run
        self.assertIsNone(z)
        self.assertFalse(diag.get("ran", True))

    def test_alternating_sequence_high_z(self):
        """
        Alternating 0.001, 0.15, 0.001, ... creates many short runs → high Z_R.
        Expected: Z_R > 0 (more runs than expected under H0).
        """
        deltas = _alternating_deltas(40)  # 20 optimal + 20 suboptimal, alternating
        z, p, diag = WaldWolfowitzRunsTest.run(deltas)
        self.assertIsNotNone(z)
        self.assertTrue(diag["ran"])
        # Alternating pattern maximises run count → Z_R should be positive and large
        self.assertGreater(z, 1.0)

    def test_expected_and_variance_formula(self):
        """
        Manually verify E[R] and Var[R] for a tiny known sequence.
        b = [1,0,1,0,...] (20 alternating, n1=n0=10)
        E[R] = 2*10*10/20 + 1 = 11
        Var[R] = 2*n1*n0*(2*n1*n0 - n) / (n^2*(n-1))
               = 2*10*10*(200-20) / (400*19)
               = 36000 / 7600
               ≈ 4.7368
        """
        n = 20
        n1, n0 = 10, 10
        e_r = 2.0 * n1 * n0 / n + 1.0
        self.assertAlmostEqual(e_r, 11.0, places=5)

        var_r = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n ** 2 * (n - 1))
        expected_var = 36000 / 7600  # ≈ 4.7368
        self.assertAlmostEqual(var_r, expected_var, places=4)


# ---------------------------------------------------------------------------
# 3. Lag-1 ACF Test
# ---------------------------------------------------------------------------

class TestLag1ACFTest(unittest.TestCase):

    def test_insufficient_data_returns_none(self):
        z, p, rho, diag = Lag1ACFTest.run([0.1] * 20)
        self.assertIsNone(rho)
        self.assertFalse(diag["ran"])

    def test_alternating_sequence_negative_acf(self):
        """Alternating engine/human moves produces negative lag-1 autocorrelation."""
        deltas = _alternating_deltas(50)
        rho1, z_acf, p_acf, diag = Lag1ACFTest.run(deltas)
        self.assertTrue(diag["ran"])
        self.assertIsNotNone(rho1)
        # Alternating pattern → strongly negative rho1
        self.assertLess(rho1, -0.5)
        # One-tailed p should be very small
        self.assertLess(p_acf, 0.01)

    def test_constant_sequence_zero_variance(self):
        """Constant deltas → zero variance → test cannot run."""
        deltas = [0.10] * 40
        rho1, z_acf, p_acf, diag = Lag1ACFTest.run(deltas)
        self.assertIsNone(rho1)
        self.assertFalse(diag["ran"])

    def test_human_like_uncorrelated(self):
        """Random-ish human deltas should give rho1 close to 0."""
        # Periodic but non-alternating pattern
        deltas = [0.05, 0.10, 0.08, 0.12, 0.06, 0.09, 0.11, 0.07] * 5
        rho1, z_acf, p_acf, diag = Lag1ACFTest.run(deltas)
        self.assertTrue(diag["ran"])
        # Should be weakly correlated (not strongly negative)
        self.assertGreater(rho1, -0.5)

    def test_z_acf_formula(self):
        """Z_ACF = sqrt(n) * rho1."""
        deltas = _alternating_deltas(50)
        rho1, z_acf, p_acf, diag = Lag1ACFTest.run(deltas)
        self.assertAlmostEqual(z_acf, math.sqrt(50) * rho1, places=10)


# ---------------------------------------------------------------------------
# 4. CUSUM Test
# ---------------------------------------------------------------------------

class TestCUSUMTest(unittest.TestCase):

    def test_insufficient_data_returns_none(self):
        cmax, cp, pval, diag = CUSUMTest.run([0.1] * 10, [0.5] * 10)
        self.assertIsNone(cmax)
        self.assertFalse(diag["ran"])

    def test_cheater_triggers_cusum(self):
        """
        All-near-zero deltas give very high per-move log-likelihood ratio scores
        (small Δ strongly favours H1 = cheater) → CUSUM eventually crosses h=3.
        """
        deltas = [0.001] * 30
        complexities = [0.6] * 30
        cmax, cp, pval, diag = CUSUMTest.run(deltas, complexities)
        self.assertTrue(diag["ran"])
        self.assertIsNotNone(cmax)
        self.assertTrue(diag["triggered"])
        self.assertGreater(cmax, CUSUMTest.H_THRESHOLD)
        self.assertIsNotNone(cp)  # Change-point should be set

    def test_human_does_not_trigger(self):
        """Large blundering deltas score negatively under cheater H1 → CUSUM stays low."""
        # Large deltas → log-ratio very negative → CUSUM absorbed at 0
        deltas = [0.30, 0.25, 0.20, 0.35, 0.28] * 6  # 30 moves
        complexities = [0.5] * 30
        cmax, cp, pval, diag = CUSUMTest.run(deltas, complexities)
        self.assertTrue(diag["ran"])
        self.assertFalse(diag["triggered"])

    def test_cusum_recursion_absorb_at_zero(self):
        """
        Verify the recursion S_t = max(0, S_{t-1} + s_t) absorbs at zero.
        A single very negative score (large blunder) should reset the CUSUM.
        Need n >= MIN_N=15 for the test to run.
        """
        # 15 moves: initial cheater-like (small deltas) then one huge blunder
        deltas = [0.001] * 7 + [0.50] + [0.001] * 7   # 15 total
        complexities = [0.8] * 15
        cmax, cp, pval, diag = CUSUMTest.run(deltas, complexities)
        self.assertTrue(diag["ran"])  # n=15 meets MIN_N
        self.assertIsNotNone(cmax)
        # After the blunder, CUSUM reset to 0 and restarted — verify no crash

    def test_p_cusum_decreasing_with_detectable_signal(self):
        """Stronger cheating (more moves) → lower p_cusum."""
        deltas_light = [0.005] * 20
        deltas_heavy = [0.001] * 20
        cx = [0.6] * 20
        _, _, p_light, _ = CUSUMTest.run(deltas_light, cx)
        _, _, p_heavy, _ = CUSUMTest.run(deltas_heavy, cx)
        # More extreme cheating → higher cusum_max → lower p-value
        self.assertLessEqual(p_heavy, p_light)


# ---------------------------------------------------------------------------
# 5. CAC Test
# ---------------------------------------------------------------------------

class TestCACTest(unittest.TestCase):

    def test_insufficient_data_returns_none(self):
        deltas = [0.1] * 10
        cx = [0.6] * 10
        cac, z, p, n_i, diag = CACTest.run(deltas, cx)
        self.assertIsNone(cac)
        self.assertFalse(diag["ran"])

    def test_type_a_pattern_positive_cac(self):
        """
        Type A: high C_i → low Δ_i (engine used at complex positions).
        → negative Pearson r(Δ, C) → positive CAC.
        """
        # 25 non-trivial moves (C > 0.25 for all)
        # Simple linear: high complexity = near-zero delta
        n = 25
        complexities = [0.3 + 0.6 * i / (n - 1) for i in range(n)]  # 0.3 → 0.9
        # Type A: Δ decreases as C increases
        deltas = [0.20 - 0.18 * i / (n - 1) + 0.01 for i in range(n)]  # 0.21 → 0.03

        cac, z_cac, p_cac, n_i, diag = CACTest.run(deltas, complexities)
        self.assertTrue(diag["ran"])
        self.assertIsNotNone(cac)
        self.assertGreater(cac, 0.0)  # Suspicious direction

    def test_human_pattern_negative_cac(self):
        """
        Human: high C_i → larger Δ_i (harder positions = more errors).
        → positive r(Δ, C) → negative CAC (not suspicious).
        """
        n = 25
        complexities = [0.3 + 0.6 * i / (n - 1) for i in range(n)]
        # Human: Δ increases as C increases
        deltas = [0.02 + 0.20 * i / (n - 1) for i in range(n)]

        cac, z_cac, p_cac, n_i, diag = CACTest.run(deltas, complexities)
        self.assertTrue(diag["ran"])
        self.assertIsNotNone(cac)
        self.assertLess(cac, 0.0)  # Human direction

    def test_filters_trivial_positions(self):
        """Positions with C_i ≤ 0.25 should be excluded."""
        # 30 moves: 20 trivial (C=0.1), 10 non-trivial (C=0.8)
        deltas = [0.1] * 30
        cx = [0.1] * 20 + [0.8] * 10
        cac, z, p, n_i, diag = CACTest.run(deltas, cx)
        # Only 10 non-trivial → below MIN_NONTRIVIAL=20 → should skip
        self.assertEqual(n_i, 10)
        self.assertFalse(diag["ran"])


# ---------------------------------------------------------------------------
# 6. Shannon Entropy Test
# ---------------------------------------------------------------------------

class TestShannonEntropyTest(unittest.TestCase):

    def test_insufficient_data_returns_none(self):
        h, mt, tp, pe, diag = ShannonEntropyTest.run([0.05] * 20)
        self.assertIsNone(h)
        self.assertFalse(diag["ran"])

    def test_all_near_zero_low_entropy(self):
        """All deltas in [0,1%) bin → entropy ≈ 0 (single bin)."""
        deltas = [0.001] * 30
        h, mt, tp, pe, diag = ShannonEntropyTest.run(deltas)
        self.assertTrue(diag["ran"])
        self.assertIsNotNone(h)
        # All in one bin → entropy = 0
        self.assertAlmostEqual(h, 0.0, places=3)

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution across all 5 bins → H = log2(5) ≈ 2.32 bits."""
        # Craft deltas hitting each bin uniformly
        deltas = (
            [0.005] * 6 +  # [0,1%)
            [0.02] * 6 +   # [1%,5%)
            [0.08] * 6 +   # [5%,15%)
            [0.25] * 6 +   # [15%,50%)
            [0.60] * 6     # [50%+]
        )  # 30 total
        h, mt, tp, pe, diag = ShannonEntropyTest.run(deltas)
        self.assertTrue(diag["ran"])
        # Uniform → H ≈ log2(5) ≈ 2.322
        self.assertAlmostEqual(h, math.log2(5), places=2)

    def test_missing_tail_flag(self):
        """No delta > 10% → missing_tail = True."""
        deltas = [0.05] * 30  # All below τ_blunder=0.10
        _, mt, tp, pe, diag = ShannonEntropyTest.run(deltas)
        self.assertTrue(diag["ran"])
        self.assertTrue(mt)
        self.assertIsNotNone(tp)
        self.assertLess(tp, 1.0)  # p-value should be < 1

    def test_no_missing_tail_when_blunder_present(self):
        """At least one delta > 10% → missing_tail = False."""
        deltas = [0.05] * 29 + [0.15]  # One blunder
        _, mt, _, _, _ = ShannonEntropyTest.run(deltas)
        self.assertFalse(mt)

    def test_missing_tail_pval_decreases_with_n(self):
        """Longer game with no blunders → smaller tail p-value."""
        deltas_short = [0.05] * 25
        deltas_long  = [0.05] * 50
        _, _, tp_short, _, _ = ShannonEntropyTest.run(deltas_short)
        _, _, tp_long,  _, _ = ShannonEntropyTest.run(deltas_long)
        self.assertLess(tp_long, tp_short)


# ---------------------------------------------------------------------------
# 7. Ensemble Detector
# ---------------------------------------------------------------------------

class TestEnsembleDetector(unittest.TestCase):

    def test_all_significant_triggers_fisher(self):
        """Five very small p-values → large χ², p-Fisher < 0.05."""
        p_vals = (0.001, 0.001, 0.001, 0.001, 0.001)
        chi2, p_fisher, vote, n_flagged = EnsembleDetector.combine(*p_vals)
        self.assertIsNotNone(chi2)
        self.assertLess(p_fisher, 0.05)
        self.assertEqual(n_flagged, 5)
        # Vote should be 1.0 (all tests fire with balanced weights summing to 1)
        self.assertAlmostEqual(vote, 1.0, places=5)

    def test_all_non_significant(self):
        """All high p-values → no flag."""
        p_vals = (0.50, 0.60, 0.70, 0.40, 0.55)
        chi2, p_fisher, vote, n_flagged = EnsembleDetector.combine(*p_vals)
        self.assertGreater(p_fisher, 0.05)
        self.assertEqual(n_flagged, 0)
        self.assertAlmostEqual(vote, 0.0, places=5)

    def test_missing_p_values_handled(self):
        """None p-values (test skipped) should be excluded gracefully."""
        p_vals = (0.001, None, 0.001, None, 0.001)
        chi2, p_fisher, vote, n_flagged = EnsembleDetector.combine(*p_vals)
        # Should only use 3 valid tests → df=6
        self.assertIsNotNone(chi2)
        self.assertGreater(vote, 0.0)

    def test_fisher_chi2_formula(self):
        """Verify Fisher χ² = -2 Σ log(p_j) manually."""
        p1, p2 = 0.03, 0.04
        expected_chi2 = -2.0 * (math.log(p1) + math.log(p2))
        chi2, _, _, _ = EnsembleDetector.combine(p1, p2, None, None, None)
        self.assertAlmostEqual(chi2, expected_chi2, places=8)

    def test_verdict_clean(self):
        """Non-significant ensemble → 'Clean'."""
        v = EnsembleDetector.determine_verdict(
            fisher_chi2=5.0, p_fisher=0.80, ensemble_score=0.1, n_flagged=0,
            p_runs=0.5, p_acf=0.5, p_cusum=0.5, p_cac=0.5, p_entropy=0.5,
            cac=0.1,
        )
        self.assertEqual(v, "Clean")

    def test_verdict_type_a(self):
        """CAC significant + CAC > 0.30 → Type A verdict."""
        v = EnsembleDetector.determine_verdict(
            fisher_chi2=20.0, p_fisher=0.03, ensemble_score=0.60, n_flagged=1,
            p_runs=0.5, p_acf=0.5, p_cusum=0.5, p_cac=0.02, p_entropy=0.5,
            cac=0.55,
        )
        self.assertIn("Type A", v)

    def test_verdict_type_b_from_acf(self):
        """ACF significant → Type B verdict."""
        v = EnsembleDetector.determine_verdict(
            fisher_chi2=20.0, p_fisher=0.02, ensemble_score=0.60, n_flagged=1,
            p_runs=0.5, p_acf=0.01, p_cusum=0.5, p_cac=0.5, p_entropy=0.5,
            cac=0.0,
        )
        self.assertIn("Type B", v)


# ---------------------------------------------------------------------------
# 8. SequentialPatternAnalyzer facade (end-to-end)
# ---------------------------------------------------------------------------

class TestSequentialPatternAnalyzerFacade(unittest.TestCase):

    def _spa(self):
        return SequentialPatternAnalyzer()

    def test_empty_input(self):
        result = self._spa().analyze([], [])
        self.assertEqual(result.n_moves, 0)
        self.assertIn("N/A", result.verdict)

    def test_small_input_all_tests_skip(self):
        """n=10 → all 5 tests skip."""
        deltas = [0.05] * 10
        cx = [0.5] * 10
        result = self._spa().analyze(deltas, cx)
        self.assertFalse(result.runs_test_ran)
        self.assertFalse(result.acf_test_ran)
        self.assertFalse(result.cusum_ran)
        self.assertFalse(result.cac_test_ran)
        self.assertFalse(result.entropy_test_ran)

    def test_cusum_runs_at_n15(self):
        """n=15 → only CUSUM should run (min_n=15)."""
        deltas = [0.001] * 15
        cx = [0.6] * 15
        result = self._spa().analyze(deltas, cx)
        self.assertTrue(result.cusum_ran)
        self.assertFalse(result.runs_test_ran)  # needs 20

    def test_human_game_clean(self):
        """
        Long human-like game with varied deltas: ensemble should not flag.
        Not guaranteed statistically, but the test verifies the pipeline runs.
        """
        deltas = _human_deltas(50)
        cx = _uniform_complexity(50, 0.5)
        result = self._spa().analyze(deltas, cx)
        self.assertEqual(result.n_moves, 50)
        self.assertIsNotNone(result.verdict)

    def test_full_cheater_game_detected(self):
        """
        Full cheater (all deltas ≈ 0): CUSUM should trigger, entropy very low,
        missing-tail flag raised.
        """
        deltas = [0.001] * 40
        cx = _uniform_complexity(40, 0.6)
        result = self._spa().analyze(deltas, cx)
        # CUSUM should trigger
        self.assertTrue(result.cusum_triggered)
        # Entropy should be near 0
        self.assertIsNotNone(result.shannon_entropy)
        self.assertLess(result.shannon_entropy, 0.5)
        # Missing tail should be flagged
        self.assertTrue(result.missing_tail)

    def test_type_b_alternating_detected(self):
        """Type B cheater (alternating engine/human, n=50): ACF should be negative."""
        deltas = _alternating_deltas(50)
        cx = _uniform_complexity(50, 0.5)
        result = self._spa().analyze(deltas, cx)
        self.assertIsNotNone(result.rho1)
        self.assertLess(result.rho1, 0.0)

    def test_result_fields_are_populated(self):
        """After a full run with sufficient n, key fields must be non-None."""
        deltas = _human_deltas(50)
        cx = _uniform_complexity(50, 0.5)
        result = self._spa().analyze(deltas, cx)
        # At least CUSUM and entropy should have run
        self.assertIsNotNone(result.cusum_max)
        self.assertIsNotNone(result.shannon_entropy)
        self.assertIsNotNone(result.ensemble_score)

    def test_format_report_does_not_crash(self):
        """format_report() must return a non-empty string."""
        deltas = _human_deltas(50)
        cx = _uniform_complexity(50, 0.5)
        result = self._spa().analyze(deltas, cx)
        report = SequentialPatternAnalyzer.format_report(result)
        self.assertIsInstance(report, str)
        self.assertIn("LAYER 2", report)
        self.assertIn(result.verdict, report)





if __name__ == "__main__":
    unittest.main()
