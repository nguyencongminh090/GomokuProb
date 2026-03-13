import unittest
from src.core.complexity import (
    calculate_complexity, 
    compute_f_opp, 
    compute_c_final,
    W_REF,
    ALPHA_MIN
)

class TestComplexity(unittest.TestCase):
    
    def test_compute_f_opp_boundary(self):
        # 1. Test F_opp = 1.0 when w_opp_best = 0.40 (boundary)
        self.assertAlmostEqual(compute_f_opp(0.40), 1.0)
        self.assertAlmostEqual(compute_f_opp(0.80), 1.0)  # Capped at 1.0
        
        # 2. Test F_opp = 0.005 when w_opp_best = 0.002 (nearly zero)
        self.assertAlmostEqual(compute_f_opp(0.002), 0.005)

    def test_compute_c_final(self):
        c_raw = 0.50
        
        # 3. Test C_final = 0.10 when trivial override = True (regardless of F_opp)
        f_opp_high = 1.0
        c_adj, c_final = compute_c_final(c_raw, f_opp_high, trivial_override=True)
        self.assertEqual(c_final, 0.10)
        
        f_opp_low = 0.05
        c_adj_low, c_final_low = compute_c_final(c_raw, f_opp_low, trivial_override=True)
        self.assertEqual(c_final_low, 0.10)
        
        # 4. Test C_final = C * ALPHA_MIN when f_opp < ALPHA_MIN
        f_opp_tiny = 0.02
        c_adj_tiny, c_final_tiny = compute_c_final(c_raw, f_opp_tiny, trivial_override=False)
        self.assertEqual(c_adj_tiny, c_raw * ALPHA_MIN)
        self.assertEqual(c_final_tiny, c_raw * ALPHA_MIN)
        
        # 5. Regression test: C_final == C_adj when opponent competitive (f_opp = 1.0)
        c_adj_comp, c_final_comp = compute_c_final(c_raw, 1.0, trivial_override=False)
        self.assertEqual(c_adj_comp, c_raw)
        self.assertEqual(c_final_comp, c_raw)

    def test_calculate_complexity_integration(self):
        # Integrate everything by simulating candidate winrates
        # Case A: Competitive opponent (w_opp_best = 0.5) -> F_opp = 1.0
        # No trivial override
        candidate_wrs = [0.60, 0.58, 0.55, 0.52]
        prev_wr = 0.50
        
        res_comp = calculate_complexity(candidate_wrs, prev_wr, w_opp_best=0.50)
        self.assertEqual(res_comp.opp_factor, 1.0)
        self.assertEqual(res_comp.adjusted_complexity, res_comp.complexity)
        self.assertEqual(res_comp.final_complexity, res_comp.complexity)
        
        # Case B: Trivial override should force final_complexity to 0.10
        # Winrate > 0.95 and variance < 0.05 (all wrs ~0.99)
        candidate_wrs_trivial = [0.99, 0.985, 0.98, 0.98]
        res_triv = calculate_complexity(candidate_wrs_trivial, prev_wr, w_opp_best=0.80)
        self.assertEqual(res_triv.final_complexity, 0.10)
        
        # Case C: Defeated opponent (w_opp_best = 0.0) -> F_opp = 0.0, C_adj = C * 0.10
        candidate_wrs_lost = [0.70, 0.60, 0.50]  # Normal complexity
        res_lost = calculate_complexity(candidate_wrs_lost, prev_wr, w_opp_best=0.0)
        self.assertEqual(res_lost.opp_factor, 0.0)
        self.assertAlmostEqual(res_lost.adjusted_complexity, res_lost.complexity * 0.10)
        self.assertAlmostEqual(res_lost.final_complexity, res_lost.complexity * 0.10)

if __name__ == '__main__':
    unittest.main()
