"""
Sequential Pattern Analysis — Layer 2 Anti-Cheating Detection
=============================================================

Implements the five statistical tests from:
    "Sequential Pattern Analysis for Detecting Sophisticated Cheating
     in Online Gomoku" (nguyen2025sequential)

Layer 2 complements the per-move Bayesian (Layer 1) analysis by exploiting
*inter-move dependencies* that random-mixing cheaters imprint on the
sequence (Δ₁,C₁), …, (Δₙ,Cₙ).

Three cheating archetypes targeted:
  * Type A — engine at complex positions (inverts complexity-accuracy relation)
  * Type B — stochastic random mixing (zig-zag ACF, run anomalies)
  * Type C — noise-camouflaged (2nd/3rd-best always, no blunders)

Five tests implemented:
  1. Wald–Wolfowitz Runs Test          (Section 4, Eq. ER/VarR/ZR)
  2. Lag-1 Autocorrelation Test (ACF)  (Section 5, Eq. acf/bartlett)
  3. CUSUM Change-Point Detection      (Section 6, Eq. cusum_score/cusum_recursion)
  4. Complexity-Accuracy Correlation   (Section 7, Eq. cac/cac_test)
  5. Shannon Entropy Test              (Section 8, Eq. entropy/tail_pval)

Ensemble:
  * Fisher's combined χ² (Section 9, Eq. fisher)
  * Weighted vote V (Section 9, Eq. vote)

IMPORTANT: Pure Python only — no numpy, no scipy.
           All distribution functions implemented via math.erf / math.lgamma.

Minimum sample sizes (per paper):
  Runs Test:    n ≥ 20
  ACF Test:     n ≥ 30
  CUSUM:        n ≥ 15
  CAC Test:     |non-trivial| ≥ 20 (non-trivial means C_i > 0.25)
  Entropy Test: n ≥ 25
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pure-Python distribution helpers (avoid numpy/scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(z: float) -> float:
    """Cumulative distribution function of the standard normal N(0,1)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_sf(z: float) -> float:
    """Survival function (1 - CDF) of the standard normal."""
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def _t_cdf(t: float, df: int) -> float:
    """
    Student-t CDF via the regularised incomplete beta function.
    Uses the relation:  CDF(t, df) = 1 - I_x(df/2, 1/2)/2   where x = df/(df+t²)
    """
    x = df / (df + t * t)
    # Regularised incomplete beta via continued fraction (Lentz)
    ib = _regularised_incomplete_beta(df / 2.0, 0.5, x)
    p = ib / 2.0  # tail probability
    if t > 0:
        return 1.0 - p
    return p


def _regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a,b) via continued fraction."""
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    # Use continued fraction representation (Lentz's algorithm)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    # Use symmetry for faster convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(b, a, 1.0 - x)
    # Lentz continued fraction
    MAX_ITER = 200
    EPSILON = 1e-12
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d
    for m in range(1, MAX_ITER + 1):
        # Even step
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        c = 1.0 + num / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= c * d
        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        c = 1.0 + num / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < EPSILON:
            break
    return front * f


def _chi2_sf(x: float, df: int) -> float:
    """
    Survival function of chi-squared distribution with df degrees of freedom.
    Uses the relation: SF(x, df) = 1 - I_{x/2}(df/2) where I is the upper
    regularised incomplete gamma.
    Implemented via the regularised incomplete gamma function.
    """
    if x <= 0.0:
        return 1.0
    return _regularised_upper_gamma(df / 2.0, x / 2.0)


def _regularised_upper_gamma(a: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(a,x) = Γ(a,x)/Γ(a)."""
    if x < 0.0:
        return 1.0
    if x == 0.0:
        return 1.0
    # For x < a+1 use series; otherwise use continued fraction
    if x < a + 1.0:
        return 1.0 - _lower_gamma_series(a, x)
    return _upper_gamma_cf(a, x)


def _lower_gamma_series(a: float, x: float) -> float:
    """Lower regularised incomplete gamma P(a,x) via series expansion."""
    MAX_ITER = 300
    EPSILON = 1e-12
    if x <= 0.0:
        return 0.0
    lngamma_a = math.lgamma(a)
    ap = a
    s = 1.0 / a
    delta = s
    for _ in range(MAX_ITER):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * EPSILON:
            break
    return s * math.exp(-x + a * math.log(x) - lngamma_a)


def _upper_gamma_cf(a: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(a,x) via continued fraction."""
    MAX_ITER = 300
    EPSILON = 1e-12
    lngamma_a = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    f = d
    for i in range(1, MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta
        if abs(delta - 1.0) < EPSILON:
            break
    return math.exp(-x + a * math.log(x) - lngamma_a) * f


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SequentialAnalysisResult:
    """
    Complete output of Layer 2 Sequential Pattern Analysis.

    Each test returns its primary statistic and a p-value.
    p-value = None means test was not run (insufficient data).
    """
    # --- Test 1: Wald-Wolfowitz Runs ---
    z_runs: Optional[float] = None          # Z_R
    p_runs: Optional[float] = None          # two-sided p-value

    # --- Test 2: Lag-1 ACF ---
    rho1: Optional[float] = None            # ρ̂₁ (lag-1 autocorrelation)
    z_acf: Optional[float] = None           # √n · ρ̂₁
    p_acf: Optional[float] = None           # one-tailed (left) p-value

    # --- Test 3: CUSUM ---
    cusum_max: Optional[float] = None       # max S_t
    cusum_triggered: bool = False           # whether max S_t > h
    change_point: Optional[int] = None      # estimated τ̂ (move index, 1-based)
    p_cusum: Optional[float] = None         # exceedance-based p-value

    # --- Test 4: CAC ---
    cac: Optional[float] = None             # Complexity-Accuracy Correlation
    z_cac: Optional[float] = None           # t-statistic
    p_cac: Optional[float] = None           # one-tailed (left) p-value

    # --- Test 5: Shannon Entropy ---
    shannon_entropy: Optional[float] = None # Ĥ in bits
    missing_tail: bool = False              # T_n = 0 flag (no blunder found)
    tail_pval: Optional[float] = None      # p-value for missing-tail test
    p_entropy: Optional[float] = None      # combined entropy p-value

    # --- Ensemble ---
    fisher_chi2: Optional[float] = None    # -2 Σ log(p_j), df=10
    p_fisher: Optional[float] = None       # p-value from χ²(10)
    ensemble_score: float = 0.0            # Weighted vote V ∈ [0, 1]
    ensemble_flags: int = 0                # Number of individual tests flagged

    # --- Metadata ---
    n_moves: int = 0
    n_nontrivial: int = 0                  # |{i: C_i > 0.25}|
    verdict: str = "N/A"                   # "Clean", "Suspicious (Type A/B/C)", "Suspicious (Mixed)"

    # Diagnostics (run conditions met?)
    runs_test_ran: bool = False
    acf_test_ran: bool = False
    cusum_ran: bool = False
    cac_test_ran: bool = False
    entropy_test_ran: bool = False


# ---------------------------------------------------------------------------
# Test 1: Wald-Wolfowitz Runs Test
# ---------------------------------------------------------------------------

class WaldWolfowitzRunsTest:
    """
    Paper Section 4: Wald-Wolfowitz Runs Test for binary-encoded move sequence.

    τ = 0.02 (matches near_optimal_threshold in DeltaModel).
    Requires n ≥ 20.
    """

    MIN_N = 20
    TAU = 0.02  # near-optimal threshold (same as delta model)

    @classmethod
    def run(cls, deltas: List[float]) -> Tuple[Optional[float], Optional[float], dict]:
        """
        Run the test.

        Returns:
            (z_runs, p_runs, diagnostics_dict)
        """
        n = len(deltas)
        if n < cls.MIN_N:
            return None, None, {"ran": False, "reason": f"n={n} < {cls.MIN_N}"}

        # Binary encoding: b_i = 1 if Δ_i ≤ τ
        b = [1 if d <= cls.TAU else 0 for d in deltas]
        n1 = sum(b)          # optimal moves
        n0 = n - n1          # suboptimal moves

        if n1 == 0 or n0 == 0:
            # Cannot compute runs test (all same symbol)
            return None, None, {"ran": False, "reason": "All moves on same side of threshold"}

        # Count runs
        runs = 1
        for i in range(1, n):
            if b[i] != b[i - 1]:
                runs += 1

        # Expected value and variance (Eq. ER, VarR)
        e_r = 2.0 * n1 * n0 / n + 1.0
        var_r_num = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n)
        var_r_den = n * n * (n - 1.0)
        if var_r_den <= 0 or var_r_num / var_r_den <= 0:
            return None, None, {"ran": False, "reason": "Variance undefined"}
        var_r = var_r_num / var_r_den
        std_r = math.sqrt(var_r)

        # Z-statistic (Eq. ZR)
        z_r = (runs - e_r) / std_r

        # Two-sided p-value
        p_runs = 2.0 * min(_norm_sf(z_r), _norm_cdf(z_r))

        diag = {"ran": True, "n": n, "n1": n1, "n0": n0, "runs": runs, "e_r": e_r}
        return z_r, p_runs, diag


# ---------------------------------------------------------------------------
# Test 2: Lag-1 Autocorrelation Test
# ---------------------------------------------------------------------------

class Lag1ACFTest:
    """
    Paper Section 5: Lag-1 Autocorrelation Test.

    Detects zig-zag pattern from engine/human alternation (Type B).
    One-tailed (left) at α=0.05 since negative autocorrelation is predicted.
    Requires n ≥ 30.
    """

    MIN_N = 30

    @classmethod
    def run(cls, deltas: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], dict]:
        """
        Returns:
            (rho1, z_acf, p_acf, diagnostics_dict)
        """
        n = len(deltas)
        if n < cls.MIN_N:
            return None, None, None, {"ran": False, "reason": f"n={n} < {cls.MIN_N}"}

        delta_bar = sum(deltas) / n
        # Numerator Σ (Δ_i - Δ̄)(Δ_{i+1} - Δ̄), i=1..n-1 (Eq. acf)
        numerator = sum(
            (deltas[i] - delta_bar) * (deltas[i + 1] - delta_bar)
            for i in range(n - 1)
        )
        # Denominator Σ (Δ_i - Δ̄)²
        denominator = sum((d - delta_bar) ** 2 for d in deltas)
        if denominator < 1e-15:
            return None, None, None, {"ran": False, "reason": "Zero variance in deltas"}

        rho1 = numerator / denominator
        # Clip to valid range (numerical safety)
        rho1 = max(-1.0, min(1.0, rho1))

        # Z statistic and one-tailed p-value (left tail, Eq. bartlett)
        z_acf = math.sqrt(n) * rho1
        p_acf = _norm_cdf(z_acf)  # P(Z ≤ z_acf); small when rho1 is negative

        return rho1, z_acf, p_acf, {"ran": True, "n": n}


# ---------------------------------------------------------------------------
# Test 3: CUSUM Change-Point Detection
# ---------------------------------------------------------------------------

class CUSUMTest:
    """
    Paper Section 6: Page's CUSUM for change-point detection.

    Uses complexity-tempered log-likelihood ratio scores.
    Threshold h = 3.0 (ARL₀ ≈ 50 under human null, per paper).
    Requires n ≥ 15.
    """

    MIN_N = 15
    H_THRESHOLD = 4.0  # Increased from 3.0 to reduce oversensitivity on strong human openings.
    
    # Defaults in case not provided, but we will pass them dynamically
    LAMBDA_HUMAN = 10.0      
    LAMBDA_CHEATER = 70.0

    @classmethod
    def _score(cls, delta: float, complexity: float, lambda_h: float, lambda_c: float) -> float:
        """
        Per-move score s_i = C_i · log(p(Δ|H1)/p(Δ|H0))  (Eq. cusum_score).

        Both H0 and H1 are Exponential distributions.
        log ratio = log(λ₁/λ₀) - (λ₁ - λ₀)·Δ
        """
        d = max(delta, 1e-4)
        log_ratio = (
            math.log(lambda_c / lambda_h)
            - (lambda_c - lambda_h) * d
        )
        return complexity * log_ratio

    @classmethod
    def run(
        cls,
        deltas: List[float],
        complexities: List[float],
        lambda_h: float = LAMBDA_HUMAN,
        lambda_c: float = LAMBDA_CHEATER,
    ) -> Tuple[Optional[float], Optional[int], Optional[float], dict]:
        """
        Returns:
            (cusum_max, change_point_index, p_cusum, diagnostics_dict)
        """
        n = len(deltas)
        if n < cls.MIN_N or len(complexities) < n:
            return None, None, None, {"ran": False, "reason": f"n={n} < {cls.MIN_N}"}

        # Compute CUSUM recursion S_t = max(0, S_{t-1} + s_t)  (Eq. cusum_recursion)
        s_values = [0.0] * n
        s_prev = 0.0
        for i in range(n):
            s_i = s_prev + cls._score(deltas[i], complexities[i], lambda_h, lambda_c)
            s_values[i] = max(0.0, s_i)
            s_prev = s_values[i]

        cusum_max = max(s_values)
        triggered = cusum_max > cls.H_THRESHOLD

        # Change-point estimate: move where accumulated sum restarted most recently
        # τ̂ = argmin_{k≤t} Σ_{i=k+1}^{t} s_i  (Eq. changepoint_estimate)
        # Approximate: find the index where S drops to 0 most recently before the peak
        change_point = None
        if triggered:
            peak_idx = s_values.index(cusum_max)
            # Walk backward to find last zero-crossing
            cp = 0
            for j in range(peak_idx, -1, -1):
                if s_values[j] <= 0.0:
                    cp = j + 1
                    break
            change_point = cp + 1  # 1-based move index

        # p-value: one-sided approximation from maximum of CUSUM path
        # Under H0, CUSUM max has approximately exponential distribution
        # with mean ~h; we approximate p = exp(-cusum_max / h)
        if cusum_max is not None and cusum_max > 0:
            p_cusum = math.exp(-cusum_max / cls.H_THRESHOLD)
            p_cusum = min(1.0, p_cusum)
        else:
            p_cusum = 1.0

        diag = {
            "ran": True, "n": n,
            "triggered": triggered,
            "cusum_max": cusum_max,
        }
        return cusum_max, change_point, p_cusum, diag


# ---------------------------------------------------------------------------
# Test 4: Complexity-Accuracy Correlation (CAC)
# ---------------------------------------------------------------------------

class CACTest:
    """
    Paper Section 7: Complexity-Accuracy Correlation.

    CAC = -ρ̂(Δ, C) restricted to non-trivial moves {i: C_i > 0.25}.
    CAC > 0 signals Type A cheating (engine at high-C positions).
    Requires |non-trivial| ≥ 20 for reliable inference (paper says 20).
    """

    MIN_NONTRIVIAL = 20
    C_THRESHOLD = 0.25

    @classmethod
    def run(
        cls,
        deltas: List[float],
        complexities: List[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], int, dict]:
        """
        Returns:
            (cac, z_cac, p_cac, n_nontrivial, diagnostics_dict)
        """
        # Filter to non-trivial moves
        pairs = [(d, c) for d, c in zip(deltas, complexities) if c > cls.C_THRESHOLD]
        n_i = len(pairs)

        if n_i < cls.MIN_NONTRIVIAL:
            return None, None, None, n_i, {"ran": False, "reason": f"|I|={n_i} < {cls.MIN_NONTRIVIAL}"}

        delta_i = [p[0] for p in pairs]
        c_i = [p[1] for p in pairs]

        # Pearson correlation r(Δ, C)  (Eq. cac)
        d_bar = sum(delta_i) / n_i
        c_bar = sum(c_i) / n_i

        cov = sum((d - d_bar) * (c - c_bar) for d, c in zip(delta_i, c_i))
        std_d = math.sqrt(sum((d - d_bar) ** 2 for d in delta_i))
        std_c = math.sqrt(sum((c - c_bar) ** 2 for c in c_i))

        if std_d < 1e-10 or std_c < 1e-10:
            return None, None, None, n_i, {"ran": False, "reason": "Zero variance in filtered sequence"}

        r = cov / (std_d * std_c)
        r = max(-1.0, min(1.0, r))  # Clip for numerical safety
        cac = -r  # Negate: CAC > 0 means Δ DECREASES as C increases (suspicious)

        # t-statistic for r  (Eq. cac_test)
        if abs(r) >= 1.0 - 1e-10:
            z_cac = math.copysign(float('inf'), r)
            p_cac = 0.0
        else:
            z_cac = r / math.sqrt((1.0 - r * r) / (n_i - 2))
            # One-tailed (left): small p when r is negative (CAC positive)
            p_cac = _t_cdf(z_cac, df=n_i - 2)

        diag = {"ran": True, "n_nontrivial": n_i, "r": r, "cac": cac}
        return cac, z_cac, p_cac, n_i, diag


# ---------------------------------------------------------------------------
# Test 5: Shannon Entropy Test
# ---------------------------------------------------------------------------

class ShannonEntropyTest:
    """
    Paper Section 8: Shannon Entropy test and missing-tail test.

    Bins: [0,1%), [1%,5%), [5%,15%), [15%,50%), [50%+]    (Eq. entropy)
    Missing tail: T_n = 0 if no Δ > τ_blunder = 0.10      (Eq. tail_pval)
    Requires n ≥ 25.
    """

    MIN_N = 25
    # Bin edges as fractions (Δ is in [0,1])
    BINS = [0.0, 0.01, 0.05, 0.15, 0.50, float('inf')]
    TAU_BLUNDER = 0.10  # 10% WR threshold

    @classmethod
    def _discretize(cls, deltas: List[float]) -> List[float]:
        """Return empirical frequency vector over the 5 bins."""
        n = len(deltas)
        counts = [0] * (len(cls.BINS) - 1)
        for d in deltas:
            for k in range(len(cls.BINS) - 1):
                if cls.BINS[k] <= d < cls.BINS[k + 1]:
                    counts[k] += 1
                    break
        return [c / n for c in counts]

    @classmethod
    def run(
        cls,
        deltas: List[float],
        blunder_rate_baseline: float = 0.20,
    ) -> Tuple[Optional[float], bool, Optional[float], Optional[float], dict]:
        """
        Args:
            deltas: Winrate loss sequence
            blunder_rate_baseline: P(Δ > τ_blunder) for the player's skill tier.
                                   Default 0.20 (20% of moves are blunders in typical games).
        Returns:
            (shannon_entropy, missing_tail, tail_pval, p_entropy, diagnostics)
        """
        n = len(deltas)
        if n < cls.MIN_N:
            return None, False, None, None, {"ran": False, "reason": f"n={n} < {cls.MIN_N}"}

        freqs = cls._discretize(deltas)

        # Shannon entropy Ĥ = -Σ p̂_k log₂(p̂_k)   (Eq. entropy)
        epsilon = 1e-12
        h_hat = -sum(p * math.log2(p) for p in freqs if p > epsilon)

        # Missing-tail test: T_n = 0 iff no blunder exists
        has_blunder = any(d > cls.TAU_BLUNDER for d in deltas)
        missing_tail = not has_blunder

        # p-value for missing-tail: P(T_n=0 | H_0) = (1-p_blunder)^n  (Eq. tail_pval)
        p_blunder = max(blunder_rate_baseline, 1e-6)
        tail_pval = (1.0 - p_blunder) ** n if missing_tail else 1.0

        # Entropy test: under null (human), H ≈ 2.0 bits for typical players.
        # Cheaters concentrate mass in bin [0,1%) → lower entropy ≈ 1.3-1.7 bits.
        # Simple heuristic p-value: map H into [0,1] relative to max entropy (log2(5)≈2.32)
        h_null = 2.0   # Expected human entropy in bits (from paper Proposition prop:entropy_gap)
        h_std  = 0.5   # Approximate std-dev of human entropy (calibrated later)
        z_h = (h_hat - h_null) / h_std
        p_entropy_raw = _norm_cdf(z_h)  # Small p when H is much lower than human baseline

        # Combine with missing-tail: take min (most suspicious signal)
        p_entropy = min(p_entropy_raw, tail_pval)

        diag = {
            "ran": True, "n": n,
            "freqs": freqs, "h_hat": h_hat,
            "has_blunder": has_blunder,
        }
        return h_hat, missing_tail, tail_pval, p_entropy, diag


# ---------------------------------------------------------------------------
# Ensemble Detector
# ---------------------------------------------------------------------------

class EnsembleDetector:
    """
    Paper Section 9: Ensemble combination of the five tests.

    Fisher's combined: χ² = -2 Σ log(p_j), df=10, theshold=18.31 at α=0.05
    Weighted vote: V = Σ w_j · 1[p_j < α_j]

    Balanced weights (Table 1): [0.20, 0.25, 0.20, 0.15, 0.20]
    """

    # Balanced weights: [Runs, ACF, CUSUM, CAC, Entropy]
    BALANCED_WEIGHTS = [0.20, 0.25, 0.20, 0.15, 0.20]
    # Per-test α thresholds (individual significance levels)
    ALPHA_PER_TEST = [0.05, 0.05, 0.05, 0.05, 0.05]
    # Fisher threshold at α=0.05, df=10
    FISHER_THRESHOLD = 18.31
    # Vote threshold for "suspicious" call
    V_THRESHOLD = 0.40

    @classmethod
    def combine(
        cls,
        p_runs: Optional[float],
        p_acf: Optional[float],
        p_cusum: Optional[float],
        p_cac: Optional[float],
        p_entropy: Optional[float],
        weights: Optional[List[float]] = None,
    ) -> Tuple[Optional[float], Optional[float], float, int]:
        """
        Combine the five p-values.

        Missing p-values (None = test not run) are excluded from Fisher's χ²
        and ignored in the weighted vote (weight redistributed).

        Returns:
            (fisher_chi2, p_fisher, ensemble_score, n_flagged)
        """
        p_vals = [p_runs, p_acf, p_cusum, p_cac, p_entropy]
        w = weights or cls.BALANCED_WEIGHTS

        # Fisher's combined p-value (only over valid p-values)
        valid_pairs = [(p, w[i]) for i, p in enumerate(p_vals) if p is not None]
        if not valid_pairs:
            return None, None, 0.0, 0

        chi2 = -2.0 * sum(math.log(max(p, 1e-300)) for p, _ in valid_pairs)
        # Degrees of freedom = 2 × number of valid tests (Fisher, 1932)
        df = 2 * len(valid_pairs)
        p_fisher = _chi2_sf(chi2, df)

        # Weighted vote (redistribute weights from missing tests)
        valid_w_total = sum(ww for _, ww in valid_pairs)
        if valid_w_total <= 0:
            return chi2, p_fisher, 0.0, 0

        n_flagged = 0
        vote = 0.0
        for i, p in enumerate(p_vals):
            if p is None:
                continue
            normalised_w = w[i] / valid_w_total
            if p < cls.ALPHA_PER_TEST[i]:
                vote += normalised_w
                n_flagged += 1

        return chi2, p_fisher, vote, n_flagged

    @classmethod
    def determine_verdict(
        cls,
        fisher_chi2: Optional[float],
        p_fisher: Optional[float],
        ensemble_score: float,
        n_flagged: int,
        p_runs: Optional[float],
        p_acf: Optional[float],
        p_cusum: Optional[float],
        p_cac: Optional[float],
        p_entropy: Optional[float],
        cac: Optional[float],
    ) -> str:
        """
        Determine the Layer 2 verdict string.

        Combination rule (architecture Section 9):
          Flag suspicious if ensemble_score > V* OR p_fisher < 0.05
        """
        is_suspicious = (
            (p_fisher is not None and p_fisher < 0.05)
            or ensemble_score > cls.V_THRESHOLD
        )
        if not is_suspicious:
            return "Clean"

        # Determine dominant archetype from which tests fired
        type_a = p_cac is not None and p_cac < 0.05 and cac is not None and cac > 0.30
        type_b = (
            (p_runs is not None and p_runs < 0.05)
            or (p_acf is not None and p_acf < 0.05)
        )
        type_c = p_entropy is not None and p_entropy < 0.05

        active = [("Type A", type_a), ("Type B", type_b), ("Type C", type_c)]
        active_types = [name for name, flag in active if flag]

        if len(active_types) == 1:
            return f"Suspicious ({active_types[0]})"
        elif len(active_types) > 1:
            return "Suspicious (Mixed)"
        else:
            # Fisher/ensemble triggered but no specific type identified
            return "Suspicious"


# ---------------------------------------------------------------------------
# Facade: SequentialPatternAnalyzer
# ---------------------------------------------------------------------------

class SequentialPatternAnalyzer:
    """
    Layer 2 Sequential Pattern Analysis — main entry point.

    Usage:
        spa = SequentialPatternAnalyzer()
        result = spa.analyze(deltas, complexities)
    """

    def analyze(
        self,
        deltas: List[float],
        complexities: List[float],
        blunder_rate_baseline: float = 0.20,
        lambda_h: float = 10.0,
        lambda_c: float = 70.0,
    ) -> SequentialAnalysisResult:
        """
        Run all five tests and the ensemble detector.

        Args:
            deltas:       Sequence of Δᵢ = W*ᵢ - Wᵖᵢ (always ≥ 0).
                          Must be in [0, 1] (fraction, NOT percentage).
            complexities: Final complexity C_final per analysed move.
                          Must have the same length as `deltas`.
            blunder_rate_baseline: P(Δ > 10%) for the player's skill tier
                                   (used by missing-tail p-value).

        Returns:
            SequentialAnalysisResult with all computed statistics.
        """
        n = len(deltas)
        result = SequentialAnalysisResult(n_moves=n)

        if n == 0:
            result.verdict = "N/A (no moves)"
            return result

        # Align lengths (safety)
        min_len = min(len(deltas), len(complexities))
        deltas = deltas[:min_len]
        complexities = complexities[:min_len]
        n = min_len
        result.n_moves = n

        # --- Test 1: Runs ---
        z_r, p_r, diag1 = WaldWolfowitzRunsTest.run(deltas)
        result.z_runs = z_r
        result.p_runs = p_r
        result.runs_test_ran = diag1.get("ran", False)

        # --- Test 2: ACF ---
        rho1, z_acf, p_acf, diag2 = Lag1ACFTest.run(deltas)
        result.rho1 = rho1
        result.z_acf = z_acf
        result.p_acf = p_acf
        result.acf_test_ran = diag2.get("ran", False)

        # --- Test 3: CUSUM ---
        cusum_max, cp, p_cusum, diag3 = CUSUMTest.run(deltas, complexities, lambda_h, lambda_c)
        result.cusum_max = cusum_max
        result.change_point = cp
        result.p_cusum = p_cusum
        result.cusum_triggered = diag3.get("triggered", False)
        result.cusum_ran = diag3.get("ran", False)

        # --- Test 4: CAC ---
        cac, z_cac, p_cac, n_nontrivial, diag4 = CACTest.run(deltas, complexities)
        result.cac = cac
        result.z_cac = z_cac
        result.p_cac = p_cac
        result.n_nontrivial = n_nontrivial
        result.cac_test_ran = diag4.get("ran", False)

        # --- Test 5: Entropy ---
        h_hat, missing_tail, tail_pval, p_entropy, diag5 = ShannonEntropyTest.run(
            deltas, blunder_rate_baseline
        )
        result.shannon_entropy = h_hat
        result.missing_tail = missing_tail
        result.tail_pval = tail_pval
        result.p_entropy = p_entropy
        result.entropy_test_ran = diag5.get("ran", False)

        # --- Ensemble ---
        chi2, p_fisher, vote, n_flagged = EnsembleDetector.combine(
            p_r, p_acf, p_cusum, p_cac, p_entropy
        )
        result.fisher_chi2 = chi2
        result.p_fisher = p_fisher
        result.ensemble_score = vote
        result.ensemble_flags = n_flagged

        # --- Verdict ---
        result.verdict = EnsembleDetector.determine_verdict(
            chi2, p_fisher, vote, n_flagged,
            p_r, p_acf, p_cusum, p_cac, p_entropy,
            cac,
        )

        return result

    @staticmethod
    def format_report(result: SequentialAnalysisResult) -> str:
        """
        Format a human-readable Layer 2 report for the STAGE log channel.
        """
        lines = [
            "",
            "=" * 50,
            "LAYER 2 — SEQUENTIAL PATTERN ANALYSIS",
            "=" * 50,
            f"  Moves analysed       : {result.n_moves}",
            f"  Non-trivial (C>0.25) : {result.n_nontrivial}",
            "",
            "  Test 1 — Runs Test:",
        ]

        def _fmt(val: Optional[float], fmt: str = ".3f") -> str:
            return f"{val:{fmt}}" if val is not None else "N/A"

        def _ran(flag: bool) -> str:
            return "" if flag else "  [skipped — insufficient data]"

        lines += [
            f"    Z_R        = {_fmt(result.z_runs)}{_ran(result.runs_test_ran)}",
            f"    p-value    = {_fmt(result.p_runs)}",
            "",
            "  Test 2 — Lag-1 ACF:",
            f"    ρ̂₁        = {_fmt(result.rho1)}{_ran(result.acf_test_ran)}",
            f"    Z_ACF      = {_fmt(result.z_acf)}",
            f"    p-value    = {_fmt(result.p_acf)}",
            "",
            "  Test 3 — CUSUM:",
            f"    max(S_t)   = {_fmt(result.cusum_max)}{_ran(result.cusum_ran)}",
            f"    triggered  = {result.cusum_triggered}",
            f"    change-pt  = move {result.change_point if result.change_point else 'N/A'}",
            f"    p-value    = {_fmt(result.p_cusum)}",
            "",
            "  Test 4 — CAC:",
            f"    CAC        = {_fmt(result.cac)}{_ran(result.cac_test_ran)}",
            f"    Z_CAC      = {_fmt(result.z_cac)}",
            f"    p-value    = {_fmt(result.p_cac)}",
            "",
            "  Test 5 — Shannon Entropy:",
            f"    Ĥ (bits)  = {_fmt(result.shannon_entropy)}{_ran(result.entropy_test_ran)}",
            f"    missing-τ  = {result.missing_tail}",
            f"    tail p-val = {_fmt(result.tail_pval)}",
            f"    p-value    = {_fmt(result.p_entropy)}",
            "",
            "  Ensemble:",
            f"    Fisher χ²  = {_fmt(result.fisher_chi2)}",
            f"    p(Fisher)  = {_fmt(result.p_fisher)}",
            f"    Vote V     = {result.ensemble_score:.3f}",
            f"    Flags      = {result.ensemble_flags}/5",
            "",
            f"  ► Layer 2 Verdict: {result.verdict}",
            "=" * 50,
        ]
        return "\n".join(lines)
