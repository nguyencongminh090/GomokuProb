"""
GomoProb: Profile Analyzer — Sprint 2 Update

Changes from original:
  - Replaces scipy stats with pure Python (norm/t CDFs from sequential_analysis.py)
  - Works with the new ProfileStore (dict rows from get_all_games)
  - Adds compute_profile_verdict() → Normal / Monitoring / Flagged
  - ProfileResult gains verdict and weighted_lambda fields
"""

import math
from typing import Optional
from dataclasses import dataclass

from src.core.sequential_analysis import _norm_sf, _t_cdf


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProfileResult:
    """Result of a single-game anomaly test against historical baseline."""
    status: str                       # "INSUFFICIENT_DATA" | "NORMAL" | "SUSPICIOUS"
    message: str                      # Human-readable explanation
    m_games: int                      # Number of games in baseline
    lambda_bar:     Optional[float] = None   # Baseline mean λ
    s_lambda:       Optional[float] = None   # Baseline std-dev of λ
    test_statistic: Optional[float] = None   # Z-score or T-score
    p_value:        Optional[float] = None
    critical_05:    Optional[float] = None
    critical_01:    Optional[float] = None
    verdict:        str = "N/A"


# ---------------------------------------------------------------------------
# ProfileAnalyzer
# ---------------------------------------------------------------------------

class ProfileAnalyzer:
    """
    Multi-Game Profile Analysis.

    Computes a statistical baseline from Tier 2 (baseline-eligible) games and
    tests whether a suspect game's λ is anomalously high.
    """

    MIN_GAMES_FOR_ANALYSIS = 2   # Need at least 2 for sample variance
    Z_TEST_THRESHOLD       = 30  # m >= 30 → Z-test; else T-test

    # ---------- pure-Python distribution helpers ----------

    @staticmethod
    def _norm_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    @staticmethod
    def _t_ppf_95(df: int) -> float:
        """Approximate t_{0.95, df} critical value."""
        # Use a lookup table for common df values, then asymptotic
        TABLE: dict[int, float] = {
            1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
            6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
            15: 1.753, 20: 1.725, 25: 1.708, 29: 1.699,
        }
        if df in TABLE:
            return TABLE[df]
        # Asymptotic approximation for large df
        return 1.645 + 0.9 / df

    @staticmethod
    def _t_ppf_99(df: int) -> float:
        TABLE: dict[int, float] = {
            1: 31.82, 2: 6.965, 3: 4.541, 4: 3.747, 5: 3.365,
            6: 3.143, 7: 2.998, 8: 2.896, 9: 2.821, 10: 2.764,
            15: 2.602, 20: 2.528, 25: 2.485, 29: 2.462,
        }
        if df in TABLE:
            return TABLE[df]
        return 2.326 + 1.1 / df

    @staticmethod
    def _t_sf(t: float, df: int) -> float:
        """Survival function (one-tailed right) of t-distribution."""
        return 1.0 - _t_cdf(t, df)

    # ---------- core test ----------

    @classmethod
    def compute_test_statistic(
        cls,
        lambda_suspect: float,
        lambda_bar: float,
        s_lambda: float,
        m: int,
    ) -> tuple[float, float, float, float]:
        """
        Compute Z-score or T-score depending on sample size m.

        Returns: (test_statistic, p_value, critical_05, critical_01)
        """
        std_err = (s_lambda / math.sqrt(m)) if s_lambda > 1e-6 else 1e-6
        test_stat = (lambda_suspect - lambda_bar) / std_err

        if m >= cls.Z_TEST_THRESHOLD:
            p_value     = _norm_sf(test_stat)
            critical_05 = 1.645
            critical_01 = 2.326
        else:
            df          = m - 1
            p_value     = cls._t_sf(test_stat, df)
            critical_05 = cls._t_ppf_95(df)
            critical_01 = cls._t_ppf_99(df)

        return test_stat, p_value, critical_05, critical_01

    @classmethod
    def analyze_suspect_game(
        cls,
        suspect_lambda: float,
        history: list,
    ) -> ProfileResult:
        """
        Perform anomaly detection against a player's baseline history.

        Args:
            suspect_lambda: λ_MLE of the game under scrutiny
            history: list of dicts from ProfileStore.get_all_games(baseline_only=True)
                     Each dict must have a 'lambda_mle' key.
        """
        m = len(history)

        if m < cls.MIN_GAMES_FOR_ANALYSIS:
            return ProfileResult(
                status  = "INSUFFICIENT_DATA",
                message = f"Need ≥ {cls.MIN_GAMES_FOR_ANALYSIS} baseline games (have {m}).",
                m_games = m,
            )

        # Extract λ values (support both dict and old ProfileRecord objects)
        lambdas = [
            r["lambda_mle"] if isinstance(r, dict) else r.lambda_mle
            for r in history
            if (r["lambda_mle"] if isinstance(r, dict) else r.lambda_mle) is not None
        ]

        if len(lambdas) < cls.MIN_GAMES_FOR_ANALYSIS:
            return ProfileResult(
                status  = "INSUFFICIENT_DATA",
                message = f"Insufficient non-null λ values (have {len(lambdas)}).",
                m_games = m,
            )

        m = len(lambdas)
        lambda_bar = sum(lambdas) / m
        variance   = sum((l - lambda_bar) ** 2 for l in lambdas) / (m - 1)
        s_lambda   = math.sqrt(variance)

        test_stat, p_value, crit_05, crit_01 = cls.compute_test_statistic(
            suspect_lambda, lambda_bar, s_lambda, m
        )

        if test_stat > crit_01:
            status  = "SUSPICIOUS"
            verdict = "Suspicious"
            msg = f"Stat={test_stat:.2f} > crit={crit_01:.2f} (p={p_value:.4f}). High confidence anomaly."
        elif test_stat > crit_05:
            status  = "SUSPICIOUS"
            verdict = "Suspicious"
            msg = f"Stat={test_stat:.2f} > crit={crit_05:.2f} (p={p_value:.4f}). Anomaly detected."
        else:
            status  = "NORMAL"
            verdict = "Normal"
            msg = f"Stat={test_stat:.2f} ≤ crit={crit_05:.2f}. Consistent with baseline."

        return ProfileResult(
            status         = status,
            message        = msg,
            m_games        = m,
            lambda_bar     = lambda_bar,
            s_lambda       = s_lambda,
            test_statistic = test_stat,
            p_value        = p_value,
            critical_05    = crit_05,
            critical_01    = crit_01,
            verdict        = verdict,
        )

    # ---------- profile verdict ----------

    @staticmethod
    def compute_profile_verdict(profile: dict) -> str:
        """
        Determine profile-level verdict from the `player_profiles` aggregate row.

        Thresholds:
            Flagged    — ≥ 3 active signals  OR  z_score_lambda > 3.0
            Monitoring — ≥ 2 active signals  OR  z_score_lambda > 2.0
            Normal     — everything else
        """
        if not profile:
            return "Normal"

        signals = [
            (profile.get("z_score_lambda")  or 0) > 2.5,
            (profile.get("mean_p_cheat")    or 0) > 0.15,
            (profile.get("quick_win_rate")  or 0) > 0.35,
            (profile.get("l2_flag_rate")    or 0) > 0.30,
            (profile.get("mean_cac")         or 0) > 0.35,
        ]
        n_signals = sum(signals)

        z_lambda = profile.get("z_score_lambda") or 0

        if n_signals >= 3 or z_lambda > 3.0:
            return "Flagged"
        elif n_signals >= 2 or z_lambda > 2.0:
            return "Monitoring"
        return "Normal"
