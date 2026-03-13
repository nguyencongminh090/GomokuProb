import math
import scipy.stats as stats
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.core.profile_store import ProfileRecord

@dataclass
class ProfileResult:
    """Result of a profile analysis comparison."""
    status: str            # "INSUFFICIENT_DATA", "NORMAL", "SUSPICIOUS"
    message: str           # Human-readable explanation
    m_games: int           # Number of games in baseline
    lambda_bar: Optional[float] = None
    s_lambda: Optional[float] = None
    test_statistic: Optional[float] = None  # Z-score or T-score
    p_value: Optional[float] = None
    critical_05: Optional[float] = None
    critical_01: Optional[float] = None


class ProfileAnalyzer:
    """
    Implements Multi-Game Profile Analysis (Paper Section 8).
    Computes statistical baseline and performs anomaly hypothesis testing.
    """
    
    MIN_GAMES_FOR_ANALYSIS = 2  # Need at least 2 games to compute variance (s_lambda)
    Z_TEST_THRESHOLD = 30       # m >= 30 uses Z-test, m < 30 uses T-test
    
    @staticmethod
    def compute_test_statistic(
        lambda_suspect: float, 
        lambda_bar: float, 
        s_lambda: float, 
        m: int
    ) -> Tuple[float, float, float, float]:
        """
        Computes the Z-score or T-score depending on sample size m.
        Returns: (test_statistic, p_value, critical_05, critical_01)
        """
        # Protect against divide-by-zero if variance is exactly 0 (e.g. all games identical)
        std_err = (s_lambda / math.sqrt(m)) if s_lambda > 1e-6 else 1e-6
        
        # Test statistic (Eq. 29)
        test_stat = (lambda_suspect - lambda_bar) / std_err
        
        if m >= ProfileAnalyzer.Z_TEST_THRESHOLD:
            # Z-test (normal distribution)
            p_value = 1.0 - stats.norm.cdf(test_stat)
            critical_05 = 1.645 # z_0.05
            critical_01 = 2.326 # z_0.01
        else:
            # T-test (t-distribution with m-1 df)
            df = m - 1
            p_value = 1.0 - stats.t.cdf(test_stat, df=df)
            critical_05 = stats.t.ppf(0.95, df=df)
            critical_01 = stats.t.ppf(0.99, df=df)
            
        return test_stat, p_value, critical_05, critical_01

    @staticmethod
    def analyze_suspect_game(suspect_lambda: float, history: List[ProfileRecord]) -> ProfileResult:
        """
        Perform anomaly detection against a player's baseline history.
        """
        m = len(history)
        
        if m < ProfileAnalyzer.MIN_GAMES_FOR_ANALYSIS:
            return ProfileResult(
                status="INSUFFICIENT_DATA",
                message=f"Need at least {ProfileAnalyzer.MIN_GAMES_FOR_ANALYSIS} baseline games (have {m}).",
                m_games=m
            )
            
        # Extract lambda_mle series
        lambdas = [record.lambda_mle for record in history]
        
        # Compute baseline statistics (Eqs. 24, 25)
        lambda_bar = sum(lambdas) / m
        
        # Sample variance (divide by m-1)
        variance = sum((l - lambda_bar) ** 2 for l in lambdas) / (m - 1)
        s_lambda = math.sqrt(variance)
        
        # Compute exact test statistic and critical values
        test_stat, p_value, crit_05, crit_01 = ProfileAnalyzer.compute_test_statistic(
            suspect_lambda, lambda_bar, s_lambda, m
        )
        
        # Escalation logic based on test statistic
        if test_stat > crit_01:
            status = "SUSPICIOUS"  # > 99% confident it's an anomaly
            msg = f"Stat={test_stat:.2f} > crit={crit_01:.2f} (p={p_value:.4f}). High confidence anomaly."
        elif test_stat > crit_05:
            status = "SUSPICIOUS"  # > 95% confident
            msg = f"Stat={test_stat:.2f} > crit={crit_05:.2f} (p={p_value:.4f}). Anomaly detected."
        else:
            status = "NORMAL"
            msg = f"Stat={test_stat:.2f} <= crit={crit_05:.2f}. Consistent with baseline."
            
        return ProfileResult(
            status=status,
            message=msg,
            m_games=m,
            lambda_bar=lambda_bar,
            s_lambda=s_lambda,
            test_statistic=test_stat,
            p_value=p_value,
            critical_05=crit_05,
            critical_01=crit_01
        )
