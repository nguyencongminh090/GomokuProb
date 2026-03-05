"""
Math utilities for analysis graphs and filtering.
V1 ProbabilityModel removed - see v2_worker.py for V2 implementation.
"""

from typing import List, Tuple
from dataclasses import dataclass
from typing import Optional


@dataclass
class MoveEvaluation:
    """Raw evaluation data from engine."""
    move_notation: str
    winrate: float  # 0.0 to 1.0
    is_best: bool = False
    score: Optional[float] = None  # Centipawn score
    mate_score: Optional[int] = None  # Positive for win, Negative for loss
    depth: int = 0
    nodes: int = 0
    salience: float = 1.0  # Visual Salience (0.0 to 2.0)


class KalmanFilter1D:
    """Simple 1D Kalman Filter for smoothing accuracy estimates."""
    
    def __init__(self, initial_state: float = 0.5, process_noise: float = 0.001, measurement_noise: float = 0.1):
        self.x = initial_state
        self.p = 1.0  # High uncertainty initially
        self.q = process_noise
        self.r = measurement_noise
        
    def update(self, measurement: float) -> Tuple[float, float]:
        # 1. Prediction
        self.p = self.p + self.q
        
        # 2. Update
        k_gain = self.p / (self.p + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p = (1 - k_gain) * self.p
        
        return self.x, self.p
        
    def batch_filter(self, measurements: List[float]) -> Tuple[List[float], List[float]]:
        """
        Returns:
            means: List of estimated skills
            covariances: List of uncertainties
        """
        means = []
        covariances = []
        
        if len(measurements) > 0:
            self.x = measurements[0]

        for z in measurements:
            x, p = self.update(z)
            means.append(x)
            covariances.append(p)
            
        return means, covariances


def calculate_ema(values: List[float], alpha: float = 0.2) -> List[float]:
    """
    Calculate Exponential Moving Average (EMA).
    EMA_t = alpha * x_t + (1 - alpha) * EMA_t-1
    """
    if not values:
        return []
        
    ema = []
    current = values[0]
    ema.append(current)
    
    for val in values[1:]:
        current = alpha * val + (1 - alpha) * current
        ema.append(current)
        
    return ema
