"""
GomoProb V2: Temporal Model

Implements temporal analysis for detecting sudden changes in player behavior.
- Moving average of performance over time
- Switch-point detection (player suddenly becomes much better/worse)
- Trend analysis
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import statistics


@dataclass
class SwitchPoint:
    """Detected switch point in player behavior."""
    move_index: int              # Where the switch occurred
    delta_before: float          # Average Δ before switch
    delta_after: float           # Average Δ after switch
    magnitude: float             # How significant the change is
    direction: str               # "improved" or "degraded"


@dataclass
class TemporalAnalysis:
    """Complete temporal analysis of a game."""
    moving_averages: List[float] = field(default_factory=list)
    switch_points: List[SwitchPoint] = field(default_factory=list)
    trend_slope: float = 0.0     # Overall trend (negative = improving)
    volatility: float = 0.0       # How much the quality varies
    is_suspicious: bool = False   # Sudden improvement detected


class TemporalModel:
    """
    Temporal analysis for anti-cheating detection.
    
    Key insight: Cheaters who toggle engine mid-game will show
    a sudden "switch point" where their play quality dramatically improves.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        switch_threshold: float = 0.1  # 10% winrate improvement = switch
    ):
        """
        Args:
            window_size: Window for moving average
            switch_threshold: Minimum Δ change to count as switch point
        """
        self.window_size = window_size
        self.switch_threshold = switch_threshold
    
    def compute_moving_average(self, deltas: List[float]) -> List[float]:
        """
        Compute moving average of delta values.
        
        Args:
            deltas: List of Δ values per move
            
        Returns:
            List of moving average values (same length as input)
        """
        if not deltas:
            return []
        
        moving_avg = []
        for i in range(len(deltas)):
            start = max(0, i - self.window_size + 1)
            window = deltas[start:i+1]
            avg = sum(window) / len(window)
            moving_avg.append(avg)
        
        return moving_avg
    
    def detect_switch_points(
        self,
        deltas: List[float],
        moving_avg: Optional[List[float]] = None
    ) -> List[SwitchPoint]:
        """
        Detect sudden changes in player performance.
        
        A switch point is where the moving average changes by more than
        switch_threshold compared to the previous window.
        
        Args:
            deltas: Raw delta values
            moving_avg: Pre-computed moving average (optional)
            
        Returns:
            List of detected SwitchPoint objects
        """
        if len(deltas) < self.window_size * 2:
            return []  # Not enough data
        
        if moving_avg is None:
            moving_avg = self.compute_moving_average(deltas)
        
        switch_points = []
        
        # Compare each point to previous window
        for i in range(self.window_size, len(moving_avg) - self.window_size):
            # Average before this point (previous window)
            before_start = max(0, i - self.window_size)
            delta_before = statistics.mean(moving_avg[before_start:i])
            
            # Average after this point (next window)
            after_end = min(len(moving_avg), i + self.window_size)
            delta_after = statistics.mean(moving_avg[i:after_end])
            
            change = delta_before - delta_after  # Positive = improved
            
            if abs(change) >= self.switch_threshold:
                direction = "improved" if change > 0 else "degraded"
                switch_points.append(SwitchPoint(
                    move_index=i,
                    delta_before=delta_before,
                    delta_after=delta_after,
                    magnitude=abs(change),
                    direction=direction
                ))
        
        # Merge nearby switch points (keep the most significant)
        merged = self._merge_nearby_switches(switch_points)
        
        return merged
    
    def _merge_nearby_switches(
        self,
        switches: List[SwitchPoint],
        min_gap: int = 5
    ) -> List[SwitchPoint]:
        """Merge switch points that are too close together."""
        if not switches:
            return []
        
        merged = []
        current = switches[0]
        
        for switch in switches[1:]:
            if switch.move_index - current.move_index < min_gap:
                # Keep the more significant one
                if switch.magnitude > current.magnitude:
                    current = switch
            else:
                merged.append(current)
                current = switch
        
        merged.append(current)
        return merged
    
    def compute_trend(self, deltas: List[float]) -> float:
        """
        Compute overall trend in delta values.
        
        Uses simple linear regression slope.
        Negative slope = player improving over time
        Positive slope = player degrading over time
        
        Returns:
            Slope of the trend line
        """
        if len(deltas) < 2:
            return 0.0
        
        n = len(deltas)
        x_mean = (n - 1) / 2
        y_mean = sum(deltas) / n
        
        numerator = sum((i - x_mean) * (d - y_mean) for i, d in enumerate(deltas))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def compute_volatility(self, deltas: List[float]) -> float:
        """
        Compute volatility (variation) in delta values.
        
        Uses coefficient of variation: std / mean
        High volatility = inconsistent play
        Low volatility = consistent (could be human or consistent cheater)
        """
        if len(deltas) < 2:
            return 0.0
        
        mean = statistics.mean(deltas)
        if mean == 0:
            return 0.0
        
        std = statistics.stdev(deltas)
        return std / mean
    
    def analyze(self, deltas: List[float]) -> TemporalAnalysis:
        """
        Perform full temporal analysis on a game.
        
        Args:
            deltas: List of Δ values per move
            
        Returns:
            TemporalAnalysis with all computed metrics
        """
        result = TemporalAnalysis()
        
        if not deltas:
            return result
        
        # Moving averages
        result.moving_averages = self.compute_moving_average(deltas)
        
        # Switch point detection
        result.switch_points = self.detect_switch_points(deltas, result.moving_averages)
        
        # Trend and volatility
        result.trend_slope = self.compute_trend(deltas)
        result.volatility = self.compute_volatility(deltas)
        
        # Flag suspicious sudden improvements
        for sp in result.switch_points:
            if sp.direction == "improved" and sp.magnitude >= self.switch_threshold * 1.5:
                result.is_suspicious = True
                break
        
        return result
