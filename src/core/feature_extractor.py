"""
GomoProb V2: Feature Extractor

Builds feature vectors from game analysis for classification.
Aggregates all metrics from delta_model, player_model, and information_theory.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.core.delta_model import GameAnalysis, MoveAnalysis
from src.core.information_theory import InformationTheory


@dataclass
class FeatureVector:
    """
    Complete feature vector for a game, used for classification.
    """
    # Delta-based features
    mean_delta: float = 0.0         # Average winrate loss
    std_delta: float = 0.0          # Variance in quality
    lambda_mle: float = 0.0         # MLE λ from Exponential model
    accuracy_index: float = 0.0     # 1/λ (lower = more suspicious)
    near_optimal_ratio: float = 0.0 # Proportion of moves with Δ < ε
    
    # Complexity-weighted features
    weighted_mean_delta: float = 0.0
    
    # Information theory features
    delta_entropy: float = 0.0      # Entropy of delta distribution
    avg_position_entropy: float = 0.0  # Average entropy of positions
    
    # Game metadata
    total_moves: int = 0
    analyzed_moves: int = 0
    forced_moves: int = 0           # Moves with only 1 good option
    
    # Classification features
    p_cheat: float = 0.0            # From Bayesian model
    classification: str = "Unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/display."""
        return {
            "mean_delta": round(self.mean_delta, 4),
            "std_delta": round(self.std_delta, 4),
            "lambda_mle": round(self.lambda_mle, 2),
            "accuracy_index": round(self.accuracy_index, 4),
            "near_optimal_ratio": round(self.near_optimal_ratio, 3),
            "delta_entropy": round(self.delta_entropy, 3),
            "total_moves": self.total_moves,
            "analyzed_moves": self.analyzed_moves,
            "forced_moves": self.forced_moves,
            "p_cheat": round(self.p_cheat, 3),
            "classification": self.classification
        }


class FeatureExtractor:
    """
    Extracts feature vectors from game analysis data.
    """
    
    def __init__(self, near_optimal_threshold: float = 0.02):
        """
        Args:
            near_optimal_threshold: Δ below this counts as "near-optimal"
        """
        self.near_optimal_threshold = near_optimal_threshold
        self.info_theory = InformationTheory()
    
    def extract(
        self,
        game_analysis: GameAnalysis,
        softmax_vectors: Optional[List[List[float]]] = None
    ) -> FeatureVector:
        """
        Extract features from game analysis.
        
        Args:
            game_analysis: GameAnalysis from DeltaModel
            softmax_vectors: Optional list of P(move | position) vectors
            
        Returns:
            FeatureVector with all computed features
        """
        fv = FeatureVector()
        
        moves = game_analysis.moves
        if not moves:
            return fv
        
        # Copy basic stats from GameAnalysis
        fv.mean_delta = game_analysis.mean_delta
        fv.std_delta = game_analysis.std_delta
        fv.lambda_mle = game_analysis.lambda_mle
        fv.accuracy_index = game_analysis.accuracy_index
        fv.near_optimal_ratio = game_analysis.near_optimal_ratio
        fv.weighted_mean_delta = game_analysis.weighted_mean_delta
        
        # Game metadata
        fv.total_moves = len(moves)
        fv.analyzed_moves = len([m for m in moves if m.delta >= 0])
        fv.forced_moves = len([m for m in moves if m.is_forced])
        
        # Information theory features
        deltas = [m.delta for m in moves]
        fv.delta_entropy = self.info_theory.entropy_from_deltas(deltas)
        
        if softmax_vectors:
            fv.avg_position_entropy = self.info_theory.move_choice_entropy(softmax_vectors)
        
        return fv
    
    def extract_temporal_features(
        self,
        moves: List[MoveAnalysis],
        window_size: int = 5
    ) -> List[float]:
        """
        Extract temporal features (moving average of delta over game).
        
        Used for switch-point detection.
        
        Args:
            moves: List of MoveAnalysis
            window_size: Window for moving average
            
        Returns:
            List of moving average delta values
        """
        if not moves:
            return []
        
        deltas = [m.delta for m in moves]
        moving_avg = []
        
        for i in range(len(deltas)):
            start = max(0, i - window_size + 1)
            window = deltas[start:i+1]
            avg = sum(window) / len(window)
            moving_avg.append(avg)
        
        return moving_avg
