"""
Context-Aware Complexity Module.

Calculates position complexity using all candidate moves and their relation
to the previous game state (context).
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ComplexityResult:
    """Result of complexity calculation."""
    complexity: float           # Overall complexity score 0.0 - 1.0
    impact_factor: float        # How much best move changes the game
    variance_factor: float      # How different are the options
    criticality_factor: float   # Is the best move uniquely important?
    phase_factor: float         # Game phase relevance
    
    # Detailed metrics
    deltas: List[float]         # Delta of each candidate vs prev_wr
    best_delta: float
    mean_delta: float
    delta_spread: float
    improvement_ratio: float    # % of moves that improve position
    
    # V3 Opponent Difficulty metrics (Paper Sec 4.4)
    opp_factor: float           # F_opp: 0.0 to 1.0
    adjusted_complexity: float  # C_adj: C * max(F_opp, 0.10)
    final_complexity: float     # C_final: final value after trivial override



@dataclass
class OpponentAnalysis:
    """
    Analysis of opponent's position AFTER human move.
    
    These metrics help understand the strategic quality of human's move
    by analyzing what options the opponent has in response.
    """
    # Core metrics
    opp_variance: float     # Variance in opponent's options (0-1)
    opp_best: float         # Opponent's best achievable WR (0-1)
    forcing_level: float    # How many moves are "killed" (WR < 10%)
    pressure: float         # WR reduction caused by human's move
    viable_count: int       # Number of "acceptable" moves (WR > 30%)
    
    # Derived insights
    move_quality: str       # "Tactical", "Strategic", "Passive", "Blunder"
    is_forcing: bool        # Did human create a forcing sequence?
    
    # Raw data
    opp_candidates: List[float]  # Opponent's candidate WRs


def calculate_opponent_metrics(
    opp_candidate_winrates: List[float],
    prev_opp_best: float = 0.5
) -> OpponentAnalysis:
    """
    Calculate 5 opponent analysis metrics.
    
    Args:
        opp_candidate_winrates: List of winrates from opponent's NBEST 5
        prev_opp_best: Opponent's best WR BEFORE human's move (for pressure calc)
        
    Returns:
        OpponentAnalysis with all 5 metrics
    """
    if not opp_candidate_winrates:
        return OpponentAnalysis(
            opp_variance=0.0,
            opp_best=0.5,
            forcing_level=0.0,
            pressure=0.0,
            viable_count=0,
            move_quality="Unknown",
            is_forcing=False,
            opp_candidates=[]
        )
    
    winrates = opp_candidate_winrates[:5]  # Max 5 candidates
    n = len(winrates)
    
    # 1. opp_variance: How different are opponent's options?
    mean_wr = sum(winrates) / n
    variance = sum((wr - mean_wr) ** 2 for wr in winrates) / n
    # Normalize to 0-1 (max variance is 0.25 when half are 0 and half are 1)
    opp_variance = min(1.0, variance / 0.25)
    
    # 2. opp_best: Opponent's best achievable winrate
    opp_best = max(winrates)
    
    # 3. forcing_level: Proportion of "killed" moves (WR < 10%)
    killed_moves = sum(1 for wr in winrates if wr < 0.10)
    forcing_level = killed_moves / n
    
    # 4. pressure: How much WR did opponent lose due to human's move?
    pressure = max(0.0, prev_opp_best - opp_best)
    
    # 5. viable_count: Number of moves with WR > 30%
    viable_count = sum(1 for wr in winrates if wr > 0.30)
    
    # Derived: is_forcing
    is_forcing = forcing_level >= 0.6 or viable_count <= 1
    
    # Derived: move_quality
    if opp_best < 0.20:
        move_quality = "Winning"      # Opponent has no good response
    elif is_forcing:
        move_quality = "Tactical"     # Forcing sequence created
    elif opp_variance > 0.3:
        move_quality = "Strategic"    # Complex position for opponent
    elif pressure > 0.15:
        move_quality = "Aggressive"   # Applied significant pressure
    elif opp_best > 0.70:
        move_quality = "Blunder"      # Gave opponent great position
    else:
        move_quality = "Passive"      # Neutral move
    
    return OpponentAnalysis(
        opp_variance=opp_variance,
        opp_best=opp_best,
        forcing_level=forcing_level,
        pressure=pressure,
        viable_count=viable_count,
        move_quality=move_quality,
        is_forcing=is_forcing,
        opp_candidates=winrates
    )



W_REF = 0.40        # Eq. (15) calibration threshold
ALPHA_MIN = 0.10    # floor in Eq. (16)

def compute_f_opp(w_opp_best: float) -> float:
    """Calculate Opponent Difficulty Factor (Eq. 15)."""
    return min(w_opp_best / W_REF, 1.0)

def compute_c_final(c_raw: float, f_opp: float, trivial_override: bool) -> tuple[float, float]:
    """Calculate C_adj (Eq. 16) and C_final (Eq. 17). Returns (c_adj, c_final)."""
    c_adj = c_raw * max(f_opp, ALPHA_MIN)
    if trivial_override:
        return c_adj, 0.10
    return c_adj, c_adj

def calculate_complexity(
    candidate_winrates: List[float], 
    prev_winrate: float,
    w_opp_best: float = 0.50
) -> ComplexityResult:
    """
    Calculate Context-Aware Complexity using all candidates.
    
    Args:
        candidate_winrates: Winrates of top N candidates (sorted desc usually)
        prev_winrate: Player's winrate BEFORE this move (after opponent's last move)
        
    Returns:
        ComplexityResult with complexity score and detailed metrics
    """
    if not candidate_winrates:
        return ComplexityResult(
            complexity=0.5, impact_factor=0.5, variance_factor=0.5,
            criticality_factor=0.5, phase_factor=0.5,
            deltas=[], best_delta=0.0, mean_delta=0.0,
            delta_spread=0.0, improvement_ratio=0.0,
            opp_factor=1.0, adjusted_complexity=0.5, final_complexity=0.5
        )
    
    n = len(candidate_winrates)
    winrates = list(candidate_winrates)
    
    # 1. Delta Vector: Compare each candidate with previous winrate
    deltas = [wr - prev_winrate for wr in winrates]
    
    # 2. Delta Metrics
    best_delta = max(deltas)
    worst_delta = min(deltas)
    mean_delta = sum(deltas) / n
    delta_spread = best_delta - worst_delta
    
    # 3. Improvement Ratio: How many moves improve the position?
    improving_count = sum(1 for d in deltas if d > 0.02)  # > 2% improvement
    improvement_ratio = improving_count / n
    
    # 4. Best Move Criticality: Is the best move uniquely important?
    if delta_spread > 0.01:
        best_criticality = (best_delta - mean_delta) / delta_spread
        best_criticality = max(0.0, min(1.0, best_criticality))
    else:
        best_criticality = 0.0
    
    # === Complexity Factors ===
    
    # A. Impact Factor: Is there a breakthrough opportunity?
    # Normalize by 25% (big swing threshold)
    impact = min(abs(best_delta) / 0.25, 1.0)
    
    # B. Variance Factor: How different are the options?
    # Normalize by 40% (max typical spread)
    variance = min(delta_spread / 0.40, 1.0)
    
    # C. Criticality Factor: Is the best move uniquely special?
    criticality = best_criticality
    
    # D. Phase Factor: Game phase relevance
    best_wr = max(winrates)
    if best_wr > 0.85:
        # Winning big - easy to convert
        phase = 0.3
    elif best_wr < 0.15:
        # Losing big - hopeless
        phase = 0.2
    else:
        # Balanced game - every move matters
        balance = 1.0 - abs(best_wr - 0.5) * 2
        phase = 0.5 + 0.5 * balance
    
    # === TRIVIAL POSITION CHECK ===
    # If variance is near 0 (all moves are ~equal) AND position is clearly won/lost,
    # then there's no real decision to make → trivial complexity
    is_trivially_won = best_wr > 0.95 and variance < 0.05
    is_trivially_lost = best_wr < 0.05 and variance < 0.05
    is_trivial = is_trivially_won or is_trivially_lost
    
    # === Combined Raw Complexity ===
    # Weight variance MORE heavily - low variance = easy decision
    c_raw = (
        0.15 * impact +
        0.40 * variance +
        0.20 * criticality +
        0.25 * phase
    )
    
    # === Opponent Adjustment (V3 Eq. 15, 16, 17) ===
    f_opp = compute_f_opp(w_opp_best)
    c_adj, c_final = compute_c_final(c_raw, f_opp, is_trivial)
    
    return ComplexityResult(
        complexity=c_raw, # keep raw for legacy / internal structure
        impact_factor=impact,
        variance_factor=variance,
        criticality_factor=criticality,
        phase_factor=phase,
        deltas=deltas,
        best_delta=best_delta,
        mean_delta=mean_delta,
        delta_spread=delta_spread,
        improvement_ratio=improvement_ratio,
        opp_factor=f_opp,
        adjusted_complexity=c_adj,
        final_complexity=c_final
    )


def get_complexity_description(complexity: float) -> str:
    """Get human-readable description of complexity level."""
    if complexity >= 0.75:
        return "Very High"
    elif complexity >= 0.55:
        return "High"
    elif complexity >= 0.35:
        return "Medium"
    elif complexity >= 0.20:
        return "Low"
    else:
        return "Trivial"


import math

def calculate_accuracy(delta: float, complexity: float, opponent_complexity: float = 0.0) -> float:
    """
    Calculate Context-Aware Accuracy + Strategic Bonus.
    
    Args:
        delta: Winrate loss (best_wr - played_wr), 0.0 to 1.0
        complexity: Position complexity (0.0 to 1.0)
        opponent_complexity: Difficulty creating for opponent (0.0 to 1.0)
        
    Returns:
        Accuracy score 0.0 to 1.0 (can exceed 1.0 briefly but capped)
    """
    # Near-optimal moves get full accuracy
    if delta <= 0.02:
        return 1.0
    
    # 1. Base Context-Aware Accuracy
    # Complexity modifier: 
    # High complexity (0.8) -> factor = 0.76 (more lenient)
    complexity_factor = 1.0 - complexity * 0.3
    effective_delta = delta * complexity_factor
    
    base_accuracy = math.exp(-5 * effective_delta)
    
    # 2. Strategic Bonus (New in V2.1)
    # Reward moves that create high complexity for opponent (Opponent Variance)
    # Only if the move is decent (base_accuracy > 0.4) to avoid boosting blunders
    if base_accuracy > 0.4 and opponent_complexity > 0.1:
        # Bonus up to 15% for max complexity (variance=1.0)
        bonus = opponent_complexity * 0.15
        base_accuracy = base_accuracy * (1.0 + bonus)
    
    return max(0.0, min(1.0, base_accuracy))


def get_accuracy_grade(accuracy: float) -> str:
    """Get letter grade for accuracy."""
    if accuracy >= 0.95:
        return "S"  # Perfect
    elif accuracy >= 0.85:
        return "A"
    elif accuracy >= 0.70:
        return "B"
    elif accuracy >= 0.55:
        return "C"
    elif accuracy >= 0.40:
        return "D"
    else:
        return "F"
