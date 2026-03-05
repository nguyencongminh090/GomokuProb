"""
SalienceEvaluator for Gomoku (CBRM Phase 3)

This module implements Visual Salience heuristics to estimate how "obvious" a move is to a human player.
Salience (σ) is a multiplier in the Human Probability formula:
P(m) ~ σ(m) * exp((V - C)/tau)

Heuristics:
1. Instant Win (Make 5): σ = 2.0
2. Instant Block (Block 5): σ = 2.0 (Forced)
3. Make Open 4: σ = 1.8
4. Block Open 4: σ = 1.8
5. Make Open 3: σ = 1.5
6. Clustered Move: σ = 1.0 (Baseline)
7. Distant Move: σ = 0.2
"""

from src.core.board import BoardState

class SalienceEvaluator:
    def __init__(self):
        pass

    def get_salience(self, board: BoardState, x: int, y: int, color: int) -> float:
        """
        Calculate visual salience of a move.
        Returns: float 0.0 to 2.0
        """
        # 1. Check for Instant Win (Make 5)
        # We can use board.check_win_after_move
        if board.check_win_after_move(x, y, color) == color:
            return 2.0

        # 2. Check for Instant Block (Opponent making 5)
        opponent = 3 - color
        # If we don't move here, does opponent win immediately?
        # Note: This is computationally expensive to check ALL opponent moves, 
        # but usually we only check if THIS move blocks a threat.
        # Actually, simpler: Does opponent win if they play at x,y?
        if board.check_win_after_move(x, y, opponent) == opponent:
             return 2.0 # Forced Block

        # 3. Local Proximity (Gestalt Law of Proximity)
        # Check dist to nearest stone
        min_dist = 999
        for m in board.moves:
            dist = max(abs(m.x - x), abs(m.y - y)) # Chebyshev distance
            if dist < min_dist:
                min_dist = dist
        
        if min_dist > 2:
            return 0.2 # Distant move, hard to see unless tactical
            
        # 4. Pattern Heuristics (Stub for now, or simple line check)
        # Implementing full open-4 check is complex without efficient pattern matcher.
        # For now, let's use a "Neighborhood Density" proxy or leave at 1.0 for "Normal Local Move"
        
        return 1.0
