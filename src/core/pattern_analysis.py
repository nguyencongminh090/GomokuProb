from typing import Dict, List, Tuple, Counter
from collections import defaultdict
import datetime
from .rapfi_patterns import RapfiPattern
from .board import BoardState, Move

class PatternAnalyzer:
    """
    Analyzes which static patterns a player 'activates' or 'blocks' with their moves.
    Maintains a histogram of move choices (Offensive vs Defensive patterns).
    """
    
    def __init__(self):
        self.history: List[dict] = []
        # (OwnPattern, OppPattern) -> Count
        self.pattern_counts: Counter[Tuple[int, int]] = Counter()
        
    def analyze_move(self, 
                     prev_static_eval: dict, 
                     move: Move, 
                     is_black: bool) -> dict:
        """
        Determine what patterns were present at the move's location BEFORE the move was made.
        
        Args:
            prev_static_eval: JSON result from YXSTATICJSON on the PRE-MOVE board.
            move: The move made by the player.
            is_black: True if the player moving is Black.
            
        Returns:
            dict containing 'own_pattern' and 'opp_pattern' codes.
        """
        if not prev_static_eval or 'patterns' not in prev_static_eval:
            return None
            
        x, y = move.x, move.y
        patterns = prev_static_eval['patterns']
        
        # Colors in JSON are "black" and "white"
        player_key = "black" if is_black else "white"
        opp_key = "white" if is_black else "black"
        
        try:
            # Get the pattern code at the move position
            # Note: Rapfi y is row, x is col
            own_code = patterns[player_key][y][x]
            opp_code = patterns[opp_key][y][x]
        except (IndexError, KeyError):
            return None
            
        result = {
            'move': move.to_algebraic(),
            'own_pattern': own_code,
            'own_desc': RapfiPattern.describe(own_code),
            'opp_pattern': opp_code,
            'opp_desc': RapfiPattern.describe(opp_code),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Update Stats
        self.pattern_counts[(own_code, opp_code)] += 1
        self.history.append(result)
        
        return result

    def get_summary(self) -> str:
        """Text summary of pattern usage."""
        if not self.pattern_counts:
            return "No pattern data."
            
        # Top 3 combinations
        top = self.pattern_counts.most_common(5)
        lines = ["Top Patterns Used:"]
        for (own, opp), count in top:
            o_name = RapfiPattern.describe(own)
            e_name = RapfiPattern.describe(opp)
            lines.append(f"  Own: {o_name:<10} | Block: {e_name:<10} : {count} times")
        return "\n".join(lines)
