"""
Core Domain Entities for Gomoku Board and Moves.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pygomo.board.bitboard import BitBoard
from pygomo.protocol.models import Move as AlgoMove

@dataclass
class Move:
    """Represents a single move on the board."""
    x: int
    y: int
    color: int  # 1=Black, 2=White
    notation: str = "" # e.g. "h8"
    comment: str = ""
    
    @property
    def tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def __str__(self):
        return f"{self.notation} ({'Black' if self.color == 1 else 'White'})"

    def to_algebraic(self, size: int = 15) -> str:
        """Return algebraic notation (e.g. h8)."""
        if self.notation:
            return self.notation
        
        # Convert coordinate to algebraic (Top-Left 0,0 origin)
        # x=0 -> a
        col_char = chr(ord('a') + self.x)
        # y=0 -> 15 (Top), y=14 -> 1 (Bottom)
        row_num = size - self.y
        
        return f"{col_char}{row_num}"

@dataclass
class GameInfo:
    """Metadata about the game."""
    black_player: str = "Unknown"
    white_player: str = "Unknown"
    result: str = "*" # 1-0, 0-1, 1/2-1/2, *
    date: str = ""
    event: str = ""
    rule: str = "Standard" # Standard, Renju, Freestyle

class BoardState:
    """
    Represents the full state of the board.
    Optimized for replay and analysis.
    """
    def __init__(self, size: int = 15):
        self.size = size
        self.moves: List[Move] = []
        self.info = GameInfo()
        self._bitboard = BitBoard(size)
        
    def add_move(self, x: int, y: int, color: int, notation: str = ""):
        self.moves.append(Move(x, y, color, notation))
        self._bitboard.place(AlgoMove((x, y)), color=color)
        
    def get_move_at(self, index: int) -> Optional[Move]:
        if 0 <= index < len(self.moves):
            return self.moves[index]
        return None
    
    
    def to_position_string(self) -> str:
        """Export to raw position string compatible with PyGomo."""
        # Simple coordinate list
        return " ".join([m.notation for m in self.moves])

    
    def check_win(self) -> int:
        """
        Check if there is a winner on the current board using persistent PyGomo BitBoard.
        Returns:
            1 if Black wins
            2 if White wins
            0 if no winner
        """
        # Use cached BitBoard directly
        win_info = self._bitboard.check_win()
        if win_info:
            return win_info.winner
            
        return 0

    def check_win_after_move(self, x: int, y: int, color: int) -> int:
        """
        Check if placing a move at x,y results in a win.
        Does not modify current board state.
        """
        # Lightweight copy of bitboard (just integer copy + history list)
        temp_bb = self._bitboard.copy()
        
        # Place move
        success = temp_bb.place(AlgoMove((x, y)), color=color)
        if not success:
            return 0 # Invalid move
            
        # Check win
        win_info = temp_bb.check_win()
        if win_info:
            return win_info.winner
            
        return 0
