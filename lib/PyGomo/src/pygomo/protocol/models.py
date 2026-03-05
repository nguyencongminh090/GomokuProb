"""
Protocol data models.

This module defines data classes for moves, evaluations,
search information, and game state.
"""

from dataclasses import dataclass, field
from typing import Union, Optional
import math
import re


@dataclass
class Move:
    """
    Represents a board position/move.
    
    Supports multiple input formats:
        - Tuple: (7, 8)
        - String numeric: "7,8"
        - String algebraic: "h8"
    
    Coordinates are 0-indexed internally.
    """
    col: int
    row: int
    
    def __init__(self, move: Union[tuple[int, int], str]):
        """
        Initialize a Move from various input formats.
        
        Args:
            move: Move in tuple, numeric string, or algebraic notation.
        """
        if isinstance(move, tuple) and len(move) == 2:
            self.col, self.row = move
        elif isinstance(move, str):
            move = move.strip().replace(" ", "")
            if "," in move:
                # Numeric format: "7,8"
                parts = move.split(",")
                self.col = int(parts[0])
                self.row = int(parts[1])
            else:
                # Algebraic format: "h8"
                self.col = ord(move[0].lower()) - ord('a')
                self.row = int(move[1:]) - 1
        else:
            raise ValueError(f"Invalid move format: {move}")
    
    def to_tuple(self) -> tuple[int, int]:
        """Return as (col, row) tuple."""
        return (self.col, self.row)
    
    def to_numeric(self) -> str:
        """Return as numeric string 'col,row'."""
        return f"{self.col},{self.row}"
    
    def to_algebraic(self) -> str:
        """Return as algebraic notation 'a1' style."""
        return f"{chr(ord('a') + self.col)}{self.row + 1}"
    
    def __str__(self) -> str:
        return self.to_algebraic()
    
    def __repr__(self) -> str:
        return f"<Move {self.to_algebraic()}>"
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Move):
            return self.col == other.col and self.row == other.row
        return False
    
    def __hash__(self) -> int:
        return hash((self.col, self.row))


@dataclass
class Evaluate:
    """
    Engine evaluation score with winrate conversion.
    
    Supports both centipawn scores and mate scores.
    """
    raw_value: str  # Original string value
    
    # Default scaling factor (can be configured)
    DEFAULT_SCALING_FACTOR: float = 400.0
    
    def score(self) -> Union[int, float, "MateScore"]:
        """
        Get the evaluation score.
        
        Returns:
            Numeric score, or MateScore for mate positions.
        """
        value = self.raw_value.strip()
        if 'm' in value.lower():
            return MateScore(value)
        
        # Try to parse as integer or float
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return 0
    
    def winrate(self, scaling_factor: Optional[float] = None) -> float:
        """
        Convert evaluation to winrate using sigmoid function.
        
        Formula: winrate = 1 / (1 + e^(-eval / scaling_factor))
        
        Args:
            scaling_factor: Controls curve steepness. Higher = flatter.
            
        Returns:
            Winrate in range [0, 1].
        """
        sf = scaling_factor or self.DEFAULT_SCALING_FACTOR
        score = self.score()
        
        if isinstance(score, MateScore):
            # Mate scores: close to 1.0 for winning, 0.0 for losing
            steps = score.steps()
            if score.is_winning():
                # Higher mate steps = slightly lower winrate
                return 1.0 - (0.01 * min(steps, 50)) / 100
            else:
                return (0.01 * min(steps, 50)) / 100
        
        # Standard sigmoid conversion
        try:
            return 1.0 / (1.0 + math.exp(-float(score) / sf))
        except (OverflowError, ValueError):
            return 0.5
    
    def winrate_percent(self, scaling_factor: Optional[float] = None) -> float:
        """Get winrate as percentage [0, 100]."""
        return self.winrate(scaling_factor) * 100
    
    def is_mate(self) -> bool:
        """Check if this is a mate score."""
        return 'm' in self.raw_value.lower()
    
    def is_winning(self) -> bool:
        """Check if position is winning (positive mate)."""
        return self.is_mate() and self.raw_value.startswith('+')
    
    def is_losing(self) -> bool:
        """Check if position is losing (negative mate)."""
        return self.is_mate() and self.raw_value.startswith('-')


@dataclass
class MateScore:
    """Represents a mate-in-N score."""
    raw_value: str
    
    def steps(self) -> int:
        """Get the number of moves to mate."""
        # Extract number from strings like "+m5", "-m3", "m7"
        match = re.search(r'\d+', self.raw_value)
        return int(match.group()) if match else 0
    
    def is_winning(self) -> bool:
        """Check if this is a winning mate."""
        return not self.raw_value.startswith('-')
    
    def __str__(self) -> str:
        sign = "" if self.is_winning() else "-"
        return f"{sign}M{self.steps()}"


@dataclass
class SearchInfo:
    """
    Parsed search information from engine MESSAGE output.
    
    Contains realtime search progress data including depth,
    evaluation, nodes, speed, and principal variation.
    """
    depth: int = 0
    sel_depth: int = 0
    eval: Evaluate = field(default_factory=lambda: Evaluate("0"))
    nodes: int = 0
    nps: int = 0  # Nodes per millisecond
    time_ms: int = 0
    pv: list[Move] = field(default_factory=list)
    multipv: int = 1
    
    @property
    def winrate(self) -> float:
        """Get winrate from evaluation."""
        return self.eval.winrate()
    
    @property
    def winrate_percent(self) -> float:
        """Get winrate as percentage."""
        return self.eval.winrate_percent()
    
    def __str__(self) -> str:
        pv_str = " ".join(str(m) for m in self.pv[:5])
        return (
            f"depth {self.depth}-{self.sel_depth} "
            f"eval {self.eval.raw_value} "
            f"({self.winrate_percent:.1f}%) "
            f"pv {pv_str}"
        )


@dataclass
class PlayResult:
    """
    Result of a thinking command (TURN, BEGIN, etc.).
    
    Contains the best move and all collected search information.
    """
    move: Move
    search_info: Optional[SearchInfo] = None  # Last/final search info
    all_info: list[SearchInfo] = field(default_factory=list)
    
    @property
    def eval(self) -> Optional[Evaluate]:
        """Get evaluation from search info."""
        return self.search_info.eval if self.search_info else None
    
    @property
    def winrate(self) -> Optional[float]:
        """Get winrate from search info."""
        return self.search_info.winrate if self.search_info else None
    
    @property
    def pv(self) -> list[Move]:
        """Get principal variation."""
        return self.search_info.pv if self.search_info else []
    
    def __str__(self) -> str:
        if self.search_info:
            return f"PlayResult({self.move}, eval={self.search_info.eval.raw_value})"
        return f"PlayResult({self.move})"


@dataclass
class BoardPosition:
    """
    Represents a board position with move history.
    
    Used for BOARD command to set up a position.
    """
    moves: list[tuple[Move, int]] = field(default_factory=list)
    
    # Color constants (matching Gomocup protocol)
    SELF = 1
    OPPONENT = 2
    WALL = 3
    
    def add_move(self, move: Move, color: int) -> None:
        """
        Add a move to the position.
        
        Args:
            move: The move to add.
            color: 1=self, 2=opponent, 3=wall.
        """
        if color not in (self.SELF, self.OPPONENT, self.WALL):
            raise ValueError(f"Invalid color: {color}")
        self.moves.append((move, color))
    
    def to_protocol_string(self) -> str:
        """
        Convert to Gomocup BOARD format.
        
        Returns:
            String in format "x,y,color\\n..." with DONE terminator.
        """
        lines = []
        for move, color in self.moves:
            lines.append(f"{move.col},{move.row},{color}")
        lines.append("DONE")
        return "\n".join(lines)
