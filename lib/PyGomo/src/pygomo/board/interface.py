"""
Board interface definitions.

Abstract base class for board implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterator
from dataclasses import dataclass

from pygomo.protocol.models import Move


# Color constants
BLACK = 1
WHITE = 2
EMPTY = 0


@dataclass
class WinInfo:
    """Information about a winning line."""
    winner: int  # BLACK or WHITE
    line: list[Move]  # The 5 winning positions
    direction: str  # 'horizontal', 'vertical', 'diagonal', 'anti-diagonal'


class IBoard(ABC):
    """
    Abstract interface for Gomoku board implementations.
    
    Provides a common API for different board representations
    (BitBoard, SimpleBoard, etc.).
    """
    
    # Color constants
    BLACK = BLACK
    WHITE = WHITE
    EMPTY = EMPTY
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Board size (e.g., 15 for 15x15)."""
        ...
    
    @property
    @abstractmethod
    def move_count(self) -> int:
        """Number of moves played."""
        ...
    
    @property
    @abstractmethod
    def current_player(self) -> int:
        """Current player to move (BLACK or WHITE)."""
        ...
    
    @property
    @abstractmethod
    def last_move(self) -> Optional[Move]:
        """Last move played, or None if empty."""
        ...
    
    @property
    @abstractmethod
    def hash(self) -> int:
        """Zobrist hash of current position."""
        ...
    
    # --- Core Operations ---
    
    @abstractmethod
    def place(self, move: Move, color: Optional[int] = None) -> bool:
        """
        Place a stone on the board.
        
        Args:
            move: Position to place.
            color: Stone color (defaults to current player).
            
        Returns:
            True if successful, False if invalid.
        """
        ...
    
    @abstractmethod
    def remove(self, move: Move) -> bool:
        """
        Remove a stone from the board.
        
        Args:
            move: Position to remove.
            
        Returns:
            True if successful, False if empty.
        """
        ...
    
    @abstractmethod
    def get(self, move: Move) -> int:
        """
        Get the stone at position.
        
        Returns:
            EMPTY, BLACK, or WHITE.
        """
        ...
    
    @abstractmethod
    def is_empty(self, move: Move) -> bool:
        """Check if position is empty."""
        ...
    
    @abstractmethod
    def is_valid(self, move: Move) -> bool:
        """Check if position is within board bounds."""
        ...
    
    # --- Game State ---
    
    @abstractmethod
    def check_win(self, move: Optional[Move] = None) -> Optional[WinInfo]:
        """
        Check for a winning condition.
        
        Args:
            move: Check around this move (optimization).
                  If None, check entire board.
                  
        Returns:
            WinInfo if there's a winner, None otherwise.
        """
        ...
    
    @abstractmethod
    def is_full(self) -> bool:
        """Check if board is completely filled."""
        ...
    
    @abstractmethod
    def get_legal_moves(self) -> list[Move]:
        """Get all legal (empty) positions."""
        ...
    
    # --- History ---
    
    @abstractmethod
    def get_move_history(self) -> list[Move]:
        """Get ordered list of moves played."""
        ...
    
    @abstractmethod
    def undo(self) -> Optional[Move]:
        """
        Undo the last move.
        
        Returns:
            The undone move, or None if no moves to undo.
        """
        ...
    
    # --- Copy/Clone ---
    
    @abstractmethod
    def copy(self) -> "IBoard":
        """Create an independent copy of the board."""
        ...
    
    # --- Iteration ---
    
    def __iter__(self) -> Iterator[tuple[Move, int]]:
        """Iterate over all positions and their states."""
        for row in range(self.size):
            for col in range(self.size):
                move = Move((col, row))
                yield move, self.get(move)
    
    def stones(self, color: int) -> Iterator[Move]:
        """Iterate over all stones of a given color."""
        for move, c in self:
            if c == color:
                yield move


class IRenjuBoard(IBoard):
    """
    Extended interface for Renju rule support.
    
    Adds forbidden move detection for Black.
    """
    
    @abstractmethod
    def is_forbidden(self, move: Move) -> bool:
        """
        Check if a move is forbidden for Black (Renju rules).
        
        Forbidden patterns:
        - Double-three (two open threes)
        - Double-four (two fours)
        - Overline (6+ in a row)
        
        Note: Only applies to Black. White has no restrictions.
        
        Returns:
            True if forbidden, False otherwise.
        """
        ...
    
    @abstractmethod
    def get_forbidden_moves(self) -> list[Move]:
        """Get all currently forbidden moves for Black."""
        ...
