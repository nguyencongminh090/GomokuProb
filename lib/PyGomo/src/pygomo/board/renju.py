"""
Renju BitBoard with forbidden move detection.

Extends BitBoard to support Renju rules where Black
has restrictions on double-three, double-four, and overline.
"""

from typing import Optional
from dataclasses import dataclass, field

from pygomo.protocol.models import Move
from pygomo.board.interface import IRenjuBoard, WinInfo, BLACK, WHITE, EMPTY
from pygomo.board.bitboard import BitBoard
from pygomo.board.zobrist import ZobristHash, get_zobrist


@dataclass
class RenjuBitBoard(BitBoard, IRenjuBoard):
    """
    BitBoard with Renju forbidden move detection.
    
    In Renju, Black (the first player) has restrictions:
    - Overline (6+): Forbidden
    - Double-four: Forbidden (two fours in different directions)
    - Double-three: Forbidden (two open threes in different directions)
    
    White has no restrictions and wins with 5 or more.
    
    Example::

        board = RenjuBitBoard(size=15)
        board.place(Move("h8"))  # Black
        
        # Check if a move would be forbidden
        if board.is_forbidden(Move("i9")):
            print("Move is forbidden!")
    """
    
    def is_forbidden(self, move: Move) -> bool:
        """
        Check if a move is forbidden for Black (Renju rules).
        
        Only applies when it's Black's turn.
        
        Args:
            move: The move to check.
            
        Returns:
            True if the move is forbidden for Black.
        """
        # Only Black has restrictions
        if self.current_player != BLACK:
            return False
        
        if not self.is_empty(move):
            return False
        
        # Check overline (6+)
        if self._check_overline(move, BLACK):
            return True
        
        # Check double-four
        if self._check_double_four(move, BLACK):
            return True
        
        # Check double-three (open threes)
        if self._check_double_three(move, BLACK):
            return True
        
        return False
    
    def get_forbidden_moves(self) -> list[Move]:
        """Get all currently forbidden moves for Black."""
        if self.current_player != BLACK:
            return []
        
        forbidden = []
        for move in self.get_legal_moves():
            if self.is_forbidden(move):
                forbidden.append(move)
        return forbidden
    
    def place(self, move: Move, color: Optional[int] = None) -> bool:
        """
        Place a stone, respecting Renju forbidden rules.
        
        If it's Black's turn and the move is forbidden, return False.
        """
        if color is None:
            color = self.current_player
        
        # Check forbidden only for Black's own moves
        if color == BLACK and self.is_forbidden(move):
            return False
        
        return super().place(move, color)
    
    def check_win(self, move: Optional[Move] = None) -> Optional[WinInfo]:
        """
        Check for win with Renju overline rule.
        
        Black's 6+ in a row is forbidden (not a win).
        White's 5+ is a win.
        """
        result = super().check_win(move)
        
        if result and result.winner == BLACK:
            # Check if it's exactly 5 (not overline)
            if len(result.line) > 5:
                return None  # Forbidden overline
            
            # Verify it's exactly 5, not part of 6+
            if move:
                for direction, (dx, dy) in [
                    ("horizontal", (1, 0)),
                    ("vertical", (0, 1)),
                    ("diagonal", (1, 1)),
                    ("anti-diagonal", (1, -1)),
                ]:
                    if result.direction == direction:
                        count = self._count_line(move, dx, dy, BLACK)
                        if count > 5:
                            return None  # Overline
        
        return result
    
    # --- Pattern Detection ---
    
    def _count_line(self, move: Move, dx: int, dy: int, color: int) -> int:
        """Count consecutive stones in a line through move."""
        count = 1
        
        # Positive direction
        x, y = move.col + dx, move.row + dy
        while 0 <= x < self._size and 0 <= y < self._size:
            if self.get(Move((x, y))) == color:
                count += 1
                x += dx
                y += dy
            else:
                break
        
        # Negative direction
        x, y = move.col - dx, move.row - dy
        while 0 <= x < self._size and 0 <= y < self._size:
            if self.get(Move((x, y))) == color:
                count += 1
                x -= dx
                y -= dy
            else:
                break
        
        return count
    
    def _check_overline(self, move: Move, color: int) -> bool:
        """Check if placing at move creates 6+ in a row."""
        # Temporarily place the stone
        bit = self._bit(move)
        self._black |= bit
        
        has_overline = False
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            if self._count_line(move, dx, dy, color) >= 6:
                has_overline = True
                break
        
        # Remove the temporary stone
        self._black &= ~bit
        
        return has_overline
    
    def _check_double_four(self, move: Move, color: int) -> bool:
        """Check if placing at move creates double-four."""
        # Temporarily place the stone
        bit = self._bit(move)
        self._black |= bit
        
        four_count = 0
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            if self._is_four(move, dx, dy, color):
                four_count += 1
        
        # Remove the temporary stone
        self._black &= ~bit
        
        return four_count >= 2
    
    def _check_double_three(self, move: Move, color: int) -> bool:
        """Check if placing at move creates double open-three."""
        # Temporarily place the stone
        bit = self._bit(move)
        self._black |= bit
        
        three_count = 0
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            if self._is_open_three(move, dx, dy, color):
                three_count += 1
        
        # Remove the temporary stone
        self._black &= ~bit
        
        return three_count >= 2
    
    def _is_four(self, move: Move, dx: int, dy: int, color: int) -> bool:
        """
        Check if there's a four (4 consecutive with one end open)
        in the given direction through the move.
        """
        # Count consecutive stones including move
        count = 1
        
        # Check positive direction
        end_pos = 0
        end_neg = 0
        
        x, y = move.col + dx, move.row + dy
        while 0 <= x < self._size and 0 <= y < self._size:
            if self.get(Move((x, y))) == color:
                count += 1
                x += dx
                y += dy
            else:
                if self.get(Move((x, y))) == EMPTY:
                    end_pos = 1  # Open end
                break
        
        x, y = move.col - dx, move.row - dy
        while 0 <= x < self._size and 0 <= y < self._size:
            if self.get(Move((x, y))) == color:
                count += 1
                x -= dx
                y -= dy
            else:
                if self.get(Move((x, y))) == EMPTY:
                    end_neg = 1  # Open end
                break
        
        # Four: 4 stones with at least one open end
        return count == 4 and (end_pos + end_neg >= 1)
    
    def _is_open_three(self, move: Move, dx: int, dy: int, color: int) -> bool:
        """
        Check if there's an open three in the given direction.
        
        Open three: 3 consecutive with both ends open,
        such that it can become an open four with one move.
        """
        count = 1
        end_pos = 0
        end_neg = 0
        
        x, y = move.col + dx, move.row + dy
        while 0 <= x < self._size and 0 <= y < self._size:
            if self.get(Move((x, y))) == color:
                count += 1
                x += dx
                y += dy
            else:
                if self.get(Move((x, y))) == EMPTY:
                    end_pos = 1
                break
        
        x, y = move.col - dx, move.row - dy
        while 0 <= x < self._size and 0 <= y < self._size:
            if self.get(Move((x, y))) == color:
                count += 1
                x -= dx
                y -= dy
            else:
                if self.get(Move((x, y))) == EMPTY:
                    end_neg = 1
                break
        
        # Open three: 3 stones with both ends open
        return count == 3 and end_pos == 1 and end_neg == 1
    
    def copy(self) -> "RenjuBitBoard":
        """Create an independent copy."""
        return RenjuBitBoard(
            _size=self._size,
            _black=self._black,
            _white=self._white,
            _move_count=self._move_count,
            _history=list(self._history),
            _hash=self._hash,
            _zobrist=self._zobrist,
        )
