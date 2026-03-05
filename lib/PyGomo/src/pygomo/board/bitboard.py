"""
BitBoard implementation for Gomoku.

Uses Python's arbitrary precision integers as bitboards
for efficient board operations.
"""

from typing import Optional
from dataclasses import dataclass, field

from pygomo.protocol.models import Move
from pygomo.board.interface import IBoard, WinInfo, BLACK, WHITE, EMPTY
from pygomo.board.zobrist import ZobristHash, get_zobrist


@dataclass
class BitBoard(IBoard):
    """
    BitBoard representation for Gomoku.
    
    Uses two Python integers as bitboards:
    - `_black`: Bit set for each black stone
    - `_white`: Bit set for each white stone
    
    For a 15x15 board, each bitboard is 225 bits.
    Bit index = row * size + col.
    
    Features:
    - O(1) place, remove, is_empty operations
    - O(1) win detection using bit shifting
    - Zobrist hashing for transposition tables
    - Move history tracking
    
    Example:
        board = BitBoard(size=15)
        board.place(Move("h8"))           # Black plays
        board.place(Move("i8"))           # White plays
        board.place(Move("h7"))           # Black plays
        
        if board.check_win(Move("h8")):
            print("Black wins!")
    """
    
    _size: int = 15
    _black: int = 0
    _white: int = 0
    _move_count: int = 0
    _history: list[Move] = field(default_factory=list)
    _hash: int = 0
    _zobrist: ZobristHash = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize Zobrist hash table."""
        if self._zobrist is None:
            self._zobrist = get_zobrist(self._size)
        if self._hash == 0:
            self._hash = self._zobrist.empty_hash
    
    # --- Properties ---
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def move_count(self) -> int:
        return self._move_count
    
    @property
    def current_player(self) -> int:
        return BLACK if self._move_count % 2 == 0 else WHITE
    
    @property
    def last_move(self) -> Optional[Move]:
        return self._history[-1] if self._history else None
    
    @property
    def hash(self) -> int:
        return self._hash
    
    # --- Bit Operations ---
    
    def _index(self, move: Move) -> int:
        """Convert Move to bit index."""
        return move.row * self._size + move.col
    
    def _bit(self, move: Move) -> int:
        """Get bit mask for a position."""
        return 1 << self._index(move)
    
    def _get_board(self, color: int) -> int:
        """Get the bitboard for a color."""
        return self._black if color == BLACK else self._white
    
    def _set_board(self, color: int, value: int) -> None:
        """Set the bitboard for a color."""
        if color == BLACK:
            self._black = value
        else:
            self._white = value
    
    # --- Core Operations ---
    
    def place(self, move: Move, color: Optional[int] = None) -> bool:
        """Place a stone on the board."""
        if not self.is_valid(move):
            return False
        if not self.is_empty(move):
            return False
        
        if color is None:
            color = self.current_player
        
        bit = self._bit(move)
        
        # Set the bit
        if color == BLACK:
            self._black |= bit
        else:
            self._white |= bit
        
        # Update hash
        self._hash = self._zobrist.update(
            self._hash, move.col, move.row, color
        )
        
        # Update history
        self._history.append(move)
        self._move_count += 1
        
        return True
    
    def remove(self, move: Move) -> bool:
        """Remove a stone from the board."""
        if not self.is_valid(move):
            return False
        
        bit = self._bit(move)
        
        # Check which color to remove
        if self._black & bit:
            color = BLACK
            self._black &= ~bit
        elif self._white & bit:
            color = WHITE
            self._white &= ~bit
        else:
            return False  # Empty
        
        # Update hash
        self._hash = self._zobrist.update(
            self._hash, move.col, move.row, color
        )
        
        # Remove from history (if it's the last move)
        if self._history and self._history[-1] == move:
            self._history.pop()
        
        self._move_count -= 1
        return True
    
    def get(self, move: Move) -> int:
        """Get the stone at position."""
        if not self.is_valid(move):
            return EMPTY
        
        bit = self._bit(move)
        if self._black & bit:
            return BLACK
        if self._white & bit:
            return WHITE
        return EMPTY
    
    def is_empty(self, move: Move) -> bool:
        """Check if position is empty."""
        if not self.is_valid(move):
            return False
        bit = self._bit(move)
        return not ((self._black | self._white) & bit)
    
    def is_valid(self, move: Move) -> bool:
        """Check if position is within board bounds."""
        return 0 <= move.col < self._size and 0 <= move.row < self._size
    
    # --- Win Detection ---
    
    def check_win(self, move: Optional[Move] = None) -> Optional[WinInfo]:
        """
        Check for a winning condition.
        
        Uses bit shifting to detect 5-in-a-row patterns.
        If move is provided, only checks lines through that position.
        """
        if move is not None:
            color = self.get(move)
            if color == EMPTY:
                return None
            return self._check_win_at(move, color)
        
        # Check entire board
        for color in (BLACK, WHITE):
            stones = self._black if color == BLACK else self._white
            
            # Check all 4 directions using bit shifting
            for direction, shift in self._get_directions():
                if self._has_five(stones, shift):
                    # Find the winning line (slower, but only called on win)
                    line = self._find_winning_line(color, direction)
                    if line:
                        return WinInfo(winner=color, line=line, direction=direction)
        
        return None
    
    def _get_directions(self) -> list[tuple[str, int]]:
        """Get direction names and their bit shifts."""
        return [
            ("horizontal", 1),
            ("vertical", self._size),
            ("diagonal", self._size + 1),
            ("anti-diagonal", self._size - 1),
        ]
    
    def _has_five(self, stones: int, shift: int) -> bool:
        """
        Check if there are 5 consecutive bits in the given direction.
        
        Uses the "parallel prefix" technique:
        - AND stones with itself shifted by 1, 2, and 4 positions
        - If any bit remains, there's a 5-in-a-row
        """
        m = stones & (stones >> shift)
        m = m & (m >> shift)
        m = m & (m >> shift)
        m = m & (m >> shift)
        return m != 0
    
    def _check_win_at(self, move: Move, color: int) -> Optional[WinInfo]:
        """Check for win centered at a specific move."""
        for direction, (dx, dy) in [
            ("horizontal", (1, 0)),
            ("vertical", (0, 1)),
            ("diagonal", (1, 1)),
            ("anti-diagonal", (1, -1)),
        ]:
            count = 1
            line = [move]
            
            # Check positive direction
            for i in range(1, 5):
                x, y = move.col + dx * i, move.row + dy * i
                if 0 <= x < self._size and 0 <= y < self._size:
                    if self.get(Move((x, y))) == color:
                        count += 1
                        line.append(Move((x, y)))
                    else:
                        break
                else:
                    break
            
            # Check negative direction
            for i in range(1, 5):
                x, y = move.col - dx * i, move.row - dy * i
                if 0 <= x < self._size and 0 <= y < self._size:
                    if self.get(Move((x, y))) == color:
                        count += 1
                        line.insert(0, Move((x, y)))
                    else:
                        break
                else:
                    break
            
            if count >= 5:
                return WinInfo(winner=color, line=line[:5], direction=direction)
        
        return None
    
    def _find_winning_line(self, color: int, direction: str) -> Optional[list[Move]]:
        """Find the actual winning line positions."""
        dx, dy = {
            "horizontal": (1, 0),
            "vertical": (0, 1),
            "diagonal": (1, 1),
            "anti-diagonal": (1, -1),
        }[direction]
        
        for row in range(self._size):
            for col in range(self._size):
                if self.get(Move((col, row))) == color:
                    line = []
                    for i in range(5):
                        x, y = col + dx * i, row + dy * i
                        if 0 <= x < self._size and 0 <= y < self._size:
                            if self.get(Move((x, y))) == color:
                                line.append(Move((x, y)))
                            else:
                                break
                    if len(line) == 5:
                        return line
        return None
    
    # --- Game State ---
    
    def is_full(self) -> bool:
        """Check if board is completely filled."""
        return self._move_count >= self._size * self._size
    
    def get_legal_moves(self) -> list[Move]:
        """Get all legal (empty) positions."""
        occupied = self._black | self._white
        moves = []
        for row in range(self._size):
            for col in range(self._size):
                idx = row * self._size + col
                if not (occupied & (1 << idx)):
                    moves.append(Move((col, row)))
        return moves
    
    # --- History ---
    
    def get_move_history(self) -> list[Move]:
        """Get ordered list of moves played."""
        return list(self._history)
    
    def undo(self) -> Optional[Move]:
        """Undo the last move."""
        if not self._history:
            return None
        
        move = self._history[-1]
        self.remove(move)
        return move
    
    # --- Copy ---
    
    def copy(self) -> "BitBoard":
        """Create an independent copy of the board."""
        return BitBoard(
            _size=self._size,
            _black=self._black,
            _white=self._white,
            _move_count=self._move_count,
            _history=list(self._history),
            _hash=self._hash,
            _zobrist=self._zobrist,  # Share zobrist table
        )
    
    # --- Display ---
    
    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        for row in range(self._size - 1, -1, -1):
            line = f"{row + 1:2} "
            for col in range(self._size):
                stone = self.get(Move((col, row)))
                if stone == BLACK:
                    line += "X "
                elif stone == WHITE:
                    line += "O "
                else:
                    line += ". "
            lines.append(line)
        
        # Column labels
        header = "   " + " ".join(chr(ord('A') + c) for c in range(self._size))
        lines.append(header)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"BitBoard(size={self._size}, moves={self._move_count})"
