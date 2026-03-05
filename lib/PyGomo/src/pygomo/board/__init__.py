"""
PyGomo Board Submodule.

Provides board representations for Gomoku:
- BitBoard: Fast bitwise operations
- RenjuBitBoard: With forbidden move detection

Example:
    from pygomo.board import BitBoard, RenjuBitBoard
    
    # Standard board
    board = BitBoard(size=15)
    board.place(Move("h8"))
    
    # Renju with forbidden detection
    renju = RenjuBitBoard(size=15)
    if renju.is_forbidden(Move("i9")):
        print("Forbidden!")
"""

from pygomo.board.interface import (
    IBoard,
    IRenjuBoard,
    WinInfo,
    BLACK,
    WHITE,
    EMPTY,
)

from pygomo.board.bitboard import BitBoard
from pygomo.board.renju import RenjuBitBoard
from pygomo.board.zobrist import ZobristHash, get_zobrist


__all__ = [
    # Interfaces
    "IBoard",
    "IRenjuBoard",
    "WinInfo",
    
    # Constants
    "BLACK",
    "WHITE",
    "EMPTY",
    
    # Implementations
    "BitBoard",
    "RenjuBitBoard",
    
    # Hashing
    "ZobristHash",
    "get_zobrist",
]
