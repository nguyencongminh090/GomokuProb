"""
Zobrist hashing for board positions.

Provides fast incremental hash updates for transposition tables.
"""

import random
from typing import Optional


class ZobristHash:
    """
    Zobrist hashing for Gomoku boards.
    
    Uses random 64-bit integers for each (position, color) pair.
    Hash is XORed incrementally as stones are added/removed.
    
    Example:
        zobrist = ZobristHash(size=15)
        
        hash_val = zobrist.empty_hash
        hash_val = zobrist.update(hash_val, Move("h8"), BLACK)  # Add stone
        hash_val = zobrist.update(hash_val, Move("h8"), BLACK)  # Remove stone (XOR again)
    """
    
    def __init__(self, size: int = 15, seed: Optional[int] = None):
        """
        Initialize Zobrist hash tables.
        
        Args:
            size: Board size.
            seed: Random seed for reproducibility.
        """
        self.size = size
        self.total_squares = size * size
        
        # Initialize RNG
        rng = random.Random(seed if seed is not None else 0x5A0B12157)
        
        # Generate random values for each (position, color) pair
        # table[color][position] -> 64-bit random int
        # color: 0=unused, 1=BLACK, 2=WHITE
        self._table = [
            [0] * self.total_squares,  # Unused (index 0)
            [rng.getrandbits(64) for _ in range(self.total_squares)],  # BLACK
            [rng.getrandbits(64) for _ in range(self.total_squares)],  # WHITE
        ]
        
        # Random value for "side to move" (optional, for distinguishing turns)
        self._side_to_move = rng.getrandbits(64)
        
        # Empty board hash
        self._empty_hash = rng.getrandbits(64)
    
    @property
    def empty_hash(self) -> int:
        """Hash value for an empty board."""
        return self._empty_hash
    
    def _index(self, col: int, row: int) -> int:
        """Convert (col, row) to linear index."""
        return row * self.size + col
    
    def get_value(self, col: int, row: int, color: int) -> int:
        """
        Get the Zobrist value for a position and color.
        
        Args:
            col: Column (0-indexed).
            row: Row (0-indexed).
            color: Stone color (1=BLACK, 2=WHITE).
            
        Returns:
            64-bit random integer for this (position, color).
        """
        if color not in (1, 2):
            return 0
        idx = self._index(col, row)
        return self._table[color][idx]
    
    def update(self, current_hash: int, col: int, row: int, color: int) -> int:
        """
        Update hash when adding or removing a stone.
        
        XOR is self-inverse: adding then removing gives original hash.
        
        Args:
            current_hash: Current hash value.
            col: Column of stone.
            row: Row of stone.
            color: Color of stone.
            
        Returns:
            Updated hash value.
        """
        return current_hash ^ self.get_value(col, row, color)
    
    def toggle_side(self, current_hash: int) -> int:
        """
        Toggle the side-to-move bit.
        
        Optional: use if you want different hashes for same position
        but different side to move.
        """
        return current_hash ^ self._side_to_move
    
    def compute_full(self, stones: list[tuple[int, int, int]]) -> int:
        """
        Compute hash from scratch for a list of stones.
        
        Args:
            stones: List of (col, row, color) tuples.
            
        Returns:
            Complete hash value.
        """
        h = self._empty_hash
        for col, row, color in stones:
            h ^= self.get_value(col, row, color)
        return h


# Singleton instances for common sizes (optional optimization)
_zobrist_cache: dict[int, ZobristHash] = {}


def get_zobrist(size: int = 15) -> ZobristHash:
    """
    Get a Zobrist hash instance for the given board size.
    
    Uses a cache to reuse instances for the same size.
    """
    if size not in _zobrist_cache:
        _zobrist_cache[size] = ZobristHash(size=size)
    return _zobrist_cache[size]
