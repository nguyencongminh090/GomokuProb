"""
Pytest configuration and fixtures.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def empty_board():
    """Create an empty 15x15 BitBoard."""
    from pygomo.board import BitBoard
    return BitBoard(_size=15)


@pytest.fixture
def empty_board_19():
    """Create an empty 19x19 BitBoard."""
    from pygomo.board import BitBoard
    return BitBoard(_size=19)


@pytest.fixture
def empty_renju_board():
    """Create an empty 15x15 RenjuBitBoard."""
    from pygomo.board import RenjuBitBoard
    return RenjuBitBoard(_size=15)
