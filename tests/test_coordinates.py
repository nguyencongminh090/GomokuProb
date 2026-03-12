"""
Tests for coordinate system conventions.

Two separate conventions coexist:
1. GameParser (human input): a15=(0,0), a1=(0,14), h8=(7,7)  [Bottom-Left = row 1]
2. PyGomo/Engine (Rapfi output): a1=(0,0),  a15=(0,14), h8=(7,7) [Top-Left = row 1]

Both produce (7,7) for the center cell, but their algebraic notation differs.
h8 in human notation = h8 in engine notation ONLY because 15 is odd (center is symmetric).

IMPORTANT: Do NOT try to unify these into the same string representation.
They serve different purposes - one is for human input, one is for engine I/O.
The INTERNAL (x,y) values are always consistent (Top-Left 0,0).
"""

import sys
import os
import unittest

# Add project root and PyGomo to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
PYGOMO_PATH = os.path.join(PROJECT_ROOT, "lib", "PyGomo", "src")
sys.path.insert(0, PYGOMO_PATH)

from src.services.parser import GameParser
from src.core.board import Move as CoreMove
from pygomo.protocol.models import Move as PyGomoMove


class TestGameParserCoordinates(unittest.TestCase):
    """GameParser uses Renju standard: Row 1 = Bottom, Row 15 = Top."""

    def test_corners(self):
        self.assertEqual(GameParser.parse_coordinate("a15"), (0, 0))   # Top-Left
        self.assertEqual(GameParser.parse_coordinate("o15"), (14, 0))  # Top-Right
        self.assertEqual(GameParser.parse_coordinate("a1"), (0, 14))   # Bottom-Left
        self.assertEqual(GameParser.parse_coordinate("o1"), (14, 14))  # Bottom-Right

    def test_center(self):
        self.assertEqual(GameParser.parse_coordinate("h8"), (7, 7))    # Center (symmetric)


class TestPyGomoMoveCoordinates(unittest.TestCase):
    """PyGomo uses Rapfi convention: Row 1 = Top, 0-indexed internally."""

    def test_parse_algebraic(self):
        # Row 1 is the TOP row = internal y=0
        self.assertEqual(PyGomoMove("a1").to_tuple(), (0, 0))    # Top-Left
        self.assertEqual(PyGomoMove("o1").to_tuple(), (14, 0))   # Top-Right
        self.assertEqual(PyGomoMove("a15").to_tuple(), (0, 14))  # Bottom-Left
        self.assertEqual(PyGomoMove("o15").to_tuple(), (14, 14)) # Bottom-Right

    def test_center(self):
        self.assertEqual(PyGomoMove("h8").to_tuple(), (7, 7))    # Center (symmetric!)

    def test_generate_algebraic(self):
        # (0,0) = Top-Left = engine "a1"
        self.assertEqual(PyGomoMove((0, 0)).to_algebraic(), "a1")
        self.assertEqual(PyGomoMove((0, 14)).to_algebraic(), "a15")

    def test_center_is_unchanged(self):
        """Center cell (7,7) has the same algebraic notation in both systems due to 15 being odd."""
        self.assertEqual(PyGomoMove((7, 7)).to_algebraic(), "h8")


class TestCrossConvention(unittest.TestCase):
    """Verify that human 'c3' means the same internal cell as engine 'c13'."""

    def test_c3_human_to_engine(self):
        # Human: c3 = col 2, row 3 from bottom = internal (2, 12)
        human = GameParser.parse_coordinate("c3")
        self.assertEqual(human, (2, 12))

        # Engine representation of the same cell (2,12) = row 12+1=13 from top
        engine_str = PyGomoMove((2, 12)).to_algebraic()
        self.assertEqual(engine_str, "c13")

    def test_g6_human_to_engine(self):
        # Human: g6 = col 6, row 6 from bottom = internal (6, 9)
        human = GameParser.parse_coordinate("g6")
        self.assertEqual(human, (6, 9))

        # Engine: (6, 9) = row 9+1=10 from top
        engine_str = PyGomoMove((6, 9)).to_algebraic()
        self.assertEqual(engine_str, "g10")


if __name__ == '__main__':
    unittest.main()
