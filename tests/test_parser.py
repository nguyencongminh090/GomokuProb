"""
Tests for GameParser.
"""
import unittest
from src.services.parser import GameParser

class TestParser(unittest.TestCase):
    
    def test_parse_simple_coord(self):
        self.assertEqual(GameParser.parse_coordinate("h8"), (7, 7))
        self.assertEqual(GameParser.parse_coordinate("a1"), (0, 0))
        self.assertEqual(GameParser.parse_coordinate("o15"), (14, 14))
        
    def test_parse_inverted_coord(self):
        self.assertEqual(GameParser.parse_coordinate("8h"), (7, 7))
        
    def test_parse_raw_string(self):
        raw = "h8 i9 h9"
        board = GameParser.parse_raw_string(raw)
        self.assertEqual(len(board.moves), 3)
        self.assertEqual(board.moves[0].x, 7)
        self.assertEqual(board.moves[0].y, 7)
        self.assertEqual(board.moves[0].color, 1) # Black
        
        self.assertEqual(board.moves[1].x, 8)
        self.assertEqual(board.moves[1].y, 8)
        self.assertEqual(board.moves[1].color, 2) # White

    def test_parse_comma_separated(self):
        raw = "h8, i9, h9"
        board = GameParser.parse_raw_string(raw)
        self.assertEqual(len(board.moves), 3)

if __name__ == '__main__':
    unittest.main()
