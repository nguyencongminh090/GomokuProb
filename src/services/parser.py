"""
Input Parser Service.
Handles SGF, PGN, and raw position strings.
Sanitizes input and detects formats automatically.
"""

import re
from typing import List, Tuple, Optional
from src.core.board import BoardState, Move, GameInfo

class GameParser:
    """
    Robust parser handling multiple formats.
    """
    
    @staticmethod
    def parse_coordinate(coord: str, size: int = 15) -> Optional[Tuple[int, int]]:
        """
        Parse "h8", "8h" to internal (x, y).
        Follows Standard Renju Notation:
        - Columns: a-o (x=0 to 14)
        - Rows: 1-15 (1 is Bottom, 15 is Top) -> y = size - row_num
        """
        coord = coord.strip().lower()
        if not coord:
            return None
            
        # Regex for Letter+Number or Number+Letter
        match_ln = re.match(r"([a-z])([0-9]{1,2})", coord)
        match_nl = re.match(r"([0-9]{1,2})([a-z])", coord)
        
        col_char = ''
        row_num = 0
        
        if match_ln:
            col_char, row_num_str = match_ln.groups()
            row_num = int(row_num_str)
        elif match_nl:
            row_num_str, col_char = match_nl.groups()
            row_num = int(row_num_str)
        else:
            return None
            
        col = ord(col_char) - ord('a')
        if col < 0 or col >= size:
            return None
            
        # Standard Renju: Row 1 is Bottom (index size-1), Row 15 is Top (index 0)
        # However, checking if user wants Matrix (1=Top).
        # Most Gomoku software uses 1=Bottom (Cartesian).
        # Let's enforce Cartesian.
        if row_num < 1 or row_num > size:
            return None
            
        row = size - row_num # 0-based index from Top
        
        return (col, row)

    @staticmethod
    def parse_raw_string(text: str) -> BoardState:
        """
        Parse raw sequence like "h8 i9 h9 j9" or "h8i9".
        """
        # Improved Regex Tokenization to handle concatenated strings
        tokens = re.findall(r"[a-z]\d{1,2}|\d{1,2}[a-z]", text.lower())
        
        board = BoardState()
        current_color = 1 # Black starts
        
        for token in tokens:
            pos = GameParser.parse_coordinate(token)
            
            if pos:
                x, y = pos
                board.add_move(x, y, current_color, notation=token)
                current_color = 3 - current_color # Switch 1 <-> 2
                
        return board

    @staticmethod
    def parse_sgf(content: str) -> BoardState:
        """
        Basic SGF parser.
        Supports AB, AW, B, W tags.
        """
        board = BoardState()
        
        # Simple regex extraction for now (robust SGF parsing is complex)
        # Find all properties inside ;...
        nodes = content.split(";")
        
        for node in nodes:
            if not node.strip():
                continue
                
            # Check for B[...] or W[...]
            black_move = re.search(r"B\[([a-zA-Z]{2})\]", node)
            white_move = re.search(r"W\[([a-zA-Z]{2})\]", node)
            
            if black_move:
                coords = black_move.group(1).lower()
                # SGF uses 'aa' for 0,0. Need to map.
                x = ord(coords[0]) - ord('a')
                y = ord(coords[1]) - ord('a')
                # Wait, Gomoku SGFs often use 'h8' standard notation inside labels but SGF coordinate system is strictly [aa]..[ss]
                # Standard Gomoku SGFs usually use standard SGF coords: a-s.
                board.add_move(x, y, 1, notation=f"{chr(x+97)}{y+1}")
                
            elif white_move:
                coords = white_move.group(1).lower()
                x = ord(coords[0]) - ord('a')
                y = ord(coords[1]) - ord('a')
                board.add_move(x, y, 2, notation=f"{chr(x+97)}{y+1}")
                
        return board
    
    @staticmethod
    def auto_detect_and_parse(content: str) -> BoardState:
        """Top-level entry point."""
        content = content.strip()
        
        if content.startswith("(;"):
            return GameParser.parse_sgf(content)
        # PGN detection usually [Event ...]
        elif content.startswith("[Event"):
            # TODO: Implement PGN
            return GameParser.parse_raw_string(content) # Fallback
        else:
            return GameParser.parse_raw_string(content)
