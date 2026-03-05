"""
Tests for BitBoard implementation.

Tests cover:
- Basic operations (place, remove, get)
- Win detection in all 4 directions
- Move history and undo
- Zobrist hashing
- Multi-size support
"""

import pytest
from pygomo.board import BitBoard, BLACK, WHITE, EMPTY
from pygomo.protocol.models import Move


class TestBitBoardBasics:
    """Test basic BitBoard operations."""
    
    def test_create_empty_board(self, empty_board):
        """Test creating an empty board."""
        assert empty_board.size == 15
        assert empty_board.move_count == 0
        assert empty_board.current_player == BLACK
        assert empty_board.last_move is None
    
    def test_create_different_sizes(self):
        """Test creating boards of different sizes."""
        for size in [9, 15, 19, 20]:
            board = BitBoard(_size=size)
            assert board.size == size
            assert board.move_count == 0
    
    def test_place_stone(self, empty_board):
        """Test placing a stone."""
        move = Move("h8")
        result = empty_board.place(move)
        
        assert result is True
        assert empty_board.get(move) == BLACK
        assert empty_board.move_count == 1
        assert empty_board.current_player == WHITE
    
    def test_place_alternates_colors(self, empty_board):
        """Test that colors alternate."""
        empty_board.place(Move("h8"))  # Black
        empty_board.place(Move("i8"))  # White
        empty_board.place(Move("h7"))  # Black
        
        assert empty_board.get(Move("h8")) == BLACK
        assert empty_board.get(Move("i8")) == WHITE
        assert empty_board.get(Move("h7")) == BLACK
    
    def test_cannot_place_on_occupied(self, empty_board):
        """Test that placing on occupied position fails."""
        move = Move("h8")
        empty_board.place(move)
        result = empty_board.place(move)  # Try again
        
        assert result is False
        assert empty_board.move_count == 1
    
    def test_cannot_place_outside_board(self, empty_board):
        """Test that placing outside board fails."""
        result = empty_board.place(Move((20, 20)))
        assert result is False
    
    def test_remove_stone(self, empty_board):
        """Test removing a stone."""
        move = Move("h8")
        empty_board.place(move)
        
        result = empty_board.remove(move)
        
        assert result is True
        assert empty_board.get(move) == EMPTY
        assert empty_board.move_count == 0
    
    def test_remove_empty_fails(self, empty_board):
        """Test that removing from empty position fails."""
        result = empty_board.remove(Move("h8"))
        assert result is False
    
    def test_is_empty(self, empty_board):
        """Test is_empty check."""
        move = Move("h8")
        assert empty_board.is_empty(move) is True
        
        empty_board.place(move)
        assert empty_board.is_empty(move) is False
    
    def test_is_valid(self, empty_board):
        """Test is_valid boundary check."""
        assert empty_board.is_valid(Move("a1")) is True
        assert empty_board.is_valid(Move("o15")) is True
        assert empty_board.is_valid(Move((0, 0))) is True
        assert empty_board.is_valid(Move((14, 14))) is True
        assert empty_board.is_valid(Move((-1, 0))) is False
        assert empty_board.is_valid(Move((15, 0))) is False


class TestWinDetection:
    """Test win detection in all 4 directions."""
    
    def test_horizontal_win(self, empty_board):
        """Test horizontal 5-in-a-row detection."""
        # Black: h8, i8, j8, k8, l8
        # White: h7, i7, j7, k7
        for i in range(5):
            empty_board.place(Move((7 + i, 7)))  # Black
            if i < 4:
                empty_board.place(Move((7 + i, 6)))  # White
        
        win = empty_board.check_win()
        assert win is not None
        assert win.winner == BLACK
        assert win.direction == "horizontal"
        assert len(win.line) == 5
    
    def test_vertical_win(self, empty_board):
        """Test vertical 5-in-a-row detection."""
        # Black: h4, h5, h6, h7, h8
        # White: i4, i5, i6, i7
        for i in range(5):
            empty_board.place(Move((7, 3 + i)))  # Black
            if i < 4:
                empty_board.place(Move((8, 3 + i)))  # White
        
        win = empty_board.check_win()
        assert win is not None
        assert win.winner == BLACK
        assert win.direction == "vertical"
    
    def test_diagonal_win(self, empty_board):
        """Test diagonal (↗) 5-in-a-row detection."""
        # Black: d4, e5, f6, g7, h8
        # White: a1, a2, a3, a4
        for i in range(5):
            empty_board.place(Move((3 + i, 3 + i)))  # Black diagonal
            if i < 4:
                empty_board.place(Move((0, i)))  # White
        
        win = empty_board.check_win()
        assert win is not None
        assert win.winner == BLACK
        assert win.direction == "diagonal"
    
    def test_anti_diagonal_win(self, empty_board):
        """Test anti-diagonal (↘) 5-in-a-row detection."""
        # Black: l4, k5, j6, i7, h8 (going from top-right to bottom-left)
        # White: a1, a2, a3, a4
        for i in range(5):
            empty_board.place(Move((11 - i, 3 + i)))  # Black anti-diagonal
            if i < 4:
                empty_board.place(Move((0, i)))  # White
        
        win = empty_board.check_win()
        assert win is not None
        assert win.winner == BLACK
        assert win.direction == "anti-diagonal"
    
    def test_no_win_with_four(self, empty_board):
        """Test that 4-in-a-row doesn't trigger win."""
        for i in range(4):
            empty_board.place(Move((7 + i, 7)))  # Black
            empty_board.place(Move((7 + i, 6)))  # White
        
        win = empty_board.check_win()
        assert win is None
    
    def test_white_can_win(self, empty_board):
        """Test that White can also win."""
        # Black moves scattered (not in a line)
        # White: h8, i8, j8, k8, l8 (horizontal 5)
        black_moves = [(0, 0), (2, 0), (4, 0), (6, 0), (8, 0)]  # a1, c1, e1, g1, i1
        white_moves = [(7, 7), (8, 7), (9, 7), (10, 7), (11, 7)]  # h8, i8, j8, k8, l8
        
        for i in range(5):
            empty_board.place(Move(black_moves[i]))  # Black scattered
            empty_board.place(Move(white_moves[i]))  # White horizontal
        
        win = empty_board.check_win()
        assert win is not None
        assert win.winner == WHITE
        assert win.direction == "horizontal"
    
    def test_check_win_at_specific_move(self, empty_board):
        """Test checking win at a specific position."""
        # Create horizontal 5
        for i in range(5):
            empty_board.place(Move((7 + i, 7)))
            if i < 4:
                empty_board.place(Move((7 + i, 6)))
        
        # Check at the last move
        win = empty_board.check_win(Move((11, 7)))
        assert win is not None
        assert win.winner == BLACK


class TestMoveHistory:
    """Test move history and undo functionality."""
    
    def test_history_records_moves(self, empty_board):
        """Test that history records moves in order."""
        moves = [Move("h8"), Move("i8"), Move("h7")]
        for m in moves:
            empty_board.place(m)
        
        history = empty_board.get_move_history()
        assert len(history) == 3
        assert history[0] == moves[0]
        assert history[1] == moves[1]
        assert history[2] == moves[2]
    
    def test_undo_removes_last_move(self, empty_board):
        """Test that undo removes the last move."""
        empty_board.place(Move("h8"))
        empty_board.place(Move("i8"))
        
        undone = empty_board.undo()
        
        assert undone == Move("i8")
        assert empty_board.move_count == 1
        assert empty_board.is_empty(Move("i8"))
        assert empty_board.current_player == WHITE
    
    def test_undo_on_empty_returns_none(self, empty_board):
        """Test that undo on empty board returns None."""
        result = empty_board.undo()
        assert result is None
    
    def test_last_move_property(self, empty_board):
        """Test last_move property."""
        assert empty_board.last_move is None
        
        empty_board.place(Move("h8"))
        assert empty_board.last_move == Move("h8")
        
        empty_board.place(Move("i8"))
        assert empty_board.last_move == Move("i8")


class TestZobristHashing:
    """Test Zobrist hashing functionality."""
    
    def test_empty_board_has_hash(self, empty_board):
        """Test that empty board has a hash."""
        assert empty_board.hash != 0
    
    def test_different_positions_different_hash(self, empty_board):
        """Test that different positions have different hashes."""
        hash1 = empty_board.hash
        
        empty_board.place(Move("h8"))
        hash2 = empty_board.hash
        
        empty_board.place(Move("i8"))
        hash3 = empty_board.hash
        
        assert hash1 != hash2
        assert hash2 != hash3
        assert hash1 != hash3
    
    def test_undo_restores_hash(self, empty_board):
        """Test that undo restores the previous hash."""
        hash_before = empty_board.hash
        
        empty_board.place(Move("h8"))
        empty_board.undo()
        
        assert empty_board.hash == hash_before
    
    def test_same_position_same_hash(self):
        """Test that same position gives same hash (transposition)."""
        board1 = BitBoard(_size=15)
        board2 = BitBoard(_size=15)
        
        # Same moves in same order
        board1.place(Move("h8"))
        board1.place(Move("i8"))
        
        board2.place(Move("h8"))
        board2.place(Move("i8"))
        
        assert board1.hash == board2.hash


class TestCopy:
    """Test board copying."""
    
    def test_copy_creates_independent_board(self, empty_board):
        """Test that copy creates an independent board."""
        empty_board.place(Move("h8"))
        copy = empty_board.copy()
        
        # Modify original
        empty_board.place(Move("i8"))
        
        # Copy should be unchanged
        assert copy.move_count == 1
        assert empty_board.move_count == 2
    
    def test_copy_preserves_state(self, empty_board):
        """Test that copy preserves all state."""
        empty_board.place(Move("h8"))
        empty_board.place(Move("i8"))
        
        copy = empty_board.copy()
        
        assert copy.size == empty_board.size
        assert copy.move_count == empty_board.move_count
        assert copy.hash == empty_board.hash
        assert copy.get_move_history() == empty_board.get_move_history()


class TestLegalMoves:
    """Test legal move generation."""
    
    def test_empty_board_all_moves_legal(self, empty_board):
        """Test that all positions are legal on empty board."""
        legal = empty_board.get_legal_moves()
        assert len(legal) == 15 * 15  # 225 moves
    
    def test_legal_moves_decrease(self, empty_board):
        """Test that legal moves decrease as stones are placed."""
        empty_board.place(Move("h8"))
        legal = empty_board.get_legal_moves()
        assert len(legal) == 15 * 15 - 1
    
    def test_is_full(self):
        """Test is_full on a very small board."""
        board = BitBoard(_size=3)
        assert board.is_full() is False
        
        # Fill the 3x3 board (9 moves)
        for row in range(3):
            for col in range(3):
                board.place(Move((col, row)))
        
        assert board.is_full() is True


class TestStringRepresentation:
    """Test string representation."""
    
    def test_str_contains_stones(self, empty_board):
        """Test that string representation shows stones."""
        empty_board.place(Move("h8"))
        empty_board.place(Move("i8"))
        
        s = str(empty_board)
        assert "X" in s  # Black
        assert "O" in s  # White
    
    def test_repr(self, empty_board):
        """Test repr format."""
        r = repr(empty_board)
        assert "BitBoard" in r
        assert "size=15" in r
