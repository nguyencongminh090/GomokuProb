#!/usr/bin/env python3
"""
Run tests manually without pytest.
Works on systems without pytest installed.
"""

import sys
import os
import traceback

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygomo.board import BitBoard, RenjuBitBoard, BLACK, WHITE, EMPTY
from pygomo.protocol.models import Move


def test(name, condition, msg=""):
    """Simple test assertion."""
    if condition:
        print(f"  ✓ {name}")
        return True
    else:
        print(f"  ✗ {name} - {msg}")
        return False


def run_bitboard_tests():
    """Run BitBoard tests."""
    print("\n=== BitBoard Tests ===\n")
    passed = 0
    failed = 0
    
    # Basic operations
    print("Basic Operations:")
    board = BitBoard(_size=15)
    passed += test("Create empty board", board.size == 15)
    passed += test("Move count is 0", board.move_count == 0)
    passed += test("Current player is Black", board.current_player == BLACK)
    
    # Place stone
    result = board.place(Move("h8"))
    passed += test("Place h8 succeeds", result is True)
    passed += test("h8 is Black", board.get(Move("h8")) == BLACK)
    passed += test("Move count is 1", board.move_count == 1)
    passed += test("Current player is White", board.current_player == WHITE)
    
    # Cannot place on occupied
    result = board.place(Move("h8"))
    passed += test("Cannot place on occupied", result is False)
    
    # Win Detection - Horizontal
    print("\nWin Detection - Horizontal:")
    board = BitBoard(_size=15)
    for i in range(5):
        board.place(Move((7 + i, 7)))  # Black h8, i8, j8, k8, l8
        if i < 4:
            board.place(Move((7 + i, 6)))  # White
    win = board.check_win()
    passed += test("Horizontal win detected", win is not None)
    passed += test("Winner is Black", win and win.winner == BLACK)
    passed += test("Direction is horizontal", win and win.direction == "horizontal")
    
    # Win Detection - Vertical
    print("\nWin Detection - Vertical:")
    board = BitBoard(_size=15)
    for i in range(5):
        board.place(Move((7, 3 + i)))  # Black h4, h5, h6, h7, h8
        if i < 4:
            board.place(Move((8, 3 + i)))  # White
    win = board.check_win()
    passed += test("Vertical win detected", win is not None)
    passed += test("Winner is Black", win and win.winner == BLACK)
    passed += test("Direction is vertical", win and win.direction == "vertical")
    
    # Win Detection - Diagonal
    print("\nWin Detection - Diagonal (↗):")
    board = BitBoard(_size=15)
    for i in range(5):
        board.place(Move((3 + i, 3 + i)))  # Black d4, e5, f6, g7, h8
        if i < 4:
            board.place(Move((0, i)))  # White
    win = board.check_win()
    passed += test("Diagonal win detected", win is not None)
    passed += test("Winner is Black", win and win.winner == BLACK)
    passed += test("Direction is diagonal", win and win.direction == "diagonal")
    
    # Win Detection - Anti-diagonal
    print("\nWin Detection - Anti-diagonal (↘):")
    board = BitBoard(_size=15)
    for i in range(5):
        board.place(Move((11 - i, 3 + i)))  # Black l4, k5, j6, i7, h8
        if i < 4:
            board.place(Move((0, i)))  # White
    win = board.check_win()
    passed += test("Anti-diagonal win detected", win is not None)
    passed += test("Winner is Black", win and win.winner == BLACK)
    passed += test("Direction is anti-diagonal", win and win.direction == "anti-diagonal")
    
    # Zobrist hashing
    print("\nZobrist Hashing:")
    board1 = BitBoard(_size=15)
    board2 = BitBoard(_size=15)
    hash_empty = board1.hash
    passed += test("Empty board has hash", hash_empty != 0)
    
    board1.place(Move("h8"))
    passed += test("Hash changes after move", board1.hash != hash_empty)
    
    board1.undo()
    passed += test("Hash restored after undo", board1.hash == hash_empty)
    
    # Move history
    print("\nMove History:")
    board = BitBoard(_size=15)
    board.place(Move("h8"))
    board.place(Move("i8"))
    board.place(Move("h7"))
    
    history = board.get_move_history()
    passed += test("History has 3 moves", len(history) == 3)
    passed += test("First move is h8", str(history[0]) == "h8")
    
    undone = board.undo()
    passed += test("Undo returns h7", str(undone) == "h7")
    passed += test("Move count is 2", board.move_count == 2)
    
    print(f"\n--- BitBoard: {passed} passed ---")
    return passed


def run_renju_tests():
    """Run RenjuBitBoard tests."""
    print("\n=== Renju Tests (Forbidden Moves) ===\n")
    passed = 0
    
    # Overline test
    print("Overline (6+):")
    board = RenjuBitBoard(_size=15)
    # Place 5 black stones in a row
    for i in range(5):
        board.place(Move((7 + i, 7)))  # Black h8-l8
        board.place(Move((0, i)))      # White
    
    # Check if 6th is forbidden
    is_forbidden_g8 = board.is_forbidden(Move((6, 7)))  # g8
    passed += test("g8 (6th stone, overline) is forbidden", is_forbidden_g8)
    
    is_forbidden_m8 = board.is_forbidden(Move((12, 7)))  # m8
    passed += test("m8 (6th stone, overline) is forbidden", is_forbidden_m8)
    
    # 5-in-a-row is NOT forbidden (it's a win)
    print("\nFive (not forbidden):")
    board = RenjuBitBoard(_size=15)
    for i in range(4):
        board.place(Move((7 + i, 7)))  # Black h8-k8
        board.place(Move((0, i)))      # White
    is_forbidden_l8 = board.is_forbidden(Move((11, 7)))  # l8 (5th stone)
    passed += test("l8 (5th stone, five) is NOT forbidden", not is_forbidden_l8)
    
    # Double-four test
    print("\nDouble-Four (4x4):")
    board = RenjuBitBoard(_size=15)
    
    # Create pattern where one move creates two fours
    # Horizontal: e8, f8, g8 (need h8 for four)
    # Vertical: h5, h6, h7 (need h8 for four)
    for i, pos in enumerate(['e8', 'f8', 'g8']):
        board.place(Move(pos))
        board.place(Move(f"a{i+1}"))
    for i, pos in enumerate(['h5', 'h6', 'h7']):
        board.place(Move(pos))
        board.place(Move(f"b{i+1}"))
    
    # h8 creates double-four
    is_forbidden_h8 = board.is_forbidden(Move("h8"))
    passed += test("h8 (double-four) is forbidden", is_forbidden_h8)
    
    # Single four is NOT forbidden
    print("\nSingle Four (not forbidden):")
    board = RenjuBitBoard(_size=15)
    board.place(Move("h8"))
    board.place(Move("a1"))
    board.place(Move("i8"))
    board.place(Move("a2"))
    board.place(Move("j8"))
    board.place(Move("a3"))
    
    is_forbidden_k8 = board.is_forbidden(Move("k8"))  # Makes single four
    passed += test("k8 (single four) is NOT forbidden", not is_forbidden_k8)
    
    # Double-three test
    print("\nDouble-Three (3x3):")
    board = RenjuBitBoard(_size=15)
    
    # Create pattern where one move creates two open threes
    # Horizontal: f8 _ h8 (placing g8 makes f8-g8-h8)
    # Vertical: g7 _ g9 (placing g8 makes g7-g8-g9)
    board.place(Move("f8"))
    board.place(Move("a1"))
    board.place(Move("h8"))
    board.place(Move("a2"))
    board.place(Move("g7"))
    board.place(Move("a3"))
    board.place(Move("g9"))
    board.place(Move("a4"))
    
    is_forbidden_g8 = board.is_forbidden(Move("g8"))
    passed += test("g8 (double-three) is forbidden", is_forbidden_g8)
    
    # Single open three is NOT forbidden
    print("\nSingle Open Three (not forbidden):")
    board = RenjuBitBoard(_size=15)
    board.place(Move("f8"))
    board.place(Move("a1"))
    board.place(Move("h8"))
    board.place(Move("a2"))
    
    is_forbidden_g8 = board.is_forbidden(Move("g8"))  # Makes single open three
    passed += test("g8 (single open three) is NOT forbidden", not is_forbidden_g8)
    
    # Forbidden move rejected by place()
    print("\nForbidden move rejected:")
    board = RenjuBitBoard(_size=15)
    for i in range(5):
        board.place(Move((7 + i, 7)))
        board.place(Move((0, i)))
    
    result = board.place(Move((6, 7)))  # Try overline
    passed += test("place() returns False for forbidden", result is False)
    passed += test("Position remains empty", board.is_empty(Move((6, 7))))
    
    # get_forbidden_moves()
    print("\nget_forbidden_moves():")
    forbidden = board.get_forbidden_moves()
    passed += test("Forbidden list includes g8", Move((6, 7)) in forbidden)
    passed += test("Forbidden list includes m8", Move((12, 7)) in forbidden)
    
    print(f"\n--- Renju: {passed} passed ---")
    return passed


def main():
    """Run all tests."""
    print("=" * 50)
    print("  PyGomo Board Module Tests")
    print("=" * 50)
    
    total = 0
    
    try:
        total += run_bitboard_tests()
    except Exception as e:
        print(f"BitBoard tests failed with exception: {e}")
        traceback.print_exc()
    
    try:
        total += run_renju_tests()
    except Exception as e:
        print(f"Renju tests failed with exception: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"  Total: {total} tests passed")
    print("=" * 50)
    
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
