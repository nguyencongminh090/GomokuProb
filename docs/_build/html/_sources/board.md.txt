# Board & Rules

PyGomo includes a highly optimized BitBoard implementation for state management and rule validation.

## Board Implementations

### BitBoard (Standard Gomoku)

`BitBoard` handles the standard Gomoku rules (freestyle or standard). It uses bitwise operations for O(1) validation and win checking.

```python
from pygomo.board import BitBoard

board = BitBoard(size=15)
board.place("h8")
is_win = board.check_win(board.get_last_move())
```

### RenjuBitBoard (Renju Rules)

`RenjuBitBoard` extends `BitBoard` to implement the complex Renju rules for Black:
*   **Double Three**: Cannot form two open threes simultaneously.
*   **Double Four**: Cannot form two fours simultaneously.
*   **Overline**: Cannot form a line of 6 or more stones (win for White, foul for Black).

```python
from pygomo.board import RenjuBitBoard

board = RenjuBitBoard(size=15)
# Check if a move is forbidden
if board.is_forbidden("j9", color=1):
    print("This move is illegal for Black in Renju!")
```

## Internal Representation

The board uses Python's arbitrary-precision integers as bitsets.
*   `_bits[0]`: Bitmask for Player 1 (Black).
*   `_bits[1]`: Bitmask for Player 2 (White).

Bit index is calculated as `row * (size + 1) + col`. The extra +1 adds a virtual "wall" between rows to prevent line detection wrapping around edges.

## Zobrist Hashing

The board maintains an incremental Zobrist hash useful for transposition tables or synchronizing state with engines.

```python
print(f"Current Hash: {board.hash:016X}")
```

## Class Reference

```{eval-rst}
.. autoclass:: pygomo.board.bitboard.BitBoard
    :members: place, remove, get, check_win, undo, is_empty
    :undoc-members:
    :show-inheritance:

.. autoclass:: pygomo.board.renju.RenjuBitBoard
    :members: is_forbidden
    :undoc-members:
    :show-inheritance:
```
