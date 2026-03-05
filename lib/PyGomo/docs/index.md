# PyGomo Developer Cookbook

**PyGomo** is designed to be the backbone of your Gomoku application. Whether you are building a GUI, a tournament manager, or an analysis bot, this cookbook provides the recipes you need to get things done.

```{toctree}
:maxdepth: 2
:hidden:

client
board
protocol
api
```

## 🏗️ Architecture Overview

Understanding the layers will help you know where to look:

1.  **`pygomo` (Top Level)**: Exports common objects like `EngineClient`, `Move`, `SearchInfo`. Start here.
2.  **`pygomo.board`**: The "Physics Engine". Handles board state, move generation (BitBoard), and Renju rules validation.
3.  **`pygomo.client`**: The High-Level API. `EngineClient` lives here.
4.  **`pygomo.protocol`**: The Translator. Converts Python objects to/from Gomocup protocol strings.
5.  **`pygomo.transport`**: The Plumber. Handles subprocess creation and non-blocking I/O.

---

## 🍳 Recipe 1: The "Hello World" Connection

The simplest way to connect to an engine. Always use the `with` statement to ensure the subprocess is killed clean.

```python
from pygomo import EngineClient

# Engine path can be absolute or relative
with EngineClient("./engines/piskvork") as engine:
    # 1. Start a game (default 15x15)
    engine.start(board_size=15)
    
    # 2. Make a move (Human plays h8)
    result = engine.turn("h8")
    
    # 3. Print result
    print(f"Engine replied: {result.move}")
```

---

## 🍳 Recipe 2: Configuring The Engine

Before starting the game, you often need to set limits. Note that `start()` resets some engine internals, so configure *after* `start()` usually (depending on engine quirk, but safely: connect -> start -> configure).

```python
with EngineClient("./engines/rapfi") as engine:
    engine.start(15)
    
    # Setup Time Control
    engine.set_time(
        turn_time_ms=5000,    # 5 seconds per move
        match_time_ms=300000, # 5 minutes total
        time_left_ms=300000   # Current remaining time
    )
    
    # Setup Rules (1 = Standard, 4 = Renju)
    engine.set_rule(1)
    
    # Advanced Config (Thread, Hash, etc.)
    engine.set_threads(4)
    engine.set_memory(1024 * 1024 * 1024) # 1GB
```

---

## 🍳 Recipe 3: Dealing with Realtime Search Info

Modern engines (like Rapfi, Yixin) provide a stream of search data while thinking. You can tap into this via the `on_info` callback.

```python
from pygomo import SearchInfo

def my_info_handler(info: SearchInfo):
    # This is called repeatedly while engine thinks!
    if info.depth > 0:
        pv_string = " ".join(str(m) for m in info.pv[:5])
        print(f"\rDepth: {info.depth} | Eval: {info.eval.raw_value} | PV: {pv_string}", end="")

# Pass the handler to any thinking command (turn, begin, board, nbest)
result = engine.turn("h8", on_info=my_info_handler)

# The final 'result' also contains the LAST search info received
print(f"\nFinal Winrate: {result.search_info.winrate_percent:.2f}%")
```

**Understanding `SearchInfo` fields:**
*   `depth`/`sel_depth`: Search depth.
*   `eval`: `Evaluate` object. Use `.raw_value` for engine score, `.score()` for int, `.winrate_percent()` for 0-100%.
*   `pv`: List of `Move` objects (Principal Variation / Best Line).
*   `nodes`/`nps`: Performance metrics.

---

## 🍳 Recipe 4: Managing Board State with BitBoard

Don't track 2D arrays yourself. Use `pygomo.board` for high-performance state management.

```python
from pygomo.board import BitBoard, RenjuBitBoard

# Use RenjuBitBoard for complex Renju rules (double-3, double-4, overline)
board = RenjuBitBoard(size=15)

# 1. Place moves
board.place("h8") # Black
board.place("h9") # White

# 2. Check for Win/Forbidden
last_move = board.get_last_move()
win_info = board.check_win(last_move)

if win_info:
    print(f"Winner: {win_info.winner}") # 1=Black, 2=White
    print(f"Line: {win_info.direction}") # e.g. [(7,7), (8,8)...]

# 3. Check Forbidden (Renju Only)
if board.is_forbidden(move="j9", color=1): # Is j9 forbidden for Black?
    print("Foul!")

# 4. Undo
board.undo() # Takes back h9
board.undo() # Takes back h8
```

---

## 🍳 Recipe 5: Synchronizing Engine with Custom Positions

If you implement `Undo`, `Redo`, or `Load Game`, you must sync the engine. The engine has its own internal board.

```python
# ... user clicks undo ...
my_local_board.undo()

# SYNC RECIPE:
# 1. Convert local board to protocol objects
protocol_position = my_local_board.to_position() 

# 2. Send BOARD command
# start_thinking=False means "just set the board, don't move yet"
engine.board(protocol_position, start_thinking=False)
```

---

## 🍳 Recipe 6: Parsing User Input Handling

Users type garbage. `Move` handles it.

```python
from pygomo import Move

try:
    m1 = Move("h8")       # Standard algebraic
    m2 = Move("H8")       # Case insensitive
    m3 = Move("7,7")      # Numeric (0-indexed)
    m4 = Move((7, 7))     # Tuple
    
    print(m1.col, m1.row) # 7, 7
except ValueError:
    print("Invalid move format")
```

---

## 🍳 Recipe 7: Building a Non-Blocking GUI Loop

If you use `asyncio` or a GUI event loop (PyQt/Tkinter), you don't want `engine.turn()` to freeze the UI. PyGomo is synchronous by default (for simplicity), but you can wrap it easily.

**Pattern for Threading (Simplest):**
```python
import threading

def worker():
    # This blocks, but it's in a thread
    result = engine.turn("h8", on_info=update_gui_progress)
    # Post result to GUI thread
    gui.post_event(GameMoveEvent(result.move))

thread = threading.Thread(target=worker)
thread.start()
```

---

## 🍳 Recipe 8: Custom Commands

Need to send a command PyGomo doesn't support explicitly?

```python
# Send raw string (fire and forget)
engine.send_raw("MY_CUSTOM_COMMAND args")

# Execute and want response (handles timeout/locking)
# This returns a CommandResult object
result = engine.execute("YXSHOWINFO") 
print(result.data) 
```

## 🛠️ Developer Checklist

When implementing a full game:

1.  [ ] **Always handle specific Exceptions**: Engines crash. Catch `RuntimeError` or timeouts on turn execution.
2.  [ ] **Hash Synchronization**: If your board supports Zobrist hashing (`board.hash`), check it against the engine occasionally (Rapfi supports `checksum` in some debug modes) to detect desyncs.
3.  [ ] **Clean Shutdown**: Ensure `engine.quit()` is called. Orphaned engine processes consume CPU.
4.  [ ] **Protocol Logging**: For debugging, you can hook into the transport or use `run_game.sh` which prints what it's doing.
