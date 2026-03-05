# Client API

The `EngineClient` is the primary entry point for interacting with Gomoku engines. It handles the complexity of process management, protocol parsing, and command execution.

## Usage Guide

### Basic Lifecycle

The recommended way to use the client is as a context manager, which ensures the engine process is properly terminated.

```python
from pygomo import EngineClient

with EngineClient("/path/to/engine") as engine:
    # 1. Start the game session
    engine.start(board_size=15)
    
    # 2. Interact
    # ...
    
    # 3. Quit happens automatically on exit, or manually:
    # engine.quit()
```

### Making Moves

The `turn` method is your main tool. It sends a move to the engine and waits for a response.

```python
# Send algebraic notation
result = engine.turn("h8")

# Send tuple (col, row) - 0-indexed
result = engine.turn((7, 7))

# Send numeric string
result = engine.turn("7,7")

if result.move:
    print(f"Engine played: {result.move}")
```

### Realtime Search Info

Engines often compute for seconds or minutes. To get feedback during this time (evaluation, winrate, principal variation), use the `on_info` callback.

```python
from pygomo import SearchInfo

def print_progress(info: SearchInfo):
    if info.depth > 0:
        print(f"Depth: {info.depth} Win Rate: {info.winrate_percent:.1f}%")

# Pass the callback to any thinking command
engine.turn("h8", on_info=print_progress)
```

### Configuration

You can configure engine parameters like time control, threads, and memory.

```python
# Set time control (5s per move, 5m total)
engine.set_time(turn_time_ms=5000, match_time_ms=300000)

# Set Hash size (MB)
engine.set_memory(1024)

# Set Threads
engine.set_threads(4)
```

### Custom Commands

For engine-specific commands not covered by the standard API, use `execute` or `send_raw`.

```python
# Execute and wait for response
result = engine.execute("YXSHOWINFO")
print(result.data)

# Fire and forget
engine.send_raw("DEBUG_MODE 1")
```

## Class Reference

```{eval-rst}
.. autoclass:: pygomo.client.engine.EngineClient
    :members: start, turn, begin, board, restart, set_time, set_rule, quit
    :undoc-members:
    :show-inheritance:
```
