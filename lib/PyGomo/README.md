# PyGomo

**PyGomo** is a clean, modern, and extensible Python library for communicating with Gomoku engines via the [Gomocup protocol](https://gomocup.org/). It is designed for developers, researchers, and AI enthusiasts who need a robust interface to interact with Gomoku AI agents.

 [![PyPI version](https://badge.fury.io/py/pygomo-lib.svg)](https://badge.fury.io/py/pygomo-lib)
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🚀 Features

*   **Robust Engine Management**: Automatically handles engine subprocesses (start, stop, restart).
*   **Gomocup Protocol Support**: Full implementation of the standard Gomocup protocol (START, TURN, BOARD, INFO, etc.).
*   **Real-time Search Info**: Parse and consume engine search data (depth, winrate, PV) in real-time via callbacks.
*   **Rich Data Models**: Strongly typed models for `Move`, `BoardPosition`, and `SearchInfo`.
*   **Extensible Architecture**: Built-in hook system and command registry for customizing behavior.
*   **Board Library**: High-performance BitBoard implementation for standard and Renju rules.
*   **Console Game**: Includes a built-in console-based client for playing against engines.

## 📦 Installation

Install via pip:

```bash
pip install pygomo-lib
```

> **Note:** The package name on PyPI is `pygomo-lib`, but the import name is `pygomo`.

## ⚡ Quick Start

Here is a minimal example of how to load an engine and play a game.

```python
from pygomo import EngineClient
from pygomo.protocol.models import SearchInfo

# Path to your Gomoku engine executable (e.g., Rapfi, Pela, Yixin)
ENGINE_PATH = "./path/to/engine.exe"

def on_search_info(info: SearchInfo):
    """Callback to print search progress in real-time."""
    print(f"Depth: {info.depth} | Winrate: {info.winrate_percent:.1f}% | PV: {info.pv[:4]}")

# Initialize and connect
with EngineClient(ENGINE_PATH) as engine:
    # 1. Start a new game (Board size 15)
    engine.start(board_size=15)

    # 2. Set strict time control (1 second per turn)
    engine.configure(timeout_turn=1000)

    # 3. Request the engine to play first (Black)
    result = engine.begin(on_info=on_search_info)
    print(f"Engine played: {result.move}")  # e.g., "h8"

    # 4. Send our move (White)
    my_move = "h9"
    result = engine.turn(my_move, on_info=on_search_info)
    print(f"Engine responded: {result.move}")
```

## 🎮 Console Game

PyGomo comes with a built-in console CLI for testing engines or playing directly in your terminal.

```bash
# Play against an engine
pygomo-console --engine ./engines/rapfi --time 5000

# Play using Renju rules
pygomo-console --engine ./engines/pela --rule renju
```

**Key Commands in Console:**
*   `move <coord>`: Play a move (e.g., `h8`).
*   `undo`: Undo the last move.
*   `swap`: Swap sides (if playing swap rule).
*   `info`: Show last search info.
*   `quit`: Exit.

## 📚 Core Concepts

### EngineClient
The `EngineClient` class is the main entry point. It wraps the raw protocol communications into high-level Python methods like `.start()`, `.turn()`, and `.board()`.

### Data Models
PyGomo uses dataclasses to represent game entities cleanly:

*   **`Move`**: Handles coordinates. Supports `(row, col)`, `"h8"` (algebraic), and `"7,7"` (numeric) formats.
*   **`Evaluate`**: Parses engine scores, supporting raw values (e.g., "150"), winrates, and mate scores (e.g., "+M5").
*   **`SearchInfo`**: Aggregates real-time search data (depth, NPS, nodes, PV).

### Board Library
Under `pygomo.board`, you'll find efficient board representations:
*   `BitBoard`: 64-bit optimized board for standard Gomoku.
*   `RenjuBitBoard`: Extends BitBoard with forbidden move logic (3x3, 4x4, overline) for Renju rules.

## 🔧 Advanced Usage

### Hooks
You can inject custom logic into the command lifecycle using hooks.

```python
from pygomo.command import HookType

def log_command(context):
    print(f"Sending command: {context.command}")

engine.hooks.on(HookType.PRE_EXECUTE)(log_command)
```

### Custom Commands
If your engine supports non-standard commands, you can send them raw:

```python
# Send "MY_CUSTOM_CMD arg1"
engine.send_raw("MY_CUSTOM_CMD arg1")
response = engine.receive_raw()
```

## 🛠️ Development

To set up a development environment:

1.  Clone the repository.
2.  Install development dependencies:
    ```bash
    pip install -e .[dev]
    ```
3.  Run tests:
    ```bash
    pytest tests/
    ```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for the Gomoku community.*
