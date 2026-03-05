"""
PyGomo - Python Gomoku Engine Communication Framework

A clean, extensible library for communicating with Gomoku engines
via the Gomocup protocol.

Example:
    from pygomo import EngineClient
    
    with EngineClient("/path/to/engine") as engine:
        engine.start(15)
        
        # With realtime search info
        def on_info(info):
            print(f"Depth {info.depth}: {info.winrate_percent:.1f}%")
        
        result = engine.turn("h8", on_info=on_info)
        print(f"Engine played: {result.move}")
        
        engine.quit()
"""

__version__ = "0.1.1"
__author__ = "PyGomo Contributors"

# Main client
from pygomo.client import EngineClient

# Protocol models
from pygomo.protocol.models import (
    Move,
    Evaluate,
    SearchInfo,
    PlayResult,
    BoardPosition,
)

# Protocol implementations
from pygomo.protocol import GomocupProtocol, IProtocol

# Transport layer
from pygomo.transport import (
    SubprocessTransport,
    OutputChannelRouter,
    TransportError,
    TransportTimeoutError,
)

# Command layer
from pygomo.command import (
    CommandRegistry,
    ICommandHandler,
    CommandContext,
    CommandResult,
    HookManager,
    HookType,
)

# Exceptions
from pygomo.exceptions import (
    PyGomoError,
    EngineError,
    ProtocolError,
    TimeoutError as PyGomoTimeoutError,
)

# Board representations
from pygomo.board import (
    BitBoard,
    RenjuBitBoard,
    IBoard,
    IRenjuBoard,
    BLACK,
    WHITE,
    EMPTY,
)

__all__ = [
    # Version
    "__version__",
    
    # Main client
    "EngineClient",
    
    # Models
    "Move",
    "Evaluate",
    "SearchInfo",
    "PlayResult",
    "BoardPosition",
    
    # Protocols
    "GomocupProtocol",
    "IProtocol",
    
    # Transport
    "SubprocessTransport",
    "OutputChannelRouter",
    "TransportError",
    "TransportTimeoutError",
    
    # Command
    "CommandRegistry",
    "ICommandHandler",
    "CommandContext",
    "CommandResult",
    "HookManager",
    "HookType",
    
    # Exceptions
    "PyGomoError",
    "EngineError",
    "ProtocolError",
    "PyGomoTimeoutError",
    
    # Board
    "BitBoard",
    "RenjuBitBoard",
    "IBoard",
    "IRenjuBoard",
    "BLACK",
    "WHITE",
    "EMPTY",
]
