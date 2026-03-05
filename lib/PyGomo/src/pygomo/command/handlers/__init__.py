# Command Handlers Package
"""
Built-in command handlers for Gomocup protocol.
"""

from pygomo.command.handlers.base import BaseCommandHandler
from pygomo.command.handlers.lifecycle import (
    StartHandler,
    RestartHandler,
    EndHandler,
    AboutHandler,
)
from pygomo.command.handlers.game import (
    TurnHandler,
    BeginHandler,
    BoardHandler,
    TakebackHandler,
)
from pygomo.command.handlers.config import InfoHandler
from pygomo.command.handlers.search import (
    StopHandler,
    NBestHandler,
)

__all__ = [
    "BaseCommandHandler",
    # Lifecycle
    "StartHandler",
    "RestartHandler",
    "EndHandler",
    "AboutHandler",
    # Game
    "TurnHandler",
    "BeginHandler",
    "BoardHandler",
    "TakebackHandler",
    # Config
    "InfoHandler",
    # Search
    "StopHandler",
    "NBestHandler",
]


def register_all_handlers(registry) -> None:
    """
    Register all built-in handlers to a registry.
    
    Args:
        registry: CommandRegistry to register handlers to.
    """
    handlers = [
        StartHandler(),
        RestartHandler(),
        EndHandler(),
        AboutHandler(),
        TurnHandler(),
        BeginHandler(),
        BoardHandler(),
        TakebackHandler(),
        InfoHandler(),
        StopHandler(),
        NBestHandler(),
    ]
    
    for handler in handlers:
        registry.register(handler)
