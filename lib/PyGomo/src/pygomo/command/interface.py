"""
Command layer interfaces.

This module defines abstract interfaces for command handlers
and data structures for command execution context.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pygomo.transport import ITransport, OutputChannelRouter
    from pygomo.protocol import IProtocol
    from pygomo.protocol.models import SearchInfo


class CommandStatus(Enum):
    """Command execution status."""
    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


@dataclass
class CommandContext:
    """
    Context passed to command handlers.
    
    Contains all dependencies and state needed for command execution.
    """
    transport: "ITransport"
    protocol: "IProtocol"
    router: "OutputChannelRouter"
    
    # Command details
    command: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    # Callbacks
    on_info: Optional[Callable[["SearchInfo"], None]] = None
    
    # Timeout settings
    timeout: Optional[float] = None
    
    # Custom data (for hooks)
    data: dict = field(default_factory=dict)


@dataclass
class CommandResult:
    """
    Result of command execution.
    
    Contains the response data and execution status.
    """
    status: CommandStatus
    data: Any = None
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Check if command succeeded."""
        return self.status == CommandStatus.SUCCESS
    
    @property
    def is_error(self) -> bool:
        """Check if command failed."""
        return self.status == CommandStatus.ERROR
    
    @classmethod
    def success(cls, data: Any = None) -> "CommandResult":
        """Create a success result."""
        return cls(status=CommandStatus.SUCCESS, data=data)
    
    @classmethod
    def error(cls, message: str) -> "CommandResult":
        """Create an error result."""
        return cls(status=CommandStatus.ERROR, error=message)
    
    @classmethod
    def timeout(cls) -> "CommandResult":
        """Create a timeout result."""
        return cls(status=CommandStatus.TIMEOUT, error="Command timed out")


class ICommandHandler(ABC):
    """
    Interface for command handlers.
    
    Each handler is responsible for executing a specific command
    and parsing its response.
    """
    
    @property
    @abstractmethod
    def command_name(self) -> str:
        """The command this handler handles."""
        ...
    
    @property
    @abstractmethod
    def aliases(self) -> list[str]:
        """Alternative names for this command."""
        ...
    
    @property
    @abstractmethod
    def requires_thinking(self) -> bool:
        """Whether this command triggers engine thinking."""
        ...
    
    @abstractmethod
    def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute the command.
        
        Args:
            context: Execution context with transport and protocol.
            
        Returns:
            CommandResult with response data.
        """
        ...
    
    @abstractmethod
    def validate_args(self, *args, **kwargs) -> bool:
        """
        Validate command arguments.
        
        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
            
        Returns:
            True if arguments are valid.
        """
        ...
