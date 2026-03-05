"""
Protocol layer interface.

This module defines the abstract protocol interface for
message serialization and response parsing.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Optional


class ResponseType(Enum):
    """Types of responses from the engine."""
    OK = auto()           # Simple acknowledgment (e.g., "OK")
    COORD = auto()        # Coordinate response (e.g., "7,8")
    MULTI_COORD = auto()  # Multiple coordinates (e.g., "7,8 8,9")
    MESSAGE = auto()      # Search info message
    INFO = auto()         # Detailed info
    ERROR = auto()        # Error response
    TEXT = auto()         # Free-form text (e.g., ABOUT)
    NONE = auto()         # No response expected


class IProtocol(ABC):
    """
    Interface for engine communication protocol.
    
    Handles command serialization and response parsing.
    Implementations should be stateless and thread-safe.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol name identifier."""
        ...
    
    @abstractmethod
    def serialize_command(self, command: str, *args: Any) -> str:
        """
        Serialize a command with arguments into protocol format.
        
        Args:
            command: Command name (e.g., "START", "TURN").
            *args: Command arguments.
            
        Returns:
            The formatted command string ready for transmission.
        """
        ...
    
    @abstractmethod
    def get_response_type(self, command: str) -> ResponseType:
        """
        Get the expected response type for a command.
        
        Args:
            command: Command name.
            
        Returns:
            The expected ResponseType.
        """
        ...
    
    @abstractmethod
    def parse_coord(self, data: str) -> tuple[int, int]:
        """
        Parse a coordinate string into (x, y).
        
        Args:
            data: Coordinate string (e.g., "7,8").
            
        Returns:
            Tuple of (x, y) coordinates.
        """
        ...
    
    @abstractmethod
    def format_coord(self, x: int, y: int) -> str:
        """
        Format coordinates into protocol string.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            
        Returns:
            Formatted coordinate string.
        """
        ...
    
    @abstractmethod
    def parse_message(self, data: str) -> dict[str, Any]:
        """
        Parse a MESSAGE line into structured data.
        
        Args:
            data: The MESSAGE line from engine.
            
        Returns:
            Dictionary with parsed fields.
        """
        ...
    
    @abstractmethod
    def is_thinking_command(self, command: str) -> bool:
        """
        Check if a command triggers engine thinking.
        
        Args:
            command: Command name.
            
        Returns:
            True if the command makes the engine think.
        """
        ...
