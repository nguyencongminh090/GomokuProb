"""
Transport layer interfaces.

This module defines abstract interfaces for engine communication,
following the Interface Segregation Principle (ISP).
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional


class TransportError(Exception):
    """Base exception for transport layer errors."""
    pass


class TransportTimeoutError(TransportError):
    """Raised when a transport operation times out."""
    pass


class TransportConnectionError(TransportError):
    """Raised when connection to engine fails."""
    pass


class IStreamReader(ABC):
    """
    Interface for reading from engine output stream.
    
    Implementations should handle buffering and line-based reading.
    """
    
    @abstractmethod
    def readline(self, timeout: Optional[float] = None) -> str:
        """
        Read a single line from the stream.
        
        Args:
            timeout: Maximum time to wait in seconds. None means no timeout.
            
        Returns:
            The line read (without trailing newline), or empty string on EOF.
            
        Raises:
            TransportTimeoutError: If timeout expires before data is available.
        """
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Close the reader and release resources."""
        ...


class IStreamWriter(ABC):
    """
    Interface for writing to engine input stream.
    """
    
    @abstractmethod
    def writeline(self, data: str) -> None:
        """
        Write a line to the stream.
        
        Args:
            data: The data to write (newline will be appended automatically).
            
        Raises:
            TransportError: If write fails.
        """
        ...
    
    @abstractmethod
    def flush(self) -> None:
        """Flush the write buffer."""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Close the writer and release resources."""
        ...


class ITransport(ABC):
    """
    Interface for engine transport layer.
    
    Manages the lifecycle of engine process and provides
    read/write access to stdin/stdout streams.
    """
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the engine process is currently running."""
        ...
    
    @property
    @abstractmethod
    def process_id(self) -> Optional[int]:
        """Get the process ID of the engine, or None if not running."""
        ...
    
    @abstractmethod
    def start(self) -> None:
        """
        Start the engine process.
        
        Raises:
            TransportConnectionError: If the engine fails to start.
            TransportError: If the engine is already running.
        """
        ...
    
    @abstractmethod
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the engine process gracefully.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown.
        """
        ...
    
    @abstractmethod
    def send(self, data: str) -> None:
        """
        Send data to the engine's stdin.
        
        Args:
            data: The command to send (newline will be appended).
            
        Raises:
            TransportError: If engine is not running.
        """
        ...
    
    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> str:
        """
        Receive a line from the engine's stdout.
        
        Args:
            timeout: Maximum time to wait in seconds.
            
        Returns:
            The received line, or empty string on EOF.
            
        Raises:
            TransportTimeoutError: If timeout expires.
            TransportError: If engine is not running.
        """
        ...
    
    @abstractmethod
    def get_reader(self) -> IStreamReader:
        """Get the stream reader for advanced usage."""
        ...
    
    @abstractmethod
    def get_writer(self) -> IStreamWriter:
        """Get the stream writer for advanced usage."""
        ...
