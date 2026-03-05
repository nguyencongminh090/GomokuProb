"""
Subprocess-based transport implementation.

This module provides a transport implementation that spawns
an engine as a subprocess and communicates via stdin/stdout.
"""

import subprocess
from threading import Lock
from typing import Optional

from pygomo.transport.interface import (
    ITransport,
    IStreamReader,
    IStreamWriter,
    TransportError,
    TransportConnectionError,
    TransportTimeoutError,
)
from pygomo.transport.reader import OutputChannelRouter


class SubprocessStreamWriter(IStreamWriter):
    """
    Stream writer that writes to subprocess stdin.
    """
    
    def __init__(self, stdin, io_logger=None):
        self._stdin = stdin
        self._lock = Lock()
        self._io_logger = io_logger  # Callback: (str, str) -> None, e.g. ("STDIN", "START 15")
    
    def writeline(self, data: str) -> None:
        """Write a line to stdin with newline appended."""
        with self._lock:
            if self._stdin is None or self._stdin.closed:
                raise TransportError("Writer is closed")
            self._stdin.write(f"{data}\n")
            self._stdin.flush()
            # Log the raw stdin
            if self._io_logger:
                try:
                    self._io_logger("STDIN", data)
                except:
                    pass
    
    def flush(self) -> None:
        """Flush the stdin buffer."""
        with self._lock:
            if self._stdin and not self._stdin.closed:
                self._stdin.flush()
    
    def close(self) -> None:
        """Close the stdin stream."""
        with self._lock:
            if self._stdin and not self._stdin.closed:
                self._stdin.close()
            self._stdin = None


class SubprocessStreamReader(IStreamReader):
    """
    Stream reader that reads from subprocess stdout via OutputChannelRouter.
    """
    
    def __init__(self, router: OutputChannelRouter, channel: str = "default"):
        self._router = router
        self._channel = channel
    
    def readline(self, timeout: Optional[float] = None) -> str:
        """
        Read a line from the specified channel.
        
        Args:
            timeout: Timeout in seconds, None for no timeout.
            
        Returns:
            The line read, or empty string on timeout/EOF.
        """
        result = self._router.get(self._channel, timeout=timeout or 0.0)
        return result
    
    def close(self) -> None:
        """Close is handled by the router."""
        pass


class SubprocessTransport(ITransport):
    """
    Transport implementation using Python subprocess.
    
    Spawns an engine executable and provides bidirectional
    communication via stdin/stdout pipes.
    
    Example:
        transport = SubprocessTransport("/path/to/engine")
        transport.start()
        transport.send("START 15")
        response = transport.receive(timeout=5.0)
        transport.stop()
    """
    
    def __init__(
        self,
        executable_path: str,
        args: Optional[list[str]] = None,
        working_directory: Optional[str] = None,
        io_logger=None,
    ):
        """
        Initialize subprocess transport.
        
        Args:
            executable_path: Path to the engine executable.
            args: Optional command line arguments.
            working_directory: Optional working directory for the process.
            io_logger: Optional callback for logging raw IO: (type, data) -> None
        """
        self._executable_path = executable_path
        self._args = args or []
        self._working_directory = working_directory
        self._io_logger = io_logger
        
        self._process: Optional[subprocess.Popen] = None
        self._router: Optional[OutputChannelRouter] = None
        self._writer: Optional[SubprocessStreamWriter] = None
        self._lock = Lock()
    
    @property
    def is_running(self) -> bool:
        """Check if the engine process is running."""
        with self._lock:
            return self._process is not None and self._process.poll() is None
    
    @property
    def process_id(self) -> Optional[int]:
        """Get the process ID of the engine."""
        with self._lock:
            return self._process.pid if self._process else None
    
    def start(self) -> None:
        """
        Start the engine subprocess.
        
        Raises:
            TransportError: If already running.
            TransportConnectionError: If process fails to start.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise TransportError("Transport is already running")
            
            try:
                cmd = [self._executable_path] + self._args
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self._working_directory,
                    bufsize=1,
                    universal_newlines=True,
                    text=True,
                )
                
                # Initialize router for stdout (with io_logger for raw logging)
                self._router = OutputChannelRouter(self._process.stdout, io_logger=self._io_logger)
                
                # Initialize writer for stdin (with io_logger for raw logging)
                self._writer = SubprocessStreamWriter(self._process.stdin, io_logger=self._io_logger)
                
            except OSError as e:
                raise TransportConnectionError(
                    f"Failed to start engine '{self._executable_path}': {e}"
                ) from e
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the engine subprocess.
        
        Attempts graceful shutdown first, then forces termination.
        
        Args:
            timeout: Time to wait for graceful shutdown.
        """
        with self._lock:
            if self._process is None:
                return
            
            # Try graceful shutdown
            if self._process.poll() is None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self._process.kill()
                    self._process.wait()
            
            # Clean up resources
            if self._writer:
                self._writer.close()
            
            self._process = None
            self._router = None
            self._writer = None
    
    def send(self, data: str) -> None:
        """
        Send a command to the engine.
        
        Args:
            data: Command string (newline appended automatically).
        """
        if not self.is_running:
            raise TransportError("Transport is not running")
        
        self._writer.writeline(data)
    
    def receive(self, timeout: Optional[float] = None) -> str:
        """
        Receive a line from the engine's default output.
        
        Args:
            timeout: Timeout in seconds.
            
        Returns:
            The received line.
        """
        if not self.is_running:
            raise TransportError("Transport is not running")
        
        return self._router.get("output", timeout=timeout or 0.0)
    
    def get_reader(self) -> IStreamReader:
        """Get a reader for the default channel."""
        if not self.is_running or self._router is None:
            raise TransportError("Transport is not running")
        return SubprocessStreamReader(self._router, "output")
    
    def get_writer(self) -> IStreamWriter:
        """Get the stdin writer."""
        if not self.is_running or self._writer is None:
            raise TransportError("Transport is not running")
        return self._writer
    
    def get_router(self) -> OutputChannelRouter:
        """
        Get the output channel router for advanced usage.
        
        Returns:
            The OutputChannelRouter instance.
        """
        if not self.is_running or self._router is None:
            raise TransportError("Transport is not running")
        return self._router
    
    def __enter__(self) -> "SubprocessTransport":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.stop()
