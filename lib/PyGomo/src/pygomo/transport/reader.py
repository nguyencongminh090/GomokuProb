"""
Output channel router for categorizing engine output.

This module provides a multi-channel message routing system
that categorizes engine output lines based on patterns.
"""

import re
from queue import Queue, Empty
from threading import Thread, Lock
from typing import TextIO, Callable, Optional


class OutputChannelRouter:
    """
    Routes engine stdout lines to categorized channels.
    
    Each channel has a filter function that determines which
    lines belong to it. Lines are read in a background thread
    and distributed to the appropriate queues.
    
    Default channels:
        - coord: Lines matching coordinate pattern (e.g., "7,8")
        - message: Lines starting with "MESSAGE"
        - info: Lines starting with "INFO"
        - error: Lines starting with "ERROR"
        - output: All other non-error lines
    
    Example:
        router = OutputChannelRouter(process.stdout)
        
        # Get a coordinate response
        coord = router.get("coord", timeout=5.0)
        
        # Get realtime search info
        message = router.get("message", timeout=0.1)
    """
    
    # Common patterns for Gomocup protocol
    COORD_PATTERN = re.compile(r"^\d+\s*,\s*\d+(\s+\d+\s*,\s*\d+)*$")
    
    def __init__(self, stream: TextIO, io_logger=None):
        """
        Initialize the router with an output stream.
        
        Args:
            stream: The stdout stream from the engine process.
            io_logger: Optional callback for logging raw stdout lines.
        """
        self._stream = stream
        self._queues: dict[str, Queue] = {}
        self._filters: dict[str, Callable[[str], bool]] = {}
        self._lock = Lock()
        self._running = True
        self._io_logger = io_logger  # Callback: (str, str) -> None, e.g. ("STDOUT", "OK")
        
        # Setup default channels
        self._setup_default_channels()
        
        # Start background reader thread
        self._thread = Thread(target=self._read_loop, daemon=True)
        self._thread.start()
    
    def _setup_default_channels(self) -> None:
        """Setup the default protocol channels."""
        self.add_channel("coord", self._is_coord)
        self.add_channel("message", self._is_message)
        self.add_channel("info", self._is_info)
        self.add_channel("error", self._is_error)
        self.add_channel("realtime", self._is_realtime)
        self.add_channel("output", self._is_output)  # Catch-all, must be last
    
    def _is_coord(self, line: str) -> bool:
        """Check if line is a coordinate response."""
        return bool(self.COORD_PATTERN.match(line))
    
    def _is_message(self, line: str) -> bool:
        """Check if line is a MESSAGE output."""
        return line.upper().startswith("MESSAGE")
    
    def _is_info(self, line: str) -> bool:
        """Check if line is an INFO output."""
        return line.upper().startswith("INFO ")
    
    def _is_error(self, line: str) -> bool:
        """Check if line is an ERROR output."""
        return line.upper().startswith("ERROR")
    
    def _is_realtime(self, line: str) -> bool:
        """Check if line is a REALTIME output."""
        return line.upper().startswith("REALTIME")
    
    def _is_output(self, line: str) -> bool:
        """Catch-all for other outputs (OK, ABOUT, etc.)."""
        return not self._is_error(line)
    
    def add_channel(
        self,
        name: str,
        filter_func: Callable[[str], bool],
    ) -> None:
        """
        Add a new output channel.
        
        Args:
            name: Channel name for retrieval.
            filter_func: Function that returns True if a line belongs to this channel.
            
        Raises:
            ValueError: If channel already exists.
        """
        with self._lock:
            if name in self._queues:
                raise ValueError(f"Channel '{name}' already exists")
            self._queues[name] = Queue()
            self._filters[name] = filter_func
    
    def remove_channel(self, name: str) -> None:
        """
        Remove an output channel.
        
        Args:
            name: Channel name to remove.
        """
        with self._lock:
            self._queues.pop(name, None)
            self._filters.pop(name, None)
    
    def _read_loop(self) -> None:
        """Background thread that reads and routes output lines."""
        while self._running:
            try:
                line = self._stream.readline()
                
                # EOF check
                if line == "":
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Log raw stdout before routing
                if self._io_logger:
                    try:
                        self._io_logger("STDOUT", line)
                    except:
                        pass
                
                # Route to appropriate channel
                with self._lock:
                    for name, filter_func in self._filters.items():
                        if filter_func(line):
                            self._queues[name].put(line)
                            break
                            
            except Exception:
                # Stream closed or error
                break
    
    def get(
        self,
        channel: str,
        timeout: float = 0.0,
        reset: bool = False,
    ) -> str:
        """
        Get the next line from a channel.
        
        Args:
            channel: Channel name to read from.
            timeout: Maximum time to wait in seconds. 0 means no wait.
            reset: If True, clear old messages before reading.
            
        Returns:
            The next line, or empty string on timeout.
            
        Raises:
            ValueError: If channel doesn't exist.
        """
        with self._lock:
            if channel not in self._queues:
                valid = ", ".join(self._queues.keys())
                raise ValueError(f"Unknown channel '{channel}'. Valid: {valid}")
            queue = self._queues[channel]
        
        if reset:
            self._clear_queue(queue)
        
        try:
            if timeout <= 0:
                return queue.get_nowait()
            return queue.get(block=True, timeout=timeout)
        except Empty:
            return ""
    
    def get_nowait(self, channel: str) -> Optional[str]:
        """
        Get a line without waiting.
        
        Args:
            channel: Channel name to read from.
            
        Returns:
            The next line, or None if queue is empty.
        """
        result = self.get(channel, timeout=0.0)
        return result if result else None
    
    def get_all(self, channel: str) -> list[str]:
        """
        Get all available lines from a channel.
        
        Args:
            channel: Channel name to read from.
            
        Returns:
            List of all available lines.
        """
        with self._lock:
            if channel not in self._queues:
                raise ValueError(f"Unknown channel '{channel}'")
            queue = self._queues[channel]
        
        lines = []
        while True:
            try:
                lines.append(queue.get_nowait())
            except Empty:
                break
        return lines
    
    def clear(self, channel: str) -> None:
        """
        Clear all pending messages in a channel.
        
        Args:
            channel: Channel name to clear.
        """
        with self._lock:
            if channel in self._queues:
                self._clear_queue(self._queues[channel])
    
    def clear_all(self) -> None:
        """Clear all pending messages in all channels."""
        with self._lock:
            for queue in self._queues.values():
                self._clear_queue(queue)
    
    @staticmethod
    def _clear_queue(queue: Queue) -> None:
        """Clear a queue without blocking."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except Empty:
                break
    
    def stop(self) -> None:
        """Stop the background reader thread."""
        self._running = False
    
    @property
    def channels(self) -> list[str]:
        """Get list of available channel names."""
        with self._lock:
            return list(self._queues.keys())
