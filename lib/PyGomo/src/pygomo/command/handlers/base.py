"""
Base command handler implementation.

This module provides a base class for command handlers
with common functionality.
"""

from abc import abstractmethod
from typing import Any, Optional

from pygomo.command.interface import (
    ICommandHandler,
    CommandContext,
    CommandResult,
)


class BaseCommandHandler(ICommandHandler):
    """
    Base class for command handlers.
    
    Provides common functionality for sending commands
    and receiving responses.
    """
    
    @property
    def aliases(self) -> list[str]:
        """Default: no aliases."""
        return []
    
    @property
    def requires_thinking(self) -> bool:
        """Default: no thinking required."""
        return False
    
    def validate_args(self, *args, **kwargs) -> bool:
        """Default: always valid."""
        return True
    
    def send_command(
        self,
        context: CommandContext,
        *args: Any,
    ) -> None:
        """
        Send command to engine.
        
        Args:
            context: Execution context.
            *args: Additional arguments.
        """
        cmd_str = context.protocol.serialize_command(
            self.command_name,
            *args,
        )
        context.transport.send(cmd_str)
    
    def receive_ok(
        self,
        context: CommandContext,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for OK response.
        
        Args:
            context: Execution context.
            timeout: Timeout in seconds.
            
        Returns:
            True if OK received.
        """
        timeout = timeout or context.timeout or 5.0
        response = context.router.get("output", timeout=timeout)
        return response.upper() == "OK"
    
    def receive_coord(
        self,
        context: CommandContext,
        timeout: Optional[float] = None,
    ) -> Optional[tuple[int, int]]:
        """
        Wait for coordinate response.
        
        Args:
            context: Execution context.
            timeout: Timeout in seconds.
            
        Returns:
            (x, y) tuple or None on timeout.
        """
        timeout = timeout or context.timeout or 30.0
        response = context.router.get("coord", timeout=timeout)
        
        if not response:
            return None
        
        return context.protocol.parse_coord(response)
    
    def collect_search_info(
        self,
        context: CommandContext,
    ) -> None:
        """
        Collect and dispatch search info messages.
        
        Polls message channel and invokes on_info callback.
        
        Args:
            context: Execution context.
        """
        if not context.on_info:
            return
        
        # Get all available messages without blocking
        messages = context.router.get_all("message")
        
        for msg in messages:
            try:
                info = context.protocol.parse_search_info(msg)
                context.on_info(info)
            except Exception:
                pass  # Skip malformed messages
