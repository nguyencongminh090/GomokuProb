"""
Command registry for handler management.

This module provides a central registry for command handlers
with plugin-style registration and lookup.
"""

from threading import RLock
from typing import Optional, Type

from pygomo.command.interface import (
    ICommandHandler,
    CommandContext,
    CommandResult,
)
from pygomo.command.hooks import HookManager, HookType


class CommandRegistry:
    """
    Central registry for command handlers.
    
    Provides:
        - Handler registration and lookup
        - Alias support
        - Hook integration
        - Plugin-style command extension
    
    Example:
        registry = CommandRegistry()
        registry.register(TurnHandler())
        registry.register(StartHandler())
        
        # Execute a command
        result = registry.execute(context)
    """
    
    def __init__(self):
        self._handlers: dict[str, ICommandHandler] = {}
        self._aliases: dict[str, str] = {}
        self._hooks = HookManager()
        self._lock = RLock()
    
    @property
    def hooks(self) -> HookManager:
        """Get the hook manager."""
        return self._hooks
    
    def register(self, handler: ICommandHandler) -> None:
        """
        Register a command handler.
        
        Args:
            handler: Handler instance to register.
            
        Raises:
            ValueError: If handler for command already exists.
        """
        with self._lock:
            name = handler.command_name.upper()
            
            if name in self._handlers:
                raise ValueError(f"Handler for '{name}' already registered")
            
            self._handlers[name] = handler
            
            # Register aliases
            for alias in handler.aliases:
                self._aliases[alias.upper()] = name
    
    def register_class(self, handler_class: Type[ICommandHandler]) -> None:
        """
        Register a handler by class (instantiates automatically).
        
        Args:
            handler_class: Handler class to instantiate and register.
        """
        self.register(handler_class())
    
    def unregister(self, command: str) -> bool:
        """
        Unregister a command handler.
        
        Args:
            command: Command name to unregister.
            
        Returns:
            True if handler was removed, False if not found.
        """
        with self._lock:
            name = command.upper()
            real_name = self._aliases.get(name, name)
            
            if real_name not in self._handlers:
                return False
            
            handler = self._handlers.pop(real_name)
            
            # Remove aliases
            for alias in handler.aliases:
                self._aliases.pop(alias.upper(), None)
            
            return True
    
    def get(self, command: str) -> Optional[ICommandHandler]:
        """
        Get handler for a command.
        
        Args:
            command: Command name or alias.
            
        Returns:
            Handler if found, None otherwise.
        """
        with self._lock:
            name = command.upper()
            real_name = self._aliases.get(name, name)
            return self._handlers.get(real_name)
    
    def has(self, command: str) -> bool:
        """Check if a command has a registered handler."""
        return self.get(command) is not None
    
    def execute(self, context: CommandContext) -> CommandResult:
        """
        Execute a command using its registered handler.
        
        Args:
            context: Command execution context.
            
        Returns:
            CommandResult from handler.
        """
        handler = self.get(context.command)
        
        if handler is None:
            return CommandResult.error(
                f"No handler registered for command: {context.command}"
            )
        
        # Validate arguments
        if not handler.validate_args(*context.args, **context.kwargs):
            return CommandResult.error(
                f"Invalid arguments for command: {context.command}"
            )
        
        # Run pre-hooks
        self._hooks.run(HookType.PRE_EXECUTE, context)
        
        try:
            # Execute command
            result = handler.execute(context)
            
            # Run post-hooks
            self._hooks.run(HookType.POST_EXECUTE, context, result)
            
            return result
            
        except Exception as e:
            # Run error hooks
            self._hooks.run(HookType.ON_ERROR, context, e)
            return CommandResult.error(str(e))
    
    def list_commands(self) -> list[str]:
        """Get list of all registered command names."""
        with self._lock:
            return list(self._handlers.keys())
    
    def list_all(self) -> list[str]:
        """Get list of all commands including aliases."""
        with self._lock:
            return list(self._handlers.keys()) + list(self._aliases.keys())
    
    def clear(self) -> None:
        """Remove all registered handlers."""
        with self._lock:
            self._handlers.clear()
            self._aliases.clear()
