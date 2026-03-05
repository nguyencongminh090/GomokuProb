"""
Hook system for command pre/post processing.

This module provides a hook mechanism for intercepting
command execution at various stages.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from threading import RLock
from typing import Any, Callable, Optional, Union

from pygomo.command.interface import CommandContext, CommandResult


class HookType(Enum):
    """Types of hooks in command lifecycle."""
    PRE_EXECUTE = auto()    # Before command execution
    POST_EXECUTE = auto()   # After successful execution
    ON_ERROR = auto()       # On execution error
    ON_INFO = auto()        # On search info received


class IHook(ABC):
    """
    Interface for hook implementations.
    
    Hooks can intercept and modify command execution.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Hook identifier."""
        ...
    
    @property
    @abstractmethod
    def hook_type(self) -> HookType:
        """Type of hook."""
        ...
    
    @property
    def priority(self) -> int:
        """
        Hook priority (lower = earlier execution).
        Default is 100.
        """
        return 100
    
    @abstractmethod
    def execute(
        self,
        context: CommandContext,
        result: Optional[CommandResult] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Execute the hook.
        
        Args:
            context: Command context.
            result: Command result (for post-execute hooks).
            error: Exception (for error hooks).
        """
        ...


# Type alias for simple function hooks
HookFunction = Callable[[CommandContext, Optional[Any]], None]


class HookManager:
    """
    Manages hooks for command lifecycle events.
    
    Supports both class-based hooks (IHook) and simple functions.
    
    Example:
        manager = HookManager()
        
        # Register a function hook
        @manager.on(HookType.PRE_EXECUTE)
        def log_command(context, _):
            print(f"Executing: {context.command}")
        
        # Register a class hook
        manager.add(MyHook())
    """
    
    def __init__(self):
        self._hooks: dict[HookType, list[tuple[int, Union[IHook, HookFunction]]]] = {
            hook_type: [] for hook_type in HookType
        }
        self._lock = RLock()
    
    def add(self, hook: IHook) -> None:
        """
        Add a hook.
        
        Args:
            hook: Hook instance to add.
        """
        with self._lock:
            hooks = self._hooks[hook.hook_type]
            hooks.append((hook.priority, hook))
            hooks.sort(key=lambda x: x[0])
    
    def add_function(
        self,
        hook_type: HookType,
        func: HookFunction,
        priority: int = 100,
    ) -> None:
        """
        Add a function as a hook.
        
        Args:
            hook_type: When to execute.
            func: Function to call.
            priority: Execution order (lower = first).
        """
        with self._lock:
            hooks = self._hooks[hook_type]
            hooks.append((priority, func))
            hooks.sort(key=lambda x: x[0])
    
    def on(
        self,
        hook_type: HookType,
        priority: int = 100,
    ) -> Callable[[HookFunction], HookFunction]:
        """
        Decorator for registering function hooks.
        
        Example:
            @manager.on(HookType.PRE_EXECUTE)
            def my_hook(context, result):
                pass
        """
        def decorator(func: HookFunction) -> HookFunction:
            self.add_function(hook_type, func, priority)
            return func
        return decorator
    
    def remove(self, hook: Union[IHook, HookFunction]) -> bool:
        """
        Remove a hook.
        
        Args:
            hook: Hook to remove.
            
        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            for hook_type in HookType:
                hooks = self._hooks[hook_type]
                for i, (_, h) in enumerate(hooks):
                    if h == hook or (isinstance(h, IHook) and h.name == getattr(hook, 'name', None)):
                        hooks.pop(i)
                        return True
            return False
    
    def run(
        self,
        hook_type: HookType,
        context: CommandContext,
        result_or_error: Any = None,
    ) -> None:
        """
        Run all hooks of a specific type.
        
        Args:
            hook_type: Type of hooks to run.
            context: Command context.
            result_or_error: Result or error for post/error hooks.
        """
        with self._lock:
            hooks = list(self._hooks[hook_type])
        
        for _, hook in hooks:
            try:
                if isinstance(hook, IHook):
                    if hook_type == HookType.ON_ERROR:
                        hook.execute(context, error=result_or_error)
                    else:
                        hook.execute(context, result=result_or_error)
                else:
                    # Function hook
                    hook(context, result_or_error)
            except Exception:
                # Hooks should not break command execution
                pass
    
    def clear(self, hook_type: Optional[HookType] = None) -> None:
        """
        Clear hooks.
        
        Args:
            hook_type: Specific type to clear, or None for all.
        """
        with self._lock:
            if hook_type:
                self._hooks[hook_type].clear()
            else:
                for ht in HookType:
                    self._hooks[ht].clear()
    
    def count(self, hook_type: Optional[HookType] = None) -> int:
        """Get number of registered hooks."""
        with self._lock:
            if hook_type:
                return len(self._hooks[hook_type])
            return sum(len(hooks) for hooks in self._hooks.values())
