"""
Configuration command handlers.

Handlers for configuration commands:
INFO
"""

from typing import Any

from pygomo.command.interface import CommandContext, CommandResult
from pygomo.command.handlers.base import BaseCommandHandler


class InfoHandler(BaseCommandHandler):
    """
    Handler for INFO command.
    
    Sets engine configuration options.
    """
    
    @property
    def command_name(self) -> str:
        return "INFO"
    
    def validate_args(self, *args, **kwargs) -> bool:
        # Requires key and value
        return len(args) >= 2
    
    def execute(self, context: CommandContext) -> CommandResult:
        if len(context.args) >= 2:
            key = context.args[0]
            value = context.args[1]
            self.send_command(context, key, value)
        elif context.kwargs:
            # Support dict-style: INFO(timeout_turn=5000)
            for key, value in context.kwargs.items():
                cmd_str = context.protocol.serialize_command(
                    "INFO",
                    key.upper(),
                    value,
                )
                context.transport.send(cmd_str)
        else:
            return CommandResult.error("INFO requires key and value")
        
        # INFO commands don't have responses
        return CommandResult.success()
    
    @staticmethod
    def format_options(options: dict[str, Any]) -> list[tuple[str, Any]]:
        """
        Format option dict for sending.
        
        Args:
            options: Dictionary of options.
            
        Returns:
            List of (key, value) tuples.
        """
        result = []
        for key, value in options.items():
            # Convert Python names to protocol names
            proto_key = key.upper().replace("_", "_")
            result.append((proto_key, value))
        return result
