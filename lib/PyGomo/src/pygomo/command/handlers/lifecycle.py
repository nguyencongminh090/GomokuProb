"""
Lifecycle command handlers.

Handlers for engine lifecycle commands:
START, RESTART, END, ABOUT
"""

from pygomo.command.interface import CommandContext, CommandResult
from pygomo.command.handlers.base import BaseCommandHandler


class StartHandler(BaseCommandHandler):
    """Handler for START command."""
    
    @property
    def command_name(self) -> str:
        return "START"
    
    def validate_args(self, *args, **kwargs) -> bool:
        # Requires board size argument
        return len(args) >= 1 and isinstance(args[0], int)
    
    def execute(self, context: CommandContext) -> CommandResult:
        board_size = context.args[0] if context.args else 15
        
        self.send_command(context, board_size)
        
        if self.receive_ok(context):
            return CommandResult.success({"board_size": board_size})
        
        return CommandResult.error("START command failed")


class RestartHandler(BaseCommandHandler):
    """Handler for RESTART command."""
    
    @property
    def command_name(self) -> str:
        return "RESTART"
    
    def execute(self, context: CommandContext) -> CommandResult:
        self.send_command(context)
        
        if self.receive_ok(context):
            return CommandResult.success()
        
        return CommandResult.error("RESTART command failed")


class EndHandler(BaseCommandHandler):
    """Handler for END command."""
    
    @property
    def command_name(self) -> str:
        return "END"
    
    def execute(self, context: CommandContext) -> CommandResult:
        self.send_command(context)
        # END doesn't expect a response
        return CommandResult.success()


class AboutHandler(BaseCommandHandler):
    """Handler for ABOUT command."""
    
    @property
    def command_name(self) -> str:
        return "ABOUT"
    
    def execute(self, context: CommandContext) -> CommandResult:
        self.send_command(context)
        
        timeout = context.timeout or 5.0
        response = context.router.get("output", timeout=timeout)
        
        if response:
            return CommandResult.success({"info": response})
        
        return CommandResult.error("No response from ABOUT command")
