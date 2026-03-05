"""
Search command handlers.

Handlers for search-related commands:
STOP, YXNBEST, YXBALANCEONE, YXBALANCETWO
"""

import time
from typing import Optional

from pygomo.command.interface import CommandContext, CommandResult
from pygomo.command.handlers.base import BaseCommandHandler
from pygomo.protocol.models import Move, PlayResult, SearchInfo


class StopHandler(BaseCommandHandler):
    """Handler for STOP command."""
    
    @property
    def command_name(self) -> str:
        return "STOP"
    
    @property
    def aliases(self) -> list[str]:
        return ["YXSTOP"]
    
    def execute(self, context: CommandContext) -> CommandResult:
        self.send_command(context)
        # STOP doesn't expect a response
        return CommandResult.success()


class NBestHandler(BaseCommandHandler):
    """
    Handler for YXNBEST command.
    
    Requests multiple best moves from engine.
    """
    
    @property
    def command_name(self) -> str:
        return "YXNBEST"
    
    @property
    def requires_thinking(self) -> bool:
        return True
    
    def validate_args(self, *args, **kwargs) -> bool:
        # Requires count argument
        return len(args) >= 1 and isinstance(args[0], int)
    
    def execute(self, context: CommandContext) -> CommandResult:
        count = context.args[0] if context.args else 1
        
        self.send_command(context, count)
        
        # Wait for coordinate response
        all_info: list[SearchInfo] = []
        timeout = context.timeout or 120.0
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                return CommandResult.timeout()
            
            coord = context.router.get("coord", timeout=min(0.1, remaining))
            
            if coord:
                result_move = Move(coord)
                
                # Collect remaining messages
                messages = context.router.get_all("message")
                for msg in messages:
                    try:
                        info = context.protocol.parse_search_info(msg)
                        all_info.append(info)
                        if context.on_info:
                            context.on_info(info)
                    except Exception:
                        pass
                
                play_result = PlayResult(
                    move=result_move,
                    search_info=all_info[-1] if all_info else None,
                    all_info=all_info,
                )
                
                return CommandResult.success(play_result)
            
            # Collect search info
            messages = context.router.get_all("message")
            for msg in messages:
                try:
                    info = context.protocol.parse_search_info(msg)
                    all_info.append(info)
                    if context.on_info:
                        context.on_info(info)
                except Exception:
                    pass


class BalanceOneHandler(BaseCommandHandler):
    """Handler for YXBALANCEONE command."""
    
    @property
    def command_name(self) -> str:
        return "YXBALANCEONE"
    
    @property
    def requires_thinking(self) -> bool:
        return True
    
    def validate_args(self, *args, **kwargs) -> bool:
        # Requires bias value
        return len(args) >= 1
    
    def execute(self, context: CommandContext) -> CommandResult:
        bias = context.args[0] if context.args else 0
        
        self.send_command(context, bias)
        
        # Similar to TURN
        timeout = context.timeout or 120.0
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                return CommandResult.timeout()
            
            coord = context.router.get("coord", timeout=min(0.1, remaining))
            
            if coord:
                result_move = Move(coord)
                return CommandResult.success(PlayResult(move=result_move))
            
            # Collect info if callback provided
            if context.on_info:
                messages = context.router.get_all("message")
                for msg in messages:
                    try:
                        info = context.protocol.parse_search_info(msg)
                        context.on_info(info)
                    except Exception:
                        pass


class BalanceTwoHandler(BaseCommandHandler):
    """Handler for YXBALANCETWO command."""
    
    @property
    def command_name(self) -> str:
        return "YXBALANCETWO"
    
    @property
    def requires_thinking(self) -> bool:
        return True
    
    def validate_args(self, *args, **kwargs) -> bool:
        return len(args) >= 1
    
    def execute(self, context: CommandContext) -> CommandResult:
        bias = context.args[0] if context.args else 0
        
        self.send_command(context, bias)
        
        timeout = context.timeout or 120.0
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                return CommandResult.timeout()
            
            coord = context.router.get("coord", timeout=min(0.1, remaining))
            
            if coord:
                # BALANCE_TWO returns two moves
                coords = coord.split()
                moves = [Move(c) for c in coords]
                return CommandResult.success({"moves": moves})
            
            if context.on_info:
                messages = context.router.get_all("message")
                for msg in messages:
                    try:
                        info = context.protocol.parse_search_info(msg)
                        context.on_info(info)
                    except Exception:
                        pass
