"""
Game command handlers.

Handlers for game-related commands:
TURN, BEGIN, BOARD, TAKEBACK
"""

import time
from typing import Optional, Callable

from pygomo.command.interface import CommandContext, CommandResult
from pygomo.command.handlers.base import BaseCommandHandler
from pygomo.protocol.models import Move, PlayResult, SearchInfo


class TurnHandler(BaseCommandHandler):
    """
    Handler for TURN command.
    
    Sends opponent's move and waits for engine's response.
    Collects search info during thinking.
    """
    
    @property
    def command_name(self) -> str:
        return "TURN"
    
    @property
    def requires_thinking(self) -> bool:
        return True
    
    def validate_args(self, *args, **kwargs) -> bool:
        # Requires move argument
        return len(args) >= 1
    
    def execute(self, context: CommandContext) -> CommandResult:
        move_arg = context.args[0]
        
        # Convert to Move if string
        if isinstance(move_arg, str):
            move = Move(move_arg)
        elif isinstance(move_arg, Move):
            move = move_arg
        else:
            return CommandResult.error(f"Invalid move: {move_arg}")
        
        # Send command
        self.send_command(context, move.to_numeric())
        
        # Collect search info while waiting
        all_info: list[SearchInfo] = []
        last_info: Optional[SearchInfo] = None
        
        timeout = context.timeout or 60.0
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                return CommandResult.timeout()
            
            # Check for final move
            coord = context.router.get("coord", timeout=min(0.1, remaining))
            
            if coord:
                # Got the result
                result_move = Move(coord)
                
                # Collect any remaining messages
                self._collect_all_info(context, all_info)
                if all_info:
                    last_info = all_info[-1]
                
                play_result = PlayResult(
                    move=result_move,
                    search_info=last_info,
                    all_info=all_info,
                )
                
                return CommandResult.success(play_result)
            
            # Collect search info
            self._collect_all_info(context, all_info, context.on_info)
    
    def _collect_all_info(
        self,
        context: CommandContext,
        all_info: list[SearchInfo],
        callback: Optional[Callable[[SearchInfo], None]] = None,
    ) -> None:
        """Collect all available search info messages."""
        messages = context.router.get_all("message")
        
        for msg in messages:
            try:
                info = context.protocol.parse_search_info(msg)
                all_info.append(info)
                if callback:
                    callback(info)
            except Exception:
                pass


class BeginHandler(BaseCommandHandler):
    """
    Handler for BEGIN command.
    
    Requests first move from engine (engine plays as black).
    """
    
    @property
    def command_name(self) -> str:
        return "BEGIN"
    
    @property
    def requires_thinking(self) -> bool:
        return True
    
    def execute(self, context: CommandContext) -> CommandResult:
        self.send_command(context)
        
        # Similar to TURN, collect info while waiting
        all_info: list[SearchInfo] = []
        last_info: Optional[SearchInfo] = None
        
        timeout = context.timeout or 60.0
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
                
                if all_info:
                    last_info = all_info[-1]
                
                play_result = PlayResult(
                    move=result_move,
                    search_info=last_info,
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


class BoardHandler(BaseCommandHandler):
    """
    Handler for BOARD command.
    
    Sets up a position and optionally starts thinking.
    """
    
    @property
    def command_name(self) -> str:
        return "BOARD"
    
    @property
    def aliases(self) -> list[str]:
        return ["YXBOARD"]
    
    @property
    def requires_thinking(self) -> bool:
        return True
    
    def execute(self, context: CommandContext) -> CommandResult:
        # Get position data from args or kwargs
        position = context.kwargs.get("position")
        start_thinking = context.kwargs.get("start_thinking", True)
        
        if not position:
            return CommandResult.error("No position provided")
        
        # Send BOARD or YXBOARD
        cmd = "BOARD" if start_thinking else "YXBOARD"
        context.transport.send(cmd)
        
        # Send position lines
        position_str = position.to_protocol_string()
        for line in position_str.split("\n"):
            context.transport.send(line)
        
        if not start_thinking:
            return CommandResult.success()
        
        # Wait for response like TURN
        all_info: list[SearchInfo] = []
        timeout = context.timeout or 60.0
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                return CommandResult.timeout()
            
            coord = context.router.get("coord", timeout=min(0.1, remaining))
            
            if coord:
                result_move = Move(coord)
                
                messages = context.router.get_all("message")
                for msg in messages:
                    try:
                        info = context.protocol.parse_search_info(msg)
                        all_info.append(info)
                    except Exception:
                        pass
                
                play_result = PlayResult(
                    move=result_move,
                    search_info=all_info[-1] if all_info else None,
                    all_info=all_info,
                )
                
                return CommandResult.success(play_result)
            
            # Collect info
            messages = context.router.get_all("message")
            for msg in messages:
                try:
                    info = context.protocol.parse_search_info(msg)
                    all_info.append(info)
                    if context.on_info:
                        context.on_info(info)
                except Exception:
                    pass


class TakebackHandler(BaseCommandHandler):
    """Handler for TAKEBACK command."""
    
    @property
    def command_name(self) -> str:
        return "TAKEBACK"
    
    def validate_args(self, *args, **kwargs) -> bool:
        # Requires position to take back
        return len(args) >= 1
    
    def execute(self, context: CommandContext) -> CommandResult:
        move_arg = context.args[0]
        
        if isinstance(move_arg, str):
            move = Move(move_arg)
        elif isinstance(move_arg, Move):
            move = move_arg
        else:
            return CommandResult.error(f"Invalid move: {move_arg}")
        
        self.send_command(context, move.to_numeric())
        
        if self.receive_ok(context):
            return CommandResult.success()
        
        return CommandResult.error("TAKEBACK command failed")
