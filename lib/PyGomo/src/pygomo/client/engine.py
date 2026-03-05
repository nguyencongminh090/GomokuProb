"""
High-level engine client API.

This module provides the main EngineClient class that serves
as the facade for engine communication.
"""

from typing import Optional, Callable, Union, Any

from pygomo.transport import SubprocessTransport, OutputChannelRouter
from pygomo.protocol import GomocupProtocol, IProtocol
from pygomo.protocol.models import (
    Move,
    PlayResult,
    SearchInfo,
    BoardPosition,
)
from pygomo.command import CommandRegistry, CommandContext, CommandResult
from pygomo.command.hooks import HookManager, HookType
from pygomo.command.handlers import register_all_handlers


class EngineClient:
    """
    High-level client for engine communication.
    
    Provides a clean, user-friendly API for interacting with
    Gomoku engines via the Gomocup protocol.
    
    Example::

        # Basic usage
        with EngineClient("/path/to/engine") as engine:
            engine.start(15)
            
            # Make a move and get response
            result = engine.turn("h8")
            print(f"Engine played: {result.move}")
            
            # With realtime callbacks
            def on_info(info: SearchInfo):
                print(f"Depth {info.depth}: {info.winrate_percent:.1f}%")
            
            result = engine.turn("i9", on_info=on_info)
            
            engine.quit()
    
    Advanced usage::

        # Access hooks for custom processing
        engine.hooks.on(HookType.PRE_EXECUTE)(my_hook)
        
        # Register custom command handler
        engine.registry.register(MyCustomHandler())
        
        # Raw command execution
        engine.execute("MYCOMMAND", arg1, arg2)
    """
    
    def __init__(
        self,
        executable_path: str,
        protocol: Optional[IProtocol] = None,
        auto_start: bool = False,
        **transport_kwargs,
    ):
        """
        Initialize the engine client.
        
        Args:
            executable_path: Path to engine executable.
            protocol: Protocol implementation (defaults to GomocupProtocol).
            auto_start: Whether to start engine immediately.
            **transport_kwargs: Additional args for transport (e.g., working_directory).
        """
        self._executable_path = executable_path
        self._protocol = protocol or GomocupProtocol()
        self._transport_kwargs = transport_kwargs
        
        # Components (initialized on start)
        self._transport: Optional[SubprocessTransport] = None
        self._router: Optional[OutputChannelRouter] = None
        self._registry = CommandRegistry()
        
        # Register built-in handlers
        register_all_handlers(self._registry)
        
        # Default settings
        self._default_timeout = 60.0
        self._board_size = 15
        self._is_started = False
        
        if auto_start:
            self.connect()
    
    # ==================== Properties ====================
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to engine."""
        return self._transport is not None and self._transport.is_running
    
    @property
    def is_started(self) -> bool:
        """Check if game has been started (START command sent)."""
        return self._is_started
    
    @property
    def process_id(self) -> Optional[int]:
        """Get engine process ID."""
        return self._transport.process_id if self._transport else None
    
    @property
    def protocol(self) -> IProtocol:
        """Get the protocol implementation."""
        return self._protocol
    
    @property
    def hooks(self) -> HookManager:
        """Get the hook manager for custom processing."""
        return self._registry.hooks
    
    @property
    def registry(self) -> CommandRegistry:
        """Get the command registry for custom handlers."""
        return self._registry
    
    @property
    def router(self) -> Optional[OutputChannelRouter]:
        """Get the output channel router."""
        return self._router
    
    # ==================== Connection ====================
    
    def connect(self) -> None:
        """
        Connect to the engine (start subprocess).
        
        Raises:
            RuntimeError: If already connected.
        """
        if self.is_connected:
            raise RuntimeError("Already connected to engine")
        
        self._transport = SubprocessTransport(
            self._executable_path,
            **self._transport_kwargs,
        )
        self._transport.start()
        self._router = self._transport.get_router()
    
    def disconnect(self, timeout: float = 5.0) -> None:
        """
        Disconnect from engine (stop subprocess).
        
        Args:
            timeout: Time to wait for graceful shutdown.
        """
        if self._transport:
            self._transport.stop(timeout=timeout)
            self._transport = None
            self._router = None
        self._is_started = False
    
    # ==================== Lifecycle Commands ====================
    
    def start(
        self,
        board_size: int = 15,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Start a new game with specified board size.
        
        Args:
            board_size: Board size (5-22, default 15).
            timeout: Command timeout.
            
        Returns:
            True if successful.
        """
        if not self.is_connected:
            self.connect()
        
        result = self._execute("START", board_size, timeout=timeout)
        
        if result.is_success:
            self._board_size = board_size
            self._is_started = True
            return True
        
        return False
    
    def restart(self, timeout: Optional[float] = None) -> bool:
        """
        Restart the current game.
        
        Returns:
            True if successful.
        """
        result = self._execute("RESTART", timeout=timeout)
        return result.is_success
    
    def quit(self) -> None:
        """Quit the engine and disconnect."""
        if self.is_connected:
            self._execute("END")
        self.disconnect()
    
    def about(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Get engine information.
        
        Returns:
            Engine info string, or None on failure.
        """
        result = self._execute("ABOUT", timeout=timeout)
        if result.is_success and result.data:
            return result.data.get("info")
        return None
    
    # ==================== Game Commands ====================
    
    def turn(
        self,
        move: Union[str, Move, tuple[int, int]],
        timeout: Optional[float] = None,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
    ) -> Optional[PlayResult]:
        """
        Send opponent's move and get engine's response.
        
        Args:
            move: Move in any format ("h8", "7,8", (7, 8), or Move).
            timeout: Maximum thinking time.
            on_info: Callback for realtime search info.
            
        Returns:
            PlayResult with engine's move, or None on failure.
        """
        if isinstance(move, tuple):
            move = Move(move)
        elif isinstance(move, str):
            move = Move(move)
        
        result = self._execute("TURN", move, timeout=timeout, on_info=on_info)
        
        if result.is_success:
            return result.data
        
        return None
    
    def begin(
        self,
        timeout: Optional[float] = None,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
    ) -> Optional[PlayResult]:
        """
        Request engine's first move (engine plays black).
        
        Args:
            timeout: Maximum thinking time.
            on_info: Callback for realtime search info.
            
        Returns:
            PlayResult with engine's move, or None on failure.
        """
        result = self._execute("BEGIN", timeout=timeout, on_info=on_info)
        
        if result.is_success:
            return result.data
        
        return None
    
    def board(
        self,
        position: BoardPosition,
        start_thinking: bool = True,
        timeout: Optional[float] = None,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
    ) -> Optional[PlayResult]:
        """
        Set up a position and optionally get engine's move.
        
        Args:
            position: BoardPosition with moves.
            start_thinking: Whether to trigger engine thinking.
            timeout: Maximum thinking time.
            on_info: Callback for realtime search info.
            
        Returns:
            PlayResult if thinking, None otherwise.
        """
        result = self._execute(
            "BOARD",
            position=position,
            start_thinking=start_thinking,
            timeout=timeout,
            on_info=on_info,
        )
        
        if result.is_success:
            return result.data
        
        return None
    
    def takeback(
        self,
        move: Union[str, Move],
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Take back a move.
        
        Args:
            move: Move to take back.
            timeout: Command timeout.
            
        Returns:
            True if successful.
        """
        if isinstance(move, str):
            move = Move(move)
        
        result = self._execute("TAKEBACK", move, timeout=timeout)
        return result.is_success
    
    def stop(self) -> None:
        """Stop engine thinking immediately."""
        self._execute("STOP")
    
    # ==================== Search Commands ====================
    
    def nbest(
        self,
        count: int = 5,
        timeout: Optional[float] = None,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
    ) -> Optional[PlayResult]:
        """
        Get multiple best moves from engine.
        
        Args:
            count: Number of best moves to request.
            timeout: Maximum thinking time.
            on_info: Callback for realtime search info.
            
        Returns:
            PlayResult with best move and all info.
        """
        result = self._execute("YXNBEST", count, timeout=timeout, on_info=on_info)
        
        if result.is_success:
            return result.data
        
    def nbest_time_limited(
        self,
        count: int = 5,
        time_limit: float = 1.0,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
    ) -> Optional[PlayResult]:
        """
        Execute YXNBEST for a fixed duration, then STOP and collect result.
        Synchronous/Blocking for time_limit duration.
        """
        if not self.is_connected:
            return None
            
        # 1. Send YXNBEST manually (bypass handler blocking)
        self.send_raw(f"YXNBEST {count}")
        
        # 2. Wait Loop
        import time
        start_time = time.time()
        
        # We need a small buffer to ensure we don't stop BEFORE the engine processes start
        time.sleep(0.001) 
        
        # Accumulate search info during the run
        all_info_accumulated: list[SearchInfo] = []
        
        while time.time() - start_time < time_limit:
            # Critical check for disconnection to prevent infinite loop and segfaults
            if not self.is_connected or not self._router:
                 return None

            if self._router:
                # Check for early finish (if engine found mate and output coord)
                coord_data = self._router.get("coord", timeout=0.05) 
                if coord_data:
                    return self._parse_and_return_result(coord_data, on_info, all_info_accumulated)
                
                # Check messages
                messages = self._router.get_all("message")
                for msg in messages:
                    try:
                        info = self._protocol.parse_search_info(msg)
                        all_info_accumulated.append(info)
                        if on_info:
                            on_info(info)
                    except: pass
            else:
                 time.sleep(0.05)
                 
        # 3. Send STOP
        self.stop()
        
        # 4. Read coordinate response
        # Wait up to 5s for response
        if self._router:
             coord_data = self._router.get("coord", timeout=5.0)
             if coord_data:
                 return self._parse_and_return_result(coord_data, on_info, all_info_accumulated)
        
        return None
    
    def _parse_and_return_result(
        self, 
        coord_data: str, 
        on_info: Optional[Callable], 
        accumulated_info: list[SearchInfo]
    ) -> PlayResult:
        """Helper to collect final messages and build PlayResult."""
        result_move = Move(coord_data)
        
        # Start with accumulated info
        all_info = list(accumulated_info)
        
        if self._router:
            # Collect remaining/final messages after STOP
            messages = self._router.get_all("message")
            for msg in messages:
                try:
                    info = self._protocol.parse_search_info(msg)
                    all_info.append(info)
                    if on_info:
                        on_info(info)
                except Exception:
                    pass
            
        return PlayResult(
            move=result_move,
            search_info=all_info[-1] if all_info else None,
            all_info=all_info,
        )
    
    def configure(self, **options) -> None:
        """
        Configure engine options.
        
        Args:
            **options: Configuration options as keyword arguments.
            
        Example::

            engine.configure(
                timeout_turn=5000,
                timeout_match=300000,
                max_memory=1073741824,
                thread_num=4,
            )
        """
        for key, value in options.items():
            self._execute("INFO", key.upper(), value)
    
    def set_time(
        self,
        turn_time_ms: Optional[int] = None,
        match_time_ms: Optional[int] = None,
        time_left_ms: Optional[int] = None,
    ) -> None:
        """
        Set time control options.
        
        Args:
            turn_time_ms: Time per move in milliseconds.
            match_time_ms: Total match time in milliseconds.
            time_left_ms: Remaining time in milliseconds.
        """
        if turn_time_ms is not None:
            self._execute("INFO", "TIMEOUT_TURN", turn_time_ms)
        if match_time_ms is not None:
            self._execute("INFO", "TIMEOUT_MATCH", match_time_ms)
        if time_left_ms is not None:
            self._execute("INFO", "TIME_LEFT", time_left_ms)
    
    def set_rule(self, rule: int) -> None:
        """
        Set game rule.
        
        Args:
            rule: Rule ID (0=freestyle, 1=standard, 4=renju).
        """
        self._execute("INFO", "RULE", rule)
    
    def set_threads(self, count: int) -> None:
        """
        Set number of search threads.
        
        Args:
            count: Number of threads.
        """
        self._execute("INFO", "THREAD_NUM", count)
    
    def set_memory(self, size_bytes: int) -> None:
        """
        Set maximum memory usage.
        
        Args:
            size_bytes: Max memory in bytes.
        """
        self._execute("INFO", "MAX_MEMORY", size_bytes)
    
    # ==================== Raw Execution ====================
    
    def execute(
        self,
        command: str,
        *args,
        timeout: Optional[float] = None,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
        **kwargs,
    ) -> CommandResult:
        """
        Execute a raw command.
        
        Args:
            command: Command name.
            *args: Positional arguments.
            timeout: Command timeout.
            on_info: Callback for search info.
            **kwargs: Keyword arguments.
            
        Returns:
            CommandResult from handler.
        """
        return self._execute(command, *args, timeout=timeout, on_info=on_info, **kwargs)
    
    def send_raw(self, command: str) -> None:
        """
        Send a raw command string directly to engine.
        
        Args:
            command: Raw command string.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to engine")
        self._transport.send(command)
    
    def receive_raw(
        self,
        channel: str = "output",
        timeout: float = 1.0,
    ) -> str:
        """
        Receive raw output from a channel.
        
        Args:
            channel: Channel name.
            timeout: Timeout in seconds.
            
        Returns:
            Raw output string.
        """
        if not self._router:
            raise RuntimeError("Not connected to engine")
        return self._router.get(channel, timeout=timeout)
    
    # ==================== Internal ====================
    
    def _execute(
        self,
        command: str,
        *args,
        timeout: Optional[float] = None,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
        **kwargs,
    ) -> CommandResult:
        """Internal command execution."""
        if not self.is_connected:
            return CommandResult.error("Not connected to engine")
        
        context = CommandContext(
            transport=self._transport,
            protocol=self._protocol,
            router=self._router,
            command=command,
            args=args,
            kwargs=kwargs,
            on_info=on_info,
            timeout=timeout or self._default_timeout,
        )
        
        return self._registry.execute(context)
    
    # ==================== Context Manager ====================
    
    def __enter__(self) -> "EngineClient":
        """Context manager entry."""
        if not self.is_connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.quit()
    
    def __del__(self) -> None:
        """Destructor for cleanup."""
        try:
            self.disconnect()
        except Exception:
            pass
