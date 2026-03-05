#!/usr/bin/env python3
"""
PyGomo Console Gomoku Game

A simple console-based Gomoku game for testing PyGomo library.
Human vs Computer (Rapfi engine).

Features:
- Standard and Swap2 rules
- Configurable board size
- Time control settings
- Start from empty or custom position
- Realtime search info display

Usage:
    python console_game.py
    python console_game.py --engine /path/to/engine
    python console_game.py --rule swap2 --time 30000
    python console_game.py --position "h8,i8,h7"  # Black, White, Black...
"""

import sys
import os
import argparse
import json
from typing import Optional



from pygomo import EngineClient, SearchInfo, Move, BoardPosition
from pygomo.board import BitBoard, RenjuBitBoard, BLACK, WHITE, EMPTY


class ConsoleBoard:
    """
    Console-based Gomoku board using BitBoard for efficient operations.
    
    Wraps BitBoard to add console display functionality with colors
    and win detection integration.
    """
    
    # ANSI colors for terminal
    RESET       = "\033[0m"          # Reset to default attributes
    BLACK_STONE = "\033[1;33m"       # Yellow for visibility
    WHITE_STONE = "\033[1;37m"       # White
    HIGHLIGHT   = "\033[1;32m"       # Green for last move
    DIM         = "\033[2m"          # Dim for dots
    WIN_COLOR   = "\033[1;31m"       # Red for winning line
    
    def __init__(self, size: int = 15, use_colors: bool = True, use_renju: bool = False):
        self.use_colors = use_colors
        
        # Use RenjuBitBoard for Renju rules, BitBoard otherwise
        if use_renju:
            self._board = RenjuBitBoard(_size=size)
        else:
            self._board = BitBoard(_size=size)
        
        self._win_line: set[tuple[int, int]] = set()
    
    # --- Delegate to BitBoard ---
    
    @property
    def size(self) -> int:
        return self._board.size
    
    @property
    def move_count(self) -> int:
        return self._board.move_count
    
    @property
    def current_player(self) -> int:
        return self._board.current_player
    
    @property
    def last_move(self) -> Optional[Move]:
        return self._board.last_move
    
    @property
    def move_history(self) -> list[Move]:
        return self._board.get_move_history()
    
    @property
    def hash(self) -> int:
        return self._board.hash
    
    def place(self, move: Move, player: Optional[int] = None) -> bool:
        """Place a stone on the board."""
        result = self._board.place(move, player)
        if result:
            # Check for win
            win_info = self._board.check_win(move)
            if win_info:
                self._win_line = {(m.col, m.row) for m in win_info.line}
        return result
    
    def is_valid(self, move: Move) -> bool:
        return self._board.is_valid(move)
    
    def is_empty(self, move: Move) -> bool:
        return self._board.is_empty(move)
    
    def get(self, move: Move) -> int:
        return self._board.get(move)
    
    def check_win(self, move: Optional[Move] = None):
        """Check for win and return WinInfo if found."""
        return self._board.check_win(move)
    
    def undo(self) -> Optional[Move]:
        """Undo the last move."""
        result = self._board.undo()
        self._win_line.clear()  # Clear win highlight
        return result
    
    # --- Display ---
    
    def _c(self, code: str) -> str:
        """Apply color if enabled."""
        return code if self.use_colors else ""
    
    def display(self, last_move: Optional[Move] = None):
        """Print the board to console with improved formatting."""
        if last_move is None:
            last_move = self.last_move
        
        # Column headers
        print()
        print("    ", end="")
        for c in range(self.size):
            print(f" {chr(ord('A') + c)}", end="")
        print()
        
        # Top border
        print("   +" + "-" * (self.size * 2 + 1) + "+")
        
        # Board rows
        for r in range(self.size - 1, -1, -1):
            print(f"{r + 1:2} |", end="")
            
            for c in range(self.size):
                cell = self._board.get(Move((c, r)))
                is_last = last_move and last_move.col == c and last_move.row == r
                is_win = (c, r) in self._win_line
                
                if cell == EMPTY:
                    print(f" {self._c(self.DIM)}.{self._c(self.RESET)}", end="")
                elif cell == BLACK:
                    if is_win:
                        print(f" {self._c(self.WIN_COLOR)}X{self._c(self.RESET)}", end="")
                    elif is_last:
                        print(f" {self._c(self.HIGHLIGHT)}X{self._c(self.RESET)}", end="")
                    else:
                        print(f" {self._c(self.BLACK_STONE)}X{self._c(self.RESET)}", end="")
                else:  # WHITE
                    if is_win:
                        print(f" {self._c(self.WIN_COLOR)}O{self._c(self.RESET)}", end="")
                    elif is_last:
                        print(f" {self._c(self.HIGHLIGHT)}O{self._c(self.RESET)}", end="")
                    else:
                        print(f" {self._c(self.WHITE_STONE)}O{self._c(self.RESET)}", end="")
            
            print(f" |{r + 1:2}")
        
        # Bottom border
        print("   +" + "-" * (self.size * 2 + 1) + "+")
        
        # Column headers
        print("    ", end="")
        for c in range(self.size):
            print(f" {chr(ord('A') + c)}", end="")
        
        # Info
        if last_move:
            print(f"   Last: {last_move}", end="")
        print(f"  Hash: {self.hash & 0xFFFF:04X}\n")
    
    def to_position(self) -> BoardPosition:
        """Convert board state to BoardPosition for engine."""
        pos = BoardPosition()
        for i, move in enumerate(self.move_history):
            color = BoardPosition.SELF if i % 2 == 0 else BoardPosition.OPPONENT
            pos.add_move(move, color)
        return pos
    
    def load_position(self, moves_str: str) -> int:
        """Load position from moves string (supports multiple formats)."""
        if not moves_str:
            return 0
        
        # Parse input (handles URLs, concatenated, or comma-separated)
        # Returns list[Move] directly
        moves = self.parse_position_input(moves_str, size=self.size)
        loaded = 0
        
        for move in moves:
            if self.place(move):
                loaded += 1
        
        return loaded
    
    @staticmethod
    def parse_position_input(input_str: str, size: int = 20) -> list[Move]:
        """
        Extract valid Gomoku moves from any input string.
        
        Uses character iteration with numeric validation.
        Returns Move objects for direct use.
        
        Args:
            input_str: Any string potentially containing moves
            size: Board size (default 20, supports any size)
        
        Examples:
            "asdfb4d5dasa2ddsd7" -> [Move("b4"), Move("d5"), Move("a2"), Move("d7")]
            "h8,i8,h7" -> [Move("h8"), Move("i8"), Move("h7")]
        """
        moves = []
        i = 0
        text = input_str.lower()
        
        while i < len(text):
            char = text[i]
            
            # Check if this is a valid column letter (a-z, supports up to 26 cols)
            if char.isalpha() and 'a' <= char <= 'z':
                col = ord(char) - ord('a')  # Convert to 0-indexed numeric
                
                # Try to read row number (1-2 digits)
                num_str = ""
                j = i + 1
                while j < len(text) and text[j].isdigit() and len(num_str) < 2:
                    num_str += text[j]
                    j += 1
                
                if num_str:
                    row = int(num_str) - 1  # Convert to 0-indexed
                    
                    # Validate: both col and row must be within board size
                    if 0 <= col < size and 0 <= row < size:
                        try:
                            moves.append(Move((col, row)))
                        except ValueError:
                            pass  # Invalid move, skip
                        i = j  # Skip past the number
                        continue
            
            i += 1
        
        return moves


class GomokuGame:
    """Console Gomoku game controller."""
    
    RULES = {
        "standard": 1,
        "freestyle": 0,
        "renju": 4,
    }
    
    # Allowed commands to send directly to engine via !command
    ALLOWED_ENGINE_COMMANDS = frozenset({"info"})
    
    def __init__(
        self,
        engine_path: str,
        board_size: int = 15,
        rule: str = "standard",
        turn_time_ms: int = 30000,
        match_time_ms: int = 300000,
        initial_position: str = "",
        time_left_ms: Optional[int] = None,
        config_file: str = "console_config.json"
    ):
        self.engine_path = engine_path
        self.board_size = board_size
        self.rule = rule
        self.turn_time_ms = turn_time_ms
        self.match_time_ms = match_time_ms
        self.initial_position = initial_position
        self.time_left_ms = time_left_ms
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.show_info = self.config.get("show_info", True)
        
        self.board = ConsoleBoard(
            size=board_size,
            use_colors=True,
            use_renju=(rule == "renju")
        )
        self.engine: Optional[EngineClient] = None
        self.human_color = 1  # 1=Black, 2=White
        self.game_over = False
        self.last_info: Optional[SearchInfo] = None

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default config if file not found or invalid
            return {
                "show_info": True,
                "pv_length": 5,
                "fields": ["depth", "eval", "winrate", "pv"]
            }
        
    def start(self):
        """Start the game with post-game menu."""
        self._print_header()
        self._setup_engine()
        
        while True:
            self._reset_board()
            self._load_initial_position()
            self._choose_color()
            self._game_loop()
            
            # Post-game menu
            action = self._post_game_menu()
            if action == "exit":
                break
            elif action == "replay":
                # Replay same position with same color
                continue
            elif action == "new":
                # Position already set in _post_game_menu (empty or custom)
                continue
        
        self._cleanup()
    
    def _reset_board(self):
        """Reset the board for a new game."""
        use_renju = self.rule.lower() == "renju"
        self.board = ConsoleBoard(
            self.board_size,
            use_colors=self.board.use_colors,
            use_renju=use_renju
        )
        self.game_over = False
        self.last_info = None
        
        # Restart engine for new game
        if self.engine:
            try:
                self.engine.restart()
                print("[*] Engine restarted for new game.")
            except Exception as e:
                print(f"[!] Failed to restart engine: {e}")
    
    def _post_game_menu(self) -> str:
        """Show post-game menu and return action."""
        print("\n" + "=" * 45)
        print("  Game ended. What would you like to do?")
        print("=" * 45)
        print("  1. Replay (same position)")
        print("  2. New game (empty board)")
        print("  3. New game (custom position)")
        print("  4. Exit")
        print("=" * 45)
        
        while True:
            try:
                choice = input("Enter choice (1/2/3/4): ").strip()
                if choice == "1":
                    return "replay"
                elif choice == "2":
                    self.initial_position = ""
                    return "new"
                elif choice == "3":
                    pos = input("Enter position (e.g. h8,i8,h7): ").strip()
                    self.initial_position = pos
                    return "new"
                elif choice == "4" or choice.lower() in ("q", "quit", "exit"):
                    return "exit"
                print("[!] Invalid choice. Enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print()
                return "exit"
    
    def _print_header(self):
        """Print game header."""
        print("=" * 55)
        print("    PyGomo Console Gomoku - Human vs Computer")
        print("=" * 55)
        print(f"  Board Size  : {self.board_size}x{self.board_size}")
        print(f"  Rule        : {self.rule.upper()}")
        print(f"  Turn Time   : {self.turn_time_ms / 1000:.1f}s")
        print(f"  Match Time  : {self.match_time_ms / 1000:.1f}s")
        if self.time_left_ms:
            print(f"  Time Left   : {self.time_left_ms / 1000:.1f}s")
        if self.initial_position:
            print(f"  Position    : {self.initial_position}")
        print("=" * 55)
    
    def _setup_engine(self):
        """Initialize and configure engine."""
        print("\n[*] Connecting to engine...")
        
        self.engine = EngineClient(
            self.engine_path,
            working_directory=os.path.dirname(self.engine_path),
        )
        self.engine.connect()
        
        # Get engine info
        about = self.engine.about(timeout=5.0)
        if about:
            print(f"[*] Engine: {about[:50]}...")
        
        # Start game
        print(f"[*] Starting game (size={self.board_size})...")
        if not self.engine.start(self.board_size):
            print("[!] Failed to start game!")
            sys.exit(1)
        
        # Configure rule
        rule_id = self.RULES.get(self.rule.lower(), 1)
        self.engine.set_rule(rule_id)
        
        # Configure time
        self.engine.set_time(
            turn_time_ms=self.turn_time_ms,
            match_time_ms=self.match_time_ms,
            time_left_ms=self.time_left_ms,
        )
        
        print("[*] Engine ready!\n")
    
    def _load_initial_position(self):
        """Load initial position if specified."""
        if self.initial_position:
            loaded = self.board.load_position(self.initial_position)
            if loaded > 0:
                print(f"[*] Loaded {loaded} moves from position.")
                
                # Send position to engine
                pos = self.board.to_position()
                self.engine.board(pos, start_thinking=False)
                print("[*] Position sent to engine.")
    
    def _choose_color(self):
        """Let player choose color."""
        # If position has odd moves, next is White
        if len(self.board.move_history) % 2 == 1:
            next_color = "White"
        else:
            next_color = "Black"
            
        print(f"\nNext to play: {next_color}")
        print("Choose your color:")
        print("  1. Black (X)")
        print("  2. White (O)")
        
        while True:
            choice = input("Enter choice (1/2): ").strip()
            if choice == "1":
                self.human_color = 1
                break
            elif choice == "2":
                self.human_color = 2
                break
            print("Invalid choice. Enter 1 or 2.")
        
        # current_player is tracked by BitBoard based on move count
    
    def _check_game_end(self, last_move: Move) -> bool:
        """Check if game ended (win or draw) and display result."""
        win_info = self.board.check_win(last_move)
        if win_info:
            self.board.display(last_move)
            winner = "Black (X)" if win_info.winner == 1 else "White (O)"
            print(f"\n{'='*50}")
            print(f"  🎉 {winner} WINS! 🎉")
            print(f"  Winning line: {win_info.direction}")
            print(f"{'='*50}\n")
            self.game_over = True
            return True
        
        # Check for draw (board full)
        if self.board.move_count >= self.board.size * self.board.size:
            self.board.display(last_move)
            print(f"\n{'='*50}")
            print(f"  DRAW! Board is full.")
            print(f"{'='*50}\n")
            self.game_over = True
            return True
        
        return False
    
    def _game_loop(self):
        """Main game loop."""
        last_move = self.board.move_history[-1] if self.board.move_history else None
        
        # If it's engine's turn, let it play first
        if self.board.current_player != self.human_color:
            if not self.board.move_history:
                # Empty board, engine plays first
                last_move = self._engine_turn(is_first=True)
            else:
                # Resume from position
                last_move = self._engine_turn_from_position()
            
            if last_move and self._check_game_end(last_move):
                return
        
        while not self.game_over:
            self.board.display(last_move)
            
            if self.board.current_player == self.human_color:
                # Human turn
                move = self._human_turn()
                if move is None:
                    if self.game_over:
                        break
                    # Swap command - let engine play this turn
                    last_move = self._engine_turn_from_position()
                    if last_move and self._check_game_end(last_move):
                        return
                    continue
                last_move = move
                
                # Check human win
                if self._check_game_end(last_move):
                    return
                
                # Engine responds
                if not self.game_over:
                    last_move = self._engine_turn(human_move=move)
                    if last_move and self._check_game_end(last_move):
                        return
            else:
                # Engine's turn (after swap)
                last_move = self._engine_turn_from_position()
                if last_move and self._check_game_end(last_move):
                    return
        
        self.board.display(last_move)
        print("\n[*] Game Over!")
    
    def _human_turn(self) -> Optional[Move]:
        """Handle human player's turn."""
        color_name = "Black (X)" if self.human_color == 1 else "White (O)"
        print(f"Your turn ({color_name})")
        print("Commands: move, quit, undo, swap, info, time <ms>, !info <key> <val>")
        
        while True:
            try:
                user_input = input("> ").strip().lower()
                
                if user_input == "quit" or user_input == "q":
                    self.game_over = True
                    return None
                
                if user_input == "undo":
                    # Take back last 2 moves (human + engine)
                    if self.board.move_count >= 2:
                        self.board.undo()  # Undo engine's move
                        self.board.undo()  # Undo human's move
                        
                        # Sync with engine
                        print("[*] Took back 2 moves. Syncing engine...")
                        self.board.display()
                        
                        # Resend board to engine
                        pos = self.board.to_position()
                        self.engine.board(pos, start_thinking=False)
                    else:
                        print("[!] No moves to undo")
                    continue
                
                if user_input == "info":
                    if self.last_info:
                        print(f"Last search: {self.last_info}")
                    else:
                        print("[!] No search info available")
                    continue
                
                if user_input.startswith("time "):
                    try:
                        new_time = int(user_input[5:])
                        self.turn_time_ms = new_time
                        self.engine.set_time(turn_time_ms=new_time)
                        print(f"[*] Turn time set to {new_time / 1000:.1f}s")
                    except ValueError:
                        print("[!] Invalid time value")
                    continue
                
                # Swap sides command (only exact 'swap', not 'swap swap')
                if user_input == "swap":
                    self.human_color = 3 - self.human_color
                    new_color = "Black (X)" if self.human_color == 1 else "White (O)"
                    print(f"[*] You are now playing as {new_color}. Engine will play this turn.")
                    return None  # Signal to let engine play
                
                # Direct engine command with ! prefix
                if user_input.startswith("!"):
                    cmd_parts = user_input[1:].split()
                    if cmd_parts:
                        cmd_name = cmd_parts[0].lower()
                        if cmd_name in self.ALLOWED_ENGINE_COMMANDS:
                            # Send to engine
                            full_cmd = user_input[1:].upper()
                            self.engine.send_raw(full_cmd)
                            print(f"[*] Sent to engine: {full_cmd}")
                        else:
                            print(f"[!] Command '{cmd_name}' not allowed. Allowed: {', '.join(self.ALLOWED_ENGINE_COMMANDS)}")
                    continue
                
                # Parse move
                move = Move(user_input)
                
                if not self.board.is_valid(move):
                    print(f"[!] Invalid position: {move}")
                    continue
                
                if not self.board.is_empty(move):
                    print(f"[!] Position already occupied: {move}")
                    continue
                
                # Place move
                self.board.place(move, self.human_color)
                return move
                
            except ValueError as e:
                print(f"[!] Invalid input: {e}")
                print("    Use format like 'h8', 'a1', 'o15'")
            except KeyboardInterrupt:
                print("\n[*] Game interrupted")
                self.game_over = True
                return None
    
    def _print_info(self, info: SearchInfo):
        """Print search info based on configuration."""
        if not self.show_info or info.depth == 0:
            return
            
        parts = []
        fields = self.config.get("fields", ["depth", "eval", "winrate", "pv"])
        pv_len = self.config.get("pv_length", 5)
        
        for field in fields:
            if field == "depth":
                parts.append(f"d{info.depth}/{info.sel_depth}")
            elif field == "eval":
                parts.append(f"ev {info.eval.raw_value:>4}")
            elif field == "winrate" or field == "wr":
                parts.append(f"WR {info.winrate_percent:4.1f}%")
            elif field == "nodes":
                parts.append(f"n {info.nodes}")
            elif field == "nps":
                parts.append(f"nps {info.nps}")
            elif field == "time":
                parts.append(f"t {info.time_ms}ms")
            elif field == "lenPV":
                parts.append(f"len {len(info.pv)}")
            elif field == "pv":
                pv_str = " ".join(str(m) for m in info.pv[:pv_len])
                parts.append(f"{pv_str}")

        print(f"  {' | '.join(parts)}")

    def _engine_turn(
        self,
        human_move: Optional[Move] = None,
        is_first: bool = False,
    ) -> Optional[Move]:
        """Handle engine's turn."""
        engine_color = 3 - self.human_color
        color_name = "Black (X)" if engine_color == 1 else "White (O)"
        print(f"\nEngine thinking ({color_name})...")
        
        def on_info(info: SearchInfo):
            """Callback for realtime search info."""
            self.last_info = info
            self._print_info(info)
        
        try:
            if is_first:
                # Engine plays first move
                result = self.engine.begin(
                    timeout=float(self.turn_time_ms) / 1000 + 5,
                    on_info=on_info,
                )
            else:
                # Respond to human move
                result = self.engine.turn(
                    human_move,
                    timeout=float(self.turn_time_ms) / 1000 + 5,
                    on_info=on_info,
                )
            
            if result and result.move:
                print(f"\n[*] Engine plays: {result.move}")
                self.board.place(result.move, engine_color)
                return result.move
            else:
                print("[!] Engine failed to return a move")
                self.game_over = True
                return None
                
        except Exception as e:
            print(f"[!] Engine error: {e}")
            self.game_over = True
            return None
    
    def _engine_turn_from_position(self) -> Optional[Move]:
        """Engine plays from loaded position."""
        engine_color = 3 - self.human_color
        color_name = "Black (X)" if engine_color == 1 else "White (O)"
        print(f"\nEngine thinking from position ({color_name})...")
        
        def on_info(info: SearchInfo):
            self.last_info = info
            self._print_info(info)
        
        try:
            # Get move from current position
            pos = self.board.to_position()
            result = self.engine.board(
                pos,
                start_thinking=True,
                timeout=float(self.turn_time_ms) / 1000 + 5,
                on_info=on_info,
            )
            
            if result and result.move:
                print(f"\n[*] Engine plays: {result.move}")
                self.board.place(result.move, engine_color)
                return result.move
            else:
                print("[!] Engine failed to return a move")
                self.game_over = True
                return None
                
        except Exception as e:
            print(f"[!] Engine error: {e}")
            self.game_over = True
            return None
    
    def _cleanup(self):
        """Clean up resources."""
        if self.engine:
            print("\n[*] Shutting down engine...")
            self.engine.quit()
            print("[*] Done.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyGomo Console Gomoku Game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Default game
  %(prog)s --rule swap2              # Swap2 rule
  %(prog)s --time 10000              # 10s per move
  %(prog)s --position "h8,i8,h7"     # Start from position
  %(prog)s --time-left 60000         # 60s remaining
        """
    )
    parser.add_argument(
        "--engine", "-e",
        default="./engine/rapfi",
        help="Path to engine executable (default: ./engine/rapfi)"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=15,
        help="Board size (default: 15)"
    )
    parser.add_argument(
        "--rule", "-r",
        choices=["standard", "freestyle", "renju"],
        default="standard",
        help="Game rule (default: standard)"
    )
    parser.add_argument(
        "--time", "-t",
        type=int,
        default=30000,
        help="Turn time in milliseconds (default: 30000)"
    )
    parser.add_argument(
        "--match-time", "-m",
        type=int,
        default=300000,
        help="Match time in milliseconds (default: 300000)"
    )
    parser.add_argument(
        "--time-left", "-l",
        type=int,
        default=None,
        help="Time left in milliseconds (for resuming games)"
    )
    parser.add_argument(
        "--position", "-p",
        type=str,
        default="",
        help="Initial position as comma-separated moves (e.g., 'h8,i8,h7')"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Don't show realtime search info"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    args = parser.parse_args()
    
    # Resolve engine path
    engine_path = args.engine
    if not os.path.isabs(engine_path):
        engine_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            engine_path
        )
    
    if not os.path.exists(engine_path):
        print(f"Error: Engine not found at {engine_path}")
        sys.exit(1)
    
    # Start game
    game = GomokuGame(
        engine_path=engine_path,
        board_size=args.size,
        rule=args.rule,
        turn_time_ms=args.time,
        match_time_ms=args.match_time,
        time_left_ms=args.time_left,
        initial_position=args.position,
    )
    
    try:
        game.start()
    except KeyboardInterrupt:
        print("\n\n[*] Game interrupted by user")
        game._cleanup()


if __name__ == "__main__":
    main()
