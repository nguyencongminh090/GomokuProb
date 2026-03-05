"""
Gomocup protocol implementation.

This module implements the Gomocup/Piskvork protocol for
engine communication.

Protocol reference:
https://plastovicka.github.io/protocl2en.htm
"""

import re
from typing import Any, Optional

from pygomo.protocol.interface import IProtocol, ResponseType
from pygomo.protocol.models import Move, Evaluate, SearchInfo


class GomocupProtocol(IProtocol):
    """
    Gomocup (Piskvork) protocol implementation.
    
    Handles serialization and parsing of Gomocup protocol messages.
    Supports both standard Gomocup and Yixin-Board extensions.
    
    Example:
        protocol = GomocupProtocol()
        cmd = protocol.serialize_command("TURN", "7,8")
        # Output: "TURN 7,8"
    """
    
    # Commands that trigger engine thinking
    THINKING_COMMANDS = frozenset({
        "BEGIN", "TURN", "BOARD",
        "YXNBEST", "YXBALANCEONE", "YXBALANCETWO",
        "YXSEARCHDEFEND", "SWAP2BOARD",
    })
    
    # Commands that expect specific response types
    RESPONSE_TYPES = {
        "START": ResponseType.OK,
        "RESTART": ResponseType.OK,
        "TAKEBACK": ResponseType.OK,
        "BEGIN": ResponseType.COORD,
        "TURN": ResponseType.COORD,
        "BOARD": ResponseType.COORD,
        "YXBOARD": ResponseType.NONE,
        "ABOUT": ResponseType.TEXT,
        "END": ResponseType.NONE,
        "STOP": ResponseType.NONE,
        "YXSTOP": ResponseType.NONE,
        "INFO": ResponseType.NONE,
        "YXNBEST": ResponseType.COORD,
        "YXSHOWINFO": ResponseType.MESSAGE,
        "YXSHOWFORBID": ResponseType.TEXT,
    }
    
    # UCI-like format patterns from Rapfi searchoutput.cpp
    # Format 1: depth 5-8 ev 123 n 1.5M n/ms 150 tm 1234 pv h8 i9
    # Format 2: depth 5-8 multipv 1 ev 123 n 1.5M n/ms 150 tm 1234 pv h8 i9
    # Format 3 (MCTS): multipv 1 ev 123 w 75.5 d 10.2 stdev 0.05 v 1M seldepth 15 n 2M n/ms 200 tm 5000 prior 0.3 pv h8 i9
    
    # NORMAL format patterns:
    # Format 1: Depth 5-8 | Eval 123 | Time 1.5s | h8 i9
    # Format 2: (1) 123 | 5-8 | h8 i9  
    # Format 3: Speed 150Kn/s | Depth 5-8 | Eval 123 | Node 1.5M | Time 1.5s
    
    @property
    def name(self) -> str:
        return "gomocup"
    
    def serialize_command(self, command: str, *args: Any) -> str:
        """
        Serialize a command with arguments.
        
        Args:
            command: Command name.
            *args: Command arguments.
            
        Returns:
            Formatted command string.
        """
        # Command is always uppercase
        cmd = command.upper()
        
        if not args:
            return cmd
        
        # Convert args to strings
        arg_strs = []
        for arg in args:
            if isinstance(arg, Move):
                arg_strs.append(arg.to_numeric())
            else:
                arg_strs.append(str(arg))
        
        return f"{cmd} {' '.join(arg_strs)}"
    
    def get_response_type(self, command: str) -> ResponseType:
        """Get expected response type for a command."""
        return self.RESPONSE_TYPES.get(
            command.upper(),
            ResponseType.NONE
        )
    
    def parse_coord(self, data: str) -> tuple[int, int]:
        """
        Parse coordinate string to (x, y).
        
        Args:
            data: Coordinate string like "7,8".
            
        Returns:
            Tuple of (x, y).
        """
        data = data.strip().replace(" ", "")
        parts = data.split(",")
        return int(parts[0]), int(parts[1])
    
    def format_coord(self, x: int, y: int) -> str:
        """Format coordinates to protocol string."""
        return f"{x},{y}"
    
    def parse_message(self, data: str) -> dict[str, Any]:
        """
        Parse a MESSAGE line into structured data.
        
        Handles multiple formats from Rapfi:
        - UCILIKE: depth X-Y ev VAL n NODES n/ms SPEED tm TIME pv MOVES
        - UCILIKE MCTS: multipv X ev VAL w WR d DR stdev SD v V seldepth SD n N n/ms SPEED tm TIME prior P pv MOVES
        - NORMAL: Depth X-Y | Eval VAL | Time T | MOVES
        
        Args:
            data: Raw MESSAGE line.
            
        Returns:
            Dictionary with parsed fields.
        """
        # Remove MESSAGE prefix if present
        if data.upper().startswith("MESSAGE"):
            data = data[7:].strip()
        
        result = {
            "depth": 0,
            "sel_depth": 0,
            "eval": "0",
            "nodes": 0,
            "nps": 0,  # n/ms (nodes per millisecond)
            "time": 0,
            "pv": [],
            "multipv": 1,
            "winrate": None,  # From MCTS output
            "drawrate": None,
            "raw": data,
        }
        
        # Try UCILIKE format with depth first
        # Pattern: depth X-Y [multipv N] ev VAL n NODES n/ms SPEED tm TIME pv MOVES
        uci_match = re.match(
            r"depth\s+(\d+)-(\d+)\s+"
            r"(?:multipv\s+(\d+)\s+)?"
            r"ev\s+([+-]?\w*\d+)\s+"
            r"n\s+([\d.]+[KMG]?)\s+"
            r"n/ms\s+(\d+)\s+"
            r"tm\s+(\d+)\s+"
            r"pv\s+(.+)",
            data, re.IGNORECASE
        )
        
        if uci_match:
            result["depth"] = int(uci_match.group(1))
            result["sel_depth"] = int(uci_match.group(2))
            if uci_match.group(3):
                result["multipv"] = int(uci_match.group(3))
            result["eval"] = uci_match.group(4)
            result["nodes"] = self._parse_node_count(uci_match.group(5))
            result["nps"] = int(uci_match.group(6))
            result["time"] = int(uci_match.group(7))
            result["pv"] = self._parse_pv(uci_match.group(8))
            return result
        
        # Try MCTS format (multipv first, no depth at start)
        # Pattern: multipv N ev VAL w WR d DR stdev SD v V seldepth SD n N n/ms SPEED tm TIME prior P pv MOVES
        mcts_match = re.match(
            r"multipv\s+(\d+)\s+"
            r"ev\s+([+-]?\w*\d+)\s+"
            r"w\s+([\d.]+)\s+"
            r"d\s+([\d.]+)\s+"
            r"stdev\s+([\d.]+)\s+"
            r"v\s+([\d.]+[KMG]?)\s+"
            r"seldepth\s+(\d+)\s+"
            r"n\s+([\d.]+[KMG]?)\s+"
            r"n/ms\s+(\d+)\s+"
            r"tm\s+(\d+)\s+"
            r"(?:prior\s+([\d.]+)\s+)?"
            r"pv\s+(.+)",
            data, re.IGNORECASE
        )
        
        if mcts_match:
            result["multipv"] = int(mcts_match.group(1))
            result["eval"] = mcts_match.group(2)
            result["winrate"] = float(mcts_match.group(3))
            result["drawrate"] = float(mcts_match.group(4))
            result["sel_depth"] = int(mcts_match.group(7))
            result["nodes"] = self._parse_node_count(mcts_match.group(8))
            result["nps"] = int(mcts_match.group(9))
            result["time"] = int(mcts_match.group(10))
            result["pv"] = self._parse_pv(mcts_match.group(12))
            return result
        
        # Try NORMAL format: Depth X-Y | Eval VAL | Time T | MOVES
        # Or: [Pondering] Depth X-Y | Eval VAL | Time T | MOVES
        normal_match = re.match(
            r"(?:\[Pondering\]\s*)?"
            r"Depth\s+(\d+)-(\d+)\s+\|\s+"
            r"Eval\s+([+-]?\w*\d+)\s+\|\s+"
            r"Time\s+([\d.]+[sm]?)\s+\|\s+"
            r"(.+)",
            data, re.IGNORECASE
        )
        
        if normal_match:
            result["depth"] = int(normal_match.group(1))
            result["sel_depth"] = int(normal_match.group(2))
            result["eval"] = normal_match.group(3)
            result["pv"] = self._parse_pv(normal_match.group(5))
            return result
        
        # Try summary format: Speed X | Depth Y-Z | Eval V | Node N | Time T
        summary_match = re.match(
            r"(?:\[Pondering\]\s*)?"
            r"Speed\s+[\d.]+[KMG]?n/s\s+\|\s+"
            r"Depth\s+(\d+)-(\d+)\s+\|\s+"
            r"Eval\s+([+-]?\w*\d+)\s+\|\s+"
            r"Node\s+([\d.]+[KMG]?)\s+\|\s+"
            r"Time\s+([\d.]+[sm]?)",
            data, re.IGNORECASE
        )
        
        if summary_match:
            result["depth"] = int(summary_match.group(1))
            result["sel_depth"] = int(summary_match.group(2))
            result["eval"] = summary_match.group(3)
            result["nodes"] = self._parse_node_count(summary_match.group(4))
            return result
        
        # Try indexed format: (1) 123 | 5-8 | h8 i9
        indexed_match = re.match(
            r"\((\d+)\)\s+([+-]?\w*\d+)\s+\|\s+"
            r"(\d+)-(\d+)\s+\|\s+"
            r"(.+)",
            data, re.IGNORECASE
        )
        
        if indexed_match:
            result["multipv"] = int(indexed_match.group(1))
            result["eval"] = indexed_match.group(2)
            result["depth"] = int(indexed_match.group(3))
            result["sel_depth"] = int(indexed_match.group(4))
            result["pv"] = self._parse_pv(indexed_match.group(5))
            return result
        
        # Fallback: try to extract any eval we can find
        eval_match = re.search(r"(?:ev|eval)\s+([+-]?\w*\d+)", data, re.IGNORECASE)
        if eval_match:
            result["eval"] = eval_match.group(1)
        
        depth_match = re.search(r"(?:depth\s+)?(\d+)-(\d+)", data, re.IGNORECASE)
        if depth_match:
            result["depth"] = int(depth_match.group(1))
            result["sel_depth"] = int(depth_match.group(2))
        
        pv_match = re.search(r"pv\s+(.+)", data, re.IGNORECASE)
        if pv_match:
            result["pv"] = self._parse_pv(pv_match.group(1))
        
        return result
    
    def parse_search_info(self, data: str) -> SearchInfo:
        """
        Parse MESSAGE line into SearchInfo object.
        
        Args:
            data: Raw MESSAGE line.
            
        Returns:
            SearchInfo with parsed data.
        """
        parsed = self.parse_message(data)
        
        return SearchInfo(
            depth=parsed["depth"],
            sel_depth=parsed["sel_depth"],
            eval=Evaluate(parsed["eval"]),
            nodes=parsed["nodes"],
            nps=parsed["nps"],
            time_ms=parsed["time"],
            pv=[Move(m) for m in parsed["pv"] if m],
            multipv=parsed["multipv"],
        )
    
    def is_thinking_command(self, command: str) -> bool:
        """Check if command triggers engine thinking."""
        return command.upper() in self.THINKING_COMMANDS
    
    def _parse_node_count(self, value: str) -> int:
        """
        Parse node count with optional suffix.
        
        Args:
            value: String like "1.5M", "500K", "1000".
            
        Returns:
            Integer node count.
        """
        value = value.strip().upper()
        
        multipliers = {
            'K': 1_000,
            'M': 1_000_000,
            'G': 1_000_000_000,
        }
        
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                return int(float(value[:-1]) * mult)
        
        try:
            return int(float(value))
        except ValueError:
            return 0
    
    def _parse_pv(self, pv_string: str) -> list[str]:
        """
        Parse principal variation string into list of moves.
        
        Args:
            pv_string: Space-separated move string.
            
        Returns:
            List of move strings.
        """
        if not pv_string:
            return []
        
        # Split by whitespace and filter empty strings
        moves = pv_string.strip().split()
        return [m for m in moves if m and m.upper() != "(NONE)"]
    
    def format_info_command(self, key: str, value: Any) -> str:
        """
        Format an INFO command.
        
        Args:
            key: Info parameter name.
            value: Parameter value.
            
        Returns:
            Formatted INFO command.
        """
        return self.serialize_command("INFO", key, value)
    
    def format_board_position(
        self,
        moves: list[tuple[Move, int]],
        start_thinking: bool = True,
    ) -> list[str]:
        """
        Format BOARD command sequence.
        
        Args:
            moves: List of (move, color) tuples.
            start_thinking: Whether to trigger thinking after setup.
            
        Returns:
            List of command strings.
        """
        commands = ["BOARD" if start_thinking else "YXBOARD"]
        
        for move, color in moves:
            commands.append(f"{move.col},{move.row},{color}")
        
        commands.append("DONE")
        return commands
