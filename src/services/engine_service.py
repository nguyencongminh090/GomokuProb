"""
Engine Service Module.
Wraps PyGomo EngineClient to provide high-level analysis capabilities.
"""

import os
import sys
from typing import List, Optional

# # Add PyGomo to path
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# PYGOMO_PATH = os.path.join(PROJECT_ROOT, "lib", "PyGomo", "src")
# if PYGOMO_PATH not in sys.path:
#     sys.path.insert(0, PYGOMO_PATH)

from PyQt6.QtCore import QObject, pyqtSignal
from pygomo import EngineClient, BoardPosition, Move as PyGomoMove, SearchInfo
from src.core.board import BoardState, Move
from src.core.math_prob import MoveEvaluation
from src.core.state import EngineState


def _engine_to_human_notation(engine_algebraic: str, board_size: int = 15) -> str:
    """Convert engine algebraic (row 1=Top) to human Renju notation (row 1=Bottom).
    
    Engine 'f11' means col=5, row_from_top=11 → internal y=10.
    Human notation for y=10: row_from_bottom = 15-10 = 5 → 'f5'.
    """
    import re
    m = re.match(r'([a-o])(\d{1,2})', engine_algebraic.lower())
    if not m:
        return engine_algebraic  # Can't parse, return as-is
    col_char = m.group(1)
    engine_row = int(m.group(2))  # 1-indexed from top
    internal_y = engine_row - 1
    human_row = board_size - internal_y  # 1-indexed from bottom
    return f"{col_char}{human_row}"

class EngineService(QObject):
    """
    Manages the lifecycle and communication with the Gomoku Engine.
    """
    
    state_changed = pyqtSignal(EngineState)
    
    def __init__(self, engine_path: str):
        super().__init__()
        self.engine_path = engine_path
        self.client: Optional[EngineClient] = None
        self.log_callback = None
        self._state = EngineState.OFF
        
    @property
    def state(self) -> EngineState:
        return self._state
        
    def _set_state(self, new_state: EngineState):
        if self._state != new_state:
            self._state = new_state
            self.state_changed.emit(new_state)
            self._log("INFO", f"Engine State changed to: {new_state.name}")
        
    def set_logger(self, callback):
        self.log_callback = callback
        
    def _log(self, type_: str, msg: str):
        if self.log_callback:
            self.log_callback(type_, msg)

    def start(self, rule: int = 0):
        """Start the engine process."""
        if self.client and self.client.is_connected:
            return
            
        self._set_state(EngineState.STARTING)
        try:
            working_dir = os.path.dirname(self.engine_path)
            
            # Pass io_logger to transport for raw stdin/stdout logging
            self.client = EngineClient(
                self.engine_path, 
                working_directory=working_dir,
                io_logger=self._log
            )
            self.client.connect()
            self._log("INFO", f"Engine started: {self.engine_path}")
            
            self.client.start(15)
            self.client.set_rule(rule)
            
            # Set Unlimited Time for Time-Wrapped mode
            self.client.execute("INFO", "TIMEOUT_MATCH", 2147483647)
            self.client.execute("INFO", "TIMEOUT_TURN", 2147483647)
            self._log("CMD", "INFO TIMEOUT_MATCH/TURN UNLIMITED")
            
            self._set_state(EngineState.IDLE)
            
        except Exception as e:
            self._log("ERROR", f"Failed to start engine: {e}")
            self._set_state(EngineState.ERROR)
        
    def shutdown(self):
        """Hard Stop: Quit and Disconnect engine."""
        if self.client:
            self._set_state(EngineState.STOPPING)
            try:
                self.client.quit()
                self.client.disconnect()
            except:
                pass
            self.client = None
            
        self._set_state(EngineState.OFF)
        
    def stop_search(self):
        """Soft Stop: Interrupt search but keep engine alive."""
        self._log("INFO", "Soft Stop requested...")
        if self.client:
            try:
                self.client.stop()
            except Exception as e:
                self._log("ERROR", f"Soft stop failed: {e}")
            
    def _setup_board_yx(self, moves: List[Move]):
        """Send YXBOARD command to setup board."""
        if not self.client:
            raise RuntimeError("Engine client is None in _setup_board_yx")
            
        self._reset_engine()
        
        if moves:
            move_strs = [f"{m.notation}({'B' if m.color == 1 else 'W'})" for m in moves]
            self._log("STAGE", f"INPUT: Position [{len(moves)} moves]: {' '.join(move_strs)}")
        else:
            self._log("STAGE", "INPUT: Empty board")
        
        self.client.send_raw("YXBOARD")
        for m in moves:
            field = m.color
            self.client.send_raw(f"{m.x},{m.y},{field}")
        self.client.send_raw("DONE")

    def _reset_engine(self):
        """Reset engine state before new position analysis."""
        if not self.client:
            return
            
        try:
            self.client.stop()
            self._log("CMD", "STOP (reset for fresh search)")
            import time
            time.sleep(0.1)
        except Exception as e:
            self._log("DEBUG", f"_reset_engine exception: {e}")

    def analyze(self, board: BoardState, top_n: int = 5, time_limit: float = 1.0, node_limit: int = 0) -> List[MoveEvaluation]:
        """Analyze a board position. Returns top N candidate moves."""
        self._set_state(EngineState.ANALYZING)
        try:
            return self._analyze_impl(board, top_n, time_limit, node_limit)
        finally:
            self._set_state(EngineState.IDLE)

    def _analyze_impl(self, board: BoardState, top_n: int = 5, time_limit: float = 1.0, node_limit: int = 0) -> List[MoveEvaluation]:
        if not self.client:
            raise RuntimeError("Engine not started")
            
        self._setup_board_yx(board.moves)
        
        candidates: List[MoveEvaluation] = []
        
        if node_limit is not None and node_limit > 0:
            self.client.execute("INFO", "MAX_NODE", node_limit)
        
        last_search_info = {}
        
        def on_search_info(info):
            """Callback for realtime search info."""
            mpv = getattr(info, 'multipv', 1)
            if mpv != 1:
                return

            depth = getattr(info, 'depth', 0)
            winrate = getattr(info, 'winrate_percent', 0.0)
            if not winrate and hasattr(info, 'winrate'):
                winrate = info.winrate * 100
                
            score = "N/A"
            if hasattr(info, 'eval'):
                if hasattr(info.eval, 'raw_value'):
                    score = info.eval.raw_value
                else:
                    score = str(info.eval)
            
            last_search_info['winrate'] = winrate
            last_search_info['score'] = score
            last_search_info['depth'] = depth

        try:
            result = self.client.nbest_time_limited(
                count=top_n, 
                time_limit=time_limit, 
                on_info=on_search_info
            )
            self._log("STAGE", f"STOP was sent via synchronous time-limit")
        except Exception as e:
            self._log("ERROR", f"NBEST failed: {e}")
            result = None
            
        if result:
            w = getattr(result, 'winrate', None)
            s = getattr(result, 'score', None)
            if s is None and hasattr(result, 'eval') and hasattr(result.eval, 'raw_value'):
                s = result.eval.raw_value
            self._log("INFO", f"NBEST Raw: Move={result.move}, Winrate={w}, Score={s}")
        else:
            self._log("INFO", "NBEST returned None")
            return candidates

        # Build candidates from all_info
        multipv_map = {}
        if hasattr(result, 'all_info') and result.all_info:
            self._log("STAGE", f"DEBUG: all_info has {len(result.all_info)} entries")
            for info in result.all_info:
                mpv = getattr(info, 'multipv', 1)
                multipv_map[mpv] = info
        elif hasattr(result, 'search_info') and result.search_info:
            multipv_map[1] = result.search_info
        
        for mpv_idx in sorted(multipv_map.keys()):
            info = multipv_map[mpv_idx]
            move_obj = info.pv[0] if info.pv else None
            engine_str = move_obj.to_algebraic() if move_obj else f"pv{mpv_idx}"
            move_str = _engine_to_human_notation(engine_str) if move_obj else engine_str
            
            # Use PyGomo's winrate logic
            wr = info.winrate if hasattr(info, 'winrate') else 0.5
            is_best = (mpv_idx == 1)
            
            raw_score = None
            mate_score = None
            if hasattr(info, 'eval') and hasattr(info.eval, 'raw_value'):
                val = info.eval.raw_value
                if isinstance(val, str):
                    val_str = val.upper()
                    if "M" in val_str:
                        try:
                            clean_str = val_str.replace("M", "").replace("+", "")
                            dist = int(clean_str)
                            if val_str.startswith("-"):
                                mate_score = -dist
                            else:
                                mate_score = dist
                        except:
                            pass
                else:
                    try:
                        raw_score = float(val)
                    except:
                        pass
            
            depth_val = getattr(info, 'depth', 0)
            nodes_val = getattr(info, 'nodes', 0)

            candidates.append(MoveEvaluation(
                move_notation=move_str,
                winrate=wr,
                is_best=is_best,
                score=raw_score,
                mate_score=mate_score,
                depth=depth_val,
                nodes=nodes_val
            ))
            self._log("STAGE", f"DEBUG Candidate {move_str}: Depth={depth_val} N={nodes_val}")
        
        if not candidates:
            # Result itself has winrate in PlayResult
            wr = result.winrate if result.winrate is not None else 0.5
            candidates.append(MoveEvaluation(
                move_notation=_engine_to_human_notation(str(result.move)) if result.move else "???",
                winrate=wr,
                is_best=True
            ))
        
        self._log("STAGE", f"=== NBEST {top_n} CANDIDATES ===")
        for i, c in enumerate(candidates):
            rank = i + 1
            self._log("STAGE", f"  #{rank}: {c.move_notation} | WR {c.winrate*100:.1f}%")
        self._log("STAGE", "=" * 30)
        
        return candidates

    def evaluate_move(self, board: BoardState, move: Move, time_limit: float = 1.0) -> MoveEvaluation:
        """Evaluate a specific move made by player."""
        temp_moves = board.moves + [move]
        self._setup_board_yx(temp_moves)
        
        last_info = {'winrate': None, 'score': None, 'mate': None, 'depth': 0}
        
        def on_search_info(info):
            mpv = getattr(info, 'multipv', 1)
            if mpv != 1:
                return

            depth = getattr(info, 'depth', 0)
            score = None
            mate_score = None
            
            if hasattr(info, 'eval') and hasattr(info.eval, 'raw_value'):
                val = info.eval.raw_value
                score = val
                if isinstance(val, str) and "M" in val.upper():
                    try:
                        clean_str = val.upper().replace("M", "").replace("+", "")
                        dist = int(clean_str)
                        if val.startswith("-"):
                            mate_score = -dist
                        else:
                            mate_score = dist
                    except:
                        pass
                elif isinstance(val, (int, float)):
                    score = float(val)

            winrate = getattr(info, 'winrate_percent', 0.0) / 100.0
            
            last_info['depth'] = depth
            last_info['score'] = score
            last_info['mate'] = mate_score
            last_info['winrate'] = winrate

        try:
            result = self.client.nbest_time_limited(
                count=1, 
                time_limit=time_limit, 
                on_info=on_search_info
            )
        except Exception as e:
            self._log("ERROR", f"Eval failed: {e}")
            result = None
        
        opponent_wr = 0.5
        opponent_score = None
        opponent_mate = None
        
        if result and result.winrate is not None:
            opponent_wr = result.winrate
            if result.eval:
                opponent_score = result.eval.raw_value
        elif last_info['winrate'] is not None:
            opponent_wr = last_info['winrate']
            opponent_score = last_info['score']
            opponent_mate = last_info['mate']
                
        # Invert Perspective (Opponent -> Player)
        player_wr = 1.0 - opponent_wr
        
        player_score = None
        player_mate = None
        
        if opponent_score is not None:
            try:
                if isinstance(opponent_score, str):
                    s = opponent_score.upper()
                    if "M" in s:
                        clean = s.replace("M", "").replace("+", "")
                        m_dist = int(clean)
                        if s.startswith("-"):
                            player_mate = m_dist
                        else:
                            player_mate = -m_dist
                    else:
                        player_score = -float(s)
                else:
                    player_score = -float(opponent_score)
            except:
                pass
                
        if opponent_mate is not None:
            player_mate = -opponent_mate
            
        return MoveEvaluation(
            move_notation=move.to_algebraic(),
            winrate=player_wr,
            score=player_score,
            mate_score=player_mate
        )
    def get_static_eval(self, board: BoardState) -> dict:
        """
        Get static evaluation and pattern data from engine.
        Returns: { 'static_eval': int, 'patterns': {'black': [[...]], 'white': [[...]]}, ... }
        """
        import json
        if not self.client:
            raise RuntimeError("Engine not started")
            
        self._set_state(EngineState.ANALYZING)
        try:
            # Setup board but DO NOT start search (YXBOARD)
            self._setup_board_yx(board.moves)
            
            # Request static JSON
            self.client.send_raw("YXSTATICJSON")
            
            # Read response directly from process stdout
            try:
                # We expect exactly one line of JSON
                line = self.client._process.stdout.readline()
                if not line:
                    return {}
                    
                line = line.strip()
                self._log("CMD", f"YXSTATICJSON Response: {line[:50]}...")
                
                if line.startswith("{"):
                    return json.loads(line)
                else:
                    self._log("ERROR", f"Unexpected YXSTATICJSON output: {line}")
                    return {}
            except Exception as e:
                self._log("ERROR", f"Failed to read/parse YXSTATICJSON: {e}")
                return {}
                
        finally:
            self._set_state(EngineState.IDLE)
