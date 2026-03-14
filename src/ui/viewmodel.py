"""
ViewModel for the Gomoku Analysis Tool.
"""

from PyQt6.QtCore import QObject, pyqtSignal, QThread
from typing import List, Optional

from src.core.board import BoardState, Move
from src.core.config import AnalysisConfig
from src.services.engine_service import EngineService, MoveEvaluation
from src.services.parser import GameParser

# V2 Imports
from src.core.v2_worker import V2AnalysisWorker, V2MoveResult, V2GameResult
from src.core.profile_store import ProfileStore

from src.core.state import SystemState, EngineState

class MainViewModel(QObject):
    """
    Main ViewModel handling UI logic.
    """
    # Signals
    game_loaded = pyqtSignal(int) # number of moves
    analysis_updated = pyqtSignal(object) # V2MoveResult
    game_result = pyqtSignal(object) # V2GameResult - final classification
    system_state_changed = pyqtSignal(object) # SystemState
    engine_state_changed = pyqtSignal(object) # EngineState
    current_move_idx = pyqtSignal(int)
    visible_step_changed = pyqtSignal(int)
    analysis_progress = pyqtSignal(int, int)  # current, total
    
    # Deprecated (kept for temporary compat if needed, but we will update MainWindow)
    # is_analyzing = pyqtSignal(bool) 
    
    def __init__(self):
        super().__init__()
        # Load from config.json or use default
        self.config = AnalysisConfig.load()
        # Ensure engine path is valid or fallback to default if missing
        if not self.config.engine_path:
             self.config.engine_path = "engine/rapfi"
             
        self.engine_service = EngineService(self.config.engine_path)
        self.profile_store = ProfileStore()
        # Proxy engine state
        self.engine_service.state_changed.connect(self.engine_state_changed.emit)
        
        # V2: No longer need math_model - V2AnalysisWorker handles analysis internally
        
        self.current_board: Optional[BoardState] = None
        self.worker: Optional[V2AnalysisWorker] = None
        self.thread: Optional[QThread] = None # Explicitly shadow QObject.thread() method
        self._state = SystemState.IDLE
        self.visible_step = 0 # 0 means empty board, N means N moves shown

    @property
    def state(self) -> SystemState:
        return self._state
        
    def _set_state(self, new_state: SystemState):
        if self._state != new_state:
            self._state = new_state
            self.system_state_changed.emit(new_state)

    # --- Navigation ---
    def set_visible_step(self, step: int):
        if not self.current_board:
            return
        
        max_step = len(self.current_board.moves)
        step = max(0, min(step, max_step)) # Clamp
        
        if self.visible_step != step:
            self.visible_step = step
            self.visible_step_changed.emit(step)
            
    def step_first(self):
        self.set_visible_step(0)
        
    def step_prev(self):
        self.set_visible_step(self.visible_step - 1)
        
    def step_next(self):
        self.set_visible_step(self.visible_step + 1)
        
    def step_last(self):
        if self.current_board:
            self.set_visible_step(len(self.current_board.moves))

    def update_config(self, new_config: AnalysisConfig):
        self.config = new_config
        # Persist changes
        self.config.save()
        
        # Update engine service path immediately if changed, 
        # though worker will set it before run.
        self.engine_service.engine_path = self.config.engine_path
        
    def load_game(self, content: str):
        """Load game from string (PGN/SGF/Raw)."""
        self.current_board = GameParser.auto_detect_and_parse(content)
        moves_count = len(self.current_board.moves)
        self.game_loaded.emit(moves_count)
        self.set_visible_step(moves_count) # Jump to end
        
    def start_analysis(self, player_name: str = "", save_to_profile: bool = False):
        if not self.current_board:
            return
        
        if self._state != SystemState.IDLE:
            return
            
        # Log start intent
        start_idx = max(0, self.config.start_move - 1)
        self.engine_service._log("INFO", f"Starting analysis from Move #{self.config.start_move} (Index {start_idx})")
        
        # Ensure previous thread is completely finished before creating a new one
        if self.thread:
            try:
                if self.thread.isRunning():
                    # This should rarely happen if state is IDLE, but if it does, we wait
                    self.thread.wait(1000)
                if self.thread.isRunning(): # Still running?
                    print("WARNING: Thread still running in start_analysis, force waiting")
                    self.thread.wait() # Blocking wait
            except RuntimeError:
                pass
            self.thread = None
            
        self.thread = QThread()
        self.worker = V2AnalysisWorker(
            self.engine_service, 
            self.current_board, 
            self.config,
            player_name=player_name,
            save_to_profile=save_to_profile
        )
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater) # REMOVED: Managed manually
        
        # V2: Signal connections
        self.worker.move_result.connect(self.analysis_updated)  # Per-move result
        self.worker.game_complete.connect(self._on_game_complete)  # Final results
        
        # Progress updates view to follow analysis
        self.worker.progress.connect(lambda cur, tot: self.current_move_idx.emit(cur - 1))
        self.worker.progress.connect(lambda cur, tot: self.set_visible_step(cur))
        self.worker.progress.connect(self.analysis_progress.emit)
        
        self.worker.finished.connect(lambda: self._set_state(SystemState.IDLE))
        
        self._set_state(SystemState.RUNNING)
        self.thread.start()
        
        
    def cleanup(self):
        """Cleanup resources before exit."""
        self._set_state(SystemState.STOPPING)
        
        # 1. Signal worker to stop logic (sets internal flag)
        if self.worker:
            self.worker.stop()
            
        # 2. Stop Engine (CRITICAL: Unblocks any blocking I/O or sleep in worker)
        if self.engine_service:
            self.engine_service.shutdown() # Full shutdown on exit
            
        # 3. Wait for thread to finish (now that it's unblocked)
        if self.thread:
            try:
                if self.thread.isRunning():
                    self.thread.quit()
                    self.thread.wait(2000) # Wait max 2s
            except RuntimeError:
                pass
            
        self.thread = None
        self.worker = None
        self._set_state(SystemState.IDLE)
            
    def stop_analysis(self):
        if self._state == SystemState.IDLE:
             return

        self._set_state(SystemState.STOPPING)
        
        # Signal Stop -> Thread will finish naturally
        if self.worker:
            self.worker.stop()
            
        if self.engine_service:
            self.engine_service.stop_search() # Soft stop for user interruption
            
        # DO NOT force quit/wait/delete thread here. 
        # Let worker finish its loop and exit.
        # start_analysis will wait for it to join if restarted immediately.
        # This prevents "QThread: Destroyed while thread is still running" crashes.
        
        # self._set_state(SystemState.IDLE) handled by worker.finished signal

    def _on_game_complete(self, result: V2GameResult):
        """Handle V2 game analysis completion."""
        # Emit final result for UI
        self.game_result.emit(result)
        
        # Log final classification
        if result.classification:
            c = result.classification
            print(f"V2 FINAL: P(Cheat)={c.p_cheat*100:.1f}% | Class={c.classification} | Conf={c.confidence*100:.0f}%")
