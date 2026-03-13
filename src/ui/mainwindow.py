"""
Main Window for Gomoku Analysis Tool.
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QDockWidget, 
                             QTextEdit, QProgressBar, QSplitter, QMenu, QTabWidget,
                             QLineEdit, QCheckBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal
import sys
import os
from datetime import datetime

from src.ui.widgets.pattern_matrix import PatternMatrixWidget
from src.ui.widgets.board_widget import BoardWidget
from src.ui.widgets.graph_widget import AnalysisGraphWidget
from src.ui.widgets.console_widget import EngineConsoleWidget
from src.ui.widgets.analysis_log_widget import AnalysisLogWidget
from src.ui.viewmodel import MainViewModel
from src.ui.dialogs import PastePositionDialog, SettingsDialog
from src.core.state import SystemState

class MainWindow(QMainWindow):
    sig_log_message = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gomoku Accuracy Tool - Anti-Cheating System")
        self.resize(1200, 800)
        
        # Setup file logging
        self._setup_file_logging()
        
        # Initialize ViewModel
        self.view_model = MainViewModel()
        self.view_model.engine_service.set_logger(self.on_engine_log)
        
        # Connect thread-safe logging signal
        self.sig_log_message.connect(self._on_log_message_safe)
        
        self.setup_ui()
        self.connect_signals()
        
    def _setup_file_logging(self):
        """Initialize file-based logging."""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"analysis_{session_id}.txt")
        
        # Open log file
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        self.log_file.write(f"=== Gomoku Analysis Log - {session_id} ===\n\n")
        self.log_file.flush()
        
    def _write_to_log_file(self, type_: str, msg: str):
        """Write a log entry to file."""
        if hasattr(self, 'log_file') and self.log_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.log_file.write(f"[{timestamp}] [{type_:6}] {msg}\n")
            self.log_file.flush()
        

    def closeEvent(self, event):
        self.view_model.cleanup()
        
        # Close log file
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.write("\n=== Session Ended ===\n")
            self.log_file.close()
            self.log_file = None
            
        event.accept()
        
    def setup_ui(self):
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter to resize Board vs Analysis Panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left: Board Area
        board_container = QWidget()
        board_layout = QVBoxLayout(board_container)
        self.board_widget = BoardWidget()
        board_layout.addWidget(self.board_widget)
        
        # Navigation Controls
        nav_layout = QHBoxLayout()
        self.btn_first = QPushButton("<<")
        self.btn_prev = QPushButton("<")
        self.btn_next = QPushButton(">")
        self.btn_last = QPushButton(">>")
        
        # Style buttons (optional, make them smaller)
        for btn in [self.btn_first, self.btn_prev, self.btn_next, self.btn_last]:
             btn.setFixedWidth(40)
             
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_first)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_last)
        nav_layout.addStretch()
        
        board_layout.addLayout(nav_layout)
        
        splitter.addWidget(board_container)
        
        # Right: Analysis Panel
        analysis_container = QWidget()
        analysis_layout = QVBoxLayout(analysis_container)
        
        # 1. Graph Widget
        self.graph_widget = AnalysisGraphWidget()
        analysis_layout.addWidget(QLabel("Realtime Accuracy Graph:"))
        analysis_layout.addWidget(self.graph_widget)
        
        # 2. Controls
        controls_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Game / Position")
        self.btn_settings = QPushButton("Settings")
        
        # V4 Profile Input
        self.player_input = QLineEdit()
        self.player_input.setPlaceholderText("Player Name (for Profile)")
        self.player_input.setFixedWidth(150)
        self.chk_save_profile = QCheckBox("Save to DB")
        self.chk_save_profile.setChecked(True)
        
        self.btn_analyze = QPushButton("Start Analysis") # Text will toggle
        self.btn_analyze.setEnabled(False) # Disable until loaded
        
        controls_layout.addWidget(self.btn_load)
        controls_layout.addWidget(self.btn_settings)
        controls_layout.addWidget(self.player_input)
        controls_layout.addWidget(self.chk_save_profile)
        controls_layout.addWidget(self.btn_analyze)
        analysis_layout.addLayout(controls_layout)
        
        # 3. Tabs: Report & Console
        self.tabs = QTabWidget()
        
        # Tab 1: Report
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.tabs.addTab(self.log_area, "Analysis Report")
        
        # Tab 2: Console
        self.console_widget = EngineConsoleWidget()
        self.tabs.addTab(self.console_widget, "Engine Console")
        
        # Tab 3: Analysis Flow
        self.analysis_log_widget = AnalysisLogWidget()
        self.tabs.addTab(self.analysis_log_widget, "Analysis Flow")
        
        # Tab 4: Pattern Matrix (Heatmap)
        self.pattern_matrix_widget = PatternMatrixWidget()
        self.tabs.addTab(self.pattern_matrix_widget, "Pattern Matrix")
        
        analysis_layout.addWidget(self.tabs)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        analysis_layout.addWidget(self.progress_bar)
        
        splitter.addWidget(analysis_container)
        splitter.setSizes([700, 500]) # Initial ratio
        
    def connect_signals(self):
        # UI -> ViewModel
        # Connect load button to a menu for options
        self.btn_load.setMenu(self.create_load_menu())
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_analyze.clicked.connect(self.toggle_analysis)
        
        self.btn_first.clicked.connect(self.view_model.step_first)
        self.btn_prev.clicked.connect(self.view_model.step_prev)
        self.btn_next.clicked.connect(self.view_model.step_next)
        self.btn_last.clicked.connect(self.view_model.step_last)
        
        # ViewModel -> UI
        self.view_model.game_loaded.connect(self.on_game_loaded)
        self.view_model.system_state_changed.connect(self.on_system_state_changed)
        self.view_model.analysis_updated.connect(self.on_analysis_result)
        self.view_model.current_move_idx.connect(self.on_move_update)
        
        self.view_model.visible_step_changed.connect(self.on_visible_step_changed)
        self.view_model.analysis_progress.connect(self.on_analysis_progress)
        
        # Graph -> Board (Blunder Visualization)
        self.graph_widget.blunders_detected.connect(self.board_widget.set_blunders)

    def create_load_menu(self):
        menu = QMenu(self)
        
        action_file = QAction("Load from File...", self)
        action_file.triggered.connect(self.load_from_file)
        menu.addAction(action_file)
        
        action_paste = QAction("Paste Position...", self)
        action_paste.triggered.connect(self.load_from_paste)
        menu.addAction(action_paste)
        
        # Set default behavior for click (optional, maybe open file?)
        # For now, let's keep the button as a dropdown primarily, 
        # or make the button click trigger the menu?
        # Standard QPushButton with menu usually needs visual cue.
        return menu

    def handle_load(self):
        # This was the old direct connection. 
        # Now we use the menu attached to the button.
        # But if the user clicks the button itself (not the arrow, if any), 
        # we might want to default to File.
        # Ideally, we change the button to "Load..." and show menu on click if not setMenu.
        # setMenu makes it a dropdown button automatically on some styles.
        # Let's fallback to showing menu if setMenu isn't visual enough?
        # Actually setMenu works well.
        pass

    def load_from_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Game", "", "Gomoku Files (*.sgf *.pgn *.txt);;All Files (*)")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.view_model.load_game(content)
            except Exception as e:
                self.log(f"Error loading file: {e}")

    def load_from_paste(self):
        dialog = PastePositionDialog(self)
        if dialog.exec():
            content = dialog.get_content()
            if content:
                self.view_model.load_game(content)
            else:
                self.log("Empty content entered.")
                
    def open_settings(self):
        dialog = SettingsDialog(self.view_model.config, self)
        if dialog.exec():
            new_config = dialog.get_config()
            self.view_model.update_config(new_config)
            self.log(f"Settings updated: Rule={new_config.rule_name}, Time={new_config.time_limit_ms}ms")
            
    def toggle_analysis(self):
        if self.view_model.state == SystemState.RUNNING:
            self.view_model.stop_analysis()
        elif self.view_model.state == SystemState.IDLE:
            player_name = self.player_input.text().strip()
            save_profile = self.chk_save_profile.isChecked()
            self.view_model.start_analysis(player_name, save_profile)

    def on_game_loaded(self, move_count):
        self.log(f"Game loaded successfully: {move_count} moves.")
        self.board_widget.set_board(self.view_model.current_board)
        self.btn_analyze.setEnabled(True)
        
    def on_system_state_changed(self, state: SystemState):
        if state == SystemState.RUNNING:
            self.btn_analyze.setText("Stop Analysis")
            self.btn_analyze.setEnabled(True)
            self.btn_load.setEnabled(False)
            self.btn_settings.setEnabled(False)
            self.player_input.setEnabled(False)
            self.chk_save_profile.setEnabled(False)
            
            # Clear logs only on new start? 
            # Ideally ViewModel controls when to clear, but UI logic is fine here.
            self.log_area.clear()
            self.graph_widget.clear()
            self.console_widget.clear()
            self.analysis_log_widget.clear()
            self.pattern_matrix_widget.clear()
            self.log("Analysis started...")
            self.progress_bar.setVisible(True)
            
        elif state == SystemState.STOPPING:
            self.btn_analyze.setText("Stopping...")
            self.btn_analyze.setEnabled(False) # Prevent double click
            self.log("Stopping analysis...")
            
        elif state == SystemState.IDLE:
            self.btn_analyze.setText("Start Analysis")
            self.btn_analyze.setEnabled(True)
            self.btn_load.setEnabled(True)
            self.btn_settings.setEnabled(True)
            self.player_input.setEnabled(True)
            self.chk_save_profile.setEnabled(True)
            self.log("Analysis finished/stopped.")
            self.progress_bar.setVisible(False)
            
    def on_move_update(self, move_idx):
        self.board_widget.set_current_move(move_idx)
    
    def on_visible_step_changed(self, step):
        self.board_widget.set_visible_count(step)
        
    def on_analysis_progress(self, current, total):
        """Update progress bar."""
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
            
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"Analyzing... {current}/{total} moves ({current/total*100:.0f}%)")
        
    def on_analysis_result(self, result):
        """Handle V2MoveResult from the V2AnalysisWorker."""
        acc_percent = result.accuracy_score * 100
        p_cheat_percent = result.p_cheat_cumulative * 100
        p_human_percent = (1 - result.p_cheat_cumulative) * 100
        
        # V2 Anti-Cheat Classification Logic using P(Cheat)
        status = "Normal"
        
        # High P(Cheat) + Low Delta = SUSPICIOUS
        if result.p_cheat_cumulative > 0.70:
            status = "🚨 CHEATER (High P(Cheat))"
        elif result.p_cheat_cumulative > 0.50:
            status = "⚠️ SUSPICIOUS (Elevated P(Cheat))"
        elif result.delta < 0.02 and not result.is_forced:
            status = "✨ Near-Optimal Move"
        elif result.delta > 0.20:
            status = "❌ Blunder (Large Delta)"
        elif result.is_forced:
            status = "🔒 Forced Move"
        
        # Detailed Log with V2 metrics
        msg = (f"Move {result.move_notation}:\n"
               f"  • Winrate: Played {result.played_winrate:.1%} | Best {result.best_winrate:.1%}\n"
               f"  • Delta (Δ): {result.delta*100:.1f}% | Accuracy: {acc_percent:.1f}%\n"
               f"  • Bayesian: P(Cheat) {p_cheat_percent:.1f}% | P(Human) {p_human_percent:.1f}%\n"
               f"  • Position: Complexity {result.position_complexity:.2f} | Forced: {result.is_forced}\n"
               f"  • Status: {status}")

        self.log(msg)
        # Update graph with V2 data
        self.graph_widget.add_data(
            result.accuracy_score, 
            result.played_winrate, 
            result.best_winrate,
            move_index=result.move_index,
            delta=result.delta,
            complexity=result.position_complexity,
            p_cheat=result.p_cheat_cumulative
        )
        
        # Update Pattern Matrix (V3)
        if result.pattern_context:
            self.pattern_matrix_widget.add_move_context(result.pattern_context)
        
    def on_engine_log(self, type_, msg):
        """
        Callback from engine background thread.
        MUST emit signal to update UI on main thread.
        """
        self.sig_log_message.emit(type_, msg)
        
    def _on_log_message_safe(self, type_, msg):
        """
        Slot handling log messages on Main Thread.
        """
        # Write ALL logs to file
        self._write_to_log_file(type_, msg)
        
        if type_ == "CMD":
            self.console_widget.log_input(msg)
        elif type_ == "STDIN":
            # Raw command sent to engine
            self.console_widget.log_input(msg)
        elif type_ == "STDOUT":
            # Raw response from engine
            self.console_widget.log_output(msg)
        elif type_ == "INFO":
            self.console_widget.log_info(msg)
        elif type_ == "STAGE":
            # Route to Analysis Flow tab with smart categorization
            if msg.startswith("===") or msg.startswith("ANALYZING"):
                self.analysis_log_widget.log_header(msg)
            elif "BEST" in msg or "#1:" in msg:
                self.analysis_log_widget.log_best(msg)
            elif "PLAYER" in msg or "Played:" in msg:
                self.analysis_log_widget.log_played(msg)
            elif "RESULT" in msg or "ACCURACY" in msg or "Regret" in msg:
                self.analysis_log_widget.log_result(msg)
            elif "Position:" in msg or "INPUT:" in msg:
                self.analysis_log_widget.log_position(msg)
            else:
                self.analysis_log_widget.log(msg)
        else:
            self.console_widget.log_output(msg)
        
    def log(self, message):
        # Also write to file
        self._write_to_log_file("REPORT", message)
        self.log_area.append(message)
