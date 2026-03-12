"""
Dialogs for Gomoku Accuracy Tool.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, 
                             QPlainTextEdit, QDialogButtonBox, QFormLayout,
                             QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QPushButton, QFileDialog, QCheckBox)
from PyQt6.QtCore import Qt
from src.core.config import AnalysisConfig

class PastePositionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paste Position")
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        lbl = QLabel("Enter position string (e.g. 'h8 i9 h9' or PGN/SGF content):")
        layout.addWidget(lbl)
        
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Paste moves here...")
        layout.addWidget(self.text_edit)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_content(self) -> str:
        return self.text_edit.toPlainText().strip()

class SettingsDialog(QDialog):
    def __init__(self, config: AnalysisConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Settings")
        self.resize(450, 550)
        self.config = config
        
        layout = QFormLayout(self)
        
        # === Engine Settings Section ===
        layout.addRow(QLabel("<b>Engine Settings</b>"))
        
        # Engine Path
        self.engine_edit = QLineEdit(self.config.engine_path)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_engine)
        layout.addRow("Engine Path:", self.engine_edit)
        layout.addRow("", btn_browse)
        
        # Rule
        self.combo_rule = QComboBox()
        self.combo_rule.addItem("Freestyle", 0)
        self.combo_rule.addItem("Standard", 1)
        self.combo_rule.addItem("Renju", 4)
        # Set current index
        index = self.combo_rule.findData(self.config.rule)
        if index >= 0:
            self.combo_rule.setCurrentIndex(index)
        layout.addRow("Game Rule:", self.combo_rule)
        
        # Time Limit
        self.spin_time = QSpinBox()
        self.spin_time.setRange(100, 60000)
        self.spin_time.setSingleStep(100)
        self.spin_time.setSuffix(" ms")
        self.spin_time.setValue(self.config.time_limit_ms)
        layout.addRow("Time per Move:", self.spin_time)
        
        # Node Limit
        self.spin_node = QSpinBox()
        self.spin_node.setRange(0, 10000000)
        self.spin_node.setSingleStep(1000)
        self.spin_node.setSpecialValueText("No Limit")
        self.spin_node.setValue(self.config.node_limit)
        layout.addRow("Node Limit:", self.spin_node)
        
        # Start Move
        self.spin_start = QSpinBox()
        self.spin_start.setRange(1, 999)
        self.spin_start.setValue(self.config.start_move)
        layout.addRow("Analyze From Move:", self.spin_start)
        
        # Analyze Side
        self.combo_side = QComboBox()
        self.combo_side.addItem("Both (Black & White)", "both")
        self.combo_side.addItem("Black Only", "black")
        self.combo_side.addItem("White Only", "white")
        # Set current index
        side_index = self.combo_side.findData(self.config.analyze_side)
        if side_index >= 0:
            self.combo_side.setCurrentIndex(side_index)
        layout.addRow("Analyze Side:", self.combo_side)
        
        # === V2 Anti-Cheat Parameters Section ===
        layout.addRow(QLabel(""))  # Spacer
        layout.addRow(QLabel("<b>V2 Anti-Cheat Parameters</b>"))
        
        # Prior P(Cheat) - as percentage
        self.spin_prior_cheat = QDoubleSpinBox()
        self.spin_prior_cheat.setRange(0.1, 50.0)
        self.spin_prior_cheat.setSingleStep(0.5)
        self.spin_prior_cheat.setSuffix(" %")
        self.spin_prior_cheat.setValue(self.config.prior_cheat * 100)
        self.spin_prior_cheat.setToolTip("Prior probability that any player is cheating (base rate)")
        layout.addRow("Prior P(Cheat):", self.spin_prior_cheat)
        
        # Lambda Human
        self.spin_lambda_human = QDoubleSpinBox()
        self.spin_lambda_human.setRange(1.0, 50.0)
        self.spin_lambda_human.setSingleStep(1.0)
        self.spin_lambda_human.setValue(self.config.lambda_human)
        self.spin_lambda_human.setToolTip("Lambda for human distribution (lower = more mistakes)")
        layout.addRow("λ Human:", self.spin_lambda_human)
        
        # Lambda Cheater
        self.spin_lambda_cheater = QDoubleSpinBox()
        self.spin_lambda_cheater.setRange(10.0, 200.0)
        self.spin_lambda_cheater.setSingleStep(5.0)
        self.spin_lambda_cheater.setValue(self.config.lambda_cheater)
        self.spin_lambda_cheater.setToolTip("Lambda for cheater distribution (higher = fewer mistakes)")
        layout.addRow("λ Cheater:", self.spin_lambda_cheater)
        
        # Threshold Suspicious - as percentage
        self.spin_threshold_sus = QDoubleSpinBox()
        self.spin_threshold_sus.setRange(10.0, 90.0)
        self.spin_threshold_sus.setSingleStep(5.0)
        self.spin_threshold_sus.setSuffix(" %")
        self.spin_threshold_sus.setValue(self.config.threshold_suspicious * 100)
        self.spin_threshold_sus.setToolTip("P(Cheat) above this → Suspicious")
        layout.addRow("Suspicious Threshold:", self.spin_threshold_sus)
        
        # Threshold Cheater - as percentage
        self.spin_threshold_cheat = QDoubleSpinBox()
        self.spin_threshold_cheat.setRange(20.0, 99.0)
        self.spin_threshold_cheat.setSingleStep(5.0)
        self.spin_threshold_cheat.setSuffix(" %")
        self.spin_threshold_cheat.setValue(self.config.threshold_cheater * 100)
        self.spin_threshold_cheat.setToolTip("P(Cheat) above this → Cheater")
        layout.addRow("Cheater Threshold:", self.spin_threshold_cheat)
        
        # Near-Optimal Threshold - as percentage
        self.spin_near_optimal = QDoubleSpinBox()
        self.spin_near_optimal.setRange(0.5, 10.0)
        self.spin_near_optimal.setSingleStep(0.5)
        self.spin_near_optimal.setSuffix(" %")
        self.spin_near_optimal.setValue(self.config.near_optimal_threshold * 100)
        self.spin_near_optimal.setToolTip("Delta below this counts as 'near-optimal' move")
        layout.addRow("Near-Optimal Δ:", self.spin_near_optimal)
        
        # === Temporal Analysis Section ===
        layout.addRow(QLabel(""))  # Spacer
        layout.addRow(QLabel("<b>Temporal Analysis</b>"))
        
        # Temporal Window Size
        self.spin_temporal_window = QSpinBox()
        self.spin_temporal_window.setRange(3, 20)
        self.spin_temporal_window.setValue(self.config.temporal_window_size)
        self.spin_temporal_window.setToolTip("Number of moves for moving average calculation")
        layout.addRow("Window Size:", self.spin_temporal_window)
        
        # Switch Detection Threshold - as percentage
        self.spin_switch_threshold = QDoubleSpinBox()
        self.spin_switch_threshold.setRange(1.0, 30.0)
        self.spin_switch_threshold.setSingleStep(1.0)
        self.spin_switch_threshold.setSuffix(" %")
        self.spin_switch_threshold.setValue(self.config.switch_detection_threshold * 100)
        self.spin_switch_threshold.setToolTip("Minimum Δ change to detect playstyle switch")
        layout.addRow("Switch Detection Δ:", self.spin_switch_threshold)
        
        # === V4 Mixture Model Section ===
        layout.addRow(QLabel(""))  # Spacer
        layout.addRow(QLabel("<b>V4 Mixture Model</b>"))
        
        # Pi (mixing weight)
        self.spin_mixture_pi = QDoubleSpinBox()
        self.spin_mixture_pi.setRange(10.0, 95.0)
        self.spin_mixture_pi.setSingleStep(5.0)
        self.spin_mixture_pi.setSuffix(" %")
        self.spin_mixture_pi.setValue(self.config.mixture_pi * 100)
        self.spin_mixture_pi.setToolTip("Weight of 'good play' component in mixture model")
        layout.addRow("π (Good Play %):", self.spin_mixture_pi)
        
        # Lambda Good
        self.spin_lambda_good = QDoubleSpinBox()
        self.spin_lambda_good.setRange(5.0, 100.0)
        self.spin_lambda_good.setSingleStep(1.0)
        self.spin_lambda_good.setValue(self.config.mixture_lambda_good)
        self.spin_lambda_good.setToolTip("Rate parameter for good plays (higher = tighter around 0)")
        layout.addRow("λ Good:", self.spin_lambda_good)
        
        # Lambda Blunder
        self.spin_lambda_blunder = QDoubleSpinBox()
        self.spin_lambda_blunder.setRange(0.5, 20.0)
        self.spin_lambda_blunder.setSingleStep(0.5)
        self.spin_lambda_blunder.setValue(self.config.mixture_lambda_blunder)
        self.spin_lambda_blunder.setToolTip("Rate parameter for blunders (lower = heavier tail)")
        layout.addRow("λ Blunder:", self.spin_lambda_blunder)
        
        # === V4 Analysis Options Section ===
        layout.addRow(QLabel(""))  # Spacer
        layout.addRow(QLabel("<b>V4 Analysis Options</b>"))
        
        # Enable Model Selection
        self.chk_model_selection = QCheckBox("Compare Exponential / Gamma / Weibull (AIC/BIC)")
        self.chk_model_selection.setChecked(self.config.enable_model_selection)
        self.chk_model_selection.setToolTip("Fit and compare distribution families after analysis")
        layout.addRow("Model Selection:", self.chk_model_selection)
        
        # Enable EM Fitting
        self.chk_em_fitting = QCheckBox("Fit mixture parameters via EM algorithm")
        self.chk_em_fitting.setChecked(self.config.enable_em_fitting)
        self.chk_em_fitting.setToolTip("Estimate optimal π, λ_good, λ_blunder from game data")
        layout.addRow("EM Fitting:", self.chk_em_fitting)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def browse_engine(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Engine Executable")
        if path:
            self.engine_edit.setText(path)
            
    def get_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            engine_path=self.engine_edit.text(),
            rule=self.combo_rule.currentData(),
            time_limit_ms=self.spin_time.value(),
            node_limit=self.spin_node.value(),
            start_move=self.spin_start.value(),
            analyze_side=self.combo_side.currentData(),
            # V2 Parameters (convert percentages back to 0-1)
            prior_cheat=self.spin_prior_cheat.value() / 100.0,
            lambda_human=self.spin_lambda_human.value(),
            lambda_cheater=self.spin_lambda_cheater.value(),
            threshold_suspicious=self.spin_threshold_sus.value() / 100.0,
            threshold_cheater=self.spin_threshold_cheat.value() / 100.0,
            near_optimal_threshold=self.spin_near_optimal.value() / 100.0,
            temporal_window_size=self.spin_temporal_window.value(),
            switch_detection_threshold=self.spin_switch_threshold.value() / 100.0,
            # V4 Mixture Model
            mixture_pi=self.spin_mixture_pi.value() / 100.0,
            mixture_lambda_good=self.spin_lambda_good.value(),
            mixture_lambda_blunder=self.spin_lambda_blunder.value(),
            # V4 Analysis Options
            enable_model_selection=self.chk_model_selection.isChecked(),
            enable_em_fitting=self.chk_em_fitting.isChecked(),
        )
