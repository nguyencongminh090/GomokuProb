import sqlite3
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QSpinBox, QPushButton, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QMessageBox, QAbstractItemView)
from PyQt6.QtCore import Qt

class DatabaseViewDialog(QDialog):
    def __init__(self, profile_store, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profile Database Viewer")
        self.resize(1000, 600)
        self.store = profile_store
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Filters and Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Player:"))
        self.player_edit = QLineEdit()
        self.player_edit.setPlaceholderText("Filter by player name...")
        self.player_edit.returnPressed.connect(self.load_data)
        controls_layout.addWidget(self.player_edit)

        controls_layout.addWidget(QLabel("Limit:"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(10, 10000)
        self.limit_spin.setSingleStep(50)
        self.limit_spin.setValue(100)
        controls_layout.addWidget(self.limit_spin)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.load_data)
        controls_layout.addWidget(self.btn_refresh)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Played At", "Player", "Outcome", "Moves", "Q. Score", 
            "Accuracy %", "Delta %", "P(Cheat) %", "Verdict", "Baseline?"
        ])
        
        # Configure table appearance
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # Columns sizing
        self.table.setColumnWidth(0, 150) # Played At
        self.table.setColumnWidth(1, 120) # Player
        self.table.setColumnWidth(2, 70)  # Outcome
        self.table.setColumnWidth(3, 60)  # Moves
        self.table.setColumnWidth(4, 70)  # Q. Score
        self.table.setColumnWidth(5, 80)  # Accuracy
        self.table.setColumnWidth(6, 70)  # Delta
        self.table.setColumnWidth(7, 80)  # P(Cheat)
        self.table.setColumnWidth(8, 120) # Verdict
        self.table.setColumnWidth(9, 70)  # Baseline?

        # Summary Stats
        self.stats_label = QLabel("Loading statistics...")
        layout.addWidget(self.stats_label)

    def load_data(self):
        player_name = self.player_edit.text().strip()
        player_filter = player_name if player_name else None
        limit = self.limit_spin.value()

        try:
            games = self.store.get_all_games(player_id=player_filter, limit=limit)
            self._populate_table(games)
            self._update_stats(player_filter)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")

    def _populate_table(self, games):
        self.table.setRowCount(len(games))
        for row, game in enumerate(games):
            
            # Formatting values safely
            played_at = game.get("played_at", "")
            played_at = played_at.replace("T", " ")[:16] if played_at else "Unknown"

            player = game.get("player_id", "")
            outcome = game.get("outcome", "")
            
            moves_anz = game.get("analyzed_moves", 0)
            moves_tot = game.get("total_moves", 0)
            moves_str = f"{moves_anz}/{moves_tot}"

            qs = game.get("quality_score")
            qs_str = f"{qs:.2f}" if qs is not None else "N/A"

            # Inverse relationship: higher accuracy usually means smaller delta and higher lambda. 
            # If the DB doesn't store a direct Accuracy metric, we might show near_optimal_pct * 100.
            near_opt = game.get("near_optimal_pct")
            acc_str = f"{near_opt * 100:.1f}%" if near_opt is not None else "N/A"

            delta = game.get("mean_delta")
            delta_str = f"{delta * 100:.1f}%" if delta is not None else "N/A"

            p_cheat = game.get("p_cheat")
            p_cheat_str = f"{p_cheat * 100:.1f}%" if p_cheat is not None else "0.0%"

            verdict = game.get("classification", "Unknown")
            
            baseline = "Yes" if game.get("is_baseline_eligible") else "No"
            
            # Assign items
            items = [
                QTableWidgetItem(played_at),
                QTableWidgetItem(player),
                QTableWidgetItem(outcome.capitalize()),
                QTableWidgetItem(moves_str),
                QTableWidgetItem(qs_str),
                QTableWidgetItem(acc_str),
                QTableWidgetItem(delta_str),
                QTableWidgetItem(p_cheat_str),
                QTableWidgetItem(verdict),
                QTableWidgetItem(baseline)
            ]

            # Right-align numeric columns
            for col in range(3, 8):
                items[col].setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            for col, item in enumerate(items):
                self.table.setItem(row, col, item)

    def _update_stats(self, player_filter):
        if not self.store.db_path:
            self.stats_label.setText("No database available.")
            return

        with sqlite3.connect(self.store.db_path) as conn:
            if player_filter:
                cursor = conn.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN is_baseline_eligible=1 THEN 1 ELSE 0 END) FROM games WHERE player_id COLLATE NOCASE = ?",
                    (player_filter,)
                )
                total, baseline = cursor.fetchone()
                total = total or 0
                baseline = baseline or 0
                self.stats_label.setText(f"Player '{player_filter}': {total} games total ({baseline} baseline tier games).")
            else:
                cursor = conn.execute("SELECT COUNT(*), COUNT(DISTINCT player_id) FROM games")
                total, unique_players = cursor.fetchone()
                total = total or 0
                unique_players = unique_players or 0
                self.stats_label.setText(f"Database summary: {total} total games across {unique_players} unique players.")
