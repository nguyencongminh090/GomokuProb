"""
Graph Widget for V2 Analysis Visualization.
"""
from PyQt6.QtWidgets import QWidget, QMenu, QInputDialog
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QAction, QPalette
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from typing import List

class AnalysisGraphWidget(QWidget):
    # Signal emitted when blunders are detected. Payload is a set of move indices (0-based)
    blunders_detected = pyqtSignal(object) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.accuracies: List[float] = []      # 0.0 to 1.0 (Blue)
        self.winrates: List[float] = []        # 0.0 to 1.0 (Orange - Played)
        self.best_winrates: List[float] = []   # 0.0 to 1.0 (Green - Best)
        
        # V2 Data
        self.deltas: List[float] = []          # 0.0 to 1.0 (Delta = best - played)
        self.complexities: List[float] = []   # 0.0 to 1.0 (Position complexity)
        self.p_cheats: List[float] = []        # 0.0 to 1.0 (Cumulative P(Cheat))
        
        # Visibility Flags
        self.show_played = True
        self.show_best = True
        self.show_blunders = True
        
        # V2 Visibility Flags
        self.show_accuracy = True   # Show Cumulative Mean Accuracy (Blue)
        self.show_complexity = True # Show Complexity
        self.show_pcheat = True     # Show P(Cheat)
        
        # Blunder Detection Configuration
        # Blunder = Delta > threshold * (1 - complexity)
        # Lower complexity = stricter threshold
        self.blunder_base_threshold = 0.15  # Base 15% drop
        self.last_emitted_blunders = set()
        
        # Store explicit move indices to handle non-contiguous data (e.g. side analysis)
        self.move_indices: List[int] = [] 
        
        self.setMinimumHeight(150)
        self.setBackgroundRole(QPalette.ColorRole.Base)
        self.setAutoFillBackground(True)
        self.setMouseTracking(True)
        self.hover_pos = None
        
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        
        def add_toggle(name, flag_name):
            action = QAction(name, self, checkable=True)
            action.setChecked(getattr(self, flag_name))
            action.triggered.connect(lambda checked: self.toggle_graph(flag_name, checked))
            menu.addAction(action)
        
        # V2 Metrics
        menu.addAction(QAction("=== V2 Metrics ===", self))
        add_toggle("Show Cumulative Accuracy (Blue)", "show_accuracy")
        add_toggle("Show Complexity (Magenta)", "show_complexity")
        add_toggle("Show P(Cheat) (Red)", "show_pcheat")
        
        menu.addSeparator()
        
        # Winrates
        menu.addAction(QAction("=== Winrates ===", self))
        add_toggle("Show Played Winrate (Orange)", "show_played")
        add_toggle("Show Best Winrate (Green)", "show_best")
        
        menu.addSeparator()
        add_toggle("Show Blunders (X)", "show_blunders")
        
        action_config = QAction(f"Set Blunder Base Threshold... (Current: {self.blunder_base_threshold*100:.0f}%)", self)
        action_config.triggered.connect(self.set_blunder_threshold)
        menu.addAction(action_config)
        
        menu.exec(event.globalPos())
        
    def set_blunder_threshold(self):
        val, ok = QInputDialog.getDouble(self, "Blunder Config", 
                                         "Base Threshold (0.05 - 0.50):\n(Adjusted by complexity)", 
                                         self.blunder_base_threshold, 0.05, 0.50, 2)
        if ok:
            self.blunder_base_threshold = val
            self.last_emitted_blunders = set()  # Force re-emit
            self.update()
        
    def toggle_graph(self, flag_name, checked):
        setattr(self, flag_name, checked)
        self.update()

    def add_accuracy(self, acc: float):
        # Backward compat
        self.add_data(acc, 0.0, 0.0)
        
    def add_data(self, acc: float, winrate: float, best_winrate: float, 
                 move_index: int = -1, delta: float = 0.0, 
                 complexity: float = 0.5, p_cheat: float = 0.0):
        """Add data point for V2 visualization."""
        self.accuracies.append(acc)
        self.winrates.append(winrate)
        self.best_winrates.append(best_winrate)
        
        # V2 Data
        self.deltas.append(delta)
        self.complexities.append(complexity)
        self.p_cheats.append(p_cheat)
        
        # Store index
        if move_index >= 0:
            self.move_indices.append(move_index)
        else:
            start = self.move_indices[-1] + 1 if self.move_indices else 0
            self.move_indices.append(start)
            
        self.update()
        
    def clear(self):
        self.accuracies = []
        self.winrates = []
        self.best_winrates = []
        self.deltas = []
        self.complexities = []
        self.p_cheats = []
        self.move_indices = []
        self.hover_pos = None
        self.last_emitted_blunders = set()
        self.blunders_detected.emit(set())
        self.update()

    def mouseMoveEvent(self, event):
        self.hover_pos = event.pos()
        self.update()

    def leaveEvent(self, event):
        self.hover_pos = None
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#FAFAFA"))
        
        w = self.width()
        h = self.height()
        padding = 20
        
        # Draw frame
        painter.setPen(QColor("#CCCCCC"))
        painter.drawRect(0, 0, w-1, h-1)
        
        if not self.accuracies:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Data")
            # Hint text
            painter.setPen(QColor("#999999"))
            painter.drawText(self.rect().adjusted(0, 40, 0, 0), Qt.AlignmentFlag.AlignCenter, "(Right-click to toggle graphs)")
            return
            
        # Draw Grid (0%, 50%, 100%)
        # Y scale: 0.0 -> h-padding, 1.0 -> padding
        def get_y(val):
            return h - padding - (val * (h - 2 * padding))
            
        painter.setPen(QPen(QColor("#EEEEEE"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(padding), int(get_y(0.0)), int(w-padding), int(get_y(0.0)))
        painter.drawLine(int(padding), int(get_y(0.5)), int(w-padding), int(get_y(0.5)))
        painter.drawLine(int(padding), int(get_y(1.0)), int(w-padding), int(get_y(1.0)))
            
        # Helper to plot a list of values
        def plot_line(values, color_hex, is_dot=False):
            count = len(values)
            if count < 2: return []
            
            step_x = (w - 2 * padding) / max(1, count - 1)
            points = []
            for i, val in enumerate(values):
                x = padding + i * step_x
                y = get_y(val)
                points.append(QPointF(x, y))
                
            pen = QPen(QColor(color_hex), 2)
            if is_dot:
                pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.drawPolyline(points)
            
            painter.setBrush(QColor(color_hex))
            for p in points:
                painter.drawEllipse(p, 3, 3)
            return points 
            
        # --- Drawing Lines ---
        
        acc_points = []  # For hover indexing guidance
        
        # Winrates
        if self.show_played:
            pts = plot_line(self.winrates, "#FF9800", is_dot=True)  # Orange
            if not acc_points: acc_points = pts
            
        if self.show_best:
            pts = plot_line(self.best_winrates, "#4CAF50", is_dot=True)  # Green
            if not acc_points: acc_points = pts
        
        # --- V2 Metrics Plotting ---
        
        # Cumulative Mean Accuracy: Running average (Blue)
        if self.show_accuracy and self.accuracies:
            cumulative_acc = []
            running_sum = 0.0
            for i, acc in enumerate(self.accuracies):
                running_sum += acc
                cumulative_acc.append(running_sum / (i + 1))
            pts = plot_line(cumulative_acc, "#2196F3")  # Blue
            if not acc_points: acc_points = pts
            
        # Complexity: 0-1 scale
        if self.show_complexity and self.complexities:
            pts = plot_line(self.complexities, "#E91E63", is_dot=True)  # Magenta/Pink
            if not acc_points: acc_points = pts
            
        # P(Cheat): 0-1 scale, cumulative
        if self.show_pcheat and self.p_cheats:
            pts = plot_line(self.p_cheats, "#F44336")  # Red
            if not acc_points: acc_points = pts
            
        # --- Blunder Detection & Drawing (Delta + Complexity) ---
        current_blunders = set()
        count = len(self.deltas) if self.deltas else 0
        step_x = (w - 2 * padding) / max(1, count - 1) if count > 1 else 0

        # Calculate Blunders using Delta + Complexity
        # Formula: blunder if delta > base_threshold * (2 - complexity)
        # Low complexity (0.2) -> threshold * 1.8 = stricter
        # High complexity (0.8) -> threshold * 1.2 = more lenient
        for i in range(count):
            delta = self.deltas[i] if i < len(self.deltas) else 0.0
            complexity = self.complexities[i] if i < len(self.complexities) else 0.5
            
            # Adjusted threshold based on complexity
            # Simple position (low complexity) = lower threshold = easier to be blunder
            adjusted_threshold = self.blunder_base_threshold * (1.5 - complexity * 0.5)
            
            if delta > adjusted_threshold:
                actual_idx = self.move_indices[i] if i < len(self.move_indices) else i
                current_blunders.add(actual_idx)
                
                # Draw only if enabled
                if self.show_blunders and step_x > 0:
                    x = padding + i * step_x
                    wr = self.winrates[i] if i < len(self.winrates) else 0.5
                    y = get_y(wr)
                    
                    # Draw Red 'X'
                    painter.setPen(QPen(QColor("red"), 2))
                    painter.drawLine(int(x - 4), int(y - 4), int(x + 4), int(y + 4))
                    painter.drawLine(int(x - 4), int(y + 4), int(x + 4), int(y - 4))
        
        # Emit signal if changed
        if current_blunders != self.last_emitted_blunders:
            self.last_emitted_blunders = current_blunders
            self.blunders_detected.emit(current_blunders)
            
        # --- Labels/Legend ---
        
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        
        # Legend (Dynamic stacking)
        legend_x = padding + 10
        
        if self.show_played:
            painter.setPen(QColor("#FF9800"))
            painter.drawText(int(legend_x), int(padding) - 5, "Played")
            legend_x += 60
            
        if self.show_best:
            painter.setPen(QColor("#4CAF50"))
            painter.drawText(int(legend_x), int(padding) - 5, "Best")
            legend_x += 50
        
        # V2 Metrics Legend
        if self.show_accuracy:
            painter.setPen(QColor("#2196F3"))
            painter.drawText(int(legend_x), int(padding) - 5, "Acc(Σ)")
            legend_x += 50
            
        if self.show_complexity:
            painter.setPen(QColor("#E91E63"))
            painter.drawText(int(legend_x), int(padding) - 5, "Cpx")
            legend_x += 35
            
        if self.show_pcheat:
            painter.setPen(QColor("#F44336"))
            painter.drawText(int(legend_x), int(padding) - 5, "P(C)")
            legend_x += 40
        
        # V2 Stats at top-right
        if self.accuracies:
            stats_x = w - 250
            mean_acc = sum(self.accuracies) / len(self.accuracies)
            mean_delta = sum(self.deltas) / len(self.deltas) if self.deltas else 0.0
            final_pcheat = self.p_cheats[-1] if self.p_cheats else 0.0
            
            painter.setPen(QColor("#333333"))
            painter.drawText(int(stats_x), int(padding) - 5, 
                           f"Acc:{mean_acc*100:.0f}% | Δ:{mean_delta*100:.1f}% | P(C):{final_pcheat*100:.0f}%")

        # --- Drops & Tooltip ---
        if self.hover_pos and acc_points:
            count = len(self.accuracies)
            step_x = (w - 2 * padding) / max(1, count - 1)
            hx = self.hover_pos.x()
            
            if step_x > 0:
                idx = round((hx - padding) / step_x)
                idx = max(0, min(idx, count - 1))
                
                # Draw vertical line
                px = padding + idx * step_x
                painter.setPen(QPen(QColor("#666666"), 1, Qt.PenStyle.DotLine))
                painter.drawLine(int(px), int(padding), int(px), int(h - padding))
                
                # Compose Tooltip Text (With Correct Explicit Index)
                actual_idx = self.move_indices[idx] if idx < len(self.move_indices) else idx
                display_move = actual_idx + 1  # Display as 1-based
                tips = [f"Move {display_move}"]
                
                # Winrates
                if self.show_played:
                    val = self.winrates[idx] if idx < len(self.winrates) else 0.0
                    tips.append(f"Played: {val*100:.1f}%")
                    
                if self.show_best:
                    val = self.best_winrates[idx] if idx < len(self.best_winrates) else 0.0
                    tips.append(f"Best: {val*100:.1f}%")
                
                # V2 Metrics
                if self.show_accuracy:
                    # Show cumulative mean accuracy up to this point
                    if idx < len(self.accuracies):
                        running_sum = sum(self.accuracies[:idx+1])
                        cumulative = running_sum / (idx + 1)
                    else:
                        cumulative = 0.0
                    tips.append(f"Acc(Σ): {cumulative*100:.0f}%")
                    
                if self.show_complexity:
                    val = self.complexities[idx] if idx < len(self.complexities) else 0.5
                    tips.append(f"Cpx: {val:.2f}")
                    
                if self.show_pcheat:
                    val = self.p_cheats[idx] if idx < len(self.p_cheats) else 0.0
                    tips.append(f"P(C): {val*100:.0f}%")
                
                tip_text = "\n".join(tips)
                
                painter.setPen(QColor("#000000"))
                painter.setBrush(QColor("#FFFFCC"))
                
                fm = painter.fontMetrics()
                lines = tip_text.split('\n')
                text_w = max(fm.horizontalAdvance(line) for line in lines)
                text_h = fm.height() * len(lines)
                pad_x = 12
                pad_y = 8
                
                box_w = text_w + 2 * pad_x
                box_h = text_h + 2 * pad_y
                
                # Position
                tx = px + 15
                ty = self.hover_pos.y() - box_h / 2
                
                if tx + box_w > w: tx = px - box_w - 15
                if ty < 0: ty = 0
                if ty + box_h > h: ty = h - box_h
                
                painter.drawRect(int(tx), int(ty), int(box_w), int(box_h))
                painter.drawText(int(tx), int(ty), int(box_w), int(box_h), Qt.AlignmentFlag.AlignCenter, tip_text)
