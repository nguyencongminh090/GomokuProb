from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QHeaderView, QLabel, QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush

from src.core.rapfi_patterns import RapfiPattern

class PatternMatrixWidget(QWidget):
    """
    Visualizes the Pattern Interaction Matrix (Heatmap).
    Rows: Opponent Patterns (Blocked)
    Cols: Own Patterns (Created/Extended)
    Cells: Count of moves matching (Own, Opp).
    """
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>Pattern Synergy Matrix</b> (Row=Opp, Col=Own)"))
        layout.addLayout(header_layout)
        
        # Table Heatmap
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        
        # Define Patterns to show (exclude rare ones to keep matrix clean if needed, 
        # or show all relevant ones. Let's show common static patterns).
        # 0=None, 2=Flex2... 13=Five
        self.relevant_codes = [
            RapfiPattern.NONE, 
            RapfiPattern.L_FLEX2, 
            RapfiPattern.K_BLOCK3,
            RapfiPattern.J_FLEX2_2X,
            RapfiPattern.H_FLEX3,
            RapfiPattern.B_FLEX4,
            RapfiPattern.A_FIVE
        ]
        
        count = len(self.relevant_codes)
        self.table.setRowCount(count)
        self.table.setColumnCount(count)
        
        labels = [RapfiPattern.describe(c).replace("_", " ") for c in self.relevant_codes]
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setVerticalHeaderLabels(labels)
        
        # Resize behavior
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.table)
        
    def update_data(self, pattern_counts: dict):
        """
        Update the heatmap with new data.
        pattern_counts: Counter/Dict {(own_code, opp_code): count}
        """
        self.table.clearContents()
        
        if not pattern_counts:
            return

        # Find max for normalization
        max_val = max(pattern_counts.values()) if pattern_counts else 1
        
        for r, opp_code in enumerate(self.relevant_codes):
            for c, own_code in enumerate(self.relevant_codes):
                
                count = pattern_counts.get((own_code, opp_code), 0)
                
                if count > 0:
                    item = QTableWidgetItem(str(count))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    # Heatmap Color (White -> Red/Blue)
                    intensity = count / max_val
                    
                    # Background
                    # White (255,255,255) -> Red (255, 0, 0)
                    bg_val = int(255 * (1 - intensity * 0.8)) 
                    color = QColor(255, bg_val, bg_val)
                    
                    if intensity > 0.5:
                         color = QColor(255, 100 + int(100*(1-intensity)), 100 + int(100*(1-intensity)))

                    item.setBackground(QBrush(color))
                    self.table.setItem(r, c, item)
                else:
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setForeground(QBrush(QColor("gray")))
                    self.table.setItem(r, c, item)
                    
    def clear(self):
        """Reset counts."""
        self.current_counts = {}
        self.table.clearContents()
        
    def add_move_context(self, ctx: dict):
        """Aggregate single move context and refresh."""
        if not ctx:
            return
            
        own = ctx.get('own_pattern', 0)
        opp = ctx.get('opp_pattern', 0)
        
        if not hasattr(self, 'current_counts'):
            self.current_counts = {}
            
        key = (own, opp)
        self.current_counts[key] = self.current_counts.get(key, 0) + 1
        
        self.update_data(self.current_counts)
