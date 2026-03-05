"""
Custom Board Widget for Gomoku.
Handles drawing of board, stones, and analysis markers.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRectF, QSize
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor, QFont, QRadialGradient

from typing import Optional
from src.core.board import BoardState, Move

class BoardWidget(QWidget):
    """
    Visual representation of the Gomoku board.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.board_size = 15
        self.board_state: Optional[BoardState] = None
        self.margin = 30
        self.cell_size = 30
        self.setMinimumSize(500, 500)
        self.visible_count = -1 # -1 means all
        self.blunder_indices = set() # Indices of moves that are blunders
        
    def set_board(self, board: BoardState):
        self.board_state = board
        self.visible_count = -1 # Reset on new board
        self.blunder_indices.clear()
        self.update() 
        
    def set_visible_count(self, count: int):
        self.visible_count = count
        self.update()
        
    def set_blunders(self, indices: set):
        self.blunder_indices = indices
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Draw Background (Wood texture or solid color)
        painter.fillRect(self.rect(), QColor("#EBB563")) # Typical Goban color
        
        # Calculate grid geometry
        size = min(self.width(), self.height())
        
        self.cell_size = size / (self.board_size + 1)
        self.margin = self.cell_size
        
        start_x = (self.width() - (self.board_size - 1) * self.cell_size) / 2
        start_y = (self.height() - (self.board_size - 1) * self.cell_size) / 2
        
        # 2. Draw Grid Lines
        pen = QPen(Qt.GlobalColor.black, 1.5)
        painter.setPen(pen)
        
        # Grid Coordinates
        letters = "ABCDEFGHIJKLMNO" # Include 'I' as requested
        font = painter.font()
        font.setBold(False)
        # Dynamic font size: 40% of cell size (fits comfortably in 1.0 cell_size margin)
        font.setPixelSize(int(self.cell_size * 0.4)) 
        painter.setFont(font)
        
        for i in range(self.board_size):
            # Horizontal (Rows)
            y = start_y + i * self.cell_size
            painter.drawLine(int(start_x), int(y), int(start_x + (self.board_size - 1) * self.cell_size), int(y))
            
            # Row Labels
            label = str(self.board_size - i)
            rect_left = QRectF(start_x - self.margin, y - self.cell_size/2, self.margin, self.cell_size)
            painter.drawText(rect_left, Qt.AlignmentFlag.AlignCenter, label)
            
            # Vertical (Cols)
            x = start_x + i * self.cell_size
            painter.drawLine(int(x), int(start_y), int(x), int(start_y + (self.board_size - 1) * self.cell_size))
            
            # Col Labels (A-O)
            if i < len(letters):
                label = letters[i]
                rect_top = QRectF(x - self.cell_size/2, start_y - self.margin, self.cell_size, self.margin)
                painter.drawText(rect_top, Qt.AlignmentFlag.AlignCenter, label)
            
        # Draw Star Points
        hoshi_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        painter.setBrush(Qt.GlobalColor.black)
        for hx, hy in hoshi_points:
            cx = start_x + hx * self.cell_size
            cy = start_y + hy * self.cell_size
            painter.drawEllipse(QPoint(int(cx), int(cy)), 3, 3)
            
        # 3. Draw Stones
        if self.board_state:
            moves = self.board_state.moves
            limit = len(moves)
            if self.visible_count >= 0:
                limit = min(limit, self.visible_count)
            
            # Only iterate up to limit
            for i in range(limit):
                move = moves[i]
                cx = start_x + move.x * self.cell_size
                cy = start_y + move.y * self.cell_size
                radius = self.cell_size * 0.45
                
                is_current_analyzed = (hasattr(self, 'current_move_index') and 
                                       self.current_move_index == i)
                
                # Check for Blunder Glow
                if i in self.blunder_indices:
                    painter.setPen(Qt.PenStyle.NoPen)
                    # Semi-transparent Red Glow
                    glow_color = QColor(255, 0, 0, 150) 
                    glow_radius = radius * 1.4
                    painter.setBrush(glow_color)
                    painter.drawEllipse(QPoint(int(cx), int(cy)), int(glow_radius), int(glow_radius))

                # Shadow
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 50))
                painter.drawEllipse(QPoint(int(cx + 2), int(cy + 2)), int(radius), int(radius))
                
                # Stone
                if move.color == 1: # Black
                    grad = QRadialGradient(cx - radius*0.4, cy - radius*0.4, radius * 1.5)
                    grad.setColorAt(0, QColor("#666666"))
                    grad.setColorAt(1, Qt.GlobalColor.black)
                    painter.setBrush(QBrush(grad))
                    painter.setPen(Qt.PenStyle.NoPen)
                    
                    text_color = Qt.GlobalColor.white
                else: # White
                    grad = QRadialGradient(cx - radius*0.4, cy - radius*0.4, radius * 1.5)
                    grad.setColorAt(0, Qt.GlobalColor.white)
                    grad.setColorAt(1, QColor("#DDDDDD"))
                    painter.setBrush(QBrush(grad))
                    painter.setPen(QPen(QColor("#999999"), 1))
                    
                    text_color = Qt.GlobalColor.black
                    
                painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))
                
                # Draw Move Number
                if is_current_analyzed:
                    text_color = QColor("red")
                    
                painter.setPen(text_color)
                font = painter.font()
                font.setBold(True)
                font.setPixelSize(int(radius)) # Scale font
                painter.setFont(font)
                rect = QRectF(cx - radius, cy - radius, radius*2, radius*2)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(i + 1))

    def set_current_move(self, idx: int):
        self.current_move_index = idx
        self.update()

    def sizeHint(self):
        return QSize(600, 600)
