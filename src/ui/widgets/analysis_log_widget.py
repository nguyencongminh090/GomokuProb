"""
Analysis Log Widget for Stage-by-Stage Analysis Logging.
Shows detailed analysis flow: candidates, player evaluation, accuracy.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtGui import QColor, QTextCursor, QTextCharFormat, QFont

class AnalysisLogWidget(QWidget):
    """Widget for displaying detailed analysis stages."""
    
    # Color scheme
    COLORS = {
        "header": "#61AFEF",    # Blue for headers
        "best": "#98C379",      # Green for best move
        "played": "#E5C07B",    # Yellow for played move
        "result": "#C678DD",    # Purple for result
        "position": "#56B6C2",  # Cyan for position/input
        "default": "#ABB2BF",   # Light gray for general text
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(
            "background-color: #282C34; color: #ABB2BF; font-family: 'Fira Code', Monospace;"
        )
        font = QFont("Fira Code")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.text_edit.setFont(font)
        
        layout.addWidget(self.text_edit)
    
    def log(self, text: str, category: str = "default"):
        """Log a message with color based on category."""
        color = self.COLORS.get(category, self.COLORS["default"])
        self._append(text, color)
        
    def log_header(self, text: str):
        self._append(text, self.COLORS["header"])
        
    def log_best(self, text: str):
        self._append(text, self.COLORS["best"])
        
    def log_played(self, text: str):
        self._append(text, self.COLORS["played"])
        
    def log_result(self, text: str):
        self._append(text, self.COLORS["result"])
        
    def log_position(self, text: str):
        self._append(text, self.COLORS["position"])
        
    def _append(self, text: str, color_hex: str):
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color_hex))
        
        cursor = self.text_edit.textCursor()
        cursor.setCharFormat(fmt)
        cursor.insertText(text + "\n")
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()
        
    def clear(self):
        self.text_edit.clear()
