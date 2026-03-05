"""
Engine Console Widget for STDIN/STDOUT Logging.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtGui import QColor, QTextCursor, QTextCharFormat, QFont

class EngineConsoleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("background-color: #1E1E1E; color: #D4D4D4; font-family: Monospace;")
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.text_edit.setFont(font)
        
        layout.addWidget(self.text_edit)
        
    def log_input(self, text: str):
        self._append(f">> {text}", "#4EC9B0") # Cyan for Input
        
    def log_output(self, text: str):
        self._append(f"{text}", "#CE9178") # Orangeish for Output
        
    def log_info(self, text: str):
        self._append(f"[INFO] {text}", "#6A9955") # Green for Info
        
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
