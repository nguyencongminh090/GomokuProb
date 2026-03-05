"""
Application Entry Point.
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
# Add PyGomo to path
project_root = os.path.dirname(current_dir)
pygomo_path = os.path.join(project_root, 'lib', 'PyGomo', 'src')
if pygomo_path not in sys.path:
    sys.path.insert(0, pygomo_path)

from PyQt6.QtWidgets import QApplication
from src.ui.mainwindow import MainWindow

def main():
    import signal
    from PyQt6.QtCore import QTimer
    
    app = QApplication(sys.argv)
    
    # Allow Ctrl+C to interrupt by letting python interpreter run periodically
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)
    
    # Optional: Apply Material styling
    # from qt_material import apply_stylesheet
    # apply_stylesheet(app, theme='dark_teal.xml')
    
    window = MainWindow()
    window.show()
    
    exit_code = 0
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        # Ensure thread cleanup runs even on crash/interrupt
        if hasattr(window, 'view_model'):
            window.view_model.cleanup()
            
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
