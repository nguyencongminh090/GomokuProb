
import sys
import os
import signal

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Importing modules...")
try:
    from PyQt6.QtWidgets import QApplication
    print("PyQt6 imported.")
except ImportError:
    print("PyQt6 not found.")

try:
    from src.services.engine_service import EngineService
    print("EngineService imported.")
except Exception as e:
    print(f"EngineService import failed: {e}")

try:
    from src.ui.viewmodel import MainViewModel
    print("MainViewModel imported.")
except Exception as e:
    print(f"MainViewModel import failed: {e}")

def verify_engine():
    print("Initializing EngineService...")
    engine_path = "/media/ngmint/Data/Programming/Python/Personal/GomoProb/engine/rapfi"
    service = EngineService(engine_path)
    print("EngineService initialized.")
    return service

def verify_ui():
    print("Initializing QApplication...")
    app = QApplication(sys.argv)
    print("QApplication initialized.")

    print("Initializing MainViewModel...")
    vm = MainViewModel()
    
    # FORCE ABSOLUTE PATH
    vm.engine_service.engine_path = "/media/ngmint/Data/Programming/Python/Personal/GomoProb/engine/rapfi"
    print(f"MainViewModel initialized. Engine path set to: {vm.engine_service.engine_path}")

    print("Starting EngineService...")
    try:
        # Simulate what update_config does
        vm.engine_service.start(rule=1)
        print("EngineService started successfully.")
    except Exception as e:
        print(f"EngineService start failed: {e}")

    try:
        from src.ui.mainwindow import MainWindow
        print("Initializing MainWindow...")
        window = MainWindow()
        window.show()
        print("MainWindow initialized and shown.")
    except Exception as e:
        print(f"MainWindow initialization failed: {e}")
    
    return app, vm

if __name__ == "__main__":
    print("Starting reproduction test...")
    app, vm = verify_ui()
    
    # Process some events to let things settle
    print("Processing events...")
    app.processEvents()
    
    print("Test complete. No immediate crash.")
    # Clean exit
    if vm.engine_service:
        vm.engine_service.stop()
    sys.exit(0)
