"""
Logging Service for Gomoku Accuracy Tool.
Provides file-based logging for debugging and analysis.
"""
import os
import logging
from datetime import datetime
from typing import Optional, Callable
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    STAGE = "STAGE"  # Analysis flow
    ENGINE = "ENGINE"  # Engine I/O

class LoggingService:
    """
    Centralized logging service with file output.
    Creates separate log files for different categories.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup loggers
        self.system_logger = self._create_logger("system", f"system_{self.session_id}.log")
        self.engine_logger = self._create_logger("engine", f"engine_{self.session_id}.log")
        self.analysis_logger = self._create_logger("analysis", f"analysis_{self.session_id}.log")
        
        # UI callback for real-time display
        self.ui_callback: Optional[Callable[[str, str], None]] = None
        
        self.log_system("INFO", f"Logging session started: {self.session_id}")
        
    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """Create a file logger."""
        logger = logging.getLogger(f"gomoprob.{name}")
        logger.setLevel(logging.DEBUG)
        
        # File handler
        filepath = os.path.join(self.log_dir, filename)
        fh = logging.FileHandler(filepath, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
        return logger
    
    def set_ui_callback(self, callback: Callable[[str, str], None]):
        """Set callback for UI updates. callback(type, message)"""
        self.ui_callback = callback
    
    def log_system(self, level: str, message: str):
        """Log system-level messages."""
        self.system_logger.info(f"[{level}] {message}")
        
    def log_engine(self, direction: str, message: str):
        """
        Log engine I/O.
        direction: 'IN' (to engine), 'OUT' (from engine), 'INFO' (parsed info)
        """
        self.engine_logger.info(f"[{direction}] {message}")
        
        # Also send to UI
        if self.ui_callback:
            if direction == "IN":
                self.ui_callback("CMD", message)
            elif direction == "OUT":
                self.ui_callback("RAW", message)
            else:
                self.ui_callback("INFO", message)
    
    def log_analysis(self, stage: str, message: str):
        """Log analysis flow stages."""
        self.analysis_logger.info(f"[{stage}] {message}")
        
        # Also send to UI
        if self.ui_callback:
            self.ui_callback("STAGE", message)
    
    def log_debug(self, context: str, message: str):
        """Log debug information to system log."""
        self.system_logger.debug(f"[{context}] {message}")
    
    def log_error(self, context: str, message: str):
        """Log errors."""
        self.system_logger.error(f"[{context}] {message}")
        if self.ui_callback:
            self.ui_callback("ERROR", f"[{context}] {message}")
    
    def log_data(self, data_type: str, data: dict):
        """Log structured data for later analysis."""
        import json
        data_str = json.dumps(data, default=str)
        self.analysis_logger.info(f"[DATA:{data_type}] {data_str}")


# Global instance
_logging_service: Optional[LoggingService] = None

def get_logging_service() -> LoggingService:
    global _logging_service
    if _logging_service is None:
        _logging_service = LoggingService()
    return _logging_service

def init_logging_service(log_dir: str = "logs") -> LoggingService:
    global _logging_service
    _logging_service = LoggingService(log_dir)
    return _logging_service
