"""
State definitions for the Gomoku Analysis Tool.
"""
from enum import Enum, auto

class SystemState(Enum):
    """Overall application state."""
    IDLE = auto()       # Application ready, waiting for user input
    RUNNING = auto()    # Analysis loop active
    STOPPING = auto()   # Transitioning to stop

class EngineState(Enum):
    """Engine process state."""
    OFF = auto()        # Process not started
    STARTING = auto()   # Process starting / Handshaking
    IDLE = auto()       # Process running, ready for commands
    ANALYZING = auto()  # Busy executing a search command
    STOPPING = auto()   # Sending QUIT/STOP
    ERROR = auto()      # Process crashed or unresponsive
