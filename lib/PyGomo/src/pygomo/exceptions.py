"""
PyGomo exceptions.

Custom exception hierarchy for the framework.
"""


class PyGomoError(Exception):
    """Base exception for all PyGomo errors."""
    pass


class EngineError(PyGomoError):
    """Error related to engine communication or execution."""
    pass


class ProtocolError(PyGomoError):
    """Error related to protocol parsing or serialization."""
    pass


class TimeoutError(PyGomoError):
    """Operation timed out."""
    pass


class CommandError(PyGomoError):
    """Error executing a command."""
    pass


class ConnectionError(PyGomoError):
    """Error connecting to engine."""
    pass


class ValidationError(PyGomoError):
    """Invalid argument or state."""
    pass
