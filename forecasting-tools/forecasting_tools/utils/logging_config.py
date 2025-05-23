"""
Logging Configuration

This module provides standardized logging configuration for the forecasting tools system,
with appropriate formatters, handlers, and filters.
"""

import os
import sys
import logging
import logging.config
import json
import socket
import platform
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import time
from datetime import datetime
import traceback
from pathlib import Path
import yaml


class LogLevel(str, Enum):
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""
    TEXT = "text"
    JSON = "json"


class SensitiveDataFilter(logging.Filter):
    """
    Filter to remove sensitive data from log records.
    
    This filter replaces sensitive data patterns with redacted placeholders.
    """
    
    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        """
        Initialize the filter with patterns to redact.
        
        Args:
            patterns: Dictionary mapping regex patterns to replacement strings
        """
        super().__init__()
        import re
        
        # Default patterns
        self.patterns = {
            r'api_key\s*=\s*["\']([^"\']+)["\']': r'api_key="***REDACTED***"',
            r'password\s*=\s*["\']([^"\']+)["\']': r'password="***REDACTED***"',
            r'token\s*=\s*["\']([^"\']+)["\']': r'token="***REDACTED***"',
            r'secret\s*=\s*["\']([^"\']+)["\']': r'secret="***REDACTED***"',
            r'key\s*=\s*["\']([^"\']{20,})["\']': r'key="***REDACTED***"',
        }
        
        # Update with custom patterns
        if patterns:
            self.patterns.update(patterns)
        
        # Compile regex patterns
        self.compiled_patterns = [(re.compile(pattern), repl) for pattern, repl in self.patterns.items()]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records to redact sensitive data.
        
        Args:
            record: LogRecord to filter
            
        Returns:
            True to include the record, False to exclude it
        """
        if record.msg and isinstance(record.msg, str):
            for pattern, repl in self.compiled_patterns:
                record.msg = pattern.sub(repl, record.msg)
        
        # Also check args if they're strings
        if record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    for pattern, repl in self.compiled_patterns:
                        args[i] = pattern.sub(repl, arg)
            record.args = tuple(args)
        
        return True


class ContextFilter(logging.Filter):
    """
    Filter to add context information to log records.
    
    This filter adds system information, request IDs, and other context.
    """
    
    def __init__(
        self, 
        app_name: str = "forecasting-tools", 
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the filter with context information.
        
        Args:
            app_name: Name of the application
            context: Additional context to include in all logs
        """
        super().__init__()
        self.app_name = app_name
        self.hostname = socket.gethostname()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter to add context to log records.
        
        Args:
            record: LogRecord to enhance
            
        Returns:
            True to include the record, False to exclude it
        """
        # Add basic context
        record.hostname = self.hostname
        record.app_name = self.app_name
        record.process_id = os.getpid()
        
        # Add timestamp in ISO format
        record.timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Add Python version
        record.python_version = platform.python_version()
        
        # Add custom context
        for key, value in self.context.items():
            setattr(record, key, value)
        
        # Add request_id if available in thread-local storage
        try:
            import threading
            if hasattr(threading.current_thread(), "request_id"):
                record.request_id = threading.current_thread().request_id
        except Exception:
            pass
        
        return True


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for log records.
    
    Formats log records as JSON for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record: LogRecord to format
            
        Returns:
            JSON string representation of the log
        """
        log_data = {
            "timestamp": getattr(record, "timestamp", datetime.utcnow().isoformat() + "Z"),
            "level": record.levelname,
            "name": record.name,
            "message": super().format(record),
            "logger": record.name,
            "path": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
            "process_id": record.process,
            "thread_id": record.thread,
            "hostname": getattr(record, "hostname", socket.gethostname()),
            "app_name": getattr(record, "app_name", "forecasting-tools")
        }
        
        # Add request_id if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add exception information if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add custom extra fields
        for key, value in vars(record).items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName", "timestamp",
                "hostname", "app_name", "process_id", "python_version"
            } and not key.startswith("_"):
                log_data[key] = value
        
        return json.dumps(log_data)


def setup_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_format: Union[str, LogFormat] = LogFormat.TEXT,
    log_file: Optional[str] = None,
    app_name: str = "forecasting-tools",
    context: Optional[Dict[str, Any]] = None,
    config_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level
        log_format: Log format (text or JSON)
        log_file: Optional file to log to
        app_name: Application name
        context: Additional context for logs
        config_file: Optional YAML config file for more complex configurations
    """
    # Convert level to string if it's an enum
    if isinstance(level, LogLevel):
        level = level.value
    
    # Convert format to string if it's an enum
    if isinstance(log_format, LogFormat):
        log_format = log_format.value
    
    # Load from config file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
        return
    
    # Create context filter
    context_filter = ContextFilter(app_name=app_name, context=context)
    sensitive_filter = SensitiveDataFilter()
    
    # Configure handlers based on parameters
    handlers: List[logging.Handler] = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if log_format == LogFormat.JSON.value:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        ))
    
    console_handler.addFilter(context_filter)
    console_handler.addFilter(sensitive_filter)
    handlers.append(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        
        if log_format == LogFormat.JSON.value:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
            ))
        
        file_handler.addFilter(context_filter)
        file_handler.addFilter(sensitive_filter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=handlers,
        force=True  # Reset existing handlers
    )
    
    # Set log levels for some noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # Log startup message
    logging.getLogger("forecasting_tools").info(
        f"Logging initialized: level={level}, format={log_format}, "
        f"file={'yes' if log_file else 'no'}"
    )


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    
    Returns:
        Unique request ID string
    """
    import uuid
    return str(uuid.uuid4())


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set a request ID for the current thread.
    
    Args:
        request_id: Request ID to set, or None to generate a new one
        
    Returns:
        The request ID that was set
    """
    import threading
    if request_id is None:
        request_id = generate_request_id()
    
    thread = threading.current_thread()
    thread.request_id = request_id
    return request_id


def get_request_id() -> Optional[str]:
    """
    Get the request ID for the current thread.
    
    Returns:
        Request ID if set, None otherwise
    """
    import threading
    thread = threading.current_thread()
    return getattr(thread, "request_id", None)


def request_context(request_id: Optional[str] = None):
    """
    Context manager for setting a request ID.
    
    Args:
        request_id: Request ID to set, or None to generate a new one
        
    Example:
        ```
        with request_context():
            # All logs in this context will have the same request_id
            logger.info("Processing request")
        ```
    """
    import contextlib
    
    @contextlib.contextmanager
    def _context_manager():
        # Save the existing request ID
        old_request_id = get_request_id()
        
        # Set the new request ID
        new_request_id = set_request_id(request_id)
        
        try:
            yield new_request_id
        finally:
            # Restore the old request ID
            import threading
            thread = threading.current_thread()
            if old_request_id is None:
                if hasattr(thread, "request_id"):
                    delattr(thread, "request_id")
            else:
                thread.request_id = old_request_id
    
    return _context_manager()


def log_execution_time(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger: Logger to use, or None to use the logger named after the module
        level: Logging level for the timing message
        
    Example:
        ```
        @log_execution_time()
        def slow_operation():
            # Function code
        ```
    """
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logger.log(
                level, 
                f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute"
            )
            return result
        return wrapper
    return decorator


def load_logging_config_from_file(config_file: str) -> None:
    """
    Load logging configuration from a YAML file.
    
    Args:
        config_file: Path to YAML config file
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Logging config file not found: {config_file}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    logging.config.dictConfig(config)
    
    logging.getLogger("forecasting_tools").info(
        f"Logging configuration loaded from {config_file}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger configured with the standard filters.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Add filters if not already present
    filter_names = [f.name for f in logger.filters]
    
    if "context_filter" not in filter_names:
        context_filter = ContextFilter()
        context_filter.name = "context_filter"
        logger.addFilter(context_filter)
    
    if "sensitive_filter" not in filter_names:
        sensitive_filter = SensitiveDataFilter()
        sensitive_filter.name = "sensitive_filter"
        logger.addFilter(sensitive_filter)
    
    return logger 