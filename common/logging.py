"""Structured logging configuration using structlog."""

import logging
from pathlib import Path

import structlog
from structlog.contextvars import merge_contextvars


def setup_logging(level: str = "INFO", log_file: str = "logs/trading_system.log") -> None:
    """Configure structlog with JSON output and file logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file.
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the log file for writing
    log_file_obj = open(log_file, "a")

    # Get the numeric log level
    log_level = getattr(logging, level.upper())

    # Configure structlog
    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=log_file_obj),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: The logger name, typically the module name (__name__).

    Returns:
        A bound logger instance from structlog.
    """
    return structlog.get_logger(name)
