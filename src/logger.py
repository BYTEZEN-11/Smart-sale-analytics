"""
Centralized logging configuration for SmartSales Analytics.
Provides structured logging, rotating file handlers, and context logging.
"""
import logging
from logging.handlers import RotatingFileHandler
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from config import (
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT
)

def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up a logger with console and rotating file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    target_file = log_file or LOG_FILE
    if target_file:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(target_file),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_dataframe_info(logger: logging.Logger, df: Any, name: str = "DataFrame") -> None:
    """
    Log DataFrame shape, memory usage, and column schema for diagnostics.

    Args:
        logger: Logger instance
        df: Pandas DataFrame
        name: Name of the DataFrame
    """
    if df is None:
        logger.warning(f"{name} is None")
        return

    try:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        logger.info(f"{name} - Shape: {df.shape} | Memory: {mem_mb:.2f} MB")
        logger.debug(f"{name} - Columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to log DataFrame info for {name}: {e}")

def log_operation(logger: logging.Logger, operation: str, **kwargs: Any) -> None:
    """
    Log an operation with contextual parameters.

    Args:
        logger: Logger instance
        operation: Operation name
        **kwargs: Operation parameters
    """
    params = ', '.join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"Operation: {operation} | Parameters: {params}")

@contextmanager
def log_execution_time(logger: logging.Logger, operation_name: str) -> Generator[None, None, None]:
    """
    Context manager to log elapsed execution time for an operation.

    Args:
        logger: Logger instance
        operation_name: Human readable operation description
    """
    start_time = time.perf_counter()
    logger.info(f"Starting {operation_name}...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"Completed {operation_name} in {elapsed:.3f}s")

default_logger = setup_logger('smartsales')
