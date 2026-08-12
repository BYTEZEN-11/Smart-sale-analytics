"""
Unit tests for centralized logging module.
"""
import logging
from pathlib import Path
import tempfile
import sys
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import setup_logger, log_dataframe_info, log_operation, log_execution_time

class TestLogger:
    """Test suite for logger functions."""

    def test_setup_logger_custom_file(self):
        """Test configuring a logger with a custom temporary file."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            temp_log = Path(f.name)

        try:
            logger = setup_logger("test_custom_logger", level="DEBUG", log_file=temp_log)
            assert logger.level == logging.DEBUG
            logger.info("Test custom log message")

            for h in list(logger.handlers):
                h.flush()

            content = temp_log.read_text(encoding='utf-8')
            assert "Test custom log message" in content
        finally:
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)
            if temp_log.exists():
                try:
                    temp_log.unlink()
                except OSError:
                    pass

    def test_log_dataframe_info(self):
        """Test logging dataframe schema and memory info."""
        test_logger = logging.getLogger("test_df_logger")
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        log_dataframe_info(test_logger, df, name="TestDF")
        log_dataframe_info(test_logger, None, name="NoneDF")

    def test_log_operation(self):
        """Test logging an operational event."""
        test_logger = logging.getLogger("test_op_logger")
        log_operation(test_logger, "DataIngest", rows=1000, status="Success")

    def test_log_execution_time(self):
        """Test operation execution timer context manager."""
        test_logger = logging.getLogger("test_timer_logger")
        with log_execution_time(test_logger, "SyntheticBenchmark"):
            total = sum(i for i in range(1000))
            assert total > 0
