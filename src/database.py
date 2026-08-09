"""
Database operations for SmartSales Analytics.
Handles SQLite connection pooling, indexing, transaction management,
parameterized queries, and analytical query execution.
"""
from contextlib import contextmanager
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import pandas as pd

from config import DATABASE_PATH, DB_CHECK_SAME_THREAD, DB_TIMEOUT, SQL_QUERIES_PATH

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception raised for SQLite database errors."""
    pass

@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections with performance pragmas and transaction safety.

    Args:
        db_path: Path to SQLite database file (defaults to config.DATABASE_PATH)

    Yields:
        sqlite3.Connection object

    Raises:
        DatabaseError: If connection or transaction fails
    """
    path = db_path or DATABASE_PATH
    conn = None

    try:
        conn = sqlite3.connect(
            str(path),
            timeout=DB_TIMEOUT,
            check_same_thread=DB_CHECK_SAME_THREAD
        )
        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")

        logger.debug(f"Database connection opened: {path}")
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
        logger.error(f"Database transaction error: {e}")
        raise DatabaseError(f"Database operation failed: {e}")
    finally:
        if conn:
            conn.close()
            logger.debug(f"Database connection closed: {path}")

def create_database(
    df: pd.DataFrame,
    db_path: Optional[Path] = None,
    table_name: str = 'sales'
) -> bool:
    """
    Create SQLite database and load cleaned data with high-coverage indexes.

    Args:
        df: Cleaned sales DataFrame
        db_path: Path to target SQLite database file
        table_name: Name of target table

    Returns:
        True if database successfully initialized, False if DataFrame empty

    Raises:
        DatabaseError: If creation or indexing fails
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided, skipping database creation")
        return False

    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with get_connection(path) as conn:
            logger.info(f"Loading {len(df):,} records into '{table_name}' table")

            load_df = df.copy()
            for col in load_df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                load_df[col] = load_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

            load_df.to_sql(table_name, conn, if_exists='replace', index=False, chunksize=5000)

            index_definitions = [
                ('idx_order_date', ['Order Date']),
                ('idx_customer', ['Customer']),
                ('idx_product', ['Product']),
                ('idx_category', ['Category']),
                ('idx_region', ['Region']),
                ('idx_year_month', ['Year-Month']),
                ('idx_cat_reg', ['Category', 'Region']),
                ('idx_state', ['State'])
            ]

            for idx_name, cols in index_definitions:
                if all(c in load_df.columns for c in cols):
                    col_str = ', '.join([f'[{c}]' for c in cols])
                    conn.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({col_str})')
                    logger.debug(f"Created index {idx_name} on ({col_str})")

        logger.info(f"Database initialized and indexed successfully at {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise DatabaseError(f"Database creation failed: {e}")

