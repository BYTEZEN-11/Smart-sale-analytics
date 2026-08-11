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

def query_database(
    query: str,
    db_path: Optional[Path] = None,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Execute a parameterized SQL query and return results as a Pandas DataFrame.

    Args:
        query: SQL query string with parameter placeholders (? or :name)
        db_path: Path to database
        params: Tuple or Dict of parameters

    Returns:
        DataFrame containing query result rows

    Raises:
        DatabaseError: If database is missing or query execution fails
    """
    path = db_path or DATABASE_PATH
    if not path.exists():
        logger.error(f"Database not found: {path}")
        raise DatabaseError(f"Database does not exist: {path}")

    try:
        with get_connection(path) as conn:
            logger.debug(f"Executing SQL query: {query.strip()[:100]}...")
            if params is not None:
                result = pd.read_sql_query(query, conn, params=params)
            else:
                result = pd.read_sql_query(query, conn)
            logger.info(f"Query returned {len(result):,} rows")
            return result
    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        raise DatabaseError(f"Query failed: {e}")

def execute_sql(
    query: str,
    db_path: Optional[Path] = None,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
) -> int:
    """
    Execute a DDL or DML statement (INSERT, UPDATE, DELETE).

    Args:
        query: SQL statement string
        db_path: Path to database
        params: Parameters for statement

    Returns:
        Count of affected rows

    Raises:
        DatabaseError: If execution fails
    """
    path = db_path or DATABASE_PATH
    try:
        with get_connection(path) as conn:
            cursor = conn.cursor()
            if params is not None:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            affected = cursor.rowcount
            logger.info(f"SQL statement executed. Rows affected: {affected}")
            return affected
    except Exception as e:
        logger.error(f"SQL execution error: {e}", exc_info=True)
        raise DatabaseError(f"Execution failed: {e}")

def table_exists(table_name: str, db_path: Optional[Path] = None) -> bool:
    """
    Check if a specific table exists in the SQLite catalog.

    Args:
        table_name: Name of table
        db_path: Path to database

    Returns:
        True if table exists, False otherwise
    """
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    try:
        result = query_database(query, db_path, params=(table_name,))
        return not result.empty
    except DatabaseError:
        return False

def get_table_info(table_name: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Retrieve PRAGMA metadata for a table (columns, types, nullability, primary key).

    Args:
        table_name: Name of table
        db_path: Path to database

    Returns:
        DataFrame with schema metadata
    """
    return query_database(f"PRAGMA table_info({table_name})", db_path)

def load_sql_queries(file_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Parse queries from sql_queries.sql into a dictionary of named queries.

    Args:
        file_path: Path to SQL script file (defaults to SQL_QUERIES_PATH)

    Returns:
        Dictionary mapping query headers to SQL text
    """
    path = file_path or SQL_QUERIES_PATH
    if not path.exists():
        logger.warning(f"SQL file not found at {path}")
        return {}

    content = path.read_text(encoding='utf-8')
    queries = {}

    pattern = r'--\s*(\d+\.?\s*[^\n]+)\n(.*?)(?=(?:--\s*\d+\.|\Z))'
    matches = re.findall(pattern, content, re.DOTALL)

    if matches:
        for title, sql_text in matches:
            clean_title = title.strip()
            clean_sql = sql_text.strip().rstrip(';')
            if clean_sql:
                queries[clean_title] = clean_sql
    else:
        raw_queries = [q.strip() for q in content.split(';') if q.strip()]
        for i, sql_text in enumerate(raw_queries, 1):
            queries[f"{i}. Query_{i}"] = sql_text

    return queries

def run_named_query(query_name_or_number: Union[str, int], db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Execute a query from sql_queries.sql by query name or number.

    Args:
        query_name_or_number: E.g., 1 or '1. Overall KPIs' or 'Overall KPIs'
        db_path: Database path

    Returns:
        DataFrame with query results
    """
    queries = load_sql_queries()
    target_key = None

    if isinstance(query_name_or_number, int):
        prefix = f"{query_name_or_number}."
        for k in queries:
            if k.startswith(prefix):
                target_key = k
                break
    else:
        for k in queries:
            if query_name_or_number.lower() in k.lower():
                target_key = k
                break

    if not target_key:
        raise DatabaseError(f"Named query '{query_name_or_number}' not found in SQL definitions")

    return query_database(queries[target_key], db_path)
