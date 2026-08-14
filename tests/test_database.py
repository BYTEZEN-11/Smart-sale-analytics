"""
Unit tests for database module.
"""
import pytest
import pandas as pd
import sqlite3
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import (
    create_database,
    query_database,
    execute_sql,
    table_exists,
    get_table_info,
    DatabaseError
)

class TestDatabase:
    """Test suite for database functions."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'Order ID': ['ORD-001', 'ORD-002', 'ORD-003'],
            'Order Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'Customer': ['Customer A', 'Customer B', 'Customer C'],
            'Product': ['Product X', 'Product Y', 'Product Z'],
            'Category': ['Electronics', 'Furniture', 'Office Supplies'],
            'Region': ['West', 'East', 'Central'],
            'Sales': [1000, 500, 750],
            'Profit': [300, 150, 200]
        })

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    def test_create_database_success(self, sample_dataframe, temp_db_path):
        """Test successful database creation."""
        result = create_database(sample_dataframe, temp_db_path)

        assert result is True
        assert temp_db_path.exists()

    def test_create_database_table_created(self, sample_dataframe, temp_db_path):
        """Test that table is created in database."""
        create_database(sample_dataframe, temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sales'")
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == 'sales'

    def test_create_database_data_inserted(self, sample_dataframe, temp_db_path):
        """Test that data is inserted correctly."""
        create_database(sample_dataframe, temp_db_path)

        result = query_database("SELECT COUNT(*) as count FROM sales", temp_db_path)

        assert result.loc[0, 'count'] == len(sample_dataframe)

    def test_create_database_empty_dataframe(self, temp_db_path):
        """Test creating database with empty DataFrame."""
        result = create_database(pd.DataFrame(), temp_db_path)
        assert result is False

    def test_query_database_success(self, sample_dataframe, temp_db_path):
        """Test successful query execution."""
        create_database(sample_dataframe, temp_db_path)

        result = query_database("SELECT * FROM sales WHERE Category='Electronics'", temp_db_path)

        assert not result.empty
        assert len(result) == 1
        assert result.loc[0, 'Category'] == 'Electronics'

    def test_query_database_with_params(self, sample_dataframe, temp_db_path):
        """Test query with parameters."""
        create_database(sample_dataframe, temp_db_path)

        result = query_database(
            "SELECT * FROM sales WHERE Sales > ?",
            temp_db_path,
            params=(600,)
        )

        assert not result.empty
        assert len(result) == 2

    def test_query_database_nonexistent_db(self, temp_db_path):
        """Test querying non-existent database."""
        with pytest.raises(DatabaseError):
            query_database("SELECT * FROM sales", temp_db_path)

    def test_table_exists_true(self, sample_dataframe, temp_db_path):
        """Test table_exists returns True for existing table."""
        create_database(sample_dataframe, temp_db_path)

        assert table_exists('sales', temp_db_path) is True

    def test_table_exists_false(self, sample_dataframe, temp_db_path):
        """Test table_exists returns False for non-existent table."""
        create_database(sample_dataframe, temp_db_path)

        assert table_exists('nonexistent_table', temp_db_path) is False

    def test_get_table_info(self, sample_dataframe, temp_db_path):
        """Test getting table information."""
        create_database(sample_dataframe, temp_db_path)

        info = get_table_info('sales', temp_db_path)

        assert not info.empty
        assert len(info) == len(sample_dataframe.columns)

    def test_execute_sql(self, sample_dataframe, temp_db_path):
        """Test executing DDL/DML statement."""
        create_database(sample_dataframe, temp_db_path)
        rows_affected = execute_sql("DELETE FROM sales WHERE Category='Electronics'", temp_db_path)
        assert rows_affected == 1
        remaining = query_database("SELECT COUNT(*) as cnt FROM sales", temp_db_path)
        assert remaining.loc[0, 'cnt'] == len(sample_dataframe) - 1

    def test_load_and_run_named_query(self, sample_dataframe, temp_db_path):
        """Test loading named queries from sql_queries.sql and executing them."""
        from src.database import load_sql_queries, run_named_query
        queries = load_sql_queries()
        assert len(queries) > 0

        create_database(sample_dataframe, temp_db_path)
        res = run_named_query(1, temp_db_path)
        assert not res.empty
        assert 'total_orders' in res.columns

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
