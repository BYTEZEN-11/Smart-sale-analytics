"""
Unit tests for UI components: data filtering, export routines, Excel generation, and insight generator.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui_components import (
    apply_filters,
    export_dataframe_to_excel,
    render_insights
)

class TestUIComponents:
    """Test suite for UI components logic and export generation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample sales data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        return pd.DataFrame({
            'Order ID': [f'ORD-{i:03d}' for i in range(50)],
            'Order Date': dates,
            'Customer': [f'Customer {i % 5}' for i in range(50)],
            'Product': [f'Product {i % 4}' for i in range(50)],
            'Category': ['Electronics'] * 25 + ['Furniture'] * 25,
            'Region': ['West'] * 30 + ['East'] * 20,
            'Sales': [100.0 + i * 10 for i in range(50)],
            'Cost': [50.0 + i * 5 for i in range(50)],
            'Profit': [50.0 + i * 5 for i in range(50)],
            'Discount': [0.1 if i % 2 == 0 else 0.3 for i in range(50)],
            'Payment Mode': ['Credit Card'] * 30 + ['UPI'] * 20,
            'Shipping Mode': ['Standard Class'] * 50
        })

    def test_apply_filters_date_range(self, sample_data):
        """Test date range filtering."""
        filters = {
            'date_range': (pd.to_datetime('2023-01-05').date(), pd.to_datetime('2023-01-15').date())
        }
        filtered = apply_filters(sample_data, filters)
        assert len(filtered) == 11
        assert filtered['Order Date'].min().date() >= pd.to_datetime('2023-01-05').date()
        assert filtered['Order Date'].max().date() <= pd.to_datetime('2023-01-15').date()

    def test_apply_filters_categories(self, sample_data):
        """Test categorical filtering."""
        filters = {
            'Category': ['Electronics']
        }
        filtered = apply_filters(sample_data, filters)
        assert len(filtered) == 25
        assert (filtered['Category'] == 'Electronics').all()

    def test_export_dataframe_to_excel(self, sample_data):
        """Test Excel byte buffer generation with multi-sheet formatting."""
        excel_bytes = export_dataframe_to_excel(sample_data)
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 1000

        import io
        excel_sheets = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
        assert 'Sales Data' in excel_sheets
        assert 'Executive Summary' in excel_sheets
        assert len(excel_sheets['Sales Data']) == 50

    def test_render_insights_generates_bullets(self, sample_data):
        """Test insight generation extracts key metrics."""
        insights = render_insights(sample_data)
        assert isinstance(insights, list)
        assert len(insights) >= 3
        assert any('insight-box' in s for s in insights)

    def test_render_insights_empty_dataframe(self):
        """Test insights gracefully handle empty dataframe."""
        insights = render_insights(pd.DataFrame())
        assert len(insights) == 1
        assert 'No transaction records' in insights[0]
