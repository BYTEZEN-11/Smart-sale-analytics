"""
Unit tests for data cleaning module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import clean_data, validate_required_columns, get_data_quality_report, DataCleaningError

class TestDataCleaning:
    """Test suite for data cleaning functions."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw sales data for testing."""
        return pd.DataFrame({
            'Order ID': ['ORD-001', 'ORD-002', 'ORD-002', 'ORD-003'],
            'Order Date': ['2023-01-01', '2023-01-02', '2023-01-02', 'invalid'],
            'Customer': ['Customer A', 'Customer B', None, 'Customer C'],
            'Product': ['Product X', 'Product Y', 'Product Z', 'Product W'],
            'Category': ['Electronics', 'Furniture', 'electronics', 'Office Supplies'],
            'Sub-Category': ['Phones', 'Chairs', 'Laptops', 'Paper'],
            'Quantity': [2, 3, 1, 5],
            'Sales': [1000, 500, 1500, 100],
            'Discount': [0.1, 0.2, None, 0.3],
            'Cost': [700, 300, 1000, 50],
            'Profit': [300, 200, 500, 50],
            'Region': ['West', 'East', 'West', 'Central'],
            'State': ['California', 'New York', 'California', 'Texas'],
            'City': ['LA', None, 'SF', 'Dallas'],
            'Payment Mode': ['Credit Card', 'Cash', 'UPI', 'Debit Card'],
            'Shipping Mode': ['Standard Class', 'First Class', 'Second Class', 'Same Day']
        })

    def test_clean_data_removes_duplicates(self, sample_raw_data):
        """Test that duplicate Order IDs are removed."""
        cleaned = clean_data(sample_raw_data)
        assert cleaned['Order ID'].nunique() == len(cleaned)
        assert len(cleaned) == 2

    def test_clean_data_standardizes_text(self, sample_raw_data):
        """Test that text columns are standardized."""
        cleaned = clean_data(sample_raw_data)
        assert all(cleaned['Category'].str.istitle())

    def test_clean_data_imputes_missing_customer(self, sample_raw_data):
        """Test that missing customer values are imputed."""
        cleaned = clean_data(sample_raw_data)
        assert cleaned['Customer'].notna().all()

    def test_clean_data_imputes_missing_city(self, sample_raw_data):
        """Test that missing city values are imputed."""
        cleaned = clean_data(sample_raw_data)
        assert cleaned['City'].notna().all()

    def test_clean_data_imputes_missing_discount(self, sample_raw_data):
        """Test that missing discount values are filled with 0."""
        cleaned = clean_data(sample_raw_data)
        assert cleaned['Discount'].notna().all()

    def test_clean_data_removes_invalid_dates(self, sample_raw_data):
        """Test that rows with invalid dates are removed."""
        cleaned = clean_data(sample_raw_data)
        assert cleaned['Order Date'].notna().all()

    def test_clean_data_creates_derived_fields(self, sample_raw_data):
        """Test that derived fields are created."""
        cleaned = clean_data(sample_raw_data)
        required_fields = ['Year', 'Month', 'Quarter', 'Year-Month', 'Profit Margin', 'Order Value']
        for field in required_fields:
            assert field in cleaned.columns

    def test_clean_data_recalculates_profit(self, sample_raw_data):
        """Test that profit is recalculated correctly."""
        cleaned = clean_data(sample_raw_data)
        expected_profit = cleaned['Sales'] - cleaned['Cost']
        pd.testing.assert_series_equal(cleaned['Profit'], expected_profit, check_names=False)

    def test_clean_data_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        with pytest.raises(DataCleaningError):
            clean_data(pd.DataFrame())

    def test_clean_data_none_input(self):
        """Test handling of None input."""
        with pytest.raises(DataCleaningError):
            clean_data(None)

    def test_validate_required_columns_success(self, sample_raw_data):
        """Test that validation passes with all required columns."""
        required = ['Order ID', 'Customer', 'Sales']
        assert validate_required_columns(sample_raw_data, required)

    def test_validate_required_columns_failure(self, sample_raw_data):
        """Test that validation fails with missing columns."""
        required = ['Order ID', 'NonExistentColumn']
        with pytest.raises(DataCleaningError):
            validate_required_columns(sample_raw_data, required)

    def test_get_data_quality_report(self, sample_raw_data):
        """Test data quality report generation."""
        report = get_data_quality_report(sample_raw_data)
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert 'duplicate_rows' in report
        assert report['total_rows'] == len(sample_raw_data)

    def test_get_data_quality_report_empty(self):
        """Test data quality report with empty DataFrame."""
        report = get_data_quality_report(pd.DataFrame())
        assert report == {}

    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection."""
        from src.data_cleaning import detect_outliers_zscore
        data = pd.Series([10, 12, 11, 10, 13, 1000])
        outliers = detect_outliers_zscore(data, threshold=2.0)
        assert outliers.iloc[-1] == True
        assert outliers.iloc[0] == False

    def test_normalize_numeric_features(self):
        """Test min-max and standard feature normalization."""
        from src.data_cleaning import normalize_numeric_features
        df = pd.DataFrame({'Sales': [100, 200, 300]})
        norm_minmax = normalize_numeric_features(df, columns=['Sales'], method='minmax')
        assert 'Sales_scaled' in norm_minmax.columns
        assert norm_minmax['Sales_scaled'].min() == 0.0
        assert norm_minmax['Sales_scaled'].max() == 1.0

        norm_std = normalize_numeric_features(df, columns=['Sales'], method='standard')
        assert 'Sales_scaled' in norm_std.columns
        assert round(norm_std['Sales_scaled'].mean(), 2) == 0.0

    def test_encode_categorical_features(self):
        """Test frequency and one-hot categorical encoding."""
        from src.data_cleaning import encode_categorical_features
        df = pd.DataFrame({'Category': ['Electronics', 'Furniture', 'Electronics']})
        freq_enc = encode_categorical_features(df, columns=['Category'], method='frequency')
        assert 'Category_freq' in freq_enc.columns
        assert freq_enc['Category_freq'].iloc[0] == pytest.approx(2/3, 0.01)

        onehot_enc = encode_categorical_features(df, columns=['Category'], method='onehot')
        assert any('Category_' in c for c in onehot_enc.columns)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
