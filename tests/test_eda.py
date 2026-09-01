"""
Unit tests for EDA (Exploratory Data Analysis) module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eda import (
    perform_eda,
    generate_dataset_overview,
    compute_statistical_summary,
    compute_correlation_matrices,
    compute_outlier_summary,
    plot_eda_correlation_heatmap,
    plot_univariate_distribution
)

class TestEDA:
    """Test suite for EDA functions."""

    @pytest.fixture
    def sample_eda_df(self):
        """Create sample dataset for testing EDA routines."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        return pd.DataFrame({
            'Order ID': [f'ORD-{i:04d}' for i in range(n)],
            'Order Date': dates,
            'Customer': [f'Customer {i % 15}' for i in range(n)],
            'Product': [f'Product {i % 5}' for i in range(n)],
            'Category': ['Electronics'] * 50 + ['Furniture'] * 50,
            'Sales': np.random.uniform(50, 1000, n),
            'Cost': np.random.uniform(30, 700, n),
            'Profit': np.random.uniform(10, 300, n),
            'Profit Margin': np.random.uniform(5, 40, n),
            'Region': ['West'] * 40 + ['East'] * 30 + ['Central'] * 30
        })

    def test_generate_dataset_overview(self, sample_eda_df):
        """Test dataset overview dictionary structure."""
        overview = generate_dataset_overview(sample_eda_df)
        assert overview['rows'] == 100
        assert overview['columns'] == 10
        assert overview['missing_cells'] == 0
        assert 'memory_mb' in overview

    def test_compute_statistical_summary(self, sample_eda_df):
        """Test statistical summary calculation."""
        stats = compute_statistical_summary(sample_eda_df)
        assert not stats.empty
        assert 'Sales' in stats.index
        assert 'Mean' in stats.columns
        assert 'Skewness' in stats.columns
        assert 'Kurtosis' in stats.columns
        assert 'IQR' in stats.columns

    def test_compute_correlation_matrices(self, sample_eda_df):
        """Test Pearson and Spearman correlation computation."""
        corrs = compute_correlation_matrices(sample_eda_df)
        assert 'pearson' in corrs
        assert 'spearman' in corrs
        assert isinstance(corrs['pearson'], pd.DataFrame)
        assert 'Sales' in corrs['pearson'].columns

    def test_compute_outlier_summary(self, sample_eda_df):
        """Test outlier summary calculation."""
        outliers = compute_outlier_summary(sample_eda_df)
        assert not outliers.empty
        assert 'Column' in outliers.columns
        assert 'Outlier Count' in outliers.columns
        assert 'Lower Fence' in outliers.columns

    def test_perform_eda_comprehensive(self, sample_eda_df):
        """Test main perform_eda function returns all necessary metrics."""
        insights = perform_eda(sample_eda_df)
        assert insights['total_sales'] > 0
        assert insights['total_profit'] > 0
        assert insights['total_orders'] == 100
        assert insights['total_customers'] == 15
        assert 'top_categories' in insights
        assert 'top_products' in insights
        assert 'statistical_summary' in insights
        assert 'correlations' in insights

    def test_empty_dataframe_handling(self):
        """Test EDA functions gracefully handle empty DataFrames."""
        empty_df = pd.DataFrame()
        assert generate_dataset_overview(empty_df)['rows'] == 0
        assert compute_statistical_summary(empty_df).empty
        assert compute_outlier_summary(empty_df).empty
        assert perform_eda(empty_df)['status'] == 'empty_dataset'

    def test_plot_functions_return_figures(self, sample_eda_df):
        """Test that plotting helper functions return valid figures."""
        heatmap_fig = plot_eda_correlation_heatmap(sample_eda_df)
        assert heatmap_fig is not None
        dist_fig = plot_univariate_distribution(sample_eda_df, 'Sales')
        assert dist_fig is not None
