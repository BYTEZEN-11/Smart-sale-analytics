"""
Unit tests for analytics module: RFM segmentation, forecasting with confidence intervals,
Pareto analysis, cohort retention, and hypothesis testing (t-test, ANOVA, Chi-Square).
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics import (
    rfm_segmentation,
    forecast_sales,
    calculate_kpis,
    get_top_performers,
    perform_two_sample_ttest,
    perform_one_way_anova,
    perform_chi_square_test,
    calculate_cohort_retention,
    calculate_pareto_analysis,
    AnalyticsError
)

class TestAnalytics:
    """Test suite for analytics functions."""

    @pytest.fixture
    def sample_sales_data(self):
        """Create sample sales data for testing."""
        np.random.seed(42)
        n = 150
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
        return pd.DataFrame({
            'Order ID': [f'ORD-{i:04d}' for i in range(n)],
            'Order Date': dates,
            'Customer': [f'Customer {i % 20:03d}' for i in range(n)],
            'Product': [f'Product {i % 10}' for i in range(n)],
            'Category': ['Electronics'] * 50 + ['Furniture'] * 50 + ['Office Supplies'] * 50,
            'Region': ['West'] * 60 + ['East'] * 50 + ['Central'] * 40,
            'Payment Mode': ['Credit Card'] * 70 + ['UPI'] * 50 + ['Cash'] * 30,
            'Sales': np.random.uniform(100, 1000, n),
            'Cost': np.random.uniform(50, 700, n),
            'Profit': np.random.uniform(20, 200, n),
            'Quantity': np.random.randint(1, 5, n)
        })

    def test_rfm_segmentation_basic(self, sample_sales_data):
        """Test basic RFM segmentation and CLV estimate columns."""
        rfm = rfm_segmentation(sample_sales_data)

        assert not rfm.empty
        assert 'Customer' in rfm.columns
        assert 'Recency' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Monetary' in rfm.columns
        assert 'R_Score' in rfm.columns
        assert 'F_Score' in rfm.columns
        assert 'M_Score' in rfm.columns
        assert 'RFM_Score' in rfm.columns
        assert 'Customer Segment' in rfm.columns
        assert 'Estimated_CLV' in rfm.columns

    def test_rfm_segmentation_customer_count(self, sample_sales_data):
        """Test that RFM returns one row per customer."""
        rfm = rfm_segmentation(sample_sales_data)
        unique_customers = sample_sales_data['Customer'].nunique()
        assert len(rfm) == unique_customers

    def test_rfm_segmentation_segment_values(self, sample_sales_data):
        """Test that customer segments are valid."""
        rfm = rfm_segmentation(sample_sales_data)
        valid_segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Lost Customers']
        assert all(rfm['Customer Segment'].isin(valid_segments))

    def test_rfm_segmentation_empty_dataframe(self):
        """Test RFM with empty DataFrame."""
        rfm = rfm_segmentation(pd.DataFrame())
        assert isinstance(rfm, pd.DataFrame)
        assert rfm.empty

    def test_forecast_sales_basic(self, sample_sales_data):
        """Test sales forecasting with confidence intervals."""
        sample_sales_data['Year-Month'] = sample_sales_data['Order Date'].dt.to_period('M').astype(str)
        forecast = forecast_sales(sample_sales_data, periods=3)

        assert not forecast.empty
        assert 'Year-Month' in forecast.columns
        assert 'Sales' in forecast.columns
        assert 'Lower_CI' in forecast.columns
        assert 'Upper_CI' in forecast.columns
        assert 'Type' in forecast.columns
        assert 'Historical' in forecast['Type'].values
        assert 'Forecast' in forecast['Type'].values

    def test_forecast_sales_periods(self, sample_sales_data):
        """Test that forecast generates correct number of periods."""
        sample_sales_data['Year-Month'] = sample_sales_data['Order Date'].dt.to_period('M').astype(str)
        periods = 6
        forecast = forecast_sales(sample_sales_data, periods=periods)
        forecast_rows = forecast[forecast['Type'] == 'Forecast']
        assert len(forecast_rows) == periods

    def test_forecast_sales_non_negative(self, sample_sales_data):
        """Test that forecasts are non-negative."""
        sample_sales_data['Year-Month'] = sample_sales_data['Order Date'].dt.to_period('M').astype(str)
        forecast = forecast_sales(sample_sales_data, periods=3)
        assert (forecast['Sales'] >= 0).all()
        assert (forecast['Lower_CI'] >= 0).all()

    def test_forecast_sales_empty_dataframe(self):
        """Test forecast with empty DataFrame."""
        forecast = forecast_sales(pd.DataFrame())
        assert forecast.empty

    def test_calculate_kpis_basic(self, sample_sales_data):
        """Test KPI calculation."""
        kpis = calculate_kpis(sample_sales_data)
        assert kpis['revenue'] > 0
        assert kpis['profit'] > 0
        assert kpis['orders'] == len(sample_sales_data)
        assert kpis['customers'] == sample_sales_data['Customer'].nunique()
        assert kpis['profit_margin'] > 0

    def test_calculate_kpis_empty(self):
        """Test KPI calculation on empty DataFrame."""
        kpis = calculate_kpis(pd.DataFrame())
        assert kpis['revenue'] == 0
        assert kpis['orders'] == 0

    def test_get_top_performers(self, sample_sales_data):
        """Test top performers aggregation."""
        top_prod = get_top_performers(sample_sales_data, group_by='Product', metric='Sales', top_n=5)
        assert len(top_prod) == 5
        assert 'Product' in top_prod.columns
        assert 'Sales' in top_prod.columns
        assert top_prod['Sales'].iloc[0] >= top_prod['Sales'].iloc[-1]

    def test_perform_two_sample_ttest(self, sample_sales_data):
        """Test two-sample t-test between two categories."""
        res = perform_two_sample_ttest(
            sample_sales_data,
            group_col='Category',
            val_col='Sales',
            group_a='Electronics',
            group_b='Furniture'
        )
        assert res['status'] == 'success'
        assert 'p_value' in res
        assert 't_statistic' in res
        assert 'is_significant' in res

    def test_perform_one_way_anova(self, sample_sales_data):
        """Test one-way ANOVA across 3 categories."""
        res = perform_one_way_anova(sample_sales_data, group_col='Category', val_col='Sales')
        assert res['status'] == 'success'
        assert 'f_statistic' in res
        assert 'p_value' in res
        assert 'is_significant' in res

    def test_perform_chi_square_test(self, sample_sales_data):
        """Test Chi-Square test of independence."""
        res = perform_chi_square_test(sample_sales_data, cat_col1='Region', cat_col2='Payment Mode')
        assert res['status'] == 'success'
        assert 'chi2_statistic' in res
        assert 'p_value' in res
        assert 'dof' in res

    def test_calculate_cohort_retention(self, sample_sales_data):
        """Test customer acquisition cohort retention calculation."""
        counts, pct = calculate_cohort_retention(sample_sales_data)
        assert isinstance(counts, pd.DataFrame)
        assert isinstance(pct, pd.DataFrame)
        assert not pct.empty
        assert (pct.iloc[:, 0] == 100.0).all()

    def test_calculate_pareto_analysis(self, sample_sales_data):
        """Test Pareto 80/20 analysis for products."""
        pareto = calculate_pareto_analysis(sample_sales_data, dimension='Product', metric='Sales')
        assert not pareto.empty
        assert 'Cumulative_Pct' in pareto.columns
        assert 'Pareto_Segment' in pareto.columns
        assert pareto['Cumulative_Pct'].iloc[-1] == 100.0
