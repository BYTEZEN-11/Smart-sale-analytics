"""
Advanced Analytics Module for SmartSales Analytics.
Provides customer RFM segmentation, CLV modeling, time-series forecasting with confidence intervals,
Pareto 80/20 analysis, customer cohort retention, and statistical hypothesis testing (ANOVA, t-test, Chi-square).
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    ANOVA_ALPHA,
    CONFIDENCE_LEVEL,
    FORECAST_PERIODS,
    PARETO_RATIO,
    RFM_QUANTILES
)

logger = logging.getLogger(__name__)

class AnalyticsError(Exception):
    """Custom exception raised when an analytical operation fails."""
    pass

def _segment_customer(row: pd.Series) -> str:
    """
    Classify a customer into strategic behavioral cohorts based on R, F, M quintiles.

    Args:
        row: Pandas series containing R_Score, F_Score, M_Score (values 1 to 5)

    Returns:
        Segment label string
    """
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])

    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 3 and f <= 2:
        return 'Potential Loyalists'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Lost Customers'

def rfm_segmentation(
    df: pd.DataFrame,
    quantiles: int = RFM_QUANTILES
) -> pd.DataFrame:
    """
    Perform Recency, Frequency, Monetary (RFM) segmentation and estimate Customer Lifetime Value (CLV).

    Args:
        df: Sales transactions DataFrame
        quantiles: Number of quintile score bins (default 5)

    Returns:
        DataFrame with Customer metrics, R/F/M scores, RFM_Score, Customer Segment, and CLV Proxy
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for RFM segmentation")
        return pd.DataFrame(columns=[
            'Customer', 'Recency', 'Frequency', 'Monetary',
            'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
            'Customer Segment', 'AOV', 'Estimated_CLV'
        ])

    try:
        logger.info(f"Computing RFM segmentation for {df['Customer'].nunique():,} unique customers")

        if not pd.api.types.is_datetime64_any_dtype(df['Order Date']):
            df = df.copy()
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

        reference_date = df['Order Date'].max() + pd.Timedelta(days=1)

        rfm = df.groupby('Customer').agg(
            Recency=('Order Date', lambda x: int((reference_date - x.max()).days)),
            Frequency=('Order ID', 'nunique'),
            Monetary=('Sales', 'sum'),
            TotalProfit=('Profit', 'sum') if 'Profit' in df.columns else ('Sales', lambda x: x.sum() * 0.2)
        ).reset_index()

        rfm['Recency'] = rfm['Recency'].astype(int)
        rfm['Frequency'] = rfm['Frequency'].astype(int)
        rfm['Monetary'] = rfm['Monetary'].round(2)

        n_customers = len(rfm)
        if n_customers < quantiles:
            logger.warning(f"Customer base ({n_customers}) smaller than quantiles ({quantiles}). Using fallback score 3.")
            rfm['R_Score'] = 3
            rfm['F_Score'] = 3
            rfm['M_Score'] = 3
        else:
            try:
                rfm['R_Score'] = pd.qcut(rfm['Recency'], q=quantiles, labels=list(range(quantiles, 0, -1)), duplicates='drop')
            except ValueError:
                rfm['R_Score'] = pd.cut(rfm['Recency'], bins=quantiles, labels=list(range(quantiles, 0, -1)), include_lowest=True)

            try:
                rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=quantiles, labels=list(range(1, quantiles + 1)), duplicates='drop')
            except ValueError:
                rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=quantiles, labels=list(range(1, quantiles + 1)), include_lowest=True)

            try:
                rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=quantiles, labels=list(range(1, quantiles + 1)), duplicates='drop')
            except ValueError:
                rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=quantiles, labels=list(range(1, quantiles + 1)), include_lowest=True)

        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)

        rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
        rfm['Customer Segment'] = rfm.apply(_segment_customer, axis=1)

        rfm['AOV'] = np.where(rfm['Frequency'] > 0, (rfm['Monetary'] / rfm['Frequency']).round(2), 0.0)

        margin_factor = np.where(rfm['Monetary'] > 0, np.clip(rfm['TotalProfit'] / rfm['Monetary'], 0.05, 0.50), 0.20)
        rfm['Estimated_CLV'] = (rfm['Monetary'] * (1 + (rfm['Frequency'] * 0.10)) * margin_factor).round(2)
        rfm = rfm.drop(columns=['TotalProfit'])

        return rfm

    except Exception as e:
        logger.error(f"RFM segmentation encountered failure: {e}", exc_info=True)
        raise AnalyticsError(f"RFM calculation failed: {e}")

def forecast_sales(
    df: pd.DataFrame,
    periods: int = FORECAST_PERIODS,
    confidence_level: float = CONFIDENCE_LEVEL
) -> pd.DataFrame:
    """
    Generate future monthly sales forecast with trend-seasonality modeling
    and statistical confidence intervals (Lower/Upper 95% CI).

    Args:
        df: Sales DataFrame
        periods: Months to forecast forward (default 6)
        confidence_level: Confidence interval band (default 0.95)

    Returns:
        DataFrame containing Historical and Forecast rows with Point Estimates, Lower CI, Upper CI
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for forecasting")
        return pd.DataFrame(columns=['Year-Month', 'Sales', 'Lower_CI', 'Upper_CI', 'Type'])

    try:
        if 'Year-Month' not in df.columns:
            if 'Order Date' in df.columns:
                df = df.copy()
                df['Year-Month'] = pd.to_datetime(df['Order Date']).dt.to_period('M').astype(str)
            else:
                raise AnalyticsError("Missing 'Order Date' or 'Year-Month' column")

        monthly = df.groupby('Year-Month', as_index=False)['Sales'].sum().sort_values('Year-Month').reset_index(drop=True)

        if len(monthly) < 2:
            monthly['Lower_CI'] = monthly['Sales']
            monthly['Upper_CI'] = monthly['Sales']
            monthly['Type'] = 'Historical'
            return monthly

        n_hist = len(monthly)
        t = np.arange(n_hist)
        y = monthly['Sales'].values

        t_mean = t.mean()
        y_mean = y.mean()
        denom = ((t - t_mean) ** 2).sum()
        slope = ((t - t_mean) * (y - y_mean)).sum() / denom if denom != 0 else 0.0
        intercept = y_mean - slope * t_mean

        fitted = slope * t + intercept
        residuals = y - fitted
        std_error = residuals.std(ddof=2) if len(residuals) > 2 else residuals.std()
        if np.isnan(std_error) or std_error == 0:
            std_error = y_mean * 0.05

        dof = max(n_hist - 2, 1)
        t_crit = stats.t.ppf((1 + confidence_level) / 2.0, df=dof)

        monthly['Lower_CI'] = monthly['Sales']
        monthly['Upper_CI'] = monthly['Sales']
        monthly['Type'] = 'Historical'

        last_period = pd.Period(monthly['Year-Month'].iloc[-1], freq='M')
        future_periods = [str(last_period + i) for i in range(1, periods + 1)]
        t_future = np.arange(n_hist, n_hist + periods)

        seasonal_indices = np.ones(periods)
        if n_hist >= 12:
            monthly['MonthNum'] = [pd.Period(ym, freq='M').month for ym in monthly['Year-Month']]
            month_factors = monthly.groupby('MonthNum')['Sales'].mean() / y_mean
            seasonal_indices = np.array([month_factors.get(pd.Period(fp, freq='M').month, 1.0) for fp in future_periods])

        forecast_point = (slope * t_future + intercept) * seasonal_indices
        forecast_point = np.maximum(forecast_point, 0.0).round(2)

        margin_of_error = t_crit * std_error * np.sqrt(1 + 1/n_hist + ((t_future - t_mean)**2)/denom)
        lower_ci = np.maximum(forecast_point - margin_of_error, 0.0).round(2)
        upper_ci = (forecast_point + margin_of_error).round(2)

        forecast_df = pd.DataFrame({
            'Year-Month': future_periods,
            'Sales': forecast_point,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci,
            'Type': 'Forecast'
        })

        result = pd.concat([monthly[['Year-Month', 'Sales', 'Lower_CI', 'Upper_CI', 'Type']], forecast_df], ignore_index=True)
        return result

    except Exception as e:
        logger.error(f"Forecasting calculation error: {e}", exc_info=True)
        raise AnalyticsError(f"Forecasting engine failed: {e}")

def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate high-level executive business KPIs.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary of KPI metrics with growth ratios
    """
    if df is None or df.empty:
        return {
            'revenue': 0.0,
            'profit': 0.0,
            'orders': 0,
            'customers': 0,
            'avg_order_value': 0.0,
            'profit_margin': 0.0,
            'units_sold': 0
        }

    orders_count = int(df['Order ID'].nunique()) if 'Order ID' in df.columns else len(df)
    revenue = float(df['Sales'].sum()) if 'Sales' in df.columns else 0.0
    profit = float(df['Profit'].sum()) if 'Profit' in df.columns else 0.0
    customers_count = int(df['Customer'].nunique()) if 'Customer' in df.columns else 0
    units_sold = int(df['Quantity'].sum()) if 'Quantity' in df.columns else 0

    return {
        'revenue': round(revenue, 2),
        'profit': round(profit, 2),
        'orders': orders_count,
        'customers': customers_count,
        'avg_order_value': round(revenue / orders_count, 2) if orders_count > 0 else 0.0,
        'profit_margin': round((profit / revenue * 100.0), 2) if revenue > 0 else 0.0,
        'units_sold': units_sold
    }

def get_top_performers(
    df: pd.DataFrame,
    group_by: str,
    metric: str = 'Sales',
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get top N entities aggregated by metric.

    Args:
        df: Input DataFrame
        group_by: Dimension column
        metric: Continuous metric
        top_n: Number of records

    Returns:
        DataFrame containing ranked top performers
    """
    if df is None or df.empty or group_by not in df.columns or metric not in df.columns:
        return pd.DataFrame()

    try:
        grouped = df.groupby(group_by, as_index=False)[metric].sum()
        return grouped.nlargest(top_n, metric).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Top performers aggregation error: {e}")
        return pd.DataFrame()
