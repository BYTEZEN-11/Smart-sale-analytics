"""
Exploratory Data Analysis (EDA) module for SmartSales Analytics.
Provides comprehensive univariate, bivariate, multivariate, statistical summary,
outlier detection, and publication-ready visualization capabilities.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns

logger = logging.getLogger(__name__)

def generate_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate high-level overview of dataset dimensions, schema, and memory.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with dataset metadata
    """
    if df is None or df.empty:
        return {'rows': 0, 'columns': 0, 'memory_mb': 0.0, 'column_types': {}}

    return {
        'rows': int(len(df)),
        'columns': int(len(df.columns)),
        'memory_mb': round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        'column_types': df.dtypes.astype(str).to_dict(),
        'missing_cells': int(df.isnull().sum().sum()),
        'missing_percentage': round((df.isnull().sum().sum() / df.size) * 100, 2) if df.size > 0 else 0.0,
        'duplicate_rows': int(df.duplicated().sum())
    }

def compute_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rigorous descriptive statistics for all numeric columns:
    Mean, Median, Mode, Variance, Std Dev, Min, Max, Skewness, Kurtosis, and IQR.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with detailed statistical metrics indexed by feature name
    """
    if df is None or df.empty:
        return pd.DataFrame()

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    records = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if len(s) == 0:
            continue

        mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25

        records.append({
            'Feature': col,
            'Count': int(len(s)),
            'Mean': round(float(s.mean()), 2),
            'Median': round(float(s.median()), 2),
            'Mode': round(float(mode_val), 2) if pd.notnull(mode_val) else np.nan,
            'Std Dev': round(float(s.std()), 2),
            'Variance': round(float(s.var()), 2),
            'Min': round(float(s.min()), 2),
            '25%': round(float(q25), 2),
            '50% (Median)': round(float(s.median()), 2),
            '75%': round(float(q75), 2),
            'Max': round(float(s.max()), 2),
            'IQR': round(float(iqr), 2),
            'Skewness': round(float(s.skew()), 3),
            'Kurtosis': round(float(s.kurtosis()), 3)
        })

    return pd.DataFrame(records).set_index('Feature')

def compute_correlation_matrices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute Pearson (linear) and Spearman (rank-order) correlation matrices for numeric features.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with 'pearson' and 'spearman' DataFrames
    """
    if df is None or df.empty:
        return {'pearson': pd.DataFrame(), 'spearman': pd.DataFrame()}

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return {'pearson': pd.DataFrame(), 'spearman': pd.DataFrame()}

    return {
        'pearson': num_df.corr(method='pearson').round(3),
        'spearman': num_df.corr(method='spearman').round(3)
    }

def compute_outlier_summary(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Calculate Tukey IQR outlier metrics for each numeric column.

    Args:
        df: Input DataFrame
        factor: Tukey multiplier

    Returns:
        DataFrame summarizing lower fence, upper fence, outlier count, and percentage
    """
    if df is None or df.empty:
        return pd.DataFrame()

    num_df = df.select_dtypes(include=[np.number])
    records = []

    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) < 4:
            continue
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        lower_fence = q25 - (factor * iqr)
        upper_fence = q75 + (factor * iqr)

        outliers = (s < lower_fence) | (s > upper_fence)
        outlier_count = int(outliers.sum())
        outlier_pct = round((outlier_count / len(s)) * 100, 2)

        records.append({
            'Column': col,
            'Q1 (25%)': round(float(q25), 2),
            'Q3 (75%)': round(float(q75), 2),
            'IQR': round(float(iqr), 2),
            'Lower Fence': round(float(lower_fence), 2),
            'Upper Fence': round(float(upper_fence), 2),
            'Outlier Count': outlier_count,
            'Outlier Pct (%)': outlier_pct
        })

    return pd.DataFrame(records)

def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute end-to-end automated exploratory data analysis.

    Args:
        df: Cleaned sales DataFrame

    Returns:
        Dictionary containing comprehensive exploratory findings and metrics
    """
    if df is None or df.empty:
        return {'status': 'empty_dataset'}

    insights = {}

    insights['overview'] = generate_dataset_overview(df)

    total_sales = float(df['Sales'].sum()) if 'Sales' in df.columns else 0.0
    total_profit = float(df['Profit'].sum()) if 'Profit' in df.columns else 0.0
    total_orders = int(df['Order ID'].nunique()) if 'Order ID' in df.columns else len(df)
    total_customers = int(df['Customer'].nunique()) if 'Customer' in df.columns else 0

    insights['total_sales'] = round(total_sales, 2)
    insights['total_profit'] = round(total_profit, 2)
    insights['total_orders'] = total_orders
    insights['total_customers'] = total_customers
    insights['avg_order_value'] = round(total_sales / total_orders, 2) if total_orders > 0 else 0.0
    insights['profit_margin'] = round((total_profit / total_sales) * 100, 2) if total_sales > 0 else 0.0

    if 'Order Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Order Date']):
        insights['date_range'] = {
            'start': df['Order Date'].min().strftime('%Y-%m-%d'),
            'end': df['Order Date'].max().strftime('%Y-%m-%d'),
            'duration_days': (df['Order Date'].max() - df['Order Date'].min()).days
        }

    if 'Category' in df.columns and 'Sales' in df.columns:
        insights['top_categories'] = df.groupby('Category')['Sales'].sum().sort_values(ascending=False).to_dict()
    if 'Product' in df.columns and 'Sales' in df.columns:
        insights['top_products'] = df.groupby('Product')['Sales'].sum().nlargest(10).to_dict()
    if 'Customer' in df.columns and 'Sales' in df.columns:
        insights['top_customers'] = df.groupby('Customer')['Sales'].sum().nlargest(10).to_dict()
    if 'Region' in df.columns and 'Sales' in df.columns:
        insights['top_regions'] = df.groupby('Region')['Sales'].sum().sort_values(ascending=False).to_dict()

    if 'Profit' in df.columns and 'Product' in df.columns:
        loss_products = df[df['Profit'] < 0]['Product'].unique()
        insights['products_with_loss_count'] = len(loss_products)
        insights['products_with_loss'] = list(loss_products)

    insights['statistical_summary'] = compute_statistical_summary(df)

    insights['correlations'] = compute_correlation_matrices(df)

    insights['outliers'] = compute_outlier_summary(df)

    return insights

