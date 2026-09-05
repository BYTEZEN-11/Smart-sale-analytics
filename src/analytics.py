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

