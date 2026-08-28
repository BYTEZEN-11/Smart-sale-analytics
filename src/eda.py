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

