"""
Data cleaning and quality assurance module for SmartSales Analytics.
Handles robust schema validation, missing value imputation, outlier detection,
business rule enforcement, type casting, and feature engineering.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import numpy as np
import pandas as pd

from config import (
    REQUIRED_COLUMNS,
    SALES_PERCENTILE_CAP,
    OUTLIER_Z_THRESHOLD
)

logger = logging.getLogger(__name__)

class DataCleaningError(Exception):
    """Custom exception raised when data cleaning operations fail."""
    pass

class SchemaValidationError(DataCleaningError):
    """Raised when incoming dataset does not adhere to required schema."""
    pass

def validate_required_columns(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> bool:
    """
    Validate that DataFrame contains all mandatory columns.

    Args:
        df: Input DataFrame
        required_cols: List of required column names (defaults to REQUIRED_COLUMNS)

    Returns:
        True if all required columns are present

    Raises:
        DataCleaningError: If required columns are missing
    """
    if df is None:
        raise DataCleaningError("Cannot validate None object as DataFrame")

    cols_to_check = required_cols or REQUIRED_COLUMNS
    missing_cols = set(cols_to_check) - set(df.columns)

    if missing_cols:
        error_msg = f"Missing required columns: {sorted(list(missing_cols))}"
        logger.error(error_msg)
        raise DataCleaningError(error_msg)

    return True

def standardize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize text casing, strip leading/trailing whitespace, and preserve acronyms.

    Args:
        df: DataFrame with raw text fields

    Returns:
        DataFrame with standardized text
    """
    df = df.copy()

    acronym_map = {
        'Upi': 'UPI',
        'upi': 'UPI',
        'Cod': 'COD',
        'cod': 'COD',
        'Id': 'ID'
    }

    text_cols = ['Customer', 'Product', 'Category', 'Sub-Category', 'Region', 'State', 'City', 'Payment Mode', 'Shipping Mode']
    for col in text_cols:
        if col in df.columns:
            cleaned_col = df[col].astype(str).str.strip().str.title()
            cleaned_col = cleaned_col.replace({'Nan': np.nan, 'None': np.nan, '': np.nan})

            for wrong, correct in acronym_map.items():
                cleaned_col = cleaned_col.replace(to_replace=r'\b' + wrong + r'\b', value=correct, regex=True)

            df[col] = cleaned_col

    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or handle missing values across business fields.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with imputed missing values
    """
    df = df.copy()

    if 'Customer' in df.columns:
        df['Customer'] = df['Customer'].fillna('Unknown Customer')

    if 'City' in df.columns and 'State' in df.columns:
        missing_city_mask = df['City'].isna() | (df['City'] == '')
        if missing_city_mask.any():
            fallback_city = df['State'].fillna('Unknown') + ' City'
            df['City'] = np.where(missing_city_mask, fallback_city, df['City'])

    if 'Discount' in df.columns:
        df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce').fillna(0.0)

    if 'Quantity' in df.columns:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1)

    for col in ['Sales', 'Cost']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate derived business metrics and temporal dimensions.

    Args:
        df: Preprocessed DataFrame

    Returns:
        DataFrame with engineered attributes
    """
    df = df.copy()

    if 'Order Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Order Date']):
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

    if 'Order Date' in df.columns:
        df['Year'] = df['Order Date'].dt.year.astype('Int64')
        df['Month'] = df['Order Date'].dt.month.astype('Int64')
        df['Quarter'] = df['Order Date'].dt.quarter.astype('Int64')
        df['Year-Month'] = df['Order Date'].dt.to_period('M').astype(str)
        df['Day of Week'] = df['Order Date'].dt.day_name()
        df['Is Weekend'] = df['Order Date'].dt.dayofweek.isin([5, 6]).astype(int)

    if 'Sales' in df.columns and 'Cost' in df.columns:
        df['Profit'] = np.round(df['Sales'] - df['Cost'], 2)

    if 'Sales' in df.columns and 'Profit' in df.columns:
        df['Profit Margin'] = np.where(
            df['Sales'] > 0,
            np.round((df['Profit'] / df['Sales']) * 100, 2),
            0.0
        )

    if 'Sales' in df.columns:
        df['Order Value'] = df['Sales']

    if 'Sales' in df.columns and 'Quantity' in df.columns:
        df['Unit Price'] = np.where(df['Quantity'] > 0, np.round(df['Sales'] / df['Quantity'], 2), 0.0)
    if 'Cost' in df.columns and 'Quantity' in df.columns:
        df['Unit Cost'] = np.where(df['Quantity'] > 0, np.round(df['Cost'] / df['Quantity'], 2), 0.0)

    return df

def normalize_numeric_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'minmax'
) -> pd.DataFrame:
    """
    Normalize numerical columns for machine learning / advanced statistical analysis.

    Args:
        df: Input DataFrame
        columns: List of columns to normalize
        method: 'minmax' for [0, 1] scaling or 'standard' for zero-mean unit-variance

    Returns:
        DataFrame containing scaled feature columns named as '<col>_scaled'
    """
    df_norm = df.copy()
    for col in columns:
        if col not in df_norm.columns:
            continue
        series = pd.to_numeric(df_norm[col], errors='coerce')
        if method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            if max_val > min_val:
                df_norm[f"{col}_scaled"] = (series - min_val) / (max_val - min_val)
            else:
                df_norm[f"{col}_scaled"] = 0.0
        elif method == 'standard':
            std_val = series.std()
            if std_val > 0:
                df_norm[f"{col}_scaled"] = (series - series.mean()) / std_val
            else:
                df_norm[f"{col}_scaled"] = 0.0
    return df_norm

def encode_categorical_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'frequency'
) -> pd.DataFrame:
    """
    Encode categorical columns for analytical modeling.

    Args:
        df: Input DataFrame
        columns: Categorical columns to encode
        method: 'frequency' (percentage of occurrences) or 'onehot' (binary dummy columns)

    Returns:
        DataFrame with encoded features
    """
    df_encoded = df.copy()
    if method == 'frequency':
        for col in columns:
            if col in df_encoded.columns:
                freq = df_encoded[col].value_counts(normalize=True)
                df_encoded[f"{col}_freq"] = df_encoded[col].map(freq).fillna(0.0)
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True, dtype=int)
    return df_encoded

