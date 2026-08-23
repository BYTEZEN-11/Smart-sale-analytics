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

def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Detect outliers using Tukey's Interquartile Range (IQR) method.

    Args:
        series: Numeric pandas Series
        factor: IQR multiplier (default 1.5 for mild outliers, 3.0 for extreme)

    Returns:
        Boolean Series indicating whether each index is an outlier
    """
    clean_series = pd.to_numeric(series, errors='coerce').dropna()
    if len(clean_series) < 4:
        return pd.Series(False, index=series.index)

    q25 = clean_series.quantile(0.25)
    q75 = clean_series.quantile(0.75)
    iqr = q75 - q25

    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)

    return (series < lower_bound) | (series > upper_bound)

def detect_outliers_zscore(series: pd.Series, threshold: float = OUTLIER_Z_THRESHOLD) -> pd.Series:
    """
    Detect outliers using standard Z-score method.

    Args:
        series: Numeric pandas Series
        threshold: Standard deviation threshold (default 3.0)

    Returns:
        Boolean Series where True indicates outlier
    """
    clean_series = pd.to_numeric(series, errors='coerce').dropna()
    std = clean_series.std()
    if std == 0 or len(clean_series) < 2:
        return pd.Series(False, index=series.index)

    mean = clean_series.mean()
    z_scores = np.abs((clean_series - mean) / std)
    outliers = z_scores > threshold
    return outliers.reindex(series.index, fill_value=False)

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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute full production data cleaning pipeline:
    1. Validation of required columns.
    2. Deduplication of records.
    3. Standardization of categorical & text fields with acronym protection.
    4. Date conversion & dropping corrupted dates.
    5. Missing value imputation with vectorized rules.
    6. Numerical boundary clipping & capping.
    7. Business rule integrity checks (Profit recalculation).
    8. Derived feature engineering (temporal, unit economics, profit margins).

    Args:
        df: Raw input DataFrame

    Returns:
        Cleaned, validated, and enriched DataFrame

    Raises:
        DataCleaningError: If validation fails or input is empty
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("Empty or invalid DataFrame provided for cleaning")
        raise DataCleaningError("Cannot clean empty or invalid DataFrame")

    initial_rows = len(df)
    logger.info(f"Initiating data cleaning pipeline. Initial row count: {initial_rows}")

    try:
        clean_df = df.copy()

        validate_required_columns(clean_df, REQUIRED_COLUMNS)

        initial_count = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=['Order ID'], keep='first')
        dupes_removed = initial_count - len(clean_df)
        if dupes_removed > 0:
            logger.info(f"Deduplication removed {dupes_removed} duplicate Order ID rows")

        clean_df = standardize_text_fields(clean_df)

        clean_df['Order Date'] = pd.to_datetime(clean_df['Order Date'], errors='coerce')
        invalid_dates_count = clean_df['Order Date'].isna().sum()
        if invalid_dates_count > 0:
            logger.warning(f"Removing {invalid_dates_count} records with unparseable Order Dates")
            clean_df = clean_df.dropna(subset=['Order Date'])

        clean_df = handle_missing_values(clean_df)

        clean_df['Quantity'] = pd.to_numeric(clean_df['Quantity'], errors='coerce').clip(lower=1, upper=10)
        clean_df['Discount'] = pd.to_numeric(clean_df['Discount'], errors='coerce').clip(lower=0.0, upper=0.55)
        clean_df['Sales'] = pd.to_numeric(clean_df['Sales'], errors='coerce')
        clean_df['Cost'] = pd.to_numeric(clean_df['Cost'], errors='coerce')

        invalid_sales_mask = clean_df['Sales'] <= 0
        if invalid_sales_mask.any():
            invalid_sales_count = invalid_sales_mask.sum()
            logger.info(f"Removing {invalid_sales_count} rows with non-positive sales")
            clean_df = clean_df[~invalid_sales_mask]

        sales_cap = clean_df['Sales'].quantile(SALES_PERCENTILE_CAP / 100.0)
        extreme_sales_count = (clean_df['Sales'] > sales_cap).sum()
        if extreme_sales_count > 0:
            logger.info(f"Capping {extreme_sales_count} extreme sales values at {SALES_PERCENTILE_CAP}th percentile (${sales_cap:,.2f})")
            clean_df['Sales'] = clean_df['Sales'].clip(upper=sales_cap)

        clean_df = engineer_features(clean_df)

        final_rows = len(clean_df)
        logger.info(f"Data cleaning pipeline successfully completed: {initial_rows} -> {final_rows} rows (Total dropped: {initial_rows - final_rows})")

        return clean_df

    except DataCleaningError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in cleaning pipeline: {e}", exc_info=True)
        raise DataCleaningError(f"Data cleaning pipeline failed: {e}")

def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an exhaustive data quality audit report.

    Args:
        df: Input DataFrame to audit

    Returns:
        Structured audit dictionary with metrics for completeness, uniqueness,
        validity, memory footprint, and column distributions.
    """
    if df is None or df.empty:
        return {}

    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    completeness = round(((total_cells - total_missing) / total_cells) * 100, 2) if total_cells > 0 else 0.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    outlier_counts = {}
    for col in numeric_cols:
        outlier_counts[col] = int(detect_outliers_iqr(df[col]).sum())

    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'completeness_score': completeness,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().mean() * 100).round(2).to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_order_ids': int(df['Order ID'].duplicated().sum()) if 'Order ID' in df.columns else 0,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'outlier_counts_iqr': outlier_counts,
        'numeric_stats': df[numeric_cols].describe().round(2).to_dict() if numeric_cols else {},
        'cardinality': {col: int(df[col].nunique()) for col in df.columns}
    }

    return report
