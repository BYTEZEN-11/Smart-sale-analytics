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

