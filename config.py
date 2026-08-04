"""
Configuration management for SmartSales Analytics Dashboard.
Centralized, environment-aware configuration with typed settings and validation.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
SRC_DIR = BASE_DIR / 'src'
LOGS_DIR = BASE_DIR / 'logs'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
TESTS_DIR = BASE_DIR / 'tests'
REPORTS_DIR = BASE_DIR / 'reports'

os.environ.setdefault('MPLCONFIGDIR', str(LOGS_DIR / '.matplotlib'))

for directory in [DATA_DIR, LOGS_DIR, NOTEBOOKS_DIR, TESTS_DIR, REPORTS_DIR, LOGS_DIR / '.matplotlib']:
    directory.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_DIR / 'sales_raw.csv'
CLEAN_DATA_PATH = DATA_DIR / 'sales_clean.csv'
DATABASE_PATH = DATA_DIR / 'smartsales.db'
SQL_QUERIES_PATH = SRC_DIR / 'sql_queries.sql'

APP_TITLE = 'SmartSales Analytics Dashboard'
APP_ICON = '📊'
APP_LAYOUT = 'wide'
APP_VERSION = '2.0.0'

RANDOM_SEED = 42
NUM_TRANSACTIONS = 25000
NUM_CUSTOMERS = 3200

RFM_QUANTILES = 5
FORECAST_PERIODS = 6
SALES_PERCENTILE_CAP = 99.5
OUTLIER_Z_THRESHOLD = 3.0
CONFIDENCE_LEVEL = 0.95
ANOVA_ALPHA = 0.05
PARETO_RATIO = 0.80

CACHE_TTL = 3600
MAX_ROWS_DISPLAY = 10000
CHUNK_SIZE = 5000

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = LOGS_DIR / 'smartsales.log'
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

DB_TIMEOUT = 30
DB_CHECK_SAME_THREAD = False

ALLOWED_EXTENSIONS: Set[str] = {'.csv', '.xlsx', '.xls'}
MAX_UPLOAD_SIZE_MB = 100

ENABLE_DATA_EXPORT = True
ENABLE_ADVANCED_ANALYTICS = True
ENABLE_FORECASTING = True
ENABLE_HYPOTHESIS_TESTING = True
ENABLE_COHORT_ANALYSIS = True

REQUIRED_COLUMNS: List[str] = [
    'Order ID', 'Order Date', 'Customer', 'Product', 'Category',
    'Sub-Category', 'Quantity', 'Sales', 'Discount', 'Cost',
    'Profit', 'Region', 'State', 'City', 'Payment Mode', 'Shipping Mode'
]

NUMERIC_COLUMNS: List[str] = ['Quantity', 'Sales', 'Discount', 'Cost', 'Profit']
CATEGORICAL_COLUMNS: List[str] = [
    'Customer', 'Product', 'Category', 'Sub-Category',
    'Region', 'State', 'City', 'Payment Mode', 'Shipping Mode'
]

def get_config(key: str, default: Optional[Any] = None) -> Any:
    """
    Get configuration value with fallback.

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value
    """
    return globals().get(key, default)

def validate_environment() -> bool:
    """
    Validate that all required directories and configurations are present.

    Returns:
        True if environment is valid, False otherwise
    """
    required_dirs = [BASE_DIR, DATA_DIR, SRC_DIR, LOGS_DIR]
    for directory in required_dirs:
        if not directory.exists():
            return False
    return True

if not validate_environment():
    raise RuntimeError("Invalid environment configuration. Required directories are missing.")
