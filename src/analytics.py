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

