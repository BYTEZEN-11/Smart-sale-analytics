"""
UI components and advanced visualizations for SmartSales Analytics Dashboard.
Provides interactive Plotly visualizations, KPI cards, Pareto charts,
cohort heatmaps, forecast confidence intervals, and Excel/CSV export capabilities.
"""
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

def render_kpi_cards(kpis: Dict[str, Any]) -> None:
    """
    Render executive KPI metric cards with modern CSS styling and gradient backdrops.

    Args:
        kpis: Dictionary containing KPI calculations
    """
    metrics = [
        ('Revenue', kpis.get('revenue', 0.0), 'revenue', '💰', '$'),
        ('Profit', kpis.get('profit', 0.0), 'profit', '📈', '$'),
        ('Orders', kpis.get('orders', 0), 'orders', '🛒', ''),
        ('Customers', kpis.get('customers', 0), 'customers', '👥', ''),
        ('Avg Order Value', kpis.get('avg_order_value', 0.0), 'aov', '💳', '$'),
        ('Profit Margin', kpis.get('profit_margin', 0.0), 'margin', '📊', '%')
    ]

    cols = st.columns(len(metrics))
    for col, (label, value, css_class, icon, prefix_or_suffix) in zip(cols, metrics):
        with col:
            if prefix_or_suffix == '%':
                formatted_value = f"{value:.1f}%"
            elif prefix_or_suffix == '$':
                formatted_value = f"${value:,.0f}" if value >= 1000 else f"${value:,.2f}"
            else:
                formatted_value = f"{value:,}" if isinstance(value, int) else f"{value:,.0f}"

            st.markdown(
                f'''<div class="kpi {css_class}">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kt">{label}</div>
                    <div class="kv">{formatted_value}</div>
                </div>''',
                unsafe_allow_html=True
            )

