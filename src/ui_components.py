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

def render_monthly_trend(df: pd.DataFrame) -> None:
    """
    Render monthly sales and profit trend chart with area fill and 3-month moving average.

    Args:
        df: Filtered sales DataFrame
    """
    if df.empty:
        st.info("No data available for sales trend analysis.")
        return

    try:
        monthly = df.groupby('Year-Month', as_index=False).agg(
            Sales=('Sales', 'sum'),
            Profit=('Profit', 'sum')
        ).sort_values('Year-Month')

        monthly['Sales_MA3'] = monthly['Sales'].rolling(3, min_periods=1).mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly['Year-Month'],
            y=monthly['Sales'],
            name='Monthly Sales',
            mode='lines+markers',
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.15)',
            hovertemplate='<b>Sales</b>: $%{y:,.0f}<br>Month: %{x}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=monthly['Year-Month'],
            y=monthly['Sales_MA3'],
            name='3-Month Moving Avg',
            mode='lines',
            line=dict(color='#f59e0b', width=2, dash='dot'),
            hovertemplate='<b>3M MA</b>: $%{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=monthly['Year-Month'],
            y=monthly['Profit'],
            name='Monthly Profit',
            mode='lines+markers',
            line=dict(color='#10b981', width=2.5),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='<b>Profit</b>: $%{y:,.0f}<extra></extra>'
        ))

        fig.update_layout(
            title={'text': '📈 Monthly Revenue, Profit & 3-Month Moving Average', 'font': {'size': 18, 'color': '#1e3a8a'}},
            height=420,
            xaxis=dict(title='Month', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
            yaxis=dict(title='Amount ($)', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering monthly trend: {e}")
        st.error("Unable to render monthly trend chart.")

def render_category_pie(df: pd.DataFrame) -> None:
    """
    Render category revenue distribution donut chart.

    Args:
        df: Filtered sales DataFrame
    """
    if df.empty:
        st.info("No category data available.")
        return

    try:
        cat_df = df.groupby('Category', as_index=False)['Sales'].sum().sort_values('Sales', ascending=False)
        colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4']

        fig = go.Figure(data=[go.Pie(
            labels=cat_df['Category'],
            values=cat_df['Sales'],
            hole=0.55,
            marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
            textposition='outside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            title={'text': '🎯 Revenue Share by Category', 'font': {'size': 18, 'color': '#1e3a8a'}},
            height=420,
            template='plotly_white',
            legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering category donut chart: {e}")
        st.error("Unable to render category chart.")

def render_regional_bar(df: pd.DataFrame) -> None:
    """
    Render regional performance comparison bar chart.

    Args:
        df: Filtered sales DataFrame
    """
    if df.empty:
        st.info("No regional data available.")
        return

    try:
        reg_df = df.groupby('Region', as_index=False).agg(
            Sales=('Sales', 'sum'),
            Profit=('Profit', 'sum')
        ).sort_values('Sales', ascending=False)

        fig = go.Figure(data=[
            go.Bar(
                name='Sales',
                x=reg_df['Region'],
                y=reg_df['Sales'],
                text=reg_df['Sales'],
                texttemplate='$%{text:,.0f}',
                textposition='outside',
                marker=dict(color='#3b82f6', line=dict(color='white', width=1.5)),
                hovertemplate='<b>%{x} Sales</b>: $%{y:,.0f}<extra></extra>'
            ),
            go.Bar(
                name='Profit',
                x=reg_df['Region'],
                y=reg_df['Profit'],
                text=reg_df['Profit'],
                texttemplate='$%{text:,.0f}',
                textposition='outside',
                marker=dict(color='#10b981', line=dict(color='white', width=1.5)),
                hovertemplate='<b>%{x} Profit</b>: $%{y:,.0f}<extra></extra>'
            )
        ])

        fig.update_layout(
            title={'text': '🌎 Regional Revenue vs Profit', 'font': {'size': 18, 'color': '#1e3a8a'}},
            barmode='group',
            height=420,
            xaxis=dict(title='Region', showgrid=False),
            yaxis=dict(title='Amount ($)', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering regional bar: {e}")
        st.error("Unable to render regional chart.")

def render_top_products(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Render top N products horizontal bar chart.

    Args:
        df: Sales DataFrame
        top_n: Number of top products
    """
    if df.empty:
        st.info("No product data available.")
        return

    try:
        prod_df = df.groupby('Product', as_index=False)['Sales'].sum().nlargest(top_n, 'Sales').sort_values('Sales')

        fig = go.Figure(data=[go.Bar(
            y=prod_df['Product'],
            x=prod_df['Sales'],
            orientation='h',
            text=prod_df['Sales'],
            texttemplate='$%{text:,.0f}',
            textposition='outside',
            marker=dict(
                color=prod_df['Sales'],
                colorscale='Blues',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>'
        )])

        fig.update_layout(
            title={'text': f'🏆 Top {top_n} Products by Revenue', 'font': {'size': 18, 'color': '#1e3a8a'}},
            height=420,
            xaxis=dict(title='Revenue ($)', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
            yaxis=dict(title='', showgrid=False),
            template='plotly_white',
            margin=dict(l=160)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering top products: {e}")
        st.error("Unable to render top products chart.")

