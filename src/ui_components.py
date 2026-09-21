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

def render_forecast_chart(forecast_df: pd.DataFrame) -> None:
    """
    Render time-series sales forecast with shaded 95% confidence interval ribbons.

    Args:
        forecast_df: DataFrame with Historical & Forecast periods and Lower_CI / Upper_CI
    """
    if forecast_df.empty:
        st.info("No forecast data available.")
        return

    try:
        hist_df = forecast_df[forecast_df['Type'] == 'Historical']
        fc_df = forecast_df[forecast_df['Type'] == 'Forecast']

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_df['Year-Month'],
            y=hist_df['Sales'],
            name='Historical Sales',
            mode='lines+markers',
            line=dict(color='#2563eb', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Historical</b><br>%{x}: $%{y:,.0f}<extra></extra>'
        ))

        if not fc_df.empty:
            bridge_x = [hist_df['Year-Month'].iloc[-1]] + list(fc_df['Year-Month'])
            bridge_y = [hist_df['Sales'].iloc[-1]] + list(fc_df['Sales'])
            bridge_upper = [hist_df['Sales'].iloc[-1]] + list(fc_df['Upper_CI'])
            bridge_lower = [hist_df['Sales'].iloc[-1]] + list(fc_df['Lower_CI'])

            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=bridge_upper,
                name='Upper 95% Confidence Band',
                mode='lines',
                line=dict(color='rgba(139, 92, 246, 0.0)'),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=bridge_lower,
                name='95% Confidence Interval',
                mode='lines',
                line=dict(color='rgba(139, 92, 246, 0.0)'),
                fill='tonexty',
                fillcolor='rgba(139, 92, 246, 0.20)',
                hovertemplate='<b>Confidence Interval</b><br>Range: $%{y:,.0f}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=bridge_y,
                name='6-Month Forecast Projection',
                mode='lines+markers',
                line=dict(color='#8b5cf6', width=3, dash='dash'),
                marker=dict(size=7, symbol='diamond'),
                hovertemplate='<b>Projected Sales</b><br>%{x}: $%{y:,.0f}<extra></extra>'
            ))

        fig.update_layout(
            title={'text': '🔮 6-Month Predictive Sales Forecast with 95% Confidence Intervals', 'font': {'size': 18, 'color': '#1e3a8a'}},
            height=460,
            xaxis=dict(title='Year-Month', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
            yaxis=dict(title='Projected Sales ($)', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering forecast chart: {e}")
        st.error("Unable to render forecast chart.")

def render_pareto_chart(pareto_df: pd.DataFrame, dimension_name: str = 'Product') -> None:
    """
    Render dual-axis Pareto 80/20 chart (Individual Bars + Cumulative % Line + 80% Threshold).

    Args:
        pareto_df: DataFrame output from calculate_pareto_analysis
        dimension_name: Label for entity dimension
    """
    if pareto_df.empty:
        st.info("No data available for Pareto analysis.")
        return

    try:
        plot_df = pareto_df.head(25)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=plot_df[dimension_name],
            y=plot_df['Sales'],
            name='Revenue ($)',
            marker=dict(color='#3b82f6', line=dict(color='white', width=1)),
            yaxis='y',
            hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=plot_df[dimension_name],
            y=plot_df['Cumulative_Pct'],
            name='Cumulative Revenue %',
            mode='lines+markers',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=6),
            yaxis='y2',
            hovertemplate='<b>Cumulative</b>: %{y:.1f}%<extra></extra>'
        ))

        fig.add_hline(
            y=80,
            line_dash='dash',
            line_color='#b91c1c',
            annotation_text='80% Revenue Cutoff (Pareto Threshold)',
            annotation_position='top left',
            yref='y2'
        )

        fig.update_layout(
            title={'text': f'🎯 Pareto (80/20) Revenue Concentration: {dimension_name}', 'font': {'size': 18, 'color': '#1e3a8a'}},
            height=460,
            template='plotly_white',
            xaxis=dict(title=dimension_name, tickangle=-45),
            yaxis=dict(title='Individual Sales ($)', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
            yaxis2=dict(
                title='Cumulative Contribution (%)',
                overlaying='y',
                side='right',
                range=[0, 105],
                showgrid=False
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering Pareto chart: {e}")
        st.error("Unable to render Pareto chart.")

def render_cohort_heatmap(retention_pct: pd.DataFrame) -> None:
    """
    Render customer cohort retention percentage heatmap.

    Args:
        retention_pct: Matrix of customer retention percentages
    """
    if retention_pct.empty:
        st.info("Insufficient longitudinal data for customer cohort matrix.")
        return

    try:
        heat_df = retention_pct.copy()
        heat_df.index = [str(idx) for idx in heat_df.index]

        fig = px.imshow(
            heat_df,
            labels=dict(x="Months After Acquisition", y="Cohort Month", color="Retention %"),
            x=[f"Month {c}" for c in heat_df.columns],
            y=heat_df.index,
            text_auto='.0f',
            aspect="auto",
            color_continuous_scale="Viridis",
            zmin=0,
            zmax=100
        )
        fig.update_layout(
            title={'text': '👥 Customer Cohort Retention Rates (%)', 'font': {'size': 18, 'color': '#1e3a8a'}},
            height=440,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering cohort heatmap: {e}")
        st.error("Unable to render cohort retention heatmap.")

def render_rfm_scatter(rfm: pd.DataFrame) -> None:
    """
    Render interactive RFM customer value scatter plot.

    Args:
        rfm: Output DataFrame from rfm_segmentation
    """
    if rfm.empty:
        st.info("No RFM data available.")
        return

    try:
        fig = px.scatter(
            rfm,
            x='Frequency',
            y='Monetary',
            size='Recency',
            color='Customer Segment',
            hover_name='Customer',
            hover_data=['R_Score', 'F_Score', 'M_Score', 'AOV'],
            title='🎯 Customer Value Map (Frequency vs. Monetary Value)',
            size_max=18,
            color_discrete_map={
                'Champions': '#10b981',
                'Loyal Customers': '#3b82f6',
                'Potential Loyalists': '#f59e0b',
                'At Risk': '#f97316',
                'Lost Customers': '#ef4444'
            }
        )
        fig.update_layout(
            template='plotly_white',
            height=460,
            xaxis=dict(title='Purchase Frequency (Orders)', showgrid=True),
            yaxis=dict(title='Monetary Spend ($)', showgrid=True)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error rendering RFM scatter: {e}")
        st.error("Unable to render customer value map.")

def render_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Render enhanced sidebar filters with date presets and multi-select dimensions.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary of user-selected filters
    """
    st.sidebar.title('📊 SmartSales Analytics')

    if df.empty or 'Order Date' not in df.columns:
        return {}

    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()

    st.sidebar.markdown('### 📅 Time Horizon')

    date_selection = st.sidebar.date_input(
        'Custom Date Range',
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        start_date, end_date = date_selection
    else:
        start_date, end_date = min_date, max_date

    st.sidebar.markdown('### 🔍 Dimensional Slicers')
    filter_keys = ['Region', 'Category', 'Customer Segment', 'Payment Mode', 'Shipping Mode']
    selected_filters = {'date_range': (start_date, end_date)}

    for key in filter_keys:
        if key in df.columns:
            options = sorted([str(x) for x in df[key].dropna().unique()])
            selected_filters[key] = st.sidebar.multiselect(key, options=options, default=[])

    return selected_filters

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply active sidebar filter conditions to DataFrame.

    Args:
        df: Input DataFrame
        filters: Filter criteria dictionary

    Returns:
        Filtered DataFrame
    """
    if df.empty or not filters:
        return df

    filtered = df.copy()

    if 'date_range' in filters and 'Order Date' in filtered.columns:
        start, end = filters['date_range']
        filtered = filtered[(filtered['Order Date'].dt.date >= start) & (filtered['Order Date'].dt.date <= end)]

    for col, values in filters.items():
        if col != 'date_range' and values and col in filtered.columns:
            filtered = filtered[filtered[col].astype(str).isin(values)]

    return filtered

