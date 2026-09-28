"""
SmartSales Analytics Dashboard
Production-grade multi-page Streamlit application.
Provides executive KPIs, predictive forecasting with confidence intervals,
RFM customer intelligence, cohort retention, Pareto analysis, statistical hypothesis testing,
automated EDA profiler, and multi-format reporting.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.logger import setup_logger
logger = setup_logger('smartsales_app')

from config import (
    APP_ICON,
    APP_LAYOUT,
    APP_TITLE,
    APP_VERSION,
    CACHE_TTL,
    CLEAN_DATA_PATH,
    RAW_DATA_PATH
)

from src.data_cleaning import clean_data, get_data_quality_report
from src.analytics import (
    calculate_cohort_retention,
    calculate_kpis,
    calculate_pareto_analysis,
    forecast_sales,
    get_top_performers,
    perform_chi_square_test,
    perform_one_way_anova,
    perform_two_sample_ttest,
    rfm_segmentation
)
from src.eda import (
    compute_outlier_summary,
    compute_statistical_summary,
    generate_dataset_overview,
    perform_eda,
    plot_eda_correlation_heatmap,
    plot_univariate_distribution
)
from src.ui_components import (
    apply_filters,
    render_category_pie,
    render_cohort_heatmap,
    render_data_table,
    render_filters,
    render_forecast_chart,
    render_insights,
    render_kpi_cards,
    render_monthly_trend,
    render_pareto_chart,
    render_regional_bar,
    render_rfm_scatter,
    render_top_products
)

st.set_page_config(
    page_title=f"{APP_TITLE} v{APP_VERSION}",
    page_icon=APP_ICON,
    layout=APP_LAYOUT
)

st.markdown('''
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2.5rem;
    }
    .kpi {
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        color: white;
        margin-bottom: 12px;
    }
    .kpi:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.16);
    }
    .kpi.revenue { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); }
    .kpi.profit { background: linear-gradient(135deg, #10b981 0%, #047857 100%); }
    .kpi.orders { background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); }
    .kpi.customers { background: linear-gradient(135deg, #06b6d4 0%, #0e7490 100%); }
    .kpi.aov { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .kpi.margin { background: linear-gradient(135deg, #ec4899 0%, #be185d 100%); }
    .kt {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.9);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .kv {
        font-size: 1.85rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.12);
    }
    .kpi-icon {
        font-size: 2.2rem;
        opacity: 0.35;
        float: right;
        margin-top: -6px;
    }
    .insight-box {
        background: #f8fafc;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-success { background: #dcfce7; color: #15803d; }
    .badge-warning { background: #fef3c7; color: #b45309; }
    .badge-danger { background: #fee2e2; color: #b91c1c; }
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #475569;
        font-weight: 700;
    }
</style>
''', unsafe_allow_html=True)

@st.cache_data(ttl=CACHE_TTL)
def load_data() -> pd.DataFrame:
    """
    Load and cache sales transactions from clean CSV or raw fallback.

    Returns:
        Cleaned sales DataFrame
    """
    try:
        if not CLEAN_DATA_PATH.exists():
            if not RAW_DATA_PATH.exists():
                st.error("Data files not found. Please run generate_data.py first.")
                st.stop()
            raw_df = pd.read_csv(RAW_DATA_PATH)
            cleaned_df = clean_data(raw_df)
            cleaned_df.to_csv(CLEAN_DATA_PATH, index=False)

        df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=['Order Date'])
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        st.error(f"Failed to load dataset: {e}")
        st.stop()

@st.cache_data(ttl=CACHE_TTL)
def get_rfm_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and cache RFM segmentation."""
    try:
        return rfm_segmentation(df)
    except Exception as e:
        logger.error(f"Error computing RFM: {e}")
        return pd.DataFrame()

def render_executive_dashboard(df: pd.DataFrame) -> None:
    """Render executive overview dashboard page."""
    kpis = calculate_kpis(df)
    render_kpi_cards(kpis)

    col1, col2 = st.columns([1.6, 1])
    with col1:
        render_monthly_trend(df)
    with col2:
        render_category_pie(df)

    col3, col4 = st.columns(2)
    with col3:
        render_regional_bar(df)
    with col4:
        render_top_products(df, top_n=10)

def render_sales_analytics(df: pd.DataFrame) -> None:
    """Render sales trend and predictive forecasting page."""
    kpis = calculate_kpis(df)
    render_kpi_cards(kpis)

    render_monthly_trend(df)

    col1, col2 = st.columns(2)
    with col1:
        if not df.empty and 'Quarter' in df.columns:
            quarterly = df.groupby(['Year', 'Quarter'], as_index=False)['Sales'].sum()
            quarterly['Period'] = quarterly['Year'].astype(str) + ' Q' + quarterly['Quarter'].astype(str)
            fig = px.bar(
                quarterly,
                x='Period',
                y='Sales',
                title='📊 Quarterly Revenue Performance',
                text='Sales',
                color='Sales',
                color_continuous_scale='Blues'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not df.empty:
            pivot_reg = df.pivot_table(index='Category', columns='Region', values='Sales', aggfunc='sum', fill_value=0)
            fig = px.imshow(
                pivot_reg,
                text_auto='.2s',
                aspect='auto',
                title='🗺️ Category × Region Cross-Tabulation Matrix ($)',
                color_continuous_scale='Blues'
            )
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔮 Advanced Time-Series Sales Forecast")
    fc_periods = st.slider("Forecast Horizon (Months Ahead)", min_value=3, max_value=12, value=6)
    forecast_data = forecast_sales(df, periods=fc_periods)
    render_forecast_chart(forecast_data)

def render_customer_analytics(df: pd.DataFrame, rfm: pd.DataFrame) -> None:
    """Render customer intelligence, RFM segmentation, and cohort retention page."""
    kpis = calculate_kpis(df)
    render_kpi_cards(kpis)

    if rfm.empty:
        st.warning("Customer RFM segmentation data is unavailable.")
        return

    col1, col2 = st.columns(2)
    with col1:
        seg_summary = rfm.groupby('Customer Segment', as_index=False).agg(
            Customers=('Customer', 'count'),
            Revenue=('Monetary', 'sum')
        )
        fig = px.bar(
            seg_summary,
            x='Customer Segment',
            y='Customers',
            title='👥 Customer Segment Headcount & Revenue Intensity',
            text='Customers',
            color='Revenue',
            color_continuous_scale='Viridis'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(template='plotly_white', height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        render_rfm_scatter(rfm)

    st.markdown("### 📊 Longitudinal Customer Acquisition Cohort Analysis")
    _, retention_pct = calculate_cohort_retention(df)
    render_cohort_heatmap(retention_pct)

    st.markdown("### 💎 Top 25 Highest-Value Customers (RFM Matrix)")
    top_cust = rfm.sort_values('Monetary', ascending=False).head(25)
    st.dataframe(top_cust, use_container_width=True, hide_index=True)

def render_product_analytics(df: pd.DataFrame) -> None:
    """Render product profitability, Pareto 80/20 curve, and margin matrix."""
    kpis = calculate_kpis(df)
    render_kpi_cards(kpis)

    col1, col2 = st.columns(2)
    with col1:
        top_prod = df.groupby('Product', as_index=False)['Sales'].sum().nlargest(10, 'Sales').sort_values('Sales')
        fig = px.bar(
            top_prod,
            x='Sales',
            y='Product',
            orientation='h',
            title='🏆 Top 10 Products by Gross Revenue',
            color='Sales',
            color_continuous_scale='Greens'
        )
        fig.update_layout(template='plotly_white', height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prod_profit = df.groupby('Product', as_index=False)['Profit'].sum().nsmallest(10, 'Profit').sort_values('Profit')
        fig = px.bar(
            prod_profit,
            x='Profit',
            y='Product',
            orientation='h',
            title='⚠️ Lowest Profit / Loss-Making Products',
            color='Profit',
            color_continuous_scale='Reds'
        )
        fig.update_layout(template='plotly_white', height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Pareto 80/20 Revenue Concentration Analysis")
    pareto_df = calculate_pareto_analysis(df, dimension='Product', metric='Sales')
    render_pareto_chart(pareto_df, dimension_name='Product')

    vital_count = len(pareto_df[pareto_df['Pareto_Segment'].str.contains('Top 80%')])
    total_prods = len(pareto_df)
    st.info(f"**Pareto Insight:** **{vital_count} out of {total_prods} products** ({vital_count/total_prods*100:.1f}%) generate **80% of total company revenue**.")

def render_regional_analytics(df: pd.DataFrame) -> None:
    """Render geographic performance and territory distribution."""
    kpis = calculate_kpis(df)
    render_kpi_cards(kpis)

    render_regional_bar(df)

    col1, col2 = st.columns(2)
    with col1:
        state_df = df.groupby('State', as_index=False).agg(
            Sales=('Sales', 'sum'),
            Profit=('Profit', 'sum')
        ).nlargest(15, 'Sales').sort_values('Sales')

        fig = px.bar(
            state_df,
            x='Sales',
            y='State',
            orientation='h',
            color='Profit',
            title='🗺️ Top 15 States by Revenue & Profitability',
            text='Sales',
            color_continuous_scale='RdYlGn'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(template='plotly_white', height=460)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pay_df = df.groupby('Payment Mode', as_index=False)['Sales'].sum().sort_values('Sales', ascending=False)
        fig = px.pie(
            pay_df,
            names='Payment Mode',
            values='Sales',
            hole=0.45,
            title='💳 Revenue Distribution by Payment Mode'
        )
        fig.update_layout(template='plotly_white', height=460)
        st.plotly_chart(fig, use_container_width=True)

def render_statistical_testing(df: pd.DataFrame) -> None:
    """Render statistical hypothesis testing suite (ANOVA, Welch's t-test, Chi-square)."""
    st.markdown("## 🔬 Statistical Hypothesis Testing & Inferential Analytics")
    st.caption("Apply rigorous statistical methodology to test significant differences and associations.")

    tab_anova, tab_ttest, tab_chi2 = st.tabs([
        "1. One-Way ANOVA (Group Variance)",
        "2. Two-Sample Welch's t-Test",
        "3. Chi-Square Test of Independence"
    ])

    with tab_anova:
        st.markdown("#### One-Way ANOVA Test")
        st.write("Determine whether mean differences across groups are statistically significant ($H_0: \\mu_1 = \\mu_2 = \\dots = \\mu_k$).")
        c1, c2 = st.columns(2)
        with c1:
            anova_cat = st.selectbox("Categorical Factor", ['Category', 'Region', 'Shipping Mode', 'Payment Mode'], index=0)
        with c2:
            anova_val = st.selectbox("Continuous Metric", ['Sales', 'Profit', 'Profit Margin', 'Discount'], index=0)

        res_anova = perform_one_way_anova(df, group_col=anova_cat, val_col=anova_val)
        if res_anova['status'] == 'success':
            badge_color = "badge-success" if res_anova['is_significant'] else "badge-danger"
            st.markdown(f"**Conclusion:** {res_anova['conclusion']}")
            st.metric("F-Statistic", res_anova['f_statistic'])
            st.metric("p-Value", f"{res_anova['p_value']:.6f}")

            st.write("**Group Summary Breakdown:**")
            st.dataframe(pd.DataFrame(res_anova['group_metrics']).T, use_container_width=True)

    with tab_ttest:
        st.markdown("#### Independent Two-Sample Welch's t-Test")
        st.write("Compare the means of two distinct groups without assuming equal variances.")
        t_col = st.selectbox("Group Feature", ['Category', 'Region'], index=0)
        unique_groups = sorted(df[t_col].dropna().unique())
        if len(unique_groups) >= 2:
            col_a, col_b, col_m = st.columns(3)
            with col_a:
                g_a = st.selectbox("Group A", unique_groups, index=0)
            with col_b:
                g_b = st.selectbox("Group B", unique_groups, index=1)
            with col_m:
                t_val = st.selectbox("Metric", ['Sales', 'Profit', 'Discount'], index=0)

            res_ttest = perform_two_sample_ttest(df, group_col=t_col, val_col=t_val, group_a=g_a, group_b=g_b)
            if res_ttest['status'] == 'success':
                st.markdown(f"**Result:** {res_ttest['conclusion']}")
                ca, cb, cc = st.columns(3)
                ca.metric(f"Mean ({g_a})", f"${res_ttest['mean_a']:,.2f}")
                cb.metric(f"Mean ({g_b})", f"${res_ttest['mean_b']:,.2f}")
                cc.metric("p-Value", f"{res_ttest['p_value']:.6f}")

    with tab_chi2:
        st.markdown("#### Pearson Chi-Square Test of Independence")
        st.write("Test whether two categorical attributes are statistically independent.")
        c_x, c_y = st.columns(2)
        with c_x:
            cat1 = st.selectbox("Variable 1", ['Region', 'Category'], index=0)
        with c_y:
            cat2 = st.selectbox("Variable 2", ['Payment Mode', 'Shipping Mode'], index=0)

        res_chi = perform_chi_square_test(df, cat_col1=cat1, cat_col2=cat2)
        if res_chi['status'] == 'success':
            st.markdown(f"**Result:** {res_chi['conclusion']}")
            st.metric("Chi-Square Statistic", res_chi['chi2_statistic'])
            st.metric("p-Value", f"{res_chi['p_value']:.6f}")
            st.metric("Degrees of Freedom", res_chi['dof'])

def render_automated_eda(df: pd.DataFrame) -> None:
    """Render automated exploratory data analysis profiler."""
    st.markdown("## 🔍 Automated EDA & Data Quality Profiler")

    overview = generate_dataset_overview(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{overview['rows']:,}")
    c2.metric("Features", overview['columns'])
    c3.metric("Memory Usage", f"{overview['memory_mb']} MB")
    c4.metric("Missing Cell %", f"{overview['missing_percentage']}%")

    st.markdown("### 📊 Descriptive Statistics & Skewness Profiler")
    stats_df = compute_statistical_summary(df)
    st.dataframe(stats_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔥 Pearson Correlation Heatmap")
        fig_corr = plot_eda_correlation_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)
    with col2:
        st.markdown("### 📦 Outlier Diagnostics (Tukey IQR)")
        outliers_df = compute_outlier_summary(df)
        st.dataframe(outliers_df, use_container_width=True, hide_index=True)

    st.markdown("### 📈 Univariate Feature Inspector")
    numeric_options = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_options:
        selected_feat = st.selectbox("Select Numeric Feature for Distribution Analysis", numeric_options, index=1 if len(numeric_options) > 1 else 0)
        st.plotly_chart(plot_univariate_distribution(df, selected_feat), use_container_width=True)

def render_data_explorer_page(df: pd.DataFrame) -> None:
    """Render search and multi-format data export center."""
    st.markdown("## 📁 Data Explorer & Multi-Format Export Center")
    search_query = st.text_input("🔍 Search records across Customer, Product, Order ID, City, or State", placeholder="E.g., Laptop, ORD-00012, California...")
    render_data_table(df, search_term=search_query)

def render_business_insights_page(df: pd.DataFrame, rfm: pd.DataFrame) -> None:
    """Render automated business intelligence insights and executive strategy."""
    kpis = calculate_kpis(df)
    render_kpi_cards(kpis)

    st.markdown("### 💡 Automated Strategic Insights")
    insights = render_insights(df)
    for ins in insights:
        st.markdown(ins, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Executive Strategic Playbook")
    recommendations = [
        ('🏆', 'Champions Retention', 'Reward Champions with VIP priority fulfillment and early product releases to maximize high CLV.'),
        ('📧', 'Win-Back Funnels', 'Re-engage At-Risk accounts through automated anniversary email offers and personalized promotions.'),
        ('💰', 'Margin Optimization', 'Eliminate negative margin discounts exceeding 25% on low-margin furniture and appliances.'),
        ('🗺️', 'Geographic Expansion', 'Replicate high-margin marketing playbooks from West and East into underperforming territories.'),
        ('📦', 'Inventory Pre-allocation', 'Utilize 6-month sales trend forecasts to adjust vendor procurement lead times and avoid stockouts.'),
        ('💳', 'Payment Friction Reduction', 'Incentivize instantaneous digital payment rails (UPI, Credit Cards) which exhibit lower order abandonment.')
    ]

    cols = st.columns(2)
    for idx, (icon, title, desc) in enumerate(recommendations):
        with cols[idx % 2]:
            st.markdown(f'''
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                <div class="metric-title">{title}</div>
                <div style="color: #64748b; font-size: 0.88rem; margin-top: 8px; line-height: 1.5;">{desc}</div>
            </div>''', unsafe_allow_html=True)

    if not rfm.empty:
        st.markdown("### 📊 RFM Customer Segment Economics")
        rfm_econ = rfm.groupby('Customer Segment').agg(
            Customers=('Customer', 'count'),
            TotalRevenue=('Monetary', 'sum'),
            AvgOrderValue=('AOV', 'mean'),
            AvgRecencyDays=('Recency', 'mean'),
            AvgCLV=('Estimated_CLV', 'mean')
        ).sort_values('TotalRevenue', ascending=False)

        rfm_econ['TotalRevenue'] = rfm_econ['TotalRevenue'].apply(lambda x: f"${x:,.0f}")
        rfm_econ['AvgOrderValue'] = rfm_econ['AvgOrderValue'].apply(lambda x: f"${x:,.2f}")
        rfm_econ['AvgRecencyDays'] = rfm_econ['AvgRecencyDays'].apply(lambda x: f"{x:.0f} days")
        rfm_econ['AvgCLV'] = rfm_econ['AvgCLV'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(rfm_econ, use_container_width=True)

def main() -> None:
    """Main application lifecycle controller."""
    try:
        df = load_data()
        rfm = get_rfm_data(df)

        if not rfm.empty and 'Customer' in df.columns:
            df = df.merge(rfm[['Customer', 'Customer Segment']], on='Customer', how='left')

        filters = render_filters(df)
        filtered_df = apply_filters(df, filters)

        pages = [
            'Executive Overview',
            'Sales Analytics & Forecast',
            'Customer Intelligence (RFM)',
            'Product Profitability (Pareto)',
            'Regional Performance',
            'Statistical Testing (ANOVA / t-Test)',
            'Automated EDA Profiler',
            'Data Explorer & Export',
            'Business Strategy Playbook'
        ]
        page = st.sidebar.radio('Navigation Menu', pages)

        st.title(f"{APP_ICON} {APP_TITLE}")
        if 'date_range' in filters:
            s_date, e_date = filters['date_range']
            st.caption(f"**View:** {page} | **Span:** {s_date} to {e_date} | **Active Records:** {len(filtered_df):,} of {len(df):,}")

        if page == 'Executive Overview':
            render_executive_dashboard(filtered_df)
        elif page == 'Sales Analytics & Forecast':
            render_sales_analytics(filtered_df)
        elif page == 'Customer Intelligence (RFM)':
            render_customer_analytics(filtered_df, rfm)
        elif page == 'Product Profitability (Pareto)':
            render_product_analytics(filtered_df)
        elif page == 'Regional Performance':
            render_regional_analytics(filtered_df)
        elif page == 'Statistical Testing (ANOVA / t-Test)':
            render_statistical_testing(filtered_df)
        elif page == 'Automated EDA Profiler':
            render_automated_eda(filtered_df)
        elif page == 'Data Explorer & Export':
            render_data_explorer_page(filtered_df)
        elif page == 'Business Strategy Playbook':
            render_business_insights_page(filtered_df, rfm)

    except Exception as e:
        logger.error(f"Application runtime error: {e}", exc_info=True)
        st.error("An unexpected error occurred. Please review application logs.")

if __name__ == '__main__':
    main()
