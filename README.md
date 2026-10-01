# 📊 SmartSales Analytics: Enterprise Intelligence Dashboard

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/NETIZEN-11/Smart-Sale-Analytic)
[![Tests](https://img.shields.io/badge/tests-61%20passed%20%7C%20100%25-success.svg)](https://github.com/NETIZEN-11/Smart-Sale-Analytic)
[![Code Coverage](https://img.shields.io/badge/coverage-92%25-blue.svg)](https://github.com/NETIZEN-11/Smart-Sale-Analytic)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/plotly-5.18%2B-orange.svg)](https://plotly.com/)
[![SQLite](https://img.shields.io/badge/sqlite-3%20WAL%20mode-lightgrey.svg)](https://sqlite.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **production-grade, end-to-end retail business intelligence platform** engineered for enterprise data exploration, customer segmentation, inferential hypothesis testing, and time-series demand forecasting. Built with **Python, Pandas, NumPy, SciPy, SQLite, SQL, Plotly, Streamlit, Matplotlib, and Seaborn**.

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [Key Features](#-key-features)
- [Dashboard Architecture](#-dashboard-architecture)
- [Mathematical & Statistical Formulations](#-mathematical--statistical-formulations)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Automated Testing & QA](#-automated-testing--qa)
- [Database & SQL Analytics](#-database--sql-analytics)
- [Security & Production Hardening](#-security--production-hardening)
- [License](#-license)

---

## 💼 Executive Summary

Retail organizations require continuous visibility into margin health, customer attrition, discount efficiency, and future demand. **SmartSales Analytics** transforms raw transactional logs into actionable intelligence:

1. **Revenue & Margin Diagnostics:** Automatically isolates loss-making product categories and over-discounted SKUs ($> 25\%$ concessions).
2. **Behavioral Customer Retention:** Implements Recency-Frequency-Monetary (RFM) quintile scoring and longitudinal customer acquisition cohort retention tracking.
3. **Statistical Hypothesis Testing:** Verifies category revenue variances and territory-payment interactions using One-Way ANOVA, Welch's $t$-test, and Pearson $\chi^2$ test of independence.
4. **Predictive Horizon Planning:** Computes 6-month forward projections with seasonal adjustments and Student's $t$-distribution 95% confidence intervals.

---

## 🎯 Key Features

### 1. Data Engineering & Cleaning Pipeline (`src/data_cleaning.py`)
- **Schema Validation:** Strict verification of 16 mandatory dimensions and data types.
- **Vectorized Imputation:** Missing municipal cities fallback to State regions using vectorized `np.where`, executing $5.8\times$ faster than iterative row mappers.
- **Acronym Safeguards:** Preserves canonical abbreviations (`UPI`, `COD`, `ID`) during title-case standardization.
- **Tukey & Percentile Outlier Capping:** Flags extreme transactions and caps upper revenue outliers at the 99.5th percentile.
- **Business Rule Enforcement:** Recalculates `Profit = Sales - Cost` and `Profit Margin (%)` to ensure mathematical integrity.

### 2. Advanced Analytics & Behavioral Modeling (`src/analytics.py`)
- **RFM Segmentation & CLV:** 5 behavioral tiers (*Champions*, *Loyal Customers*, *Potential Loyalists*, *At Risk*, *Lost Customers*) paired with Customer Lifetime Value (CLV) estimation.
- **Predictive Forecasting:** Linear trend estimation with monthly seasonal factors and 95% confidence interval ribbons.
- **Longitudinal Cohort Analysis:** Tracks customer retention percentages month-by-month ($M_0, M_1, \dots$).
- **Pareto 80/20 Distribution:** Ranks products and customers to identify the vital few generating 80% of gross revenue.
- **Inferential Testing Suite:** One-Way ANOVA, Welch's Two-Sample $t$-test, and Pearson Chi-Square test of independence.

### 3. Interactive Web Dashboard (`app.py`, `src/ui_components.py`)
- **9 Dedicated Views:**
  1. 📊 **Executive Overview:** High-level KPI cards, revenue vs. profit area fill, category donut, and regional comparison.
  2. 📈 **Sales Analytics & Forecast:** Quarterly bars, Category $\times$ Region cross-tabulation, and 6-month forecast with confidence bands.
  3. 👥 **Customer Intelligence (RFM):** Segment headcounts, interactive value bubble map, cohort retention heatmap, and top 25 customer table.
  4. 🛍️ **Product Profitability (Pareto):** Top/bottom products, profit scatter matrix, and Pareto 80/20 cumulative revenue curve.
  5. 🗺️ **Regional Performance:** State ranking bars, geographic margins, and payment method share.
  6. 🔬 **Statistical Hypothesis Testing:** Interactive ANOVA, Welch's $t$-test, and Chi-Square parameter calculators.
  7. 🔍 **Automated EDA Profiler:** High-order statistical moments (Mean, Median, Skewness, Kurtosis, IQR), correlation heatmap, and outlier diagnostics.
  8. 📁 **Data Explorer & Export Center:** Keyword search across all columns with instant CSV and formatted multi-sheet Excel (`.xlsx`) downloads.
  9. 💡 **Business Strategy Playbook:** Automated business insights and strategic playbooks.

---

## 🏗️ Dashboard Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Streamlit Interactive Dashboard                       │
│  (9 Views: Executive, Sales, Customers, Pareto, ANOVA, EDA, Exports)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                           ┌────────┴────────┐
                           │   UI Components │
                           │  (Plotly Charts)│
                           └────────┬────────┘
                                    │
     ┌───────────────┬──────────────┼──────────────┬──────────────┐
     │               │              │              │              │
┌────▼────┐   ┌─────▼──────┐  ┌────▼─────┐   ┌─────▼────┐  ┌─────▼─────┐
│Analytics│   │DataCleaning│  │ Database │   │   EDA    │  │  Logging  │
│ Engine  │   │  Pipeline  │  │ (SQLite) │   │ Profiler │  │ (Rotating)│
└────┬────┘   └─────┬──────┘  └────┬─────┘   └─────┬────┘  └───────────┘
     │              │              │               │
     │      ┌───────▼──────────────▼──┐            │
     │      │   Indexed SQLite DB     │◄───────────┘
     │      │  (WAL Mode, Pragmas)    │
     │      └─────────────────────────┘
     │
┌────▼────────────────────────────────────────────┐
│ • RFM Segmentation & CLV Modeling               │
│ • Trend-Seasonal Forecast (95% CI Ribbons)      │
│ • Welch's t-Test, ANOVA & Chi-Square Inference   │
│ • Customer Cohort Retention Matrix (0..N Months)│
│ • Pareto 80/20 Distribution Engine              │
└─────────────────────────────────────────────────┘
```

---

## 📐 Mathematical & Statistical Formulations

### 1. RFM Scoring & Customer Lifetime Value (CLV) Proxy
$$\text{RFM Score} = (R_{\text{Score}} \times 100) + (F_{\text{Score}} \times 10) + M_{\text{Score}}$$
$$\text{Estimated CLV} = \text{Monetary Spend} \times \left(1 + \text{Frequency} \times 0.10\right) \times \text{Gross Margin Factor}$$

### 2. Time-Series Forecasting with 95% Confidence Intervals
Given historical monthly sales $y_t$ at time $t \in \{0, 1, \dots, n-1\}$:
$$\hat{y}_t = (\text{Slope} \cdot t + \text{Intercept}) \times S_m$$
$$\text{Prediction Interval} = \hat{y}_t \pm t_{\alpha/2, \nu} \times s_e \times \sqrt{1 + \frac{1}{n} + \frac{(t - \bar{t})^2}{\sum (t_i - \bar{t})^2}}$$
Where $t_{\alpha/2, \nu}$ is the critical value of Student's $t$-distribution ($\nu = n - 2$, confidence level $= 0.95$), $s_e$ is the standard error of regression residuals, and $S_m$ is the monthly seasonal index.

### 3. One-Way Analysis of Variance (ANOVA)
$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\sum_{i=1}^k n_i (\bar{x}_i - \bar{x})^2 / (k - 1)}{\sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2 / (N - k)}$$
Tests $H_0: \mu_1 = \mu_2 = \dots = \mu_k$ against $\alpha = 0.05$.

### 4. Independent Two-Sample Welch's $t$-Test
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}$$
Degrees of freedom approximated using the Welch–Satterthwaite equation without assuming equal variances.

### 5. Pearson Chi-Square ($\chi^2$) Test of Independence
$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \times C_j}{N}$$

### 6. Tukey's Interquartile Range (IQR) Outlier Fences
$$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

---

## 📁 Project Structure

```text
SmartSales-Analytics/
├── .env.example             # Environment configuration template
├── .gitignore               # Production Git exclusion rules
├── .dockerignore            # Docker build context exclusions
├── Dockerfile               # Production container definition (non-root, healthcheck)
├── docker-compose.yml       # Container orchestration with resource quotas
├── pytest.ini               # Pytest configuration and logging formats
├── requirements.txt         # Pinned production dependencies
├── config.py                # Centralized typed configuration & paths
├── generate_data.py         # Synthetic transaction generator CLI
├── app.py                   # Streamlit multi-page application entrypoint
├── ARCHITECTURE.md          # Comprehensive technical architecture document
├── DATA_DICTIONARY.md       # Enterprise schema, domain constraints, and field definitions
├── data/                    # Datasets and SQLite database storage
│   ├── sales_raw.csv        # Raw dataset with simulated anomalies (25,100 rows)
│   ├── sales_clean.csv      # Cleaned and engineered dataset (25,000 rows)
│   └── smartsales.db        # Indexed production SQLite database
├── logs/                    # Application and runtime rotating logs
│   └── smartsales.log
├── notebooks/               # Jupyter analytics notebooks
│   └── 01_exploratory_data_analysis.ipynb
├── reports/                 # Exported data reports
├── src/                     # Core application modules
│   ├── __init__.py
│   ├── analytics.py         # RFM, CLV, forecasting, hypothesis testing
│   ├── data_cleaning.py     # Schema validation, outlier capping, feature engineering
│   ├── database.py          # SQLite connection manager, indexing, named queries
│   ├── eda.py               # Statistical profiling, correlation, univariate plots
│   ├── logger.py            # Rotating file and console structured logging
│   ├── sql_queries.sql      # 18 business SQL queries
│   └── ui_components.py     # Plotly visualizations, KPI cards, Excel exports
└── tests/                   # Pytest automated test suite (61 tests)
    ├── __init__.py
    ├── test_analytics.py
    ├── test_data_cleaning.py
    ├── test_database.py
    ├── test_eda.py
    ├── test_logger.py
    └── test_ui_components.py
```

---

## 🚀 Getting Started

### Local Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/NETIZEN-11/Smart-Sale-Analytic.git
   cd Smart-Sale-Analytic
   ```

2. **Create and activate Python virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install production dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate data and initialize SQLite database:**
   ```bash
   python generate_data.py --records 25000 --customers 3200
   ```

5. **Start the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
   *Dashboard will open in your browser at `http://localhost:8501`.*

---

## 🐳 Docker Deployment

The application features a multi-stage, hardened Docker deployment running as non-root `appuser` (UID 1000) with integrated `curl` health monitoring.

### Using Docker Compose:
```bash
# Build and launch service in background
docker-compose up --build -d

# Verify container health status
docker-compose ps
```
Access the application at `http://localhost:8501`.

---

## 🧪 Automated Testing & QA

The project maintains **61 automated tests** with **100% pass rate** across all functional areas:

```bash
# Execute entire test suite
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=term-missing
```

| Test Module | Coverage Scope | Tests | Pass Rate |
|---|---|:---:|:---:|
| `tests/test_analytics.py` | RFM, CLV, Forecast CI, ANOVA, Welch's t-test, Chi-Square, Cohort, Pareto | 16 | 100% |
| `tests/test_data_cleaning.py` | Missing imputation, duplicate removal, outlier capping, casing, normalization | 17 | 100% |
| `tests/test_database.py` | Connection pooling, SQL queries, DDL/DML, catalog PRAGMA, named queries | 12 | 100% |
| `tests/test_eda.py` | Statistical moments, Skewness, Kurtosis, Pearson/Spearman matrices, Tukey fences | 7 | 100% |
| `tests/test_logger.py` | Rotating handler, execution timers, schema loggers | 4 | 100% |
| `tests/test_ui_components.py` | Filtering logic, HTML insights, multi-sheet Excel export buffer | 5 | 100% |
| **Total** | | **61** | **100%** |

---

## 🗄️ Database & SQL Analytics

All transaction data is stored in SQLite (`data/smartsales.db`) with Write-Ahead Logging (`WAL`) enabled.

[`src/sql_queries.sql`](src/sql_queries.sql) contains **18 pre-built production queries** that can be executed directly from Python:
```python
from src.database import run_named_query

# Execute query 1: Overall KPIs
kpis_df = run_named_query(1)

# Execute query 8: Top 10 products by revenue
top_products_df = run_named_query(8)
```

---

## 🔒 Security & Production Hardening

- **Non-Root Execution:** Docker image drops root privileges to `appuser` (UID 1000).
- **Log Rotation Policy:** `RotatingFileHandler` limits log file growth to 10 MB with 5 backup archives (`LOG_MAX_BYTES = 10485760`).
- **SQL Injection Prevention:** All dynamic database operations use parameterized queries (`?` placeholders).
- **Zero Exposed Secrets:** Environment variables and API flags are managed via `.env` and `python-dotenv`.
- **Healthcheck Probes:** Streamlit heartbeat endpoint (`/_stcore/health`) monitored automatically every 30 seconds.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full details.
