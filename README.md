# 📈 Trading Guide App

An interactive, multi‑page **Streamlit** application for end‑to‑end equity analysis: **CAPM Beta**, **CAPM Return**, **Stock Analysis**, and **Stock Price Prediction**. Built with Python, Plotly, and scikit‑learn, it fetches market data, cleans it, visualizes insights, and runs classical finance and ML workflows in a clean UI.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Problem Statement](#problem-statement)
4. [Solution Statement](#solution-statement)
5. [Objective](#objective)
6. [Dataset](#dataset)
7. [Data Cleaning & Preparation](#data-cleaning--preparation)
8. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
9. [Tools and Technologies](#tools-and-technologies)
10. [Methods](#methods)
11. [Key Insights](#key-insights)
12. [Dependencies & Installment](#dependencies--installation)
13. [Repository Structure](#repository-structure)
14. [Dashboard](#dashboard)
15. [How to Run the Project](#how-to-run-the-project)
16. [Results and Conclusion](#results-and-conclusion)
17. [Future Work](#future-work)
18. [Keywords](#keywords)

---

## Project Overview

The Trading Guide App is an interactive, end-to-end equity analysis and forecasting platform built with Python, Streamlit, Plotly, and machine learning libraries. It brings together classical finance concepts like CAPM (Capital Asset Pricing Model) with modern time-series forecasting models (ARIMA, scaled regression, etc.) to provide a unified environment for analyzing stocks.

The app is modular and multi-page that helps users explore stocks and understand risk/return characteristics via **CAPM**. It fetches real-time market data, cleans and transforms it, visualizes historical behavior and patterns through dynamic dashboards, and applies forecasting techniques for short-horizon predictions.

The Trading Guide App is designed for learners, analysts, and enthusiasts who want a single place to dive into markets, experiment with models, and turn raw data into actionable insights—all in an interactive, hands-on way.

---

## Features

- **Market Data Integration**  
  Fetch real-time and historical stock data using **Yahoo Finance (yfinance)**.

- **Visualization Dashboard**  
  Interactive candlestick charts, moving averages, RSI, MACD, Bollinger Bands, and custom Plotly charts.

- **Statistical Finance Tools**  
  Capital Asset Pricing Model (CAPM), Beta calculation, portfolio returns, and risk analysis.

- **Forecasting Models**  
  Linear Regression, ARIMA, and ML-based predictive models.

- **Custom Utilities**  
  Reusable functions for training, evaluation, and visualization under `utils/`.

- **Streamlit Web App**  
  User-friendly interface for real-time exploration and prediction.

 ---

## Problem Statement

In today’s financial markets, retail investors, students, and even professionals face a fragmented landscape when it comes to stock analysis and forecasting. Data often comes from multiple sources, visualization requires technical expertise, and applying financial models such as CAPM or machine learning techniques demands coding skills and familiarity with specialized libraries. As a result, learners struggle to connect theoretical finance concepts with practical, real-world applications, while analysts and enthusiasts lack an integrated, interactive tool that combines risk/return analysis, technical indicators, and forecasting in one place.

---

## Solution Statement 

The Trading Guide App solves this problem by offering a modular, interactive web application that unifies data collection, cleaning, visualization, financial modeling, and forecasting. Built with Python, Streamlit, Plotly, scikit-learn, and statsmodels, the app enables users to:

- Select one or multiple stock tickers and compare them against a market benchmark
- Inspect raw and normalized prices, returns, and correlations through intuitive visualizations
- Estimate Beta (both static and rolling) and expected return using CAPM
- Train simple machine learning models to generate near-term forecasts and evaluate performance

By combining classical finance models with modern machine learning techniques in an accessible, no-code environment, the Trading Guide App bridges the gap between theory and practice—empowering learners, analysts, and enthusiasts to turn raw data into actionable insights.

---

## Objective

- Provide a **guided workflow** for CAPM Beta and CAPM Return analysis
- Offer **intuitive visualizations** (normalized prices, scatter w/ regression, rolling Beta)
- Enable **baseline forecasting** with standard ML models and clear evaluation metrics
- Be **extensible**: utilities in `utils/` separate data, modeling, and plotting concerns

---

## Dataset

**Source**: By default, data are fetched programmatically from public market data providers—primarily via yfinance, with optional pandas_datareader adapters for indices or macro data.

**Entities**
- **Stocks**: User‑selected tickers (e.g., AAPL, MSFT, TCS.NS)  
- **Benchmark / Market Index**: User‑selected or predefined (e.g., `^GSPC` for S&P 500 or `^NSEI` for NIFTY 50)

**Typical Schema** (per ticker)
- `Date` (datetime index)
- `Open, High, Low, Close, Adj Close`
- `Volume`

**Frequency**: Daily bars (can be extended to weekly/monthly).  
**Lookback**: Configurable **number of years** on the CAPM pages; rolling windows are set inside page controls or defaults.

---

## Data Cleaning & Preparation

1. **Column Standardization**: Ensure `Date` index sorted ascending; enforce `datetime64` type.  
2. **Alignment**: Outer/inner join prices on **common trading days** across tickers + benchmark.  
3. **Missing Data**: Handle NA via forward fill for prices; **drop** rows still NA before computing returns.  
4. **Returns**: Compute **simple** or **log** returns from `Adj Close`.  
5. **Outliers** (optional): Winsorize extreme returns to reduce leverage on regression fits.  
6. **Normalization**: Scale each series to 100 at start date for cross‑series comparability.  
7. **Feature Engineering** (Prediction): Rolling means/EMAs, lags, momentum, RSI/MACD (if enabled), target horizon creation.

All steps are encapsulated in utility functions inside `utils/` for reuse across pages.

---

## Exploratory Data Analysis (EDA)

- **DataFrame preview**: `head()` and `tail()` for quick sanity checks
- **Price charts** (per‑ticker & combined)
- **Normalized prices** (indexed to 100)
- **Daily returns distribution** & summary stats (optional)
- **Correlation matrix** for multi‑stock selections (optional)
- **Rolling metrics**: rolling mean/volatility; **Rolling Beta** vs benchmark
- **Scatter: Stock Returns vs Market Returns** with regression line (Beta = slope)

---

## Tools and Technologies

- **Language**: Python 3.9+
- **Framework**: Streamlit (multi‑page app under `pages/`)
- **Data**: pandas, numpy, `yfinance` (or CSV input)
- **Modeling**: scikit‑learn (regression/forecast baselines), statsmodels (OLS for CAPM)
- **Visualization**: Plotly (interactive charts)

---

## Methods

### 1) CAPM Beta Estimation
- **Returns**: $r_i = \frac{P_t - P_{t-1}}{P_{t-1}}$ (or log returns)  
- **Regression**: $r_i = \alpha + \beta r_m + \varepsilon$ where $r_m$ is market returns.  
- **Interpretation**: $\beta > 1$ = more volatile than market; $\beta < 1$ = less volatile.

### 2) CAPM Expected Return
- **Formula**: $\mathbb{E}[r_i] = r_f + \beta_i (\mathbb{E}[r_m] - r_f)$  
- **Alpha**: $\alpha = r_i - (r_f + \beta (r_m - r_f))$ on realized samples.

### 3) Rolling Beta
- **Windowed** OLS over a sliding window (e.g., 60/90/120 trading days) to assess **Beta stability** across regimes.

### 4) Forecasting (Stock Prediction)
- **Preprocessing**: train/test split by time, scaling if needed, lag features/rolling statistics.  
- **Models**: Baselines via `utils/model_train.py` (e.g., Linear Regression, Random Forest; can be extended to ARIMA/LSTM).  
- **Evaluation**: RMSE, MAE, and $R^2$; plot **Actual vs Predicted**.

---

## Key Insights

- **Beta** is the slope of stock vs market returns; it summarizes **systematic risk**.
- **Rolling Beta** uncovers regime shifts—stable vs time‑varying risk exposure.
- **Normalized prices** reveal **relative performance** independent of scale.
- **CAPM return** provides a **theoretical expectation**; compare with realized returns to reason about **alpha**.
- **Forecasting** outputs are **baseline** guides—use with caution; always validate against out‑of‑sample data.

---

## Dependencies & Installation

This project integrates **market data collection, time-series modeling, statistical finance, and interactive dashboards** into a single Streamlit app.  

### Dependencies

#### Data & Numerics
- **pandas** – tabular data manipulation  
- **numpy** – numerical computing  
- **pandas-datareader** – alternate market data sources  

#### Market Data
- **yfinance** – primary market data provider  
- **pandas-datareader** *(optional)* – adapters for other APIs  

#### Web App
- **streamlit** – interactive web application framework  

#### Visualization
- **plotly.graph_objects** – rich interactive charts  
- **utils/plotly_figure.py** – centralized custom figure builders  

#### Modeling & Stats
- **scikit-learn** – preprocessing, estimators, evaluation metrics  
- **statsmodels** – OLS regression, ARIMA time-series models  

#### Time-Series & Technical Indicators
- **TA-Lib** *(optional, binary install required)* – technical analysis functions  

#### Utilities
- **datetime** – handling time ranges  
- **os** – filesystem and environment management  

#### Project Utilities (`utils/`)
- **capm_functions.py** — CAPM & Beta calculations using OLS and rolling windows  
- **model_train.py** — data prep, feature engineering, model training & evaluation  
- **plotly_figure.py** — shared Plotly figure builders and rendering helpers  

---

## Repository Structure
```
Trading-Guide-App/
├─ Trading_App.py                       # Entry script for Streamlit (or Home.py)
├─ pages/
│  ├─ __init__.py
│  ├─ CAPM_beta.py              # CAPM Beta analysis page
│  ├─ CAPM_return.py            # CAPM Return analysis page
│  ├─ stock_analysis.py         # EDA & technical analysis page
│  ├─ stock_prediction.py       # ML forecasting page
├─ utils/
│  ├─ capm_functions.py         # Beta/return calc, benchmark helpers
│  ├─ model_train.py            # Train/evaluate models, metrics
│  ├─ plotly_figure.py          # Centralized Plotly figure builders
├─ images/
│  └─ image/                   # Logos, screenshots
├─ requirements.txt             # Python dependencies
└─ README.md                    # This file
```

---

## Dashboard

### 1) Landing Page
- **Description** of the app and links to all modules: CAPM Beta, CAPM Return, Stock Analysis, Stock Prediction.

### 2) CAPM Beta Page
**Inputs**: Stock tickers (multi‑select), benchmark index, **number of years**, rolling window; **Run Analysis** button.  
**Outputs**:
- DataFrame **head** / **tail**
- **Price of selected stocks** (Plotly)
- **Normalized prices** (indexed to 100)
- **Calculated Beta values** via linear regression
- **Beta comparison**: bar/line across tickers vs market
- **Scatter**: Stock Returns vs Market Returns **with regression line**
- **Rolling Beta**: windowed estimation over time

### 3) CAPM Return Page
**Inputs**: Stock ticker, benchmark, **number of years**, risk‑free rate (fixed or input).  
**Outputs**:
- DataFrame **head** / **tail**
- **Price of all the stocks** (if multiple)  
- **Normalized prices**
- **Calculated Beta** (reused or recomputed)
- **Calculated Return using CAPM**
- **CAPM Results Table**: **Beta, Alpha, Expected Return**

### 4) Stock Analysis Page
**Inputs**: Stock ticker, start date, end date.  
**Outputs**:  
- Price charts (line/candlestick)
- Moving averages (SMA/EMA), daily returns, and volatility overlays

### 5) Stock Prediction Page
**Workflow**:
- Select ticker and lookback/forecast horizon
- Feature engineering (lags/rolling features)
- Train/test split and **model training** via `utils/model_train.py`
- **Actual vs Predicted** chart + **metrics** (RMSE, MAE, R²)

---

## How to Run the Project

### 1) Prerequisites
- Python **3.9+**
- Git

### 2) Setup

```bash
# Clone
git clone https://github.com/<your-username>/trading-guide-app.git
cd trading-guide-app

# (Optional) create & activate venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) API keys
cp .env.example .env  # add keys if your utils support external APIs
```

### 3) Run
```bash
# If your entry file is app.py
streamlit run app.py

# If you use Streamlit's default multipage pattern with Home.py
streamlit run Home.py
```
Then open the local URL printed in the terminal. Use the sidebar or landing links to navigate between pages.

---

## Results and Conclusion

- Implemented a **reproducible** CAPM workflow: **Beta** (static & rolling) and **expected return**.
- Delivered **interactive EDA** (normalized prices, scatter/regression, rolling metrics).
- Provided **baseline forecasting** with interpretable metrics and visual diagnostics.
- The modular `utils/` design makes it easy to extend data sources, models, and charts.

---

## Future Work

- Advanced forecasting (Prophet/ARIMA/LSTM), hyperparameter tuning
- Caching & data layer (parquet/SQLite) for faster reloads
- CI/CD and containerization for deployment (Docker, GH Actions)

---

## Keywords

`Python` · `Streamlit` · `Plotly` · `Pandas` · `yfinance` · `Numpy` · `scikit-learn` · `Time Series Forecasting` · `ARIMA` · `Linear Regression` · `CAPM` · `Beta Analysis` · `Expected Return` · `Stock Price Prediction` · `Data Visualization` · `Finance Analystics` . `Interactive Dashboard` . `Talib` 

