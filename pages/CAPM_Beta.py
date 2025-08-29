# pages/capm_beta.py
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.capm_functions import interactive_plot, normalize, daily_return, calculate_beta, calculate_capm

st.set_page_config(page_title="CAPM Beta Analysis", layout="wide")
st.title("📊 CAPM Beta Analysis")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Settings")
tickers = st.sidebar.multiselect(
    "Choose stocks (select 1+)",
    ('TSLA', 'AAPL', 'GOOGL', 'AMZN', 'NFLX', 'MSFT', 'MGM', 'NVDA', 'META', 'INTC'),
    default=['TSLA', 'AAPL', 'AMZN', 'GOOGL']
)

years = st.sidebar.number_input("Number of years back", min_value=1, max_value=10, value=2)

run = st.sidebar.button("Run Analysis")

if not run:
    st.info("Choose tickers and number of years, then click **Run Analysis**.")
    st.stop()

# ---------------- Build date range ----------------
try:
    end = datetime.date.today()
    start = datetime.date(end.year - int(years), end.month, end.day)
except Exception:
    st.error("Error computing date range. Try selecting fewer years or a different system date.")
    st.stop()

# Always ensure at least one ticker selected
if not tickers:
    st.error("Please select at least one ticker.")
    st.stop()

# ---------------- Download S&P 500 (prefer FRED, fallback to yfinance) ----------------
sp500_df = None
try:
    sp500_raw = web.DataReader(['sp500'], 'fred', start, end)
    # sp500_raw is a DataFrame with column 'sp500' and DatetimeIndex
    sp500_series = sp500_raw['sp500']
    sp500_df = sp500_series.reset_index()
    sp500_df.columns = ['Date', 'sp500']
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date']).dt.date
except Exception:
    # Fallback to yfinance ^GSPC
    st.warning("Could not fetch 'sp500' from FRED; falling back to yfinance '^GSPC'.")
    try:
        gspc = yf.download("^GSPC", start=start, end=end, progress=False)
        if gspc is None or gspc.empty:
            raise RuntimeError("yfinance returned empty for ^GSPC")
        spcol = 'Adj Close' if 'Adj Close' in gspc.columns else 'Close'
        sp500_df = gspc[[spcol]].reset_index().rename(columns={spcol: 'sp500'})
        sp500_df['Date'] = pd.to_datetime(sp500_df['Date']).dt.date
    except Exception as e:
        st.error(f"Failed to obtain market series for S&P500: {e}")
        st.stop()

# ---------------- Download each stock separately and assemble dataframe ----------------
stocks_df = pd.DataFrame()
failed_tickers = []
for t in tickers:
    try:
        df = yf.download(t, start=start, end=end, progress=False)
        if df is None or df.empty:
            failed_tickers.append(t)
            continue
        # prefer Adj Close but fall back to Close
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close' if 'Close' in df.columns else None
        if col is None:
            failed_tickers.append(t)
            continue
        series = df[col].copy()
        series.index = pd.to_datetime(series.index)
        series = series.resample('D').ffill()  # fill missing calendar days to help merge
        stocks_df[t] = series
    except Exception:
        failed_tickers.append(t)

if failed_tickers:
    st.warning(f"Some tickers had no data and were skipped: {', '.join(failed_tickers)}")

if stocks_df.empty:
    st.error("No valid stock time series were downloaded. Try different tickers or a different date range.")
    st.stop()

# reset index and prepare for merge (ensure 'Date' is first column)
stocks_df = stocks_df.reset_index().rename(columns={'index': 'Date'})
stocks_df['Date'] = pd.to_datetime(stocks_df['Date']).dt.date

# Ensure sp500_df has Date column and is date type
if 'Date' not in sp500_df.columns:
    st.error("Market (sp500) data returned in unexpected format.")
    st.stop()

# Merge on Date using inner join so we only keep dates where both exist
merged = pd.merge(stocks_df, sp500_df, on='Date', how='inner')

# Reorder columns: Date, stocks..., sp500
stock_columns = [c for c in merged.columns if c not in ['Date', 'sp500']]
merged = merged[['Date'] + stock_columns + ['sp500']]

# Show basic dataframe
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Dataframe head")
    st.dataframe(merged.head(), use_container_width=True)
with col2:
    st.markdown("### Dataframe tail")
    st.dataframe(merged.tail(), use_container_width=True)

# ---------------- Plot prices and normalized prices ----------------
st.markdown("### Price of selected stocks")
try:
    st.plotly_chart(interactive_plot(merged), use_container_width=True)
except Exception:
    # interactive_plot expects first column Date and numeric others; fall back to simple plotly
    st.line_chart(merged.set_index('Date').iloc[:, :-1])

st.markdown("### Normalized prices (start = 1)")
try:
    norm_df = normalize(merged)
    st.plotly_chart(interactive_plot(norm_df), use_container_width=True)
except Exception:
    st.line_chart(merged.set_index('Date').iloc[:, :-1].apply(lambda x: x / x.iloc[0]))

# ---------------- Calculate daily returns ----------------
try:
    stocks_daily_return = daily_return(merged)
except Exception as e:
    st.error(f"Failed to compute daily returns: {e}")
    st.stop()

# ---------------- Calculate Beta & CAPM ----------------
st.markdown("### Calculated Beta values (via linear regression vs market)")

beta_results = []
for stock in stock_columns:
    try:
        b, a = calculate_beta(stocks_daily_return, stock)
        beta_results.append({'Stock': stock, 'Beta': round(b, 4), 'Alpha': round(a, 6)})
    except Exception as e:
        beta_results.append({'Stock': stock, 'Beta': None, 'Alpha': None})

beta_df = pd.DataFrame(beta_results).sort_values(by='Beta', ascending=False)
st.dataframe(beta_df, use_container_width=True)

# ---------------- Beta bar chart ----------------
st.markdown("### Beta Comparison")
fig_bar = px.bar(
    beta_df.dropna(subset=['Beta']),
    x='Stock', y='Beta', text='Beta',
    color='Beta', color_continuous_scale='RdBu', title="Stock Betas vs Market"
)
fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_bar.update_layout(yaxis=dict(title="Beta Value"), uniformtext_minsize=8, uniformtext_mode='hide')
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- Scatter + regression (stock vs market returns) ----------------
st.markdown("### Scatter: Stock Returns vs Market Returns (with regression line)")
valid_stocks = [r['Stock'] for r in beta_results if r['Beta'] is not None]
if valid_stocks:
    selected_stock = st.selectbox("Choose a stock to inspect", valid_stocks)
    mask = merged['Date'].notna()
    x = stocks_daily_return['sp500']
    y = stocks_daily_return[selected_stock]
    valid_mask = x.notna() & y.notna()
    if valid_mask.sum() < 10:
        st.info("Not enough overlapping return observations to show scatter/regression.")
    else:
        slope, intercept = np.polyfit(x[valid_mask], y[valid_mask], 1)
        reg_y = intercept + slope * x[valid_mask]

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=x[valid_mask], y=y[valid_mask], mode='markers', name='Daily returns'))
        fig_sc.add_trace(go.Line(x=x[valid_mask], y=reg_y, name=f'Regression (beta={slope:.3f})'))
        fig_sc.update_layout(title=f"{selected_stock} vs Market Returns", xaxis_title="Market daily return (%)", yaxis_title=f"{selected_stock} daily return (%)")
        st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.info("No valid beta values to display scatter plot.")

# ---------------- Rolling Beta ----------------
st.markdown("### Rolling Beta (windowed estimation)")
rolling_window = st.slider("Rolling window (days)", min_value=20, max_value=252, value=60)
if len(stocks_daily_return) <= rolling_window:
    st.info(f"Not enough rows ({len(stocks_daily_return)}) for rolling beta with window {rolling_window}.")
else:
    # compute rolling beta for selected_stock (or the first valid)
    rstock = selected_stock if valid_stocks else stock_columns[0]
    rolling_betas = []
    dates = []
    for i in range(rolling_window, len(stocks_daily_return)):
        win = stocks_daily_return.iloc[i - rolling_window:i]
        try:
            b, _ = calculate_beta(win, rstock)
        except Exception:
            b = np.nan
        rolling_betas.append(b)
        dates.append(stocks_daily_return['Date'].iloc[i])

    rolling_df = pd.DataFrame({"Date": pd.to_datetime(dates), "Rolling Beta": rolling_betas})
    fig_roll = px.line(rolling_df, x='Date', y='Rolling Beta', title=f"{rstock} Rolling Beta ({rolling_window}-day)")
    fig_roll.add_hline(y=1, line_dash="dash", line_color="red")
    st.plotly_chart(fig_roll, use_container_width=True)

# ---------------- Interpretation ----------------
st.markdown("### Interpretation / Notes")
st.markdown(
    """
- **Beta > 1** : Stock historically amplified market moves (more volatile).
- **Beta < 1** : Stock historically muted market moves (less volatile).
- **Beta ≈ 1** : Stock moves with the market on average.
\n
**Notes & robustness:**
- Beta is a historical *statistical* measure (depends on chosen period and frequency).
- If some tickers returned no data they are skipped (you saw a warning above).
- The page computes returns in *percentage* (daily pct change × 100) to match your CAPM pipeline.
"""
)
