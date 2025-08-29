# imports
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt

# Import from your updated plotly_figure.py
from utils.plotly_figure import (
    plotly_table,
    candlestick,
    close_chart,
    RSI,
    MACD,
    Moving_average
)

# Setting page config
st.set_page_config(
    page_title="Stock Analysis",
    page_icon=":page_with_curl:",
    layout="wide"
)

st.title("Stock Analysis")

# Ticker and date inputs
col1, col2, col3 = st.columns(3)

today = dt.date.today()

with col1:
    ticker = st.text_input("Stock Ticker", "TSLA")
with col2:
    start_date = st.date_input("Choose Start Date", dt.date(today.year - 1, today.month, today.day))
with col3:
    end_date = st.date_input("Choose End Date", today)

# Display basic company info (with safe lookups)
st.subheader(ticker)
stock = yf.Ticker(ticker)
info = stock.info
st.write(info.get('longBusinessSummary', 'No summary available'))
st.write("**Sector:**", info.get('sector', 'N/A'))
st.write("**Full Time Employees:**", info.get('fullTimeEmployees', 'N/A'))
st.write("**Website:**", info.get('website', 'N/A'))

# Create two columns for financial metrics
col1, col2 = st.columns(2)

with col1:
    df = pd.DataFrame({
        'Metric': ['Market Cap', 'Beta', 'EPS', 'PE Ratio'],
        'Value': [
            info.get('marketCap', 'N/A'),
            info.get('beta', 'N/A'),
            info.get('trailingEps', 'N/A'),
            info.get('trailingPE', 'N/A')
        ]
    })
    fig_df = plotly_table(df)
    st.plotly_chart(fig_df, use_container_width=True)

with col2:
    df = pd.DataFrame({
        'Metric': [
            'Quick Ratio',
            'Revenue per share',
            'Profit Margins',
            'Debt to Equity',
            'Return on Equity'
        ],
        'Value': [
            info.get('quickRatio', 'N/A'),
            info.get('revenuePerShare', 'N/A'),
            info.get('profitMargins', 'N/A'),
            info.get('debtToEquity', 'N/A'),
            info.get('returnOnEquity', 'N/A')
        ]
    })
    fig_df = plotly_table(df)
    st.plotly_chart(fig_df, use_container_width=True)

# Download historical stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Metrics row
col1, col2, col3 = st.columns(3)

last_close = float(data['Close'].iloc[-1])
prev_close = float(data['Close'].iloc[-2])
daily_change = last_close - prev_close
daily_change_percent = (daily_change / prev_close) * 100

col1.metric(
    label="Daily Change",
    value=f"{last_close:.2f} USD",
    delta=f"{daily_change:.2f} ({daily_change_percent:.2f}%)"
)

# Last 10 days table (date only in index)
last_10_df = data.tail(10).copy()
last_10_df.index = last_10_df.index.date
last_10_df = last_10_df.round(3).reset_index().rename(columns={'index': 'Date'})
fig_df = plotly_table(last_10_df)

st.write('### Historical Data (Last 10 Days)')
st.plotly_chart(fig_df, use_container_width=True)

# Row 1: Period selection buttons
cols = st.columns([1] * 7)  # 7 buttons
num_period = ''

period_buttons = [
    ('5D', '5d'),
    ('1M', '1mo'),
    ('6M', '6mo'),
    ('YTD', 'ytd'),
    ('1Y', '1y'),
    ('5Y', '5y'),
    ('MAX', 'max')
]

for idx, (label, value) in enumerate(period_buttons):
    with cols[idx]:
        if st.button(label):
            num_period = value

# Row 2: Chart type and indicator selection
col_chart_type, col_indicator, _ = st.columns([1, 1, 4])

with col_chart_type:
    chart_type = st.selectbox('', ('Candle', 'Line'))

with col_indicator:
    if chart_type == 'Candle':
        indicators = st.selectbox('', ('RSI', 'MACD'))
    else:
        indicators = st.selectbox('', ('RSI', 'Moving Average', 'MACD'))

ticker = yf.Ticker(ticker)
new_df1 = ticker.history(period = 'max')
data1 = ticker.history(period = 'max')
if num_period == '':

    if chart_type == 'Candle' and indicators == 'RSI':
        st.plotly_chart(candlestick(data1, '1y'), use_container_width=True)
        st.plotly_chart(RSI(data1, '1y'), use_container_width=True)

    if chart_type == 'Candle' and indicators == 'MACD':
        st.plotly_chart(candlestick(data1, '1y'), use_container_width=True)
        st.plotly_chart(MACD(data1, '1y'), use_container_width=True)

    if chart_type == 'Line' and indicators == 'RSI':
        st.plotly_chart(close_chart(data1, '1y'), use_container_width=True)
        st.plotly_chart(RSI(data1, '1y'), use_container_width=True)

    if chart_type == 'Line' and indicators == 'MACD':
        st.plotly_chart(close_chart(data1, '1y'), use_container_width=True)
        st.plotly_chart(MACD(data1, '1y'), use_container_width=True)

    if chart_type == 'Line' and indicators == 'Moving Average':
        st.plotly_chart(Moving_average(data1, '1y'), use_container_width=True)
        
else:
    if chart_type == 'Candle' and indicators == 'MACD':
        st.plotly_chart(candlestick(new_df1, num_period), use_container_width=True)
        st.plotly_chart(MACD(new_df1, num_period), use_container_width=True)

    if chart_type == 'Candle' and indicators == 'RSI':
        st.plotly_chart(candlestick(new_df1, num_period), use_container_width=True)
        st.plotly_chart(RSI(new_df1, num_period), use_container_width=True)

    if chart_type == 'Line' and indicators == 'MACD':
        st.plotly_chart(close_chart(new_df1, num_period), use_container_width=True)
        st.plotly_chart(MACD(new_df1, num_period), use_container_width=True)

    if chart_type == 'Line' and indicators == 'RSI':
        st.plotly_chart(close_chart(new_df1, num_period), use_container_width=True)
        st.plotly_chart(RSI(new_df1, num_period), use_container_width=True)

    if chart_type == 'Line' and indicators == 'Moving Average':
        st.plotly_chart(Moving_average(new_df1, num_period), use_container_width=True)

    if chart_type == 'Candle' and indicators == 'Moving Average':
        st.plotly_chart(candlestick(new_df1, num_period), use_container_width=True)
        st.plotly_chart(Moving_average(new_df1, num_period), use_container_width=True)

