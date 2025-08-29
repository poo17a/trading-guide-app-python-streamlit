# Importing necessary libraries
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_datareader.data as web
import datetime
from utils import capm_functions

st.set_page_config(page_title="CAPM Return Calculator", 
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")

st.title("Capital Asset Pricing Model (CAPM) Return Calculator")

# Getting input from user
col1, col2 = st.columns([1,1])
with col1:
    stocks_list = st.multiselect(
        "Choose Four Stocks",
        ('TSLA', 'AAPL', 'GOOGL', 'AMZN', 'NFLX', 'MSFT', 'MGM', 'NVDA', 'META', 'INTC'),
        ['TSLA', 'AAPL', 'AMZN', 'GOOGL']
    ) 
with col2:
    year = st.number_input("Number of Years", 1, 10)  

# Downloading data safely
try:
    end = datetime.date.today()
    start = datetime.date(end.year - year, end.month, end.day)

    SP500 = web.DataReader(['sp500'], 'fred', start, end)
    stocks_df = pd.DataFrame()

    for stock in stocks_list:
        data = yf.download(stock, period=f'{year}y')
        stocks_df[stock] = data['Close']

    stocks_df.reset_index(inplace=True)
    SP500.reset_index(inplace=True)
    SP500.columns = ['Date', 'sp500']
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    SP500['Date'] = pd.to_datetime(SP500['Date'])
    stocks_df['Date'] = stocks_df['Date'].dt.date
    SP500['Date'] = SP500['Date'].dt.date
    stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

# Display head & tail of the dataframe
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Dataframe head")
    st.dataframe(stocks_df.head(), use_container_width=True)
with col2:
    st.markdown("### Dataframe tail")
    st.dataframe(stocks_df.tail(), use_container_width=True)

# Plot stock prices
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Price of all the Stocks")
    st.plotly_chart(capm_functions.interactive_plot(stocks_df), use_container_width=True)
with col2:
    st.markdown("### Price of all the Stocks (After Normalizing)")
    st.plotly_chart(capm_functions.interactive_plot(capm_functions.normalize(stocks_df)), use_container_width=True)

# Calculate daily returns
stocks_daily_return = capm_functions.daily_return(stocks_df)

# Calculate Beta values
beta = {}
try:
    for stock in stocks_daily_return.columns[1:]:
        covariance = stocks_daily_return[stock].cov(stocks_daily_return['sp500'])
        variance = stocks_daily_return['sp500'].var()
        beta[stock] = covariance / variance
except Exception as e:
    st.error(f"Error calculating Beta: {e}")
    st.stop()

# Create a dataframe for Beta values
beta_df = pd.DataFrame(list(beta.items()), columns=['Stock', 'Beta Value'])
beta_df['Beta Value'] = beta_df['Beta Value'].round(2)

with col1:
    st.markdown("### Calculated Beta Value")
    st.dataframe(beta_df, use_container_width=True)

# CAPM return calculation
rf = 0  # Risk-free rate
rm = stocks_daily_return['sp500'].mean() * 252  # Annualized market return

return_value = [round(rf + (b * (rm - rf)), 2) for b in beta.values()]

return_df = pd.DataFrame({
    'Stock': list(beta.keys()),
    'Return Value': return_value
})

with col2:
    st.markdown("### Calculated Return using CAPM")
    st.dataframe(return_df, use_container_width=True)

# CAPM calculation (Beta, Alpha, and Expected Returns)
capm_results = capm_functions.calculate_capm(stocks_daily_return, risk_free_rate=0.04)  # Example: 4% RF rate

# Display results
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### CAPM Results (Beta, Alpha, Return)")
    st.dataframe(capm_results, use_container_width=True)
