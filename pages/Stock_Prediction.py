import streamlit as st
import pandas as pd
from utils.model_train import get_data, get_rolling_mean, get_differencing_order, scaling, evaluate_model, get_forecast, inverse_scaling
from utils.plotly_figure import plotly_table
import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("📊 Stock Prediction")

col1, _, _ = st.columns(3)
with col1:
    ticker = st.text_input('Stock Ticker', 'AAPL')

st.subheader(f"Predicting Next 30 Days Close Price for: {ticker}")

# Step 1: Get and process data
close_price = get_data(ticker)  # Series
rolling_price = get_rolling_mean(close_price)

# Step 2: Scale data
differencing_order = get_differencing_order(rolling_price)
scaled_data, scaler = scaling(rolling_price)

# Step 3: Evaluate model
rmse = evaluate_model(scaled_data, differencing_order)
st.markdown(f"<p style='font-size:14px;'><b>Model RMSE Score:</b> {rmse}</p>", unsafe_allow_html=True)

# Step 4: Forecast
forecast = get_forecast(scaled_data, differencing_order)
forecast['Close'] = inverse_scaling(scaler, forecast['Close']).flatten()

# Step 5: Show forecast table
st.write('##### Forecast Data (Next 30 Business Days)')
fig_tail = plotly_table(forecast.round(2))
fig_tail.update_layout(height=240)
st.plotly_chart(fig_tail, use_container_width=True)

# Step 6: Merge for plotting
full_data = pd.concat([rolling_price, forecast['Close']])
full_data.index.name = "Date"

# Step 7: Plot past vs future close prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=rolling_price.index, y=rolling_price.values, mode='lines', name='Past Close Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Close'], mode='lines', name='Forecast Close Price', line=dict(color='orange', dash='dot')))
fig.update_layout(
    title=f"{ticker} Close Price & 30-Day Forecast",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)
