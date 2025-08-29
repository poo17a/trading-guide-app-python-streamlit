import os
import streamlit as st

# Page config
st.set_page_config(
    page_title="Trading App",
    page_icon="📉",
    layout="wide"
)

# Title & subtitle
st.title("Trading Guide App 📊")
st.subheader("Your one-stop platform for stock analysis, prediction, and risk assessment.")

# Image section
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "images", "professional_trading_dashboard.png")

# Display image
st.image(image_path, use_container_width=True)
st.markdown("### Explore our features below:")

# Services
st.markdown("## 📌 Our Services")

st.page_link("pages/stock_analysis.py", label="📈 Stock Analysis", icon="📊")
st.write("Analyze historical stock data with interactive charts, technical indicators, and performance metrics.")

st.page_link("pages/stock_prediction.py", label="🤖 Stock Prediction", icon="📉")
st.write("Forecast the next 30 days of stock closing prices using historical trends and AI-powered models.")

st.page_link("pages/CAPM_Return.py", label="📐 CAPM Return", icon="📏")
st.write("Calculate the expected return of assets using the Capital Asset Pricing Model (CAPM).")

st.page_link("pages/CAPM_Beta.py", label="📊 CAPM Beta", icon="📊")
st.write("Measure a stock's Beta (market risk) and estimate expected returns based on its volatility.")

# Footer
st.markdown("---")
st.caption("📅 Updated for 2025 | 🚀 Powered by Streamlit")

