import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

def get_data(ticker):
    """
    Downloads 2 years of stock data for the given ticker and returns the Close price as a Series.
    """
    df = yf.download(ticker, period="2y")
    return df["Close"]  # Return Series

def get_rolling_mean(close_price, window=7):
    """
    Calculates rolling mean for the provided close price.
    Handles Series, DataFrame, or ndarray inputs.
    """
    if isinstance(close_price, np.ndarray):
        close_price = pd.Series(close_price.flatten())
    elif isinstance(close_price, pd.DataFrame):
        close_price = close_price.squeeze()
    return close_price.rolling(window=window).mean().dropna()

def scaling(series):
    """
    Scales a Series using MinMaxScaler and returns the scaled array + scaler object.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    return scaled, scaler

def inverse_scaling(scaler, scaled_data):
    """
    Inverse transforms scaled data back to the original scale.
    """
    return scaler.inverse_transform(np.array(scaled_data).reshape(-1, 1))

def get_differencing_order(series):
    """
    Returns differencing order for ARIMA. Here fixed at 1 for simplicity.
    """
    return 1  # Can be tuned or made dynamic later

def evaluate_model(scaled_data, diff_order):
    """
    Fits ARIMA model and returns RMSE score.
    """
    model = ARIMA(scaled_data, order=(5, diff_order, 0))
    model_fit = model.fit()
    return round(np.sqrt(model_fit.mse), 3)

def get_forecast(scaled_data, diff_order, steps=30):
    """
    Forecasts 'steps' future values using ARIMA and returns a DataFrame with Date index.
    """
    model = ARIMA(scaled_data, order=(5, diff_order, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    
    # Create DataFrame with business day index starting from tomorrow
    forecast_df = pd.DataFrame(forecast, columns=["Close"])
    forecast_df.index = pd.date_range(
        start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
        periods=steps,
        freq="B"  # Business days only
    )
    return forecast_df
