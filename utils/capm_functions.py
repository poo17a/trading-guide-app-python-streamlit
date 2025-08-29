import plotly.graph_objects as go
import numpy as np
import pandas as pd

def interactive_plot(df):
    """
    Creates an interactive line chart for stock prices.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'Date' column and stock price columns.
    
    Returns:
        plotly.graph_objs._figure.Figure: Interactive Plotly figure.
    """
    try:
        if 'Date' not in df.columns:
            raise ValueError("DataFrame must contain a 'Date' column.")
        
        fig = go.Figure()
        for col in df.columns[1:]:
            fig.add_scatter(x=df['Date'], y=df[col], name=col)
        
        fig.update_layout(
            width=450,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig
    except Exception as e:
        raise RuntimeError(f"Error in interactive_plot: {e}")


def normalize(df):
    """
    Normalizes stock prices based on their initial value.
    
    Parameters:
        df (pd.DataFrame): DataFrame with stock price columns (first column should be 'Date').
    
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    try:
        df = df.copy()
        for col in df.columns[1:]:
            df[col] = df[col] / df[col].iloc[0]
        return df
    except Exception as e:
        raise RuntimeError(f"Error in normalize: {e}")


def daily_return(df):
    """
    Calculates daily percentage returns for each stock.
    
    Parameters:
        df (pd.DataFrame): DataFrame with stock price columns.
    
    Returns:
        pd.DataFrame: DataFrame with daily returns in percentage.
    """
    try:
        df_daily_return = df.copy()
        for col in df.columns[1:]:
            df_daily_return[col] = df[col].pct_change().fillna(0) * 100
        return df_daily_return
    except Exception as e:
        raise RuntimeError(f"Error in daily_return: {e}")


def calculate_beta(stocks_daily_return, stock):
    """
    Calculates Beta and Alpha for a given stock using linear regression.
    
    Parameters:
        stocks_daily_return (pd.DataFrame): DataFrame with daily returns (must include 'sp500').
        stock (str): Stock column name.
    
    Returns:
        tuple: (beta, alpha)
    """
    try:
        if 'sp500' not in stocks_daily_return.columns:
            raise ValueError("DataFrame must contain 'sp500' column for market returns.")
        
        b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[stock], 1)
        return b, a
    except Exception as e:
        raise RuntimeError(f"Error in calculate_beta for {stock}: {e}")

def calculate_capm(stocks_daily_return, risk_free_rate=0.0):
    """
    Calculates Beta, Alpha, and CAPM expected return for all stocks in the DataFrame.
    
    Parameters:
        stocks_daily_return (pd.DataFrame): Daily returns DataFrame (must include 'sp500').
        risk_free_rate (float): Risk-free rate (default = 0).
    
    Returns:
        pd.DataFrame: DataFrame with columns [Stock, Beta, Alpha, CAPM_Return].
    """
    try:
        if 'sp500' not in stocks_daily_return.columns:
            raise ValueError("DataFrame must contain 'sp500' column for market returns.")
        
        results = []
        market_return = stocks_daily_return['sp500'].mean() * 252  # Annualized market return
        
        for stock in stocks_daily_return.columns[1:]:  # Exclude Date column if present
            beta, alpha = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[stock], 1)
            capm_return = risk_free_rate + beta * (market_return - risk_free_rate)
            results.append({
                "Stock": stock,
                "Beta": round(beta, 2),
                "Alpha": round(alpha, 2),
                "CAPM_Return": round(capm_return, 2)
            })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        raise RuntimeError(f"Error in calculate_capm: {e}")
