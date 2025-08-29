import plotly.graph_objects as go
import pandas as pd
import talib as ta

# ====== TABLE FUNCTION ======
def plotly_table(df, header_color='darkblue', row_color='lightgrey'):
    # Ensure DataFrame has proper column names
    df = df.copy()
    # If the first column has empty name, fix it
    if df.columns.size > 0 and df.columns[0] == '':
        df.columns = ["Metric", "Value"]

    # If index is meaningful (dates or labels), reset it so it's shown as a column
    if df.index.name or df.index.dtype == 'O' or pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.reset_index()

    # Convert date columns to string format for display
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d')

    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color=header_color,
                font=dict(color='white', size=12),
                align='left',
                height=28
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color=[row_color, 'white'],
                align='left',
                height=24
            )
        )]
    )

    # Auto adjust height to avoid large white space
    fig.update_layout(
        height=min(420, 40 + len(df) * 28),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def filter_data_by_period(data, period):
    if data is None or data.empty:
        return data
    p = str(period).lower()
    if p == "max" or p == "all":
        return data
    if p == "ytd":
        start_of_year = pd.Timestamp.today().replace(month=1, day=1)
        index_naive = data.index.tz_localize(None) if data.index.tz is not None else data.index
        return data.loc[index_naive >= start_of_year]
    # mapping shortcuts
    period_map = {
        "1mo": "1M",
        "3mo": "3M",
        "6mo": "6M",
        "1y": "1Y",
        "2y": "2Y",
        "5y": "5Y",
        "5d": "5D"
    }
    if p in period_map:
        return data.last(period_map[p])
    # if passed a pandas-friendly string like '6mo', attempt to use .last()
    try:
        return data.last(period)
    except Exception:
        return data

# ====== CHART FUNCTIONS ======
def candlestick(data, period):
    df = filter_data_by_period(data, period)
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=f"Candlestick Chart - {period}", xaxis_rangeslider_visible=False)
    return fig

def close_chart(data, period):
    df = filter_data_by_period(data, period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.update_layout(title=f"Close Price - {period}")
    return fig

def RSI(data, period):
    df = filter_data_by_period(data, period)
    rsi = ta.RSI(df['Close'], timeperiod=14)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title=f"RSI - {period}")
    return fig

def MACD(data, period):
    df = filter_data_by_period(data, period)
    macd, signal, hist = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal'))
    fig.add_trace(go.Bar(x=df.index, y=hist, name='Histogram'))
    fig.update_layout(title=f"MACD - {period}")
    return fig

def Moving_average(data, period):
    df = filter_data_by_period(data, period).copy()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA 20'))
    fig.update_layout(title=f"Moving Average - {period}")
    return fig

# ====== Forecast plot helper ======
def plot_forecast(historical: pd.Series, forecast: pd.DataFrame, rolling: pd.Series = None):
    """
    Plot historical close as one color and forecast Close as another color.
    Adds a vertical separator line between last historical day and first forecast day,
    and shades the forecast area.
    - historical: pd.Series of Close prices indexed by date
    - forecast: pd.DataFrame with a 'Close' column indexed by forecast dates
    - rolling: optional pd.Series (rolling mean) to overlay
    """
    hist = pd.Series(historical).dropna().rename("Close")
    fc = forecast.copy()
    if 'Close' not in fc.columns and fc.shape[1] >= 1:
        fc.columns = ['Close']

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, mode='lines', name='Historical Close',
        line=dict(width=2)
    ))

    # Rolling (if provided)
    if rolling is not None and not rolling.empty:
        r = pd.Series(rolling).dropna()
        fig.add_trace(go.Scatter(
            x=r.index, y=r.values, mode='lines', name='Rolling Mean (7d)',
            line=dict(width=1, dash='dash')
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc['Close'], mode='lines+markers', name='Forecast (30 days)',
        line=dict(width=2, dash='dot')
    ))

    # Shade forecast area
    fig.add_vrect(
        x0=fc.index.min(), x1=fc.index.max(),
        fillcolor="LightSalmon", opacity=0.08, layer="below", line_width=0
    )

    # Vertical separator between history and forecast
    sep_x = hist.index.max() + pd.Timedelta(days=0.5)
    fig.add_vline(x=sep_x, line=dict(color="gray", dash="dash"), annotation_text="Forecast starts", annotation_position="top left")

    fig.update_layout(
        title="Historical Close and 30-day Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
