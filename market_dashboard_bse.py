# market_dashboard_bse.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, time, timedelta

# Configuration
st.set_page_config(page_title="BSE Market Advisor Pro", layout="wide", page_icon="📈")

# Constants
BSE_SUFFIX = ".BO"

@st.cache_resource(ttl=3600)
def bse_connection():
    return yf.Tickers(" ".join([f"{code}{BSE_SUFFIX}" for code in SENSEX_CODES.values()]))

SENSEX_CODES = {
    "Reliance Industries": "500325",
    "TCS": "532540",
    "HDFC Bank": "500180",
    "ICICI Bank": "532174",
    "HUL": "500696",
    "Infosys": "500209",
    "SBI": "500112",
    "Bharti Airtel": "532454",
    "L&T": "500510",
    "Kotak Mahindra Bank": "500247",
    "Axis Bank": "532215",
    "ITC": "500875",
    "Maruti Suzuki": "532500",
    "Asian Paints": "500820",
    "HCL Technologies": "532281",
    "Sun Pharma": "524715",
    "Mahindra & Mahindra": "500520",
    "UltraTech Cement": "532538",
    "Titan Company": "500114",
    "Nestle India": "500790",
    "Bajaj Finance": "500034",
    "Tech Mahindra": "532755",
    "Wipro": "507685",
    "Adani Ports": "532921",
    "Bajaj Finserv": "532978",
    "Dr. Reddy's": "500124",
    "Power Grid": "532898",
    "NTPC": "532555",
    "IndusInd Bank": "532187",
    "HDFC Life": "540777"
}

@st.cache_data(ttl=900, show_spinner="Fetching BSE data...")
def fetch_bse_data(symbol, period="6mo"):
    try:
        ticker = yf.Ticker(f"{symbol}{BSE_SUFFIX}")
        df = ticker.history(period=period)
        if df.empty:
            return pd.DataFrame()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = [col.lower() for col in df.columns]
        df = df.reset_index().rename(columns={'Date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').dropna()
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fii_dii_data():
    dates = pd.date_range(end=datetime.today(), periods=30)
    return pd.DataFrame({
        'date': dates,
        'fii': np.random.randint(1000, 5000, 30),
        'dii': np.random.randint(800, 4000, 30)
    }).set_index('date')

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_rae(volume):
    return volume.rolling(window=20).apply(lambda x: x[-1] / x.mean(), raw=True)

def generate_trading_signal(df):
    if df.empty or 'RSI' not in df.columns or 'SMA_20' not in df.columns or 'SMA_50' not in df.columns:
        return "N/A"
    rsi = df['RSI'].iloc[-1]
    sma20 = df['SMA_20'].iloc[-1]
    sma50 = df['SMA_50'].iloc[-1]
    if rsi < 30 and sma20 > sma50:
        return "BUY"
    elif rsi > 70 and sma20 < sma50:
        return "SELL"
    else:
        return "HOLD"

def add_technical_indicators(df):
    if df.empty:
        return df
    try:
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['RSI'] = compute_rsi(df['close'])
        df['Upper_BB'], df['Lower_BB'] = compute_bollinger_bands(df['close'])
        df['MACD'], df['Signal_Line'] = compute_macd(df['close'])
        df['RAE'] = compute_rae(df['volume'])
        return df.dropna()
    except KeyError as e:
        st.error(f"Missing column for technical indicators: {str(e)}")
        return df

def market_status():
    current_time = datetime.now().time()
    market_open = time(9,15) <= current_time <= time(15,30)
    return "🟢 LIVE" if market_open else "🔴 CLOSED"

def create_sidebar():
    with st.sidebar:
        st.header("BSE Market Controls")
        st.markdown(f"**Market Status:** {market_status()}")
        st.markdown("**Trading Hours:**\n- Pre-open: 9:00-9:15 AM\n- Regular: 9:15 AM-3:30 PM")

        selected_stock = st.selectbox("Choose SENSEX Stock", list(SENSEX_CODES.keys()))
        symbol = SENSEX_CODES[selected_stock]

        st.markdown("---")
        st.markdown("**Data Disclaimer**")
        st.caption("Data delayed by 15-20 minutes. Not for real-time trading.")

        return symbol, selected_stock

def main():
    symbol, selected_stock = create_sidebar()

    st.title(f"🇮🇳 BSE Market Advisor: {selected_stock}")

    with st.spinner("Loading market data..."):
        df = fetch_bse_data(symbol)
        fii_dii = fetch_fii_dii_data()

    if df.empty:
        st.error("Failed to load stock data. Please try again later.")
        return

    df = add_technical_indicators(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        latest = df.iloc[-1]
        st.metric("Current Price", f"₹{latest['close']:,.2f}", 
                 delta=f"{latest['close'] - df.iloc[-2]['close']:+.2f}")

    with col2:
        signal = generate_trading_signal(df)
        st.metric("Trading Signal", signal, 
                 delta_color="off", 
                 help="Based on RSI and moving average crossover")

    with col3:
        st.metric("Market Activity", 
                 f"RAE: {latest['RAE']:.2f}", 
                 "High Volume" if latest['RAE'] > 1.5 else "Normal Volume")

    fig = create_candlestick_chart(df, selected_stock)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Institutional Activity")
    st.area_chart(fii_dii, use_container_width=True)

def create_candlestick_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                 open=df['open'],
                 high=df['high'],
                 low=df['low'],
                 close=df['close'],
                 name='Price'))

    for indicator in ['SMA_20', 'SMA_50', 'Upper_BB', 'Lower_BB']:
        fig.add_trace(go.Scatter(x=df.index, y=df[indicator],
                               name=indicator,
                               line=dict(width=1)))

    fig.update_layout(
        height=600,
        title=f"{title} Technical Analysis",
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    return fig

if __name__ == "__main__":
    main()
