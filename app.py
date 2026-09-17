import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

st.set_page_config(page_title="Stock Trend Prediction", layout="wide")

st.title("📈 Stock Price Trend Prediction using Stacked LSTM")
st.markdown("Interactive forecasting dashboard powered by Deep Sequential Neural Networks.")

# Sidebar Settings
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2016-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Fetch Data
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

with st.spinner(f"Fetching data for {ticker}..."):
    try:
        data = load_data(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

if data.empty:
    st.error("No historical data found. Please check ticker symbol.")
    st.stop()

st.subheader(f"Recent Data Summary ({ticker})")
st.dataframe(data.tail(5), use_container_width=True)

# Technical Indicators
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Indicator Plots
st.subheader("Technical Indicators (Moving Averages & RSI)")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(data.index, data['Close'], label='Close Price', color='black')
ax1.plot(data.index, data['MA50'], label='50 SMA', color='blue', linestyle='--')
ax1.plot(data.index, data['MA200'], label='200 SMA', color='red', linestyle='--')
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(data.index, data['RSI'], label='RSI (14)', color='purple')
ax2.axhline(70, linestyle='--', color='red', label='Overbought (70)')
ax2.axhline(30, linestyle='--', color='green', label='Oversold (30)')
ax2.set_ylabel("RSI")
ax2.legend()
ax2.grid(True, alpha=0.3)

st.pyplot(fig)

# Forecasting Section
st.subheader("LSTM Price Trend Forecasting")
if st.button("Generate Trend Forecast"):
    with st.spinner("Processing sequences and generating forecast..."):
        close_vals = data[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_vals = scaler.fit_transform(close_vals)
        
        LOOKBACK = 60
        if len(scaled_vals) <= LOOKBACK + 50:
            st.warning("Insufficient data points for 60-day window forecasting. Select an earlier start date.")
            st.stop()
            
        test_data = scaled_vals[-120:]
        X_test, y_test = [], []
        for i in range(LOOKBACK, len(test_data)):
            X_test.append(test_data[i - LOOKBACK:i, 0])
            y_test.append(test_data[i, 0])
            
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Build inference model architecture
        model = Sequential([
            Input(shape=(X_test.shape[1], 1)),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Fast fine-tune on recent history
        X_train_fast = []
        y_train_fast = []
        for i in range(LOOKBACK, len(scaled_vals) - 60):
            X_train_fast.append(scaled_vals[i - LOOKBACK:i, 0])
            y_train_fast.append(scaled_vals[i, 0])
        X_train_fast = np.reshape(np.array(X_train_fast), (len(X_train_fast), LOOKBACK, 1))
        y_train_fast = np.array(y_train_fast)
        
        model.fit(X_train_fast[-300:], y_train_fast[-300:], epochs=5, batch_size=32, verbose=0)
        
        preds_scaled = model.predict(X_test)
        preds = scaler.inverse_transform(preds_scaled)
        actual = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
        
        fig2, ax = plt.subplots(figsize=(10, 4))
        ax.plot(actual, color='black', label='Actual Price')
        ax.plot(preds, color='red', linestyle='--', label='LSTM Predicted Trend')
        ax.set_xlabel("Recent Trading Days")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)
        st.success("Trend forecasting completed successfully!")
