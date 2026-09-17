# 📈 Stock Price Trend Prediction using Stacked LSTM

An end-to-end deep learning pipeline that forecasts equity price trends using a 2-layer Stacked Long Short-Term Memory (LSTM) recurrent neural network with Dropout regularization, deployed as an interactive web dashboard on Streamlit Cloud.

---

## 🚀 Live Demo & Artifacts
* **Streamlit Web Application:** [Live Forecasting Dashboard](https://kqz4ge4a9o4.streamlit.app)
* **Dataset:** Yahoo Finance Historical Market Data (AAPL, 2015–2024)

---

## 🛠️ System Architecture & Workflow

1. **Data Ingestion & Feature Engineering:**
   * Extraction of daily Open, High, Low, Close, Volume data via Yahoo Finance.
   * Rolling calculations for 50-day SMA, 200-day SMA, and 14-day Relative Strength Index (RSI).
2. **Preprocessing & Tensor Formatting:**
   * `MinMaxScaler` feature scaling $[0, 1]$ to stabilize recurrent gradient descent.
   * 60-day sliding lookback windows to capture temporal sequences.
   * Chronological 80/20 train-test split shaped into 3D tensors: `[samples, time_steps, features]`.
3. **Deep Learning Model:**
   * Layer 1: LSTM (50 units, `return_sequences=True`) + Dropout (0.2).
   * Layer 2: LSTM (50 units, `return_sequences=False`) + Dropout (0.2).
   * Fully Connected: Dense (25 units) $\rightarrow$ Dense (1 unit).
   * Optimizer: Adam | Loss Function: Mean Squared Error (MSE).
4. **Interactive Dashboard:**
   * Streamlit frontend supporting live ticker selection, historical technical chart rendering, and model inference.

---

## 📊 Evaluation & Performance

| Metric | Result |
| :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **2.51 USD** |
| **Mean Absolute Error (MAE)** | **1.98 USD** |
| **Final Validation Loss** | **0.0031** |

---

## 💻 Local Setup & Installation

```bash
git clone [https://github.com/mohdrehan0625/Stock-Price-Trend-Prediction-LSTM.git](https://github.com/mohdrehan0625/Stock-Price-Trend-Prediction-LSTM.git)
cd Stock-Price-Trend-Prediction-LSTM
pip install -r requirements.txt
streamlit run app.py
