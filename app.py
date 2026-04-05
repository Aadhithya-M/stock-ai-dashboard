
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

st.title("📊 AI Stock Market Dashboard")
st.markdown("AI powered stock analysis and prediction")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Stock Settings")

stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "TSLA")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))

# ----------------------------
# Download Data
# ----------------------------
df = yf.download(stock_symbol, start=start_date)

# Fix MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()

if df.empty:
    st.error("No data found. Try another stock symbol.")
    st.stop()

# ----------------------------
# Dashboard Metrics
# ----------------------------
last_price = float(df["Close"].iloc[-1])
high_price = float(df["High"].max())
low_price = float(df["Low"].min())

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(last_price,2))
col2.metric("Highest Price", round(high_price,2))
col3.metric("Lowest Price", round(low_price,2))

# ----------------------------
# Candlestick Chart
# ----------------------------
st.subheader("🕯 Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)])

fig.update_layout(height=600, title=f"{stock_symbol} Stock Price")

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Closing Price Chart
# ----------------------------
st.subheader("📈 Closing Price Chart")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    mode="lines",
    name="Close Price"
))

fig2.update_layout(height=500)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Machine Learning Model
# ----------------------------
st.subheader("🤖 Machine Learning Model")

df_ml = df[["Close"]].copy()
df_ml["Prediction"] = df_ml["Close"].shift(-1)

X = np.array(df_ml.drop(["Prediction"], axis=1))[:-1]
y = np.array(df_ml["Prediction"])[:-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.write("Model Accuracy:", round(accuracy,4))

# ----------------------------
# 7 Day Prediction
# ----------------------------
st.subheader("📅 Next 7 Day Prediction")

current_price = float(df["Close"].iloc[-1])
predictions = []

for i in range(7):
    pred = model.predict(np.array([[current_price]]))
    next_price = float(pred[0])
    predictions.append(next_price)
    current_price = next_price

future_dates = pd.date_range(
    start=df.index[-1],
    periods=8,
    freq="D"
)[1:]

prediction_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": predictions
})

st.dataframe(prediction_df)

# ----------------------------
# Prediction Graph
# ----------------------------
st.subheader("📉 Prediction Graph")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    name="Actual Price"
))

fig3.add_trace(go.Scatter(
    x=future_dates,
    y=predictions,
    mode="lines+markers",
    name="Predicted Price"
))

fig3.update_layout(height=500, title="7 Day Stock Prediction")

st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Data Table
# ----------------------------
st.subheader("📋 Stock Data Table")
st.dataframe(df.tail(20))
