import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Choose stock
stock = "AAPL"

# Download stock data
data = yf.download(stock, start="2020-01-01", end="2024-01-01")

# Use closing price
data = data[['Close']]

# Create prediction column
data['Prediction'] = data['Close'].shift(-1)

# Prepare dataset
X = np.array(data.drop(['Prediction'], axis=1))[:-1]
y = np.array(data['Prediction'])[:-1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Predict next day
last_price = data['Close'].iloc[-1].values[0]
next_day = model.predict([[last_price]])
print("Last Price:", last_price)
print("Predicted Next Day Price:", next_day[0])

# Graph
plt.figure(figsize=(10,5))
plt.plot(y_test)
plt.plot(predictions)
plt.legend(["Actual Price","Predicted Price"])
plt.title("Stock Price Prediction")
plt.show()