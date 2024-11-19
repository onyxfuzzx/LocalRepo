import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import yfinance as yf
from datetime import datetime, timedelta
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# UTF-8 encoding for the output
sys.stdout.reconfigure(encoding='utf-8')

# Download data using yfinance directly
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data found for the given ticker and date range.")
    return data

# Preprocess the data
def preprocess_data(data):
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create datasets with a specified time step
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Make predictions
def make_predictions(model, X_train, X_test, scaler):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    return train_predict, test_predict

# Simulate future predictions
def simulate_future_predictions(model, scaled_data, scaler, future_days, time_step=60):
    future_dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=future_days, freq='B')
    last_60_days = scaled_data[-time_step:]
    last_60_days_scaled = last_60_days.reshape(1, last_60_days.shape[0], 1)

    future_predictions = []
    for _ in range(future_days):
        prediction = model.predict(last_60_days_scaled)
        future_prediction = scaler.inverse_transform(prediction)
        future_predictions.append(future_prediction[0, 0])

        prediction_reshaped = prediction.reshape(1, 1, 1)
        last_60_days_scaled = np.append(last_60_days_scaled[:, 1:, :], prediction_reshaped, axis=1)

    future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Future Predicted Stock Price'])
    return future_df

# Plot results
def plot_results(data, train_predict, test_predict, time_step):
    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data['Close'], label='Actual Stock Price')
    train_dates = data.index[time_step:time_step + len(train_predict)]
    plt.plot(train_dates, train_predict, label='Train Predicted Stock Price')
    test_dates = data.index[time_step + len(train_predict) + 1:time_step + len(train_predict) + 1 + len(test_predict)]
    plt.plot(test_dates, test_predict, label='Test Predicted Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Plot future predictions
def plot_future_predictions(data, future_df):
    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data['Close'], label='Actual Stock Price')
    plt.plot(future_df.index, future_df['Future Predicted Stock Price'], label='Future Predicted Stock Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main function to run the entire process
def run_predictions():
    # Parameters
    ticker = ticker_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    try:
        future_days = int(future_days_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Future days must be an integer.")
        return

    # Download data
    try:
        data = download_data(ticker, start_date, end_date)
    except Exception as e:
        messagebox.showerror("Data Download Error", f"Error downloading data: {e}")
        return

    # Preprocess data
    scaled_data, scaler = preprocess_data(data)

    # Create datasets
    X, y = create_dataset(scaled_data, time_step=60)
    if X.shape[0] == 0:
        messagebox.showerror("Data Error", "Insufficient data to create training datasets.")
        return
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build and train the model
    model = build_model((X_train.shape[1], 1))
    train_model(model, X_train, y_train, epochs=100, batch_size=32)

    # Make predictions
    train_predict, test_predict = make_predictions(model, X_train, X_test, scaler)

    # Plot results
    plot_results(data, train_predict, test_predict, 60)

    # Simulate future predictions
    future_df = simulate_future_predictions(model, scaled_data, scaler, future_days, time_step=60)

    # Plot future predictions
    plot_future_predictions(data, future_df)

    # Print future predictions
    print("\nFuture Predicted Stock Prices for the Next {} Business Days:".format(future_days))
    print(future_df)

# Create the main window
root = tk.Tk()
root.title("Stock Price Prediction")

# Create and place the widgets
ticker_label = ttk.Label(root, text="Ticker Symbol:")
ticker_label.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
ticker_entry = ttk.Entry(root)
ticker_entry.grid(column=1, row=0, padx=10, pady=10)

start_date_label = ttk.Label(root, text="Start Date (YYYY-MM-DD):")
start_date_label.grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
start_date_entry = ttk.Entry(root)
start_date_entry.grid(column=1, row=1, padx=10, pady=10)

end_date_label = ttk.Label(root, text="End Date (YYYY-MM-DD):")
end_date_label.grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
end_date_entry = ttk.Entry(root)
end_date_entry.grid(column=1, row=2, padx=10, pady=10)

future_days_label = ttk.Label(root, text="Future Days (Business Days):")
future_days_label.grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
future_days_entry = ttk.Entry(root)
future_days_entry.grid(column=1, row=3, padx=10, pady=10)

run_button = ttk.Button(root, text="Run Predictions", command=run_predictions)
run_button.grid(column=0, row=4, columnspan=2, padx=10, pady=10)

# RUNNNNNNN
root.mainloop()