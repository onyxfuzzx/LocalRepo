# Stock Price Prediction

This project uses Long Short-Term Memory (LSTM) networks to predict stock prices. The model is trained on historical stock data and can make predictions for future stock prices based on the provided data.

## Features

- Downloads historical stock data using `yfinance`.
- Preprocesses the data using `MinMaxScaler` from `scikit-learn`.
- Builds and trains an LSTM model using `Keras` (part of `TensorFlow`).
- Makes predictions on both training and test datasets.
- Simulates future stock price predictions for a specified number of business days.
- Provides a graphical user interface (GUI) using `tkinter` for easy interaction.

## Requirements

To run this project, you need to have the following dependencies installed. You can install them using `pip`:

```sh
pip install -r requirements.txt
