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
```
## The requirements.txt file includes:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `yfinance`

# How to Use
Clone the Repository

First, clone this repository to your local machine:
```sh
git clone https://github.com/your-username/stock-price-prediction.git
```
```sh
cd stock-price-prediction
```
Install Dependencies

Navigate to the project directory and install the required dependencies:
```sh
pip install -r requirements.txt
```
Run the Application

Execute the script to start the GUI application:

```sh
python main.py
```
**Input Parameters:**

- Ticker Symbol: Enter the stock ticker symbol (e.g., AAPL for Apple Inc.).
- Start Date: Enter the start date of the historical data in the format YYYY-MM-DD.
- End Date: Enter the end date of the historical data in the format YYYY-MM-DD.
- Future Days: Enter the number of business days for which you want to predict future stock prices.
- Run Predictions
- Click the "Run Predictions" button to start the prediction process.



### The application will:
- Download the historical stock data.
- Preprocess the data.
- Build and train the LSTM model.
- Make predictions on the training and test datasets.
- Simulate future stock price predictions.
- Display the results in plots and print the future predictions in the console.


# Example
## **Here is an example of how to use the application:**
- Enter AAPL as the ticker symbol.
- Enter 2022-01-01 as the start date.
- Enter 2023-01-01 as the end date.
- Enter 30 as the number of future days.
- Click **"Run Predictions"**.

The application will generate plots showing the actual stock prices, training and test predictions, and future predicted stock prices. It will also print the future predictions in the console.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.

Contact
If you have any questions or need further assistance, please contact **Zaid Shaikh** at zaid83560@gmail.com .
