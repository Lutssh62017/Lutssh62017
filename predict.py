import requests
import pandas as pd
import numpy as np
import time
import schedule
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Step 1: Fetch BTC/USD historical data from CoinGecko
def fetch_historical_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '30',  # Past 30 days
        'interval': '10m'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises exception for bad response
        data = response.json()
        return data['prices']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data: {e}")
        return []

# Step 2: Prepare data for training
def prepare_data(prices):
    if len(prices) == 0:
        print("No historical data available")
        return None, None, None
    
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Normalize the price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['price'] = scaler.fit_transform(df[['price']])

    # Prepare the data for LSTM model
    X_train = []
    y_train = []
    time_step = 60  # Use past 60 time steps (10-minute intervals) to predict the next one

    for i in range(time_step, len(df)):
        X_train.append(df['price'][i-time_step:i].values)
        y_train.append(df['price'][i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler

# Step 3: Build the LSTM model
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Step 4: Train the model with historical data
def train_model(X_train, y_train):
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    return model

# Step 5: Fetch the latest price
def fetch_latest_price():
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises exception for bad response
        data = response.json()
        latest_price = data['bitcoin']['usd']
        return latest_price
    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest price: {e}")
        return None

# Step 6: Update the model with the latest price
def update_model(model, scaler):
    latest_price = fetch_latest_price()
    if latest_price is None:
        return

    # Normalize the latest price using the same scaler
    latest_price_normalized = scaler.transform([[latest_price]])[0][0]

    # Predict future price using the updated model
    input_data = np.array([[latest_price_normalized]])
    input_data = np.reshape(input_data, (1, input_data.shape[0], 1))
    predicted_price = model.predict(input_data)

    # Inverse transform the predicted price to original scale
    predicted_price = scaler.inverse_transform(predicted_price)
    print(f"Predicted next price: {predicted_price[0][0]:.2f} USD")
    
    return predicted_price[0][0]

# Step 7: Main logic to run the script
def main():
    print("Fetching historical data...")
    prices = fetch_historical_data()

    if not prices:
        print("No historical data fetched. Exiting.")
        return
    
    # Prepare data and train model
    X_train, y_train, scaler = prepare_data(prices)
    
    if X_train is None or y_train is None or scaler is None:
        print("Data preparation failed. Exiting.")
        return
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Fetch new price every 10 minutes and update predictions
    print("Setting up schedule to update model every 10 minutes...")
    schedule.every(10).minutes.do(lambda: update_model(model, scaler))

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()
