from flask import Flask, render_template, request, send_from_directory, jsonify
import pandas as pd
import numpy as np
import requests
import pymysql.cursors
import yfinance as yf
import json
from influxdb_client.client import query_api
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import plotly.io as pio
import plotly.graph_objs as go
import diskcache as dc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
import matplotlib
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import redis
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from data_fetcher import calculate_macd
from neural_network import build_lstm_model, train_model, evaluate_model, predict_next_week_prices, perform_hyperparameter_tuning

# Import custom modules
from hyperparameter_tuning import random_search  # Assuming this is a custom module for hyperparameter tuning

# Set up Flask app and other configurations
app = Flask(__name__, static_url_path='/static')
cache = dc.Cache(".cache")
matplotlib.use('Agg')
pd.options.display.float_format = '{:.2f}'.format

# Initialize Redis client
redis_client = redis.StrictRedis(host='192.168.10.10', port=6379, db=0)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize InfluxDB client
influxdb_url = "http://192.168.10.10:8086"
influxdb_token = "9R592h_lq35-3DuWguaSiueHit64JwwpIinsUyANTvxqBGp8vBN8pL6smtwMFjgmVJbglqYkTjuzU1e0_qOUtA=="
influxdb_org = "DevOps-Tech"
influxdb_bucket = "crypto_coins_bkt"
influxdb_client = InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
write_api = influxdb_client.write_api(write_options=SYNCHRONOUS)


def get_db_connection():
    return pymysql.connect(
        host=os.getenv('MYSQL_HOST', '192.168.10.10'),
        port=int(os.getenv('MYSQL_PORT', '3306')),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', 'Kubernetes@1993'),
        db=os.getenv('MYSQL_DB', 'crypto_coins'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


def write_to_influxdb(df, measurement):
    points = []
    for index, row in df.iterrows():
        point = Point(measurement)
        for col, val in row.items():
            if col == "time":
                point = point.time(val)
            elif isinstance(val, dict):
                # Example: Flatten the dictionary or convert to individual fields
                for key, value in val.items():
                    point = point.field(f"{col}_{key}", value)
            else:
                point = point.field(col, val)
        points.append(point)
    write_api.write(bucket=influxdb_bucket, org=influxdb_org, record=points)
    logging.info(f"Written data to InfluxDB: {measurement}")


def get_influxdb_data():
    query = f'from(bucket:"{influxdb_bucket}") |> range(start: -7d) |> filter(fn: (r) => r["_measurement"] == ' \
            f'"crypto_prices")'
    result = query_api.query_data_frame(org=influxdb_org, query=query)
    return result.set_index('time')


# Function to fetch data from database with Redis caching
def fetch_data_from_db():
    cached_data = redis_client.get('crypto_data')
    if cached_data:
        return json.loads(cached_data)
    else:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM crypto_coins")
                data = cursor.fetchall()
                redis_client.set('crypto_data', json.dumps(data), ex=3600)  # Set expiration to 1 hour
        finally:
            connection.close()
        return data


# Function to fetch cryptocurrency data from CoinGecko API
def get_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "inr",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": False,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    logging.info("Fetched cryptocurrency data.")
    return df


# Function to save data to MySQL database
def save_to_mysql_database(df, table_name):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                name VARCHAR(255),
                id VARCHAR(255) PRIMARY KEY,
                current_price DOUBLE,
                expected_return DOUBLE,
                high_24h DOUBLE,
                low_24h DOUBLE,
                last_updated DATETIME,
                price_change_24h DOUBLE,
                price_change_percentage_24h DOUBLE,
                ath DOUBLE,
                ath_date DATE,
                ath_change_percentage DOUBLE,
                atl_change_percentage DOUBLE,
                market_cap DOUBLE,
                market_cap_change_24h DOUBLE,
                market_cap_change_percentage_24h DOUBLE,
                market_cap_rank INT,
                circulating_supply DOUBLE,
                symbol VARCHAR(10),
                price_day_1 DOUBLE,
                price_day_2 DOUBLE,
                price_day_3 DOUBLE,
                price_day_4 DOUBLE,
                price_day_5 DOUBLE,
                price_day_6 DOUBLE,
                price_day_7 DOUBLE
            )
            """
            cursor.execute(create_table_query)

            for index, row in df.iterrows():
                insert_query = f"""
                    INSERT INTO {table_name} (name, id, current_price, expected_return, high_24h, low_24h,
                                              last_updated, price_change_24h, price_change_percentage_24h,
                                              ath, ath_date, ath_change_percentage, atl_change_percentage,
                                              market_cap, market_cap_change_24h, market_cap_change_percentage_24h,
                                              market_cap_rank, circulating_supply, symbol,
                                              price_day_1, price_day_2, price_day_3, price_day_4,
                                              price_day_5, price_day_6, price_day_7)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    name=VALUES(name), current_price=VALUES(current_price), expected_return=VALUES(expected_return),
                    high_24h=VALUES(high_24h), low_24h=VALUES(low_24h), last_updated=VALUES(last_updated),
                    price_change_24h=VALUES(price_change_24h), price_change_percentage_24h=VALUES(price_change_percentage_24h),
                    ath=VALUES(ath), ath_date=VALUES(ath_date), ath_change_percentage=VALUES(ath_change_percentage),
                    atl_change_percentage=VALUES(atl_change_percentage), market_cap=VALUES(market_cap),
                    market_cap_change_24h=VALUES(market_cap_change_24h),
                    market_cap_change_percentage_24h=VALUES(market_cap_change_percentage_24h),
                    market_cap_rank=VALUES(market_cap_rank), circulating_supply=VALUES(circulating_supply),
                    symbol=VALUES(symbol),
                    price_day_1=VALUES(price_day_1), price_day_2=VALUES(price_day_2), price_day_3=VALUES(price_day_3),
                    price_day_4=VALUES(price_day_4), price_day_5=VALUES(price_day_5), price_day_6=VALUES(price_day_6),
                    price_day_7=VALUES(price_day_7)
                """
                cursor.execute(insert_query, (
                    row['name'], row['id'], row['current_price'], row['expected_return'],
                    row['high_24h'], row['low_24h'], row['last_updated'],
                    row['price_change_24h'], row['price_change_percentage_24h'],
                    row['ath'], row['ath_date'], row['ath_change_percentage'], row['atl_change_percentage'],
                    row['market_cap'], row['market_cap_change_24h'], row['market_cap_change_percentage_24h'],
                    row['market_cap_rank'], row['circulating_supply'], row['symbol'],
                    row['price_day_1'], row['price_day_2'], row['price_day_3'],
                    row['price_day_4'], row['price_day_5'], row['price_day_6'], row['price_day_7']
                ))
        connection.commit()
    finally:
        connection.close()
    # Write to InfluxDB
    write_to_influxdb(df, table_name)


# Function to fetch filtered cryptocurrency data
def get_filtered_crypto_data():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM crypto_coins WHERE expected_return >= 1 ORDER BY expected_return DESC")
            filtered_data = cursor.fetchall()
    finally:
        connection.close()
    return filtered_data


# Function to analyze cryptocurrency data
def analyze_crypto_data(df):
    if 'price_change_percentage_24h' in df.columns:
        df["expected_return"] = (df["price_change_percentage_24h"] / 100) + 1
    else:
        logging.warning("Column 'price_change_percentage_24h' is missing in the API response.")
        df["expected_return"] = 1  # Default to 1 if the column is missing
    for day in range(1, 8):
        df[f"price_day_{day}"] = df["current_price"] * (df["expected_return"] ** day)

    df.sort_values(by="expected_return", ascending=False, inplace=True)
    save_to_mysql_database(df, "crypto_coins")
    return df


# Function to build and train the LSTM model
def build_and_train_model(X_train, y_train, params):
    model = build_lstm_model(input_shape=(X_train.shape[1], 1), units=params['units'], dropout=params['dropout'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model


# Function to perform hyperparameter tuning
def perform_hyperparameter_tuning(X_train, y_train):
    param_distributions = {
        'units': [50, 100, 150, 200],
        'dropout': [0.2, 0.3, 0.4, 0.5]
    }
    # Perform hyperparameter tuning using custom logic or external libraries
    best_params = random_search(build_lstm_model, param_distributions, X_train, y_train)
    return best_params


# Function to predict crypto prices
def predict_crypto_prices(df):
    predictions = {}
    X_tests = {}

    def train_and_predict(coin):
        try:
            coin_df = df[df['name'] == coin]
            if len(coin_df) > 1:
                X = coin_df.select_dtypes(include=[np.number]).drop('current_price', axis=1)
                y = coin_df['current_price']
                X = X.fillna(0)
                y = y.fillna(0)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

                best_params = perform_hyperparameter_tuning(X_train, y_train)

                model = train_model(X_train, y_train, units=best_params['units'], dropout=best_params['dropout'])

                predictions[coin] = model.predict(X_test).flatten().tolist()
                X_tests[coin] = X_test
        except Exception as e:
            logging.error(f"Error in train_and_predict for {coin}: {str(e)}")

    with ThreadPoolExecutor() as executor:
        executor.map(train_and_predict, df['name'].unique())

    return predictions, X_tests


def fetch_historical_data(symbol, start_date, end_date):
    try:
        # Fetch historical data using Yahoo Finance API
        df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error


def preprocess_data(df):
    try:
        # Calculate technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)  # RSI calculation
        df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])

        # Drop NaN values
        df.dropna(inplace=True)

        # Scale numerical features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[['Close', 'SMA_20', 'EMA_10', 'RSI', 'MACD', 'MACD_signal']])
        df[['Close', 'SMA_20', 'EMA_10', 'RSI', 'MACD', 'MACD_signal']] = scaled_features

        return df
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        return pd.DataFrame()


def calculate_macd(series):
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return macd, macd_signal


def calculate_rsi(series, period):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# Example function to fetch and preprocess data
def fetch_and_preprocess_data(symbol, start_date, end_date):
    df = fetch_historical_data(symbol, start_date, end_date)
    if not df.empty:
        df_preprocessed = preprocess_data(df)
        return df_preprocessed
    else:
        return pd.DataFrame()


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(X_train, y_train, units=50, dropout=0.2, epochs=50, batch_size=32):
    """
    Train an LSTM model on the given training data.

    Parameters:
        X_train (numpy.ndarray): Input features for training.
        y_train (numpy.ndarray): Target variable for training.
        units (int): Number of units/neurons in LSTM layers (default is 50).
        dropout (float): Dropout rate between LSTM layers (default is 0.2).
        epochs (int): Number of training epochs (default is 50).
        batch_size (int): Batch size for training (default is 32).

    Returns:
        tensorflow.keras.models.Sequential or None: Trained LSTM model if successful, None if an error occurs.
    """
    try:
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=units, dropout=dropout)
        logging.info(f"Training LSTM model with units={units}, dropout={dropout}, epochs={epochs}, batch_size={batch_size}")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        logging.info("Model training completed successfully.")
        return model
    except Exception as e:
        logging.error(f"Error occurred during model training: {str(e)}")
        return None


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a trained LSTM model using mean squared error (MSE) and mean absolute error (MAE).

    Parameters:
        model (tensorflow.keras.models.Sequential): Trained LSTM model.
        X_test (numpy.ndarray): Input features for testing.
        y_test (numpy.ndarray): Target variable for testing.

    Returns:
        float, float: Mean squared error (MSE) and mean absolute error (MAE) of the model predictions.
    """
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return mse, mae
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {str(e)}")
        return None, None


def build_stacked_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_next_week_prices(model, latest_data):
    # Assuming latest_data is the preprocessed DataFrame containing recent data
    # Prepare input for prediction (reshape if using LSTM)
    X = latest_data.values.reshape((1, latest_data.shape[0], latest_data.shape[1]))
    # Predict next week's prices
    predicted_prices = model.predict(X)
    return predicted_prices.flatten()[0]  # Return the predicted price


# Function to save predictions to MySQL
def save_predictions_to_mysql(predictions):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            for coin, price in predictions.items():
                if isinstance(price, list):
                    price = price[0]
                price = float(price)
                insert_query = """
                    INSERT INTO price_predictions (cryptocurrency, predicted_price)
                    VALUES (%s, %s)
                """
                cursor.execute(insert_query, (coin, price))
        connection.commit()
    finally:
        connection.close()


# Function to save data to JSON file
def save_to_json_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)


# Placeholder function for next week price prediction
def predict_next_week_price(coin):
    return 0.0  # Placeholder value


# Function to evaluate model performance
def evaluate_model_performance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2


# Flask route for the index page
@app.route('/')
def index():
    try:
        sort_field = request.args.get('sort_field', 'name')
        sort_order = request.args.get('sort_order', 'asc')
        crypto_df = get_crypto_data()
        if not crypto_df.empty:
            if 'current_price' not in crypto_df.columns:
                raise ValueError("API response does not contain 'current_price' column.")
            top_investments = analyze_crypto_data(crypto_df)

            # Sorting
            valid_fields = ['name', 'current_price', 'expected_return', 'market_cap_rank']
            if sort_field not in valid_fields:
                sort_field = 'name'
            if sort_order not in ['asc', 'desc']:
                sort_order = 'asc'
            top_investments.sort_values(by=sort_field, ascending=(sort_order == 'asc'), inplace=True)

            predictions, _ = predict_crypto_prices(top_investments)
            for coin, prediction in predictions.items():
                if not isinstance(prediction, list):
                    predictions[coin] = prediction.tolist()
            save_predictions_to_mysql(predictions)
            raw_data_filename = 'raw_data.json'
            save_to_json_file(top_investments.to_dict(orient='records'), raw_data_filename)
            next_week_predictions = {}
            for coin in top_investments['name'].unique():
                next_week_predictions[coin] = predict_next_week_price(coin)
            page = request.args.get('page', 1, type=int)
            per_page = 100
            total = len(top_investments)
            start = (page - 1) * per_page
            end = start + per_page
            paginated_top_investments = top_investments[start:end]
            return render_template('index.html',
                                   top_investments=paginated_top_investments.to_dict(orient='records'),
                                   prediction_dict=predictions,
                                   next_week_predictions=next_week_predictions,
                                   raw_data_filename=raw_data_filename,
                                   page=page,
                                   total=total,
                                   per_page=per_page,
                                   sort_field=sort_field,
                                   sort_order=sort_order)
        else:
            raise ValueError("Failed to fetch crypto data from CoinGecko API.")
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        return render_template('error.html', error_message=str(e))


# Function to add technical indicators
def add_technical_indicators(df):
    df['SMA'] = df['Close'].rolling(window=20).mean()  # Example: Simple Moving Average (SMA)
    return df


# Route for displaying candlestick graph for a cryptocurrency
# Route for displaying candlestick graph for a cryptocurrency
@app.route('/candlestick/<symbol>')
def show_candlestick(symbol):
    try:
        # Fetch ticker symbol from the database
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT symbol FROM crypto_coins WHERE name = %s OR symbol = %s", (symbol, symbol))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"No data found for symbol or name: {symbol}")
            ticker = result['symbol']

        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Fetch historical data for the cryptocurrency using ticker
        df = fetch_historical_data(ticker, start_date, end_date)

        # Add technical indicators (e.g., SMA) to the dataframe
        df = add_technical_indicators(df)

        # Generate the candlestick graph using Plotly
        candlestick_fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol
        )])

        # Add SMA to the candlestick graph
        candlestick_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA'],
            mode='lines',
            name='SMA',
            line=dict(color='orange', width=2)
        ))

        candlestick_fig.update_layout(
            title=f'Candlestick Graph for {symbol}',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )

        # Convert the figure to HTML
        graph_html = pio.to_html(candlestick_fig, full_html=False)

        # Check if the request is AJAX for updating the graph
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'index': df.index.format(),
                'Open': df['Open'].tolist(),
                'High': df['High'].tolist(),
                'Low': df['Low'].tolist(),
                'Close': df['Close'].tolist()
            })

        # Render the HTML template with the candlestick graph
        return render_template('candlestick.html', symbol=symbol, graph_html=graph_html)

    except Exception as e:
        # Handle errors gracefully
        logging.error(f"Error in show_candlestick route for symbol {symbol}: {str(e)}")
        return render_template('error.html', error_message=str(e))

    finally:
        connection.close()


# Flask route for API to fetch historical price data
@app.route('/api/historical-price', methods=['GET'])
def get_historical_price_data_api():
    try:
        symbol = request.args.get('symbol')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Fetch historical data from a data source (e.g., Yahoo Finance)
        if symbol and start_date and end_date:
            df = fetch_historical_data(symbol, start_date, end_date)

            # Convert DataFrame to JSON format
            data_json = df.reset_index().to_json(orient='records')

            return jsonify(data_json)

        else:
            return jsonify({'error': 'Missing parameters: symbol, start_date, end_date'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Function to fetch available cryptocurrencies from MySQL
def fetch_available_cryptocurrencies():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT name FROM crypto_coins ORDER BY name")
            available_coins = [row['name'] for row in cursor.fetchall()]
    finally:
        connection.close()
    return available_coins


# Flask route for the candlestick chart
@app.route('/candlestick_chart')
def candlestick_chart():
    try:
        available_coins = fetch_available_cryptocurrencies()
        return render_template('candlestick_chart.html', available_coins=available_coins)
    except Exception as e:
        logging.error(f"Error in candlestick_chart route: {str(e)}")
        return "An error occurred."


# Flask route for handling errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


# Flask route for handling server errors
@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
