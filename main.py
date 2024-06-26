import threading
import time
from datetime import datetime, date
from flask import Flask, render_template, request, send_from_directory, jsonify
import pandas as pd
import numpy as np
import requests
import pymysql.cursors
import yfinance as yf
import json
from influxdb_client.client import query_api
from pygments.lexers import go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
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
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from feature_engineering import extract_social_media_sentiment_features
from neural_network import build_and_train_model, predict_crypto_prices, evaluate_model_performance
from cross_validation import perform_cross_validation, split_data, evaluate_model

# Set up Flask app and other configurations
app = Flask(__name__, static_url_path='/static')
cache = dc.Cache(".cache")
matplotlib.use('Agg')
pd.options.display.float_format = '{:.2f}'.format

# Initialize Redis client
redis_client = redis.StrictRedis(host='192.168.10.10', port=6379, db=0)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize InfluxDB client
influxdb_url = "http://192.168.10.10:8086"
influxdb_token = "FiAmaj78_PagPTglX8Zth2mcH4-5ZehRpFh5pX3uET2e2QUlBTtPqZJY74_jnhj7eHTnMKZgZlJ-E-kyrP-sQA=="
influxdb_org = "DevOps-Tech"
influxdb_bucket = "crypto_coins_bkt"
influxdb_client = InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
write_api = influxdb_client.write_api(write_options=SYNCHRONOUS)


# Custom JSON Encoder for datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


# Function to write data to InfluxDB
def write_to_influxdb(df, measurement):
    points = []
    try:
        for index, row in df.iterrows():
            point = Point(measurement)
            for col, val in row.items():
                if col == "time":
                    point = point.time(val)
                elif isinstance(val, dict):
                    for key, value in val.items():
                        point = point.field(f"{col}_{key}", value)
                else:
                    point = point.field(col, val)
            points.append(point)
        write_api.write(bucket=influxdb_bucket, org=influxdb_org, record=points)
        logging.info(f"Written data to InfluxDB: {measurement}")
    except Exception as e:
        logging.error(f"Error writing data to InfluxDB: {str(e)}")
        # Add additional logging for HTTP response details if available
        if hasattr(e, 'response') and e.response:
            logging.error(f"HTTP response code: {e.response.status_code}")
            logging.error(f"HTTP response body: {e.response.text}")


def get_influxdb_data():
    query = f'from(bucket:"{influxdb_bucket}") |> range(start: -7d) |> filter(fn: (r) => r["_measurement"] == ' \
            f'"current_price")'
    result = query_api.query_data_frame(org=influxdb_org, query=query)
    return result.set_index('time')


# Check if the key exists and get its value
cached_data = redis_client.get('crypto_data')

# Check if data is cached
if cached_data:
    print("Data is cached in Redis.")
    # Print the cached data if needed
    # print(cached_data.decode())  # Decoding the byte data to string
else:
    print("Data is not cached in Redis.")


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


# Function to fetch data from database with Redis caching
def fetch_data_from_db():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM crypto_coins")
            data = cursor.fetchall()
            redis_client.set('crypto_data', json.dumps(data, cls=CustomJSONEncoder))  # Update Redis cache
            logging.info("Updated data in Redis cache.")
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
                ath DATETIME,
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
                # Convert last_updated to the correct format
                last_updated = datetime.strptime(row['last_updated'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d %H:%M:%S')

                # Convert ath_date to the correct format
                ath_date = datetime.strptime(row['ath_date'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%d')

                # Insert query
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
                    row['high_24h'], row['low_24h'], last_updated,
                    row['price_change_24h'], row['price_change_percentage_24h'],
                    row['ath'], ath_date, row['ath_change_percentage'], row['atl_change_percentage'],
                    row['market_cap'], row['market_cap_change_24h'], row['market_cap_change_percentage_24h'],
                    row['market_cap_rank'], row['circulating_supply'], row['symbol'],
                    row['price_day_1'], row['price_day_2'], row['price_day_3'],
                    row['price_day_4'], row['price_day_5'], row['price_day_6'], row['price_day_7']
                ))
            connection.commit()
            logging.info(f"Data saved to MySQL table: {table_name}")
    except Exception as e:
        logging.error(f"Error saving data to MySQL: {str(e)}")
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


# Function to build and train the LSTM model (from neural_network.py)
def build_and_train_model(X_train, y_train, params):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(params['units'], return_sequences=True),
        Dropout(params['dropout']),
        LSTM(params['units']),
        Dropout(params['dropout']),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model


# Function to perform hyperparameter tuning with GridSearchCV
def perform_hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params


# Function to normalize and scale data
def normalize_and_scale_data(df):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[df_scaled.columns] = scaler.fit_transform(df_scaled[df_scaled.columns])
    return df_scaled


# Function to handle missing data
def handle_missing_data(df):
    df_filled = df.fillna(method='ffill').fillna(method='bfill')
    return df_filled


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


# Function to fetch data from MySQL database
def fetch_data_from_mysql():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM crypto_coins")
            data = cursor.fetchall()
    finally:
        connection.close()
    return data


# Function to predict crypto prices and calculate evaluation metrics (from neural_network.py)
def predict_crypto_prices(df):
    predictions = {}
    evaluation_metrics = {}

    def train_and_predict(coin):
        coin_df = df[df['name'] == coin]
        if len(coin_df) > 1:
            X = coin_df.select_dtypes(include=[np.number]).drop('current_price', axis=1)
            y = coin_df['current_price']
            X = X.fillna(0)
            y = y.fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
            # Perform hyperparameter tuning
            best_params = perform_hyperparameter_tuning(X_train, y_train)

            # Build and train model with the best parameters
            model = build_and_train_model(X_train, y_train, best_params)

            # Predict prices
            predictions[coin] = model.predict(X_test).flatten().tolist()

            # Calculate evaluation metrics
            mae, mse, r2 = evaluate_model_performance(y_test, predictions[coin])
            evaluation_metrics[coin] = {
                'MAE': mae,
                'MSE': mse,
                'R2': r2
            }

    with ThreadPoolExecutor() as executor:
        executor.map(train_and_predict, df['name'].unique())

    return predictions, evaluation_metrics


# Function to fetch data from InfluxDB
def fetch_data_from_influxdb(coin_name):
    query_api = influxdb_client.query_api()
    query = f'from(bucket: "{influxdb_bucket}") |> range(start: -1d) |> filter(fn: (r) => r["_measurement"] == "{coin_name}")'
    result = query_api.query(org=influxdb_org, query=query)
    return result


# Function to fetch price predictions from MySQL
def fetch_price_predictions():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM price_predictions")
            data = cursor.fetchall()
    finally:
        connection.close()
    return data


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


# Assuming your Flask app and required functions are defined above
def scheduled_task():
    while True:
        try:
            # Fetch cryptocurrency data from CoinGecko API
            df = get_crypto_data()

            # Analyze cryptocurrency data
            analyzed_df = analyze_crypto_data(df)

            # Save analyzed data to MySQL database
            save_to_mysql_database(analyzed_df, "crypto_coins")

            # Write analyzed data to InfluxDB
            write_to_influxdb(analyzed_df, "current_price")

            logging.info(f"Data updated and saved at: {datetime.now()}")

        except Exception as e:
            logging.error(f"Error in scheduled task: {str(e)}")

        time.sleep(120)  # Run every 2 minutes (adjust interval as needed)


# Start the scheduled task in a separate thread
scheduled_task_thread = threading.Thread(target=scheduled_task)
scheduled_task_thread.start()


# Scheduled task to periodically update Redis with fresh data from MySQL
def scheduled_redis_update():
    while True:
        try:
            fetch_data_from_db()
        except Exception as e:
            logging.error(f"Error in scheduled Redis update: {str(e)}")

        time.sleep(120)  # Run every 2 minutes (adjust interval as needed)


# Start the scheduled Redis update task in a separate thread
scheduled_redis_update_thread = threading.Thread(target=scheduled_redis_update)
scheduled_redis_update_thread.start()


# Flask route for the index page
@app.route('/')
def index():
    try:
        sort_field = request.args.get('sort_field', 'expected_return')
        sort_order = request.args.get('sort_order', 'desc')

        # Fetch data from wherever you store it (e.g., Redis, MySQL)
        # Modify this part based on your actual data fetching mechanism
        cached_data = redis_client.get('crypto_data')
        if cached_data:
            crypto_df = pd.DataFrame(json.loads(cached_data))
        else:
            logging.info("Fetching data from MySQL as Redis cache is empty or outdated.")
            crypto_data = fetch_data_from_db()
            if not crypto_data:
                return "Failed to fetch crypto data from both Redis cache and MySQL."
            crypto_df = pd.DataFrame(crypto_data)
            # Update Redis cache
            redis_client.set('crypto_data', json.dumps(crypto_data, cls=CustomJSONEncoder))

        # Handling 'page' parameter with default value of 1
        page = request.args.get('page', 1, type=int)

        # Sorting
        valid_fields = list(crypto_df.columns)  # Use all columns as valid sort fields
        if sort_field not in valid_fields:
            sort_field = 'expected_return'
        if sort_order not in ['asc', 'desc']:
            sort_order = 'desc'

        crypto_df.sort_values(by=sort_field, ascending=(sort_order == 'asc'), inplace=True)

        # Pagination
        per_page = 100
        total = len(crypto_df)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_crypto_df = crypto_df.iloc[start:end].copy()

        # Convert DataFrame to a list of dictionaries (JSON serializable)
        top_investments = paginated_crypto_df.to_dict(orient='records')

        return render_template('index.html',
                               top_investments=top_investments,
                               page=page,
                               total=total,
                               per_page=per_page,
                               sort_field=sort_field,
                               sort_order=sort_order)

    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        return "An error occurred. Please try again later.", 500


# Function to add technical indicators
def add_technical_indicators(df):
    df['SMA'] = df['Close'].rolling(window=20).mean()  # Example: Simple Moving Average (SMA)
    return df


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

        if not start_date or not end_date:
            # Default period if no dates provided
            period = '6mo'
            interval = '1d'
        else:
            # Custom period based on the dates provided
            period = '1mo'
            interval = '1d'

        # Fetch historical data for the cryptocurrency using ticker
        if period:
            df = yf.download(ticker, period=period, interval=interval)
        else:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        # Check if the dataframe is empty (which means no data was fetched)
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

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

    except ValueError as e:
        # Handle specific errors related to data not found or other expected issues
        error_message = f"Error: {str(e)}"
        logging.error(error_message)
        return render_template('error.html', error_message=error_message)

    except Exception as e:
        # Fetch the list of available coins from the database
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT name, symbol FROM crypto_coins")
                available_coins = cursor.fetchall()
        except Exception as db_error:
            logging.error(f"Database error: {str(db_error)}")
            available_coins = []

        # Handle unexpected errors gracefully
        error_message = f"Error: {str(e)}"
        logging.error(error_message)
        return render_template('error.html', error_message=error_message, available_coins=available_coins)

    finally:
        if 'connection' in locals() and connection.open:
            connection.close()


# Endpoint to handle sentiment analysis
@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    if 'symbol' in data and 'text' in data:
        symbol = data['symbol']
        text = data['text']
        sentiment_features = extract_social_media_sentiment_features(symbol, text)
        return jsonify(sentiment_features), 200
    else:
        return jsonify({'error': 'Invalid request. Required fields: symbol, text'}), 400


# Flask route for prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Assuming data contains necessary features for prediction
        df = pd.DataFrame(data)  # Example: Convert JSON data to DataFrame
        predictions, evaluation_metrics = predict_crypto_prices(df)
        return jsonify({'predictions': predictions, 'metrics': evaluation_metrics}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Assuming you have data ready to be used
data = pd.read_json('raw_data.json')  # Replace with your data loading mechanism


# Flask route for model training and evaluation
@app.route('/train-model', methods=['POST'])
def train_model():
    try:
        X_train, X_test, y_train, y_test = split_data(data)  # Implement split_data function accordingly
        model = RandomForestRegressor()  # Initialize your model here
        model.fit(X_train, y_train)

        # Perform cross-validation and evaluation
        cv_results = perform_cross_validation(model, X_train, y_train)  # Implement perform_cross_validation function accordingly
        evaluation = evaluate_model(model, X_test, y_test)  # Implement evaluate_model function accordingly

        return jsonify({
            'cv_results': cv_results,
            'evaluation': evaluation
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Add a route to serve static files, if necessary
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


# Function to create price predictions table
def create_price_predictions_table():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            create_table_query = """
                CREATE TABLE IF NOT EXISTS price_predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    cryptocurrency VARCHAR(255),
                    predicted_price DOUBLE
                )
            """
            cursor.execute(create_table_query)
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    create_price_predictions_table()  # Ensure the table is created before running the app
    app.run(host='0.0.0.0', debug=True)
