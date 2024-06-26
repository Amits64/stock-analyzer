# apis.py

import requests
from LinearRegression import r2_score
from flask import Flask, jsonify, request, render_template
import pymysql.cursors
import yfinance as yf
import pandas as pd
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from main import predict_next_week_price, get_filtered_crypto_data, process_data_update, logger, evaluate_model, \
    fetch_data_from_db, save_to_mysql_database, get_crypto_data, random_forest_predictive_model
import plotly.graph_objects as go
from neural_network import train_model, build_lstm_model

# Initialize Flask app
app = Flask(__name__)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


# Function to establish database connection
def get_db_connection():
    try:
        return pymysql.connect(
            host=os.getenv('MYSQL_HOST', '192.168.10.10'),
            port=int(os.getenv('MYSQL_PORT', '3306')),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', 'Kubernetes@1993'),
            db=os.getenv('MYSQL_DB', 'crypto_coins'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        return None


# Function to fetch available cryptocurrencies from MySQL
def fetch_available_cryptocurrencies():
    connection = get_db_connection()
    if not connection:
        return []

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT name FROM crypto_coins ORDER BY name")
            available_coins = [row['name'] for row in cursor.fetchall()]
            return available_coins
    except Exception as e:
        logging.error(f"Error fetching available cryptocurrencies: {str(e)}")
        return []
    finally:
        connection.close()


# Function to fetch historical data from Yahoo Finance
def fetch_historical_data(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        return hist
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return pd.DataFrame()


# API to predict next week's prices
@app.route('/api/predict-next-week-price/<symbol>', methods=['GET'])
def predict_next_week_price_api(symbol):
    try:
        prediction = predict_next_week_price(symbol)  # Implement this function as per your prediction logic

        return jsonify({'symbol': symbol, 'predicted_price': prediction})

    except Exception as e:
        logging.error(f"Error predicting next week price for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500


# API to fetch available cryptocurrencies
@app.route('/api/available-cryptocurrencies', methods=['GET'])
def fetch_available_cryptocurrencies_api():
    try:
        available_coins = fetch_available_cryptocurrencies()

        return jsonify(available_coins)

    except Exception as e:
        logging.error(f"Error fetching available cryptocurrencies: {str(e)}")
        return jsonify({'error': str(e)}), 500


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


# Flask route for API to fetch current price data
@app.route('/api/current-price', methods=['GET'])
def get_current_price_data_api():
    try:
        symbol = request.args.get('symbol')

        if symbol:
            # Fetch current data from CoinGecko API or database
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=inr"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if symbol in data:
                return jsonify(data[symbol])
            else:
                raise ValueError("Symbol not found in CoinGecko API response.")
        else:
            raise ValueError("Missing required parameter: symbol.")
    except Exception as e:
        logging.error(f"Error in get_current_price_data_api: {str(e)}")
        return jsonify({"error": str(e)}), 400


# Flask route for API to fetch filtered crypto data
@app.route('/api/filtered-crypto-data', methods=['GET'])
def get_filtered_crypto_data_api():
    try:
        filtered_data = get_filtered_crypto_data()
        return jsonify(filtered_data)
    except Exception as e:
        logging.error(f"Error in get_filtered_crypto_data_api: {str(e)}")
        return jsonify({"error": str(e)}), 400


# Flask route for API to fetch predicted prices
@app.route('/api/predicted-prices', methods=['GET'])
def get_predicted_prices_api():
    try:
        # Fetch data from the database
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT cryptocurrency, predicted_price FROM price_predictions")
            data = cursor.fetchall()

        return jsonify(data)
    except Exception as e:
        logging.error(f"Error in get_predicted_prices_api: {str(e)}")
        return jsonify({"error": str(e)}), 400
    finally:
        connection.close()


def generate_candlestick_chart(historical_data):
    fig = go.Figure(data=[go.Candlestick(
        x=historical_data.index,
        open=historical_data['Open'],
        high=historical_data['High'],
        low=historical_data['Low'],
        close=historical_data['Close']
    )])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')

    # Return the HTML representation of the plot
    return fig.to_html(full_html=False)


# Route to display candlestick chart for a specific cryptocurrency
from datetime import datetime, timedelta


@app.route('/candlestick_chart/<string:coin_id>', methods=['GET'])
def candlestick_chart(coin_id):
    try:
        # Calculate start_date (1 month ago from today)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        # Fetch historical data for the cryptocurrency
        historical_data = fetch_historical_data(coin_id, start_date, end_date)
        if historical_data.empty:
            return jsonify({'error': 'No data found for the given cryptocurrency.'}), 404

        # Generate candlestick chart
        candlestick = generate_candlestick_chart(historical_data)

        return render_template('candlestick_chart.html', candlestick=candlestick)
    except Exception as e:
        logging.error(f"Error in candlestick_chart endpoint: {str(e)}")
        return jsonify({'error': 'Failed to fetch candlestick chart data.'}), 500


@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    try:
        # Implement your logic to fetch data here
        data = fetch_data_from_db()  # Example function to fetch data from database
        return jsonify({'data': data})  # Example: Return fetched data as JSON
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching data'}), 500


@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        process_data_update(data)
        return jsonify({"message": "Data processed successfully"}), 200
    except Exception as e:
        logging.error(f"Error processing webhook: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/fetch-data', methods=['GET'])
def fetch_data():
    try:
        data = get_crypto_data()
        save_to_mysql_database(data, 'crypto_coins')
        return jsonify({'message': 'Data fetched and stored successfully.'})
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return jsonify({'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = fetch_data_from_db()
        df = pd.DataFrame(data)
        features = df[
            ['price_day_1', 'price_day_2', 'price_day_3', 'price_day_4', 'price_day_5', 'price_day_6', 'price_day_7']]
        target = df['current_price']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Scale the data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build, train and evaluate the model
        model = build_lstm_model(X_train_scaled.shape[1])
        history = train_model(model, X_train_scaled, y_train)
        evaluation = evaluate_model(model, X_test_scaled, y_test)

        predictions = model.predict(X_test_scaled)
        metrics = {
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mean_squared_error(y_test, predictions),
            "R2": r2_score(y_test, predictions)
        }

        return jsonify({'evaluation': evaluation, 'metrics': metrics})
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)})


@app.route('/api/crypto_data', methods=['GET'])
def get_crypto_data():
    data = fetch_data_from_db()
    return jsonify(data)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    model, mae, mse, r2 = random_forest_predictive_model(df)
    predictions = model.predict(df[['market_cap', 'volume', 'circulating_supply']])
    response = {
        'predictions': predictions.tolist(),
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
        'r2_score': r2
    }
    return jsonify(response)


# Main entry point of the application
if __name__ == '__main__':
    app.run(debug=True)
