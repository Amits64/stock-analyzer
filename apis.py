# apis.py

from flask import Flask, jsonify, request
import pymysql.cursors
import yfinance as yf
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)


# Function to establish database connection
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


# API to fetch filtered cryptocurrency data
@app.route('/api/crypto-data', methods=['GET'])
def get_filtered_crypto_data_api():
    try:
        min_expected_return = float(request.args.get('min_expected_return', 1.0))

        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT name, expected_return, current_price FROM crypto_coins WHERE expected_return >= %s", (min_expected_return,))
            filtered_data = cursor.fetchall()

        return jsonify(filtered_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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


# Function to fetch historical data from Yahoo Finance
def fetch_historical_data(symbol, start_date, end_date):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date, end=end_date)
    return hist


# API to predict next week's prices
@app.route('/api/predict-next-week-price/<symbol>', methods=['GET'])
def predict_next_week_price_api(symbol):
    try:
        prediction = predict_next_week_price(symbol)  # Implement this function as per your prediction logic

        return jsonify({'symbol': symbol, 'predicted_price': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# API to fetch available cryptocurrencies
@app.route('/api/available-cryptocurrencies', methods=['GET'])
def fetch_available_cryptocurrencies_api():
    try:
        available_coins = fetch_available_cryptocurrencies()

        return jsonify(available_coins)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# API to analyze cryptocurrency data
@app.route('/api/analyze-crypto-data', methods=['GET'])
def analyze_crypto_data_api():
    try:
        crypto_df = get_crypto_data()  # Implement this function to fetch and process crypto data
        analyzed_data = analyze_crypto_data(crypto_df)  # Implement this function to analyze crypto data

        return jsonify(analyzed_data.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Function to fetch crypto data (example)
def get_crypto_data():
    # Example function to fetch data from an API or database
    return pd.DataFrame()


# Function to analyze crypto data (example)
def analyze_crypto_data(df):
    # Example function to analyze data
    return df


# Placeholder function for next week price prediction
def predict_next_week_price(coin):
    return 0.0  # Placeholder value


# Main entry point of the application
if __name__ == '__main__':
    app.run(debug=True)
