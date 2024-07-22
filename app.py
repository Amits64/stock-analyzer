# app.py

import json
import os
import logging
from flask import Flask, jsonify, request, render_template
import subprocess
import base64

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Endpoint to fetch data
@app.route('/fetch-data', methods=['POST'])
def fetch_data():
    logger.debug('Fetching data...')
    try:
        result = subprocess.run(['python', 'data_fetcher.py'], check=True, capture_output=True, text=True)
        logger.debug(f'Subprocess stdout: {result.stdout}')
        logger.debug(f'Subprocess stderr: {result.stderr}')
        return jsonify({'status': 'Data fetched successfully'}), 200
    except subprocess.CalledProcessError as e:
        logger.error(f'Failed to fetch data: {str(e)}')
        logger.error(f'Subprocess stderr: {e.stderr}')
        return jsonify({'error': f'Failed to fetch data: {str(e)}'}), 500


# Endpoint to process data
@app.route('/process-data', methods=['POST'])
def process_data():
    logger.debug('Processing data...')
    try:
        raw_data = request.get_data(as_text=True)
        if not raw_data:
            logger.warning('No data received')
            return jsonify({'error': 'No data received'}), 400

        raw_data = raw_data.strip()
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            logger.error(f'Invalid JSON: {str(e)}')
            return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400

        result = subprocess.run(['python', 'data_processor.py'], check=True, capture_output=True, text=True)
        logger.debug(f'Subprocess stdout: {result.stdout}')
        logger.debug(f'Subprocess stderr: {result.stderr}')
        return jsonify({'status': 'Data processed successfully'}), 200
    except subprocess.CalledProcessError as e:
        logger.error(f'Failed to process data: {str(e)}')
        logger.error(f'Subprocess stderr: {e.stderr}')
        return jsonify({'error': f'Failed to process data: {str(e)}'}), 500


# Endpoint to generate forecast
@app.route('/generate-forecast', methods=['POST'])
def generate_forecast():
    coin_name = request.json.get('coin_name')
    if not coin_name:
        return jsonify({'error': 'Coin name is required'}), 400

    # Set the filename for saving the forecast plot
    forecast_file = f'plots/{coin_name}_forecast.png'
    if os.path.exists(forecast_file):
        return jsonify({'image_url': f'/static/{coin_name}_forecast.png'})

    # Run the forecasting script with the selected coin
    try:
        # Adjust the command based on your script and arguments
        subprocess.run(['python', 'forecast_script.py', '--coin', coin_name], check=True)
        return jsonify({'image_url': f'/static/{coin_name}_forecast.png'})
    except subprocess.CalledProcessError:
        return jsonify({'error': 'Failed to generate forecast'}), 500


# Route to render forecast page
@app.route('/forecast')
def forecast_page():
    return render_template('forecast.html')


if __name__ == '__main__':
    app.run(debug=True)
