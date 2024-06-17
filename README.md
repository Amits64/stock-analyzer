# Crypto Coins Prediction Dashboard

This Flask application provides a dashboard for analyzing and predicting cryptocurrency prices using various APIs and machine learning models.

## Overview

The application fetches data from CoinGecko API, stores it in MySQL database, and performs predictive analytics using LSTM models. Predictions are stored in MySQL and visualized using Plotly for candlestick charts.

## Features

- Fetches cryptocurrency data from CoinGecko API.
- Stores data in MySQL database.
- Predicts cryptocurrency prices using LSTM models.
- Visualizes candlestick charts using Plotly.
- Provides error handling and logging.

 ## Requirements
    
    Python 3.x
    Flask
    pandas
    numpy
    requests
    pymysql
    yfinance
    influxdb-client
    scikit-learn
    plotly
    diskcache
    tensorflow
    matplotlib
    redis
    
    
Install dependencies using:
  ```
  pip install -r requirements.txt
  ```


## Configuration

Set the following environment variables:

- `MYSQL_HOST`: Hostname of MySQL server.
- `MYSQL_PORT`: Port number of MySQL server.
- `MYSQL_USER`: MySQL username.
- `MYSQL_PASSWORD`: MySQL password.
- `MYSQL_DB`: MySQL database name.
- `INFLUXDB_URL`: URL of InfluxDB server.
- `INFLUXDB_TOKEN`: Authentication token for InfluxDB.
- `INFLUXDB_ORG`: InfluxDB organization.
- `INFLUXDB_BUCKET`: InfluxDB bucket name.
- `REDIS_HOST`: Hostname of Redis server.
- `REDIS_PORT`: Port number of Redis server.
- `REDIS_DB`: Redis database number.

## Setup

1. Clone the repository:
  ```
  git clone https://github.com/yourusername/crypto-coins-prediction.git
  cd crypto-coins-prediction
  ```


2. Install dependencies:
  ```
  pip install -r requirements.txt
  ```

3. Run the Flask application:
  ```
  python app.py
  ```

## Usage

- Access the dashboard at [http://localhost:5000](http://localhost:5000) after starting the Flask server.
- Navigate through the dashboard to view cryptocurrency data, predictions, and candlestick charts.
- Handle errors gracefully with custom error pages.

## Contributing

Contributions are welcome! Please create issues for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

