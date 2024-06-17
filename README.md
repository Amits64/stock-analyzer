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
- API integration for data retrival.

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

3. Install required Docker Images
  ```
  docker run -d --name mysql-custom -e MYSQL_ROOT_PASSWORD=<PASSWORD> -d mysql:tag
  ```
  ```
  docker run -d --name redis-custom -d redis:latest
  ```
  ```
  docker run -d \
   -p 8086:8086 \
   -v "$PWD/data:/var/lib/influxdb2" \
   -v "$PWD/config:/etc/influxdb2" \
   influxdb:2
  ``` 
5. Run the Flask application:
  ```
  python app.py
  ```

## To use different APIs, use below commands:
  ```
  curl -G http://localhost:5000/api/crypto-data --data-urlencode "expected_return=1.0"
  ```

  ```
  curl -G http://localhost:5000/api/historical-price --data-urlencode "symbol=BTC" --data-urlencode "start_date=2022-01-01" --data-urlencode "end_date=2022-01-07"
  ```

  ```
  curl http://localhost:5000/api/predict-next-week-price/BTC
  ```

  ```
  curl http://localhost:5000/api/available-cryptocurrencies
  ```

  ```
  curl http://localhost:5000/api/analyze-crypto-data
  ```
## Screenshots:

![image](https://github.com/Amits64/stock-analyzer/assets/135766785/030c8299-2aa2-4844-8638-9d2b6613be2d)

![newplot (1)](https://github.com/Amits64/stock-analyzer/assets/135766785/9ba7e7a0-77c8-4c63-a0ff-f7eff4ecdc51)



## Usage

- Access the dashboard at [http://localhost:5000](http://localhost:5000) after starting the Flask server.
- Navigate through the dashboard to view cryptocurrency data, predictions, and candlestick charts.
- Handle errors gracefully with custom error pages.

## Contributing

Contributions are welcome! Please create issues for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

