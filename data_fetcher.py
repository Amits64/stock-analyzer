import json
import os
import pandas as pd
import requests
import logging
import redis
import pymysql
from pymysql.err import MySQLError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Redis connection
redis_host = '192.168.10.10'  # Replace with your Redis server host
redis_port = 6379  # Default Redis port
redis_db = 0  # Default Redis database index

# Initialize Redis connection
r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

# MySQL connection parameters
mysql_host = '192.168.10.10'  # Replace with your MySQL server host
mysql_user = 'root'       # Replace with your MySQL username
mysql_password = 'Kubernetes@1993'  # Replace with your MySQL password
mysql_database = 'crypto_coins'  # Replace with your MySQL database name


def fetch_all_cryptocurrencies(days_back=30):
    base_url = "https://api.coingecko.com/api/v3"
    endpoint = "/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,  # Adjust as needed
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "1h,24h,7d",  # Include price change percentages
    }

    try:
        response = requests.get(base_url + endpoint, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
        if not data:
            logging.warning("No data fetched from API.")
            return pd.DataFrame()  # Return an empty DataFrame if no data
        df = pd.DataFrame(data, columns=["id", "symbol", "name", "current_price", "price_change_percentage_24h"])
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


def save_data_to_json(data, filename="data/raw_data.json"):
    try:
        data_dict = data.to_dict(orient="records")
        with open(filename, "w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        logging.info(f"Data saved to {filename}")
    except IOError as e:
        logging.error(f"An error occurred while saving data to {filename}: {e}")


def save_data_to_redis(data):
    try:
        data_dict = data.to_dict(orient="records")
        for record in data_dict:
            key = f"crypto:{record['id']}"
            r.set(key, json.dumps(record))
        logging.info("Data saved to Redis")
    except redis.RedisError as e:
        logging.error(f"An error occurred while saving data to Redis: {e}")


def save_data_to_mysql(data):
    connection = None
    try:
        connection = pymysql.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database
        )
        cursor = connection.cursor()

        # Create table if it does not exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cryptocurrencies (
            id VARCHAR(255) PRIMARY KEY,
            symbol VARCHAR(255),
            name VARCHAR(255),
            current_price DECIMAL(18, 6),
            price_change_percentage_24h DECIMAL(5, 2)
        );
        """
        cursor.execute(create_table_query)

        # Insert data into the table
        insert_query = """
        INSERT INTO cryptocurrencies (id, symbol, name, current_price, price_change_percentage_24h)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            symbol = VALUES(symbol),
            name = VALUES(name),
            current_price = VALUES(current_price),
            price_change_percentage_24h = VALUES(price_change_percentage_24h);
        """
        data_records = data.to_dict(orient="records")
        cursor.executemany(insert_query, [(record['id'], record['symbol'], record['name'], record['current_price'], record['price_change_percentage_24h']) for record in data_records])
        connection.commit()

        logging.info("Data saved to MySQL")
    except MySQLError as e:
        logging.error(f"An error occurred while saving data to MySQL: {e}")
    finally:
        if connection:
            if connection.open:
                cursor.close()
                connection.close()


def main():
    all_crypto_data = fetch_all_cryptocurrencies(days_back=30)
    if not all_crypto_data.empty:
        save_data_to_json(all_crypto_data)
        save_data_to_redis(all_crypto_data)
        save_data_to_mysql(all_crypto_data)
        logging.info(f"Data for {len(all_crypto_data)} cryptocurrencies saved to data/raw_data.json, Redis, and MySQL")
    else:
        logging.warning("No data available to save.")


if __name__ == '__main__':
    main()
