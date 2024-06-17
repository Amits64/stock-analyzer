import pandas as pd
import yfinance as yf
import requests
import logging


def fetch_historical_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        # Fetch additional data from another source, e.g., CoinGecko
        coingecko_url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': 'max',
            'interval': 'daily'
        }
        response = requests.get(coingecko_url, params=params)
        data = response.json()
        # Convert to DataFrame and merge
        df_coingecko = pd.DataFrame(data['prices'], columns=['date', 'price'])
        df_coingecko['date'] = pd.to_datetime(df_coingecko['date'], unit='ms')
        df_coingecko.set_index('date', inplace=True)
        df = df.merge(df_coingecko, how='outer', left_index=True, right_index=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return pd.DataFrame()


def preprocess_data(df):
    try:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        return pd.DataFrame()


def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal


def fetch_available_cryptocurrencies():
    return ['bitcoin', 'ethereum', 'litecoin', 'ripple']
