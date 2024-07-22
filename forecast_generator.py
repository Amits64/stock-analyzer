import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import ta  # Technical Analysis library
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Utility function to save plots
def save_plot(fig, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300)
    logging.info(f"Plot saved to {filename}")


# Load and preprocess data
def load_data(filename):
    """Load data from a CSV file."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File {filename} does not exist")
    df = pd.read_csv(filename)
    logging.info(f"Data loaded from {filename}. Shape: {df.shape}")
    return df


def preprocess_data(df):
    """Preprocess the data by cleaning and transforming."""
    required_columns = ['current_price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Expected columns {missing_columns} not found.")

    df = df[['current_price']].copy()
    df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce')
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("Processed data is empty after cleaning.")

    # Add technical indicators
    df['SMA_7'] = ta.trend.sma_indicator(df['current_price'], window=7)
    df['SMA_21'] = ta.trend.sma_indicator(df['current_price'], window=21)
    df['EMA_7'] = ta.trend.ema_indicator(df['current_price'], window=7)
    df['EMA_21'] = ta.trend.ema_indicator(df['current_price'], window=21)
    df['RSI'] = ta.momentum.rsi(df['current_price'], window=14)
    df['MACD'] = ta.trend.macd(df['current_price'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['current_price'])
    df['MACD_Diff'] = ta.trend.macd_diff(df['current_price'])
    df['BB_High'] = ta.volatility.bollinger_hband(df['current_price'])
    df['BB_Low'] = ta.volatility.bollinger_lband(df['current_price'])

    # Drop rows with NaN values generated by indicators
    df.dropna(inplace=True)

    logging.info(f"Data after preprocessing. Shape: {df.shape}")
    return df


def check_data_variance(prices):
    """Check if the data has low variance."""
    variance = np.var(prices)
    logging.info(f"Variance of prices: {variance:.4f}")
    if variance < 1e-6:
        logging.error("Data has very low variance, which may be constant or nearly constant.")
        return False
    return True


def check_stationarity(prices):
    """Check if the time series data is stationary."""
    result = adfuller(prices)
    p_value = result[1]
    logging.info(f'ADF Statistic: {result[0]}')
    logging.info(f'p-value: {p_value}')

    if p_value > 0.05:
        logging.warning("Data is not stationary. Differencing or transformation may be required.")
        return False
    return True


def optimize_arima_order(prices):
    """Optimize ARIMA parameters using a grid search."""

    def objective_function(params):
        p, d, q = int(params[0]), int(params[1]), int(params[2])
        if p < 0 or d < 0 or q < 0:
            return np.inf
        try:
            model = ARIMA(prices, order=(p, d, q))
            model_fit = model.fit()
            return model_fit.aic
        except Exception as e:
            logging.warning(f"ARIMA model failed for parameters (p={p}, d={d}, q={q}): {e}")
            return np.inf

    grid = product(range(5), repeat=3)  # Increased range for better tuning
    best_aic = np.inf
    best_order = None

    for p, d, q in grid:
        try:
            aic = objective_function((p, d, q))
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
        except Exception as e:
            logging.warning(f"ARIMA optimization failed for parameters (p={p}, d={d}, q={q}): {e}")

    logging.info(f'Best ARIMA order found: {best_order}')
    return best_order


def fit_sarimax_model(prices, exog, order, seasonal_order):
    """Fit the SARIMAX model and return forecast and confidence intervals."""
    if np.var(prices) < 1e-6:
        logging.error("Cannot fit SARIMAX model as the data is constant or nearly constant.")
        return None, None, None

    # Ensure exog has the same index as prices
    exog_df = pd.DataFrame(exog, index=prices.index,
                           columns=['SMA_7', 'SMA_21', 'EMA_7', 'EMA_21', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
                                    'BB_High', 'BB_Low'])

    try:
        model = SARIMAX(prices, exog=exog_df, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=30, exog=exog_df[-30:])
        conf_int = model_fit.get_forecast(steps=30, exog=exog_df[-30:]).conf_int(alpha=0.05)
        residuals = model_fit.resid
        return forecast, conf_int, residuals
    except Exception as e:
        logging.error(f"Failed to fit SARIMAX model: {e}")
        return None, None, None


def plot_forecast(prices, forecast, conf_int, residuals, coin_name):
    """Plot the forecast along with the historical data and confidence intervals."""
    if forecast is None or conf_int is None or residuals is None:
        logging.error("Unable to plot forecast due to model fitting issues.")
        return

    # Rename confidence interval columns to match the expected names
    conf_int.columns = ['lower_bound', 'upper_bound']

    # Create a larger figure
    fig, axs = plt.subplots(2, 2, figsize=(22, 18))  # Increased figure size
    fig.subplots_adjust(hspace=0.31, wspace=0.093)  # Increased space between subplots

    # Plot historical prices and forecast
    axs[0, 0].plot(prices, label='Historical Prices', color='blue')
    axs[0, 0].plot(range(len(prices), len(prices) + len(forecast)), forecast, label='Forecast', color='red')

    lower_bound = conf_int['lower_bound']
    upper_bound = conf_int['upper_bound']

    axs[0, 0].fill_between(range(len(prices), len(prices) + len(forecast)),
                           lower_bound, upper_bound, color='red', alpha=0.3, label='Confidence Interval')

    axs[0, 0].set_title(f'{coin_name} Price Forecast')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot residuals
    axs[0, 1].plot(residuals, color='purple')
    axs[0, 1].set_title('Residuals')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Residual')
    axs[0, 1].grid(True)

    # Plot ACF and PACF of residuals
    plot_acf(residuals, ax=axs[1, 0], lags=40)
    axs[1, 0].set_title('ACF of Residuals')
    axs[1, 0].grid(True)

    plot_pacf(residuals, ax=axs[1, 1], lags=40)
    axs[1, 1].set_title('PACF of Residuals')
    axs[1, 1].grid(True)

    # Save plots
    save_plot(fig, f'data/{coin_name}_forecast.png')

    logging.info(f"Forecast plot for {coin_name} saved.")


def main():
    """Main function to execute the forecasting process."""
    try:
        filename = 'data/processed_data.csv'  # Path to your data file
        df = load_data(filename)
        df = preprocess_data(df)

        prices = df['current_price']
        exog = df[
            ['SMA_7', 'SMA_21', 'EMA_7', 'EMA_21', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'BB_High', 'BB_Low']]

        if not check_data_variance(prices):
            return

        if not check_stationarity(prices):
            logging.info("Data is not stationary. Differencing will be applied.")
            # Apply differencing if needed
            prices = prices.diff().dropna()

        # Optimize ARIMA parameters
        arima_order = optimize_arima_order(prices)

        # Fit SARIMAX model
        seasonal_order = (1, 1, 1, 12)  # Example seasonal order
        forecast, conf_int, residuals = fit_sarimax_model(prices, exog, order=arima_order,
                                                          seasonal_order=seasonal_order)

        coin_name = 'crypto_coins'  # Change to dynamic coin name if needed
        plot_forecast(prices, forecast, conf_int, residuals, coin_name)
        logging.info("Forecasting process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    main()