import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def arima_forecast(df, coin_id, forecast_steps=30):
    series = df['price']  # Adjust column name as necessary

    # Use a subset of data for training to speed up computation
    series_subset = series.tail(100)  # Adjust the number of samples as needed

    # Define ARIMA model parameters
    p, d, q = 1, 1, 1  # Example parameters, adjust as needed

    try:
        # Fit ARIMA model
        arima_model = ARIMA(series_subset, order=(p, d, q))
        arima_results = arima_model.fit()

        # Generate forecasts
        arima_forecast = arima_results.get_forecast(steps=forecast_steps)

        # Plot the forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series[-100:], label='Historical Data')  # Plot only subset for visualization
        ax.plot(arima_forecast.predicted_mean, label='Forecast', color='red')
        ax.fill_between(arima_forecast.index,
                        arima_forecast.conf_int()[:, 0],
                        arima_forecast.conf_int()[:, 1],
                        color='gray', alpha=0.3)

        # Plot formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'ARIMA Forecast for {coin_id}')
        ax.legend()

        # Save the figure
        fig_path = f'static/{coin_id}_arima_forecast.png'
        fig.savefig(fig_path)

        return fig_path, arima_forecast.predicted_mean, arima_forecast.conf_int()

    except Exception as e:
        # Handle errors
        print(f"Error in ARIMA forecast: {e}")
        return None, None, None
