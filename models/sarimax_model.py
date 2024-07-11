import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


def sarimax_forecast(df, coin_id, forecast_steps=30):
    series = df['price']  # Adjust column name as necessary

    # Define SARIMAX model parameters
    p, d, q = 1, 1, 1  # Example non-seasonal parameters, adjust as needed
    P, D, Q, m = 1, 1, 1, 12  # Example seasonal parameters, adjust as needed

    try:
        # Fit SARIMAX model
        sarimax_model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, m))
        sarimax_results = sarimax_model.fit()

        # Generate forecasts
        sarimax_forecast = sarimax_results.get_forecast(steps=forecast_steps)

        # Plot the forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series[-100:], label='Historical Data')
        ax.plot(sarimax_forecast.predicted_mean, label='Forecast', color='green')
        ax.fill_between(sarimax_forecast.index,
                        sarimax_forecast.conf_int()[:, 0],
                        sarimax_forecast.conf_int()[:, 1],
                        color='gray', alpha=0.3)

        # Plot formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'SARIMAX Forecast for {coin_id}')
        ax.legend()

        # Save the figure
        fig_path = f'static/{coin_id}_sarimax_forecast.png'
        fig.savefig(fig_path)

        return fig_path, sarimax_forecast.predicted_mean, sarimax_forecast.conf_int()

    except Exception as e:
        # Handle errors
        print(f"Error in SARIMAX forecast: {e}")
        return None, None, None
