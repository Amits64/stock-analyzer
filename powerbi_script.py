import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_forecast(prices, forecast, conf_int, residuals, coin_name):
    fig, axs = plt.subplots(2, 2, figsize=(22, 18))
    fig.subplots_adjust(hspace=0.31, wspace=0.093)

    # Historical prices and forecast
    axs[0, 0].plot(prices, label='Historical Prices', color='blue')
    axs[0, 0].plot(range(len(prices), len(prices) + len(forecast)), forecast, label='Forecast', color='red')
    axs[0, 0].fill_between(range(len(prices), len(prices) + len(forecast)),
                           conf_int['lower_bound'], conf_int['upper_bound'], color='red', alpha=0.3, label='Confidence Interval')
    axs[0, 0].set_title(f'{coin_name} Price Forecast')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Residuals distribution
    sns.histplot(residuals, kde=True, color='blue', ax=axs[0, 1])
    axs[0, 1].set_title('Residuals Distribution')
    axs[0, 1].set_xlabel('Residuals')
    axs[0, 1].set_ylabel('Frequency')

    # ACF of residuals
    plot_acf(residuals, lags=30, ax=axs[1, 0])
    axs[1, 0].set_title('ACF of Residuals')

    # PACF of residuals
    plot_pacf(residuals, lags=30, ax=axs[1, 1])
    axs[1, 1].set_title('PACF of Residuals')

    plt.tight_layout()
    plt.savefig('data/forecast_results.png', dpi=300)
    plt.close()
