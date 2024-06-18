# neural_network.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


def build_lstm_model(input_shape, units=50, dropout=0.2):
    try:
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units))
        model.add(Dropout(dropout))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        logging.error(f"Error occurred during model building: {str(e)}")
        return None


def train_model(X_train, y_train, units=50, dropout=0.2, epochs=50, batch_size=32):
    try:
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=units, dropout=dropout)
        if model is None:
            return None
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return model
    except Exception as e:
        logging.error(f"Error occurred during model training: {str(e)}")
        return None


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return mse, mae
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {str(e)}")
        return None, None


def predict_next_week_prices(model, latest_data):
    try:
        X = latest_data.values.reshape((1, latest_data.shape[0], latest_data.shape[1]))
        predicted_prices = model.predict(X)
        return predicted_prices.flatten()[0]
    except Exception as e:
        logging.error(f"Error occurred during price prediction: {str(e)}")
        return None


def perform_hyperparameter_tuning(X_train, y_train):
    try:
        # Dummy implementation for hyperparameter tuning
        best_params = {'units': 50, 'dropout': 0.2}
        return best_params
    except Exception as e:
        logging.error(f"Error occurred during hyperparameter tuning: {str(e)}")
        return None
