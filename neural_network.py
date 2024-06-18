# Neural_Network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error


def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(X_train, y_train, units=50, dropout=0.2, epochs=50, batch_size=32):
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=units, dropout=dropout)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae


def predict_next_week_prices(model, latest_data):
    X = latest_data.values.reshape((1, latest_data.shape[0], latest_data.shape[1]))
    predicted_prices = model.predict(X)
    return predicted_prices.flatten()[0]


def perform_hyperparameter_tuning(X_train, y_train):
    # Dummy implementation for hyperparameter tuning
    best_params = {'units': 50, 'dropout': 0.2}
    return best_params
