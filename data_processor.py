import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, \
    StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import joblib
import redis
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Redis
redis_client = redis.StrictRedis(host='192.168.10.10', port=6379, db=0, decode_responses=True)

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_data(filepath):
    """Load data from JSON file."""
    if redis_client.exists(filepath):
        logging.info(f"Loading data from Redis cache for {filepath}")
        data = json.loads(redis_client.get(filepath))
    else:
        with open(filepath, 'r') as file:
            data = json.load(file)
        redis_client.set(filepath, json.dumps(data))
        logging.info(f"Data loaded from {filepath}. Shape: {len(data)} records")
    df = pd.DataFrame(data)
    logging.info(f"Data columns: {df.columns.tolist()}")
    logging.info(f"First few rows: {df.head()}")
    return df


def simulate_high_low(df):
    """Simulate high and low price columns if missing."""
    if 'high_price' not in df.columns or 'low_price' not in df.columns:
        logging.warning("Simulating high and low price columns.")
        df['high_price'] = df['current_price'] * (1 + np.random.uniform(0.01, 0.05, size=len(df)))
        df['low_price'] = df['current_price'] * (1 - np.random.uniform(0.01, 0.05, size=len(df)))
    return df


def add_features(df):
    """Add advanced features for modeling."""
    if 'high_price' not in df.columns or 'low_price' not in df.columns:
        logging.warning("High price and low price columns are missing.")
        # Handle missing columns appropriately
        return df

    # Moving Averages
    df['sma_5'] = df['current_price'].rolling(window=5).mean()
    df['sma_20'] = df['current_price'].rolling(window=20).mean()
    df['ema_5'] = df['current_price'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['current_price'].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df['current_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Additional RSI Periods
    gain_28 = (delta.where(delta > 0, 0)).rolling(window=28).mean()
    loss_28 = (-delta.where(delta < 0, 0)).rolling(window=28).mean()
    rs_28 = gain_28 / loss_28
    df['rsi_28'] = 100 - (100 / (1 + rs_28))

    # Lag Features
    df['lag_1'] = df['current_price'].shift(1)
    df['lag_2'] = df['current_price'].shift(2)
    df['lag_3'] = df['current_price'].shift(3)
    df['lag_5'] = df['current_price'].shift(5)

    # Bollinger Bands
    df['bollinger_mid'] = df['current_price'].rolling(window=20).mean()
    df['bollinger_std'] = df['current_price'].rolling(window=20).std()
    df['bollinger_upper'] = df['bollinger_mid'] + (df['bollinger_std'] * 2)
    df['bollinger_lower'] = df['bollinger_mid'] - (df['bollinger_std'] * 2)

    # MACD
    df['macd'] = df['current_price'].ewm(span=12, adjust=False).mean() - df['current_price'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # ATR (Average True Range)
    high_low = df['high_price'] - df['low_price']
    high_prev_close = np.abs(df['high_price'] - df['current_price'].shift(1))
    low_prev_close = np.abs(df['low_price'] - df['current_price'].shift(1))
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # Rate of Change
    df['roc'] = df['current_price'].pct_change(periods=12)

    df.dropna(inplace=True)
    return df


def preprocess_data(df):
    """Preprocess data for modeling."""
    df = add_features(df)
    logging.info(f"Data after feature engineering. Shape: {df.shape}")

    features = ['current_price', 'price_change_percentage_24h', 'sma_5', 'sma_20', 'ema_5', 'ema_20', 'rsi', 'lag_1', 'lag_2']
    X = df[features]
    y = df['current_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    numeric_features = ['current_price', 'price_change_percentage_24h', 'sma_5', 'sma_20', 'ema_5', 'ema_20', 'rsi', 'lag_1', 'lag_2']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(score_func=f_regression, k=10)),
                ('pca', PCA(n_components=0.95))
            ]), numeric_features)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


def build_model(preprocessor):
    """Build a sophisticated machine learning pipeline with hyperparameter tuning."""
    base_models = [
        ('lr', LinearRegression()),
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('elasticnet', ElasticNet(max_iter=10000)),
        ('rf', RandomForestRegressor()),
        ('extra_trees', ExtraTreesRegressor()),
        ('gbm', GradientBoostingRegressor()),
        ('hist_gbm', HistGradientBoostingRegressor())
    ]

    meta_model = GradientBoostingRegressor()

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', StackingRegressor(estimators=base_models, final_estimator=meta_model))
    ])

    param_grid = {
        'regressor__final_estimator__n_estimators': [100, 200],
        'regressor__final_estimator__learning_rate': [0.01, 0.1],
        'poly__degree': [2],
        'preprocessor__num__pca__n_components': [0.95, 0.9]
    }

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    logging.info("Model pipeline and parameter grid defined.")
    return grid_search


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the model using the test data."""
    logging.info("Starting model training...")
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"Mean Absolute Error: {mae}")
    logging.info(f"R^2 Score: {r2}")

    joblib.dump(model, 'model.pkl')
    logging.info("Model saved to 'model.pkl'")


def main():
    raw_data_file = 'data/raw_data.json'
    processed_data_file = 'data/processed_data.csv'

    df = load_data(raw_data_file)
    df = simulate_high_low(df)  # Ensure high_price and low_price are available

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    model = build_model(preprocessor)
    logging.info("Starting model training...")
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    evaluate_model(model, X_train, y_train, X_test, y_test)

    df.to_csv(processed_data_file, index=False)
    logging.info(f"Processed data saved to {processed_data_file}")


if __name__ == "__main__":
    main()
