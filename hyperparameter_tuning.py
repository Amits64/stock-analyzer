# hyperparameter_tuning.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


def grid_search(model, param_grid, X_train, y_train):
    """
    Perform Grid Search for hyperparameter tuning.

    Parameters:
        model: Machine learning model
        param_grid: Dictionary with parameters names (string) as keys and lists of parameter settings to try as values
        X_train: Training features
        y_train: Training target variable

    Returns:
        Best parameters found by Grid Search
    """
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def random_search(model, param_distributions, X_train, y_train):
    """
    Perform Random Search for hyperparameter tuning.

    Parameters:
        model: Machine learning model
        param_distributions: Dictionary with parameters names (string) as keys and distributions or lists of parameters to try
        X_train: Training features
        y_train: Training target variable

    Returns:
        Best parameters found by Random Search
    """
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5)
    random_search.fit(X_train, y_train)
    return random_search.best_params_
