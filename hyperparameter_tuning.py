from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import logging
import os
import joblib

logging.basicConfig(level=logging.INFO)


def grid_search(model, param_grid, X_train, y_train, scoring='accuracy', save_path=None):
    """
    Perform Grid Search for hyperparameter tuning.

    Parameters:
        model: Machine learning model.
        param_grid: Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
        X_train: Training features.
        y_train: Training target variable.
        scoring: Scoring method for evaluation. Default is 'accuracy'.
        save_path: Optional. Path to save the best model found. Default is None (not saving).

    Returns:
        Tuple of best parameters found by Grid Search and corresponding cross-validation scores.
    """
    try:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        cv_results = grid_search.cv_results_
        mean_scores = cv_results['mean_test_score']

        logging.info(f"Best parameters found: {best_params}")
        logging.info(f"Cross-validation scores:\n{mean_scores}")

        if save_path:
            best_model = grid_search.best_estimator_
            joblib.dump(best_model, os.path.join(save_path, 'best_model.pkl'))
            logging.info(f"Best model saved at {os.path.join(save_path, 'best_model.pkl')}")

        return best_params, mean_scores

    except Exception as e:
        logging.error(f"Error during Grid Search: {str(e)}")
        raise


def random_search(model, param_distributions, X_train, y_train, scoring='accuracy', save_path=None):
    """
    Perform Random Search for hyperparameter tuning.

    Parameters:
        model: Machine learning model.
        param_distributions: Dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
        X_train: Training features.
        y_train: Training target variable.
        scoring: Scoring method for evaluation. Default is 'accuracy'.
        save_path: Optional. Path to save the best model found. Default is None (not saving).

    Returns:
        Tuple of best parameters found by Random Search and corresponding cross-validation scores.
    """
    try:
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, scoring=scoring, verbose=1,
                                           n_jobs=-1)
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        cv_results = random_search.cv_results_
        mean_scores = cv_results['mean_test_score']

        logging.info(f"Best parameters found: {best_params}")
        logging.info(f"Cross-validation scores:\n{mean_scores}")

        if save_path:
            best_model = random_search.best_estimator_
            joblib.dump(best_model, os.path.join(save_path, 'best_model.pkl'))
            logging.info(f"Best model saved at {os.path.join(save_path, 'best_model.pkl')}")

        return best_params, mean_scores

    except Exception as e:
        logging.error(f"Error during Random Search: {str(e)}")
        raise
