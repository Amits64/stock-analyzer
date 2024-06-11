# Hyperparameter_tuning

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


def grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def random_search(model, param_distributions, X_train, y_train):
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5)
    random_search.fit(X_train, y_train)
    return random_search.best_params_


def perform_k_fold_cross_validation(model, X, y, cv=5):
    """
    Perform k-fold cross-validation for evaluating model performance.

    Parameters:
        model: Machine learning model
        X: Features
        y: Target variable
        cv: Number of folds for cross-validation

    Returns:
        Mean cross-validated score
    """
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean()
