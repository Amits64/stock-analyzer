# cross-validation

from sklearn.model_selection import cross_val_score


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
