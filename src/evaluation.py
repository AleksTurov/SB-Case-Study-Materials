from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using RMSE, R2, and MAE metrics.
    
    :param y_true: True target values.
    :param y_pred: Predicted target values.
    :return: Dictionary with RMSE, R2, MAE scores.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae
    }
    
    return metrics