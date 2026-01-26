import pandas as pd, numpy as np 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def metrics(y_true:np.array, y_pred:np.array):
    
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse = np.power(mse, 0.5)
    return pd.DataFrame([mae, mape, mse, rmse], index=['MAE', 'MAPE', 'MSE', 'RMSE']).T
