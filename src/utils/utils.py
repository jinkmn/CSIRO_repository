import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calc_metrics(y_true, y_pred):
    """
    RMSEとR2スコアを計算して返す
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2