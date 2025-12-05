import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calc_metrics(y_true, y_pred):
    """
    RMSEとR2スコアを計算して返す
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def calc_weighted_metrics(y_true, y_pred, target_weights):
    """
    画像の数式に基づき、全要素を一括として重み付きRMSEとR2を計算する
    
    Args:
        y_true: (N, 5) の正解ラベル
        y_pred: (N, 5) の予測値
        target_weights: (5,) の各ターゲットに対する重みリスト
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    target_weights = np.array(target_weights)
    w_col = target_weights.reshape(-1, 1)
    
    W = np.tile(w_col, (1, y_true.shape[1]))
    
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    W_flat = W.ravel()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat, sample_weight=W_flat)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_true_flat, y_pred_flat, sample_weight=W_flat)
    
    return rmse, r2