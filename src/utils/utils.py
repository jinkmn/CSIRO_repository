import numpy as np
import random
import os
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

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
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    target_weights = np.array(target_weights)
    
    # 形状合わせ: (5, N) -> (5, N) の重み行列を作成
    if y_true.shape != y_pred.shape:
         raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
         
    W = np.tile(target_weights[:, np.newaxis], (1, y_true.shape[1]))
        
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    W_flat = W.ravel()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat, sample_weight=W_flat)
    rmse = np.sqrt(mse)
    
    r2 = r2_score(y_true_flat, y_pred_flat, sample_weight=W_flat)
    
    return rmse, r2

def preprocess_train_data(train_df, n_splits=5):
    """
    1. 画像ごとに1行になるようにピボット（データの整列問題を解決）
    2. Group(Sampling_Date) かつ Stratified(State) でFoldを作成
    """
    # ターゲット名をカラムに変換 (Wide形式へ)
    # これにより image_path と target の対応関係が完全に保証されます
    pivot_df = train_df.pivot_table(
        index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
        columns='target_name', 
        values='target'
    ).reset_index()

    # Foldの作成
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    pivot_df['fold'] = -1
    
    # y=State (層化), groups=Sampling_Date (グループ)
    # StratifiedGroupKFold は y を層化の基準、groups をグループの基準とします
    for fold, (_, val_idx) in enumerate(sgkf.split(pivot_df, pivot_df['State'], groups=pivot_df['Sampling_Date'])):
        pivot_df.loc[val_idx, 'fold'] = fold
        
    return pivot_df

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BiomassWeightedMSELoss(nn.Module):
    def __init__(self, weights=[0.1, 0.1, 0.1, 0.5, 0.2], device="cuda"):
        """
        weights: コンペ評価指標の重みリスト
                 [Clover, Dead, Green, Total, GDM] の順と仮定
        """
        super().__init__()
        self.weights = torch.tensor(weights).to(device)
        self.mse = nn.MSELoss(reduction='none') 

    def forward(self, preds, targets):
        """
        preds:   Modelの出力 (Batch, 3) -> [Clover, Dead, Green] と仮定
        targets: 正解ラベル (Batch, 5) -> [Clover, Dead, Green, Total, GDM]
        """
        
        loss = self.mse(preds, targets) 
        weighted_loss = torch.sum(loss * self.weights)
        return weighted_loss