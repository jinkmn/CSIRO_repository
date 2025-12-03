# bin/run_ml.py
import sys
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
import gc
from tqdm import tqdm
from sklearn.linear_model import Lasso
from PIL import Image


# srcパスの追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import get_unique_image_paths, prepare_train_xy, parse_test_row, TARGET_COLUMNS
from src.utils.utils import calc_metrics

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"=== Experiment: {cfg.exp_name} ===")
    ROOT = cfg.dir.data_dir
    
    # --- 1. 特徴抽出器の準備 ---
    extractor = hydra.utils.instantiate(cfg.feature)
    
    # --- 2. 学習データの読み込みと特徴抽出 ---
    train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    unique_train_images = train_df.drop_duplicates(subset=['image_path']).reset_index()

    targets = [[] for _ in range(5)]
    target_mapping = {"Dry_Clover_g": 0, "Dry_Dead_g": 1, "Dry_Green_g": 2, "Dry_Total_g": 3, "GDM_g": 4}  
    train_df['target_name'] = train_df['sample_id'].apply(lambda x: x.split('__')[1])
    
    # ★ ここで preprocessing の関数を使用
    train_img_paths = [os.path.join(ROOT, p) for p in unique_train_images['image_path']]

    limit = cfg.dir.data_limit
    if limit is not None:
        print(f"⚠️ Debug Mode: Limiting data to {limit} samples.")
        train_img_paths = train_img_paths[:limit]
    
    print(f"Extracting features for {len(train_img_paths)} training images...")
    embeds_np = extractor.extract(train_img_paths)

    for _, row in train_df.iterrows():
        target_idx = target_mapping[row['target_name']] 
        targets[target_idx].append(torch.tensor([[row['target']]]))

    regressors = [[None for _ in range(5)] for _ in range(5)]
    # Initialize an array to store OOF predictions
    oof_preds_np = np.zeros((len(embeds_np), 5))

    print("  Training Lasso regression models...")
    for i in range(5): # For each target (Dry_Clover_g, Dry_Dead_g, ...)
        targets_np = np.array(torch.cat(targets[i]))
        key = [k for k, v in target_mapping.items() if v == i]
        print(f" Training for target: {key}")
        
        # Split using KFold (more robust than random split)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
    
        for fold, (train_idxs, val_idxs) in enumerate(kf.split(embeds_np)):
            X_train, y_train = embeds_np[train_idxs], targets_np[train_idxs]
            X_val, y_val = embeds_np[val_idxs], targets_np[val_idxs]
            
            reg = hydra.utils.instantiate(cfg.model)
            reg.fit(X_train, y_train)
            
            # Calculate and save OOF predictions
            preds = reg.predict(X_val).flatten()
            oof_preds_np[val_idxs, i] = preds
            rmse, r2 = calc_metrics(y_val, preds)
            fold_scores.append((rmse, r2))
            print(f"  Fold {fold}: RMSE={rmse:.4f}, R2={r2:.4f}")
    
            regressors[i][fold] = reg # Also save the model for test prediction
        
        target_columns = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        oof_df = pd.DataFrame(oof_preds_np, columns=target_columns)
        oof_df['image_path'] = unique_train_images['image_path']
        oof_df.to_csv(f'oof_model_{cfg.exp_name}.csv', index=False)
        all_rmse =  fold_scores[0][0]*0.1 + fold_scores[1][0]*0.1 + fold_scores[2][0]*0.1 + fold_scores[3][0]*0.5 + fold_scores[4][0]*0.2
        all_r2 = fold_scores[0][1]*0.1 + fold_scores[1][1]*0.1 + fold_scores[2][1]*0.1 + fold_scores[3][1]*0.5 + fold_scores[4][1]*0.2
        print(f"Average RMSE: {all_rmse:.4f}, Average R2: {all_r2:.4f}")


    print("  Running predictions on test data...")
    test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    unique_test_images = test_df.drop_duplicates(subset=['image_path']).reset_index()
    # ★ ここで preprocessing の関数を使用
    test_img_paths = [os.path.join(ROOT, p) for p in unique_test_images['image_path']]
    test_embeds_np = extractor.extract(test_img_paths)
    img_names = []
    for img_path in tqdm(test_df['image_path'].unique(), desc="Extracting test features"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_names.append(img_name)
    test_embeds = {img_name: embed for img_name, embed in zip(img_names, test_embeds_np)}

    predictions, sample_ids = [], []

    for _, entry in test_df.iterrows():
        sample_id = entry['sample_id']
        img_name, target_name = sample_id.split('__')
        X_test = np.array(test_embeds[img_name]).reshape(1, -1)
        target_idx = target_mapping[target_name]
        fold_preds = [reg.predict(X_test) for reg in regressors[target_idx]]
        prediction = np.mean(fold_preds)
        predictions.append(max(0.0, prediction))
        sample_ids.append(sample_id)

    submission = pd.DataFrame({'sample_id': sample_ids, 'target': predictions})
    submission.sort_values('sample_id').reset_index(drop=True)
    submission.to_csv(f'submission_{cfg.exp_name}', index=False)


if __name__ == "__main__":
    main()