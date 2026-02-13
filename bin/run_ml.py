import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, Normalizer
from PIL import Image
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import get_unique_image_paths, prepare_train_xy, parse_test_row, TARGET_COLUMNS
from src.utils.utils import calc_metrics, calc_weighted_metrics, preprocess_train_data

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wandb.init(
        project='CSIRO_Competition', 
        name=cfg.exp_name,       
        config=OmegaConf.to_container(cfg, resolve=True), 
        reinit=True,
        group=f'{cfg.model._target_.split(".")[-1]}',  
        job_type="train",
        mode=cfg.get("wandb_mode", "online")
    )

    print(f"=== Experiment: {cfg.exp_name} ===")
    ROOT = cfg.dir.data_dir

    output_dir = os.path.join(os.getcwd(), cfg.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = hydra.utils.instantiate(cfg.feature)
    
    raw_train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    
    train_df = preprocess_train_data(raw_train_df, n_splits=5)
    
    target_columns = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    target_mapping = {col: i for i, col in enumerate(target_columns)}
    
    train_img_paths = [os.path.join(ROOT, p) for p in train_df['image_path']]

    limit = cfg.dir.data_limit
    if cfg.exp_name == 'test' :
        print(f"⚠️ Debug Mode: Limiting data to {limit} samples.")
        train_img_paths = train_img_paths[:limit]
        train_df = train_df.iloc[:limit] 
    
    print(f"Extracting features for {len(train_img_paths)} training images...")
    embeds_np = extractor.extract(train_img_paths)

    targets_matrix = train_df[target_columns].values  # Shape: (N, 5)

    regressors = [[None for _ in range(5)] for _ in range(5)]
    # Initialize an array to store OOF predictions
    oof_preds_np = np.zeros((len(embeds_np), 5))
    
    metrics_log = {}

    print("  Training Lasso regression models...")
    
    for i, target_name in enumerate(target_columns): 
        # このターゲットの正解ラベル (N,)
        y_target_all = targets_matrix[:, i]
        
        print(f" Training for target: {target_name}")
        
        fold_rmse_list = []
        fold_r2_list = []
    
        for fold in range(5):
            # Boolean indexing でマスクを作成
            train_mask = (train_df['fold'] != fold).values
            val_mask = (train_df['fold'] == fold).values
            
            X_train, y_train = embeds_np[train_mask], y_target_all[train_mask]
            X_val, y_val = embeds_np[val_mask], y_target_all[val_mask]
            
            # モデル学習
            reg = hydra.utils.instantiate(cfg.model)
            preds, reg = reg.fit_predict(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
            oof_preds_np[val_mask, i] = preds
            
            rmse, r2 = calc_metrics(y_val, preds)
            fold_rmse_list.append(rmse)
            fold_r2_list.append(r2)
            
            print(f"  Fold {fold}: RMSE={rmse:.4f}, R2={r2:.4f}")
    
            regressors[i][fold] = reg # Also save the model for test prediction

        avg_rmse = np.mean(fold_rmse_list)
        avg_r2 = np.mean(fold_r2_list)
        print(f"Target {target_name} - Average RMSE: {avg_rmse:.4f}, Average R2: {avg_r2:.4f}")
        
        # WandBログ用に記録
        metrics_log[f"{target_name}_rmse"] = avg_rmse
        metrics_log[f"{target_name}_r2"] = avg_r2
    
    valid_weights = [0.1, 0.1, 0.1, 0.5, 0.2]

    overall_rmse, overall_r2 = calc_weighted_metrics(targets_matrix.T, oof_preds_np.T, valid_weights)
    
    print(f"\n=== Overall CV Score ===")
    print(f"Weighted RMSE: {overall_rmse:.5f}")
    print(f"Weighted R2:   {overall_r2:.5f}")
    
    metrics_log["overall_weighted_rmse"] = overall_rmse
    metrics_log["overall_weighted_r2"] = overall_r2
    wandb.log(metrics_log)
    
    # OOF保存 (train_dfベースなのでシンプルになります)
    oof_df = train_df.copy()
    for i, col in enumerate(target_columns):
        oof_df[f"pred_{col}"] = oof_preds_np[:, i]
        
    oof_save_path = os.path.join(output_dir, f'oof_model_{cfg.exp_name}.csv')
    oof_df.to_csv(oof_save_path, index=False)

    # --- Test Prediction ---
    print("  Running predictions on test data...")
    test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    
    unique_test_images = test_df.drop_duplicates(subset=['image_path']).reset_index()
    test_img_paths = [os.path.join(ROOT, p) for p in unique_test_images['image_path']]
    
    test_embeds_np = extractor.extract(test_img_paths)
    
    # 画像名とEmbeddingの辞書作成
    img_names = []
    for img_path in unique_test_images['image_path']:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_names.append(img_name)
    test_embeds = {img_name: embed for img_name, embed in zip(img_names, test_embeds_np)}

    predictions, sample_ids = [], []

    # テストデータの予測ループ
    # test.csvは1行=1ターゲット(Long format)なので、そのままイテレート
    for _, entry in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        sample_id = entry['sample_id']
        img_name, target_name = sample_id.split('__')
        
        if img_name in test_embeds:
            X_test = np.array(test_embeds[img_name]).reshape(1, -1)
            target_idx = target_mapping[target_name]
            fold_preds = []
            for reg in regressors[target_idx]:
                pred = reg.predict(X_test)
                fold_preds.append(pred[0])
            prediction = np.mean(fold_preds)
            predictions.append(max(0.0, prediction)) 

        else:
            predictions.append(0.0)
            
        sample_ids.append(sample_id)

    submission = pd.DataFrame({'sample_id': sample_ids, 'target': predictions})
    submission = submission.reset_index(drop=True)
    sub_save_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(sub_save_path, index=False)
    print("Submission file 'submission.csv' created.")
    
    wandb.finish()

if __name__ == "__main__":
    main()