# bin/run_ml.py
import sys
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np

# srcパスの追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 【New】前処理モジュールのインポート
from src.data.preprocessing import get_unique_image_paths, prepare_train_xy, parse_test_row, TARGET_COLUMNS

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"=== Experiment: {cfg.exp_name} ===")
    root = cfg.dir.data_dir
    
    # --- 1. 特徴抽出器の準備 ---
    extractor = hydra.utils.instantiate(cfg.feature)
    
    # --- 2. 学習データの読み込みと特徴抽出 ---
    train_df = pd.read_csv(os.path.join(root, "train.csv"))
    
    # ★ ここで preprocessing の関数を使用
    train_img_paths = get_unique_image_paths(train_df, root)

    limit = cfg.dir.data_limit
    if limit is not None:
        print(f"⚠️ Debug Mode: Limiting data to {limit} samples.")
        train_img_paths = train_img_paths[:limit]
    
    print(f"Extracting features for {len(train_img_paths)} training images...")
    train_embeds = extractor.extract(train_img_paths)
    
    # 辞書化 (画像パス -> 特徴量)
    # パスはフルパスになっているので、ファイル名または相対パスで辞書化するか、
    # get_unique_image_pathsの実装に合わせてキーを調整する必要があります。
    # 今回は get_unique_image_paths がフルパスを返すので、キーも元DFのimage_pathに合わせるため工夫します。
    # (train_df['image_path'] は相対パス)
    
    # 辞書のキーを「相対パス(train_dfに入ってる値)」にする
    rel_paths = [os.path.relpath(p, root) for p in train_img_paths]
    # Windows環境などでパス区切り文字がズレないように注意が必要ですが、基本はこれでOK
    path_to_embed = {p: embed for p, embed in zip(rel_paths, train_embeds)}
    
    # --- 3. データの整形 (X, yの作成) ---
    # ★ ここで preprocessing の関数を使用
    targets_data = prepare_train_xy(train_df, path_to_embed)

    # --- 4. 回帰モデルの学習 ---
    print("Training Regressors...")
    trained_models = {}
    
    for i in range(5):
        target_name = TARGET_COLUMNS[i]
        X = np.array(targets_data[i]['X'])
        y = np.array(targets_data[i]['y'])
        
        # モデル生成
        model = hydra.utils.instantiate(cfg.model)
        model.n_splits = cfg.training.n_splits
        model.random_state = cfg.training.random_state
        
        # 学習
        oof_preds = model.fit_predict(X, y)
        trained_models[i] = model
        
        mse = np.mean((y - oof_preds)**2)
        print(f"  Target: {target_name} | MSE: {mse:.4f}")

    # --- 5. テストデータの推論 ---
    print("Running inference on test data...")
    test_df = pd.read_csv(os.path.join(root, "test.csv"))
    
    # ★ ここでも preprocessing の関数を利用
    test_img_paths = get_unique_image_paths(test_df, root)
    test_embeds = extractor.extract(test_img_paths)
    
    rel_test_paths = [os.path.relpath(p, root) for p in test_img_paths]
    test_path_to_embed = {p: embed for p, embed in zip(rel_test_paths, test_embeds)}
    
    predictions = []
    sample_ids = []
    
    for _, row in test_df.iterrows():
        # ★ ヘルパー関数でパース
        sample_id, img_path, t_idx = parse_test_row(row)
        
        if t_idx is None: 
            predictions.append(0)
            sample_ids.append(sample_id)
            continue

        X_test = test_path_to_embed[img_path].reshape(1, -1)
        pred = trained_models[t_idx].predict(X_test)[0]
        
        predictions.append(max(0.0, pred))
        sample_ids.append(sample_id)
        
    # 保存
    submission = pd.DataFrame({'sample_id': sample_ids, 'target': predictions})
    save_path = os.path.join(os.getcwd(), "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Saved submission to {save_path}")

if __name__ == "__main__":
    main()