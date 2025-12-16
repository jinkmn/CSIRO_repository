import sys
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import hydra

# srcをインポート可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 自作モジュールのインポート
from src.data import DualStreamDataset, TransformFactory, PredictionProcessor
from src.data.preprocessing import get_unique_image_paths # 必要に応じて

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"=== Inference: {cfg.exp_name} ===")
    
    # ------------------------------------------------------------------
    # 0. 設定とパスの準備
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    root_dir = Path(cfg.dir.data_dir)
    weight_dir = Path(cfg.dir.model_weight_dir)
    
    # テストデータの読み込み
    test_csv_path = root_dir / "test.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")
    
    test_df = pd.read_csv(test_csv_path)
    
    # ★ デバッグ用: データ件数の制限 (local.yamlで data_limit: 10 としている場合など)
    limit = cfg.dir.data_limit
    if limit is not None:
        print(f"⚠️ Debug Mode: Limiting test data to {limit} samples.")
        # 画像パス単位で制限するため、ユニークな画像パスを取得してフィルタリング
        unique_paths = test_df['image_path'].unique()[:limit]
        test_df = test_df[test_df['image_path'].isin(unique_paths)].reset_index(drop=True)


    test_df_unique = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    print(f"Test images: {len(test_df_unique)}")

    # ------------------------------------------------------------------
    # 1. モデルのロード (5 Folds)
    # ------------------------------------------------------------------
    print("Loading models...")
    models = []
    n_folds = cfg.inference.n_folds
    
    for fold in range(n_folds):
        weight_path = weight_dir / f"best_model_fold{fold}.pth"
        
        if not weight_path.exists():
            print(f"⚠️ Warning: Model weight not found: {weight_path}")
            print("Skipping this fold (Ensure you downloaded weights!)")
            continue
            
        # Hydraでモデル定義を呼び出し (src.models.BiomassModel)
        model = hydra.utils.instantiate(cfg.model)
        
        # 重みをロード (module.除去などを含む)
        model.load_weights(str(weight_path), device)
        model.to(device)
        model.eval()
        models.append(model)
        
    if not models:
        raise RuntimeError("No models loaded! Check your weight directory path.")
    print(f"Loaded {len(models)} models.")

    # ------------------------------------------------------------------
    # 2. TTA (Test-Time Augmentation) 推論ループ
    # ------------------------------------------------------------------
    tf_factory = TransformFactory(img_size=cfg.inference.img_size)
    tta_transforms = tf_factory.get_tta_transforms() # [Original, HFlip, VFlip]
    
    # 全TTAビューの結果を保存するリスト
    # 構造: [{'total': [N], 'gdm': [N], 'green': [N]}, ...]
    all_view_preds = []

    print(f"Starting Inference with {len(tta_transforms)} TTA views...")

    # --- TTA Loop ---
    for i, transform in enumerate(tta_transforms):
        print(f"--- TTA View {i+1}/{len(tta_transforms)} ---")
        
        # Dataset & DataLoader作成
        dataset = DualStreamDataset(
            df=test_df_unique,
            root_dir=str(root_dir), # 画像フォルダのルート (必要に応じて /test を付与)
            transform=transform,
            return_target=False
        )
        
        loader = DataLoader(
            dataset,
            batch_size=cfg.inference.batch_size,
            shuffle=False,
            num_workers=cfg.inference.num_workers,
            pin_memory=True
        )
        
        # 1つのビューに対する予測結果格納用
        view_preds = {'total': [], 'gdm': [], 'green': []}
        
        # --- Batch Loop ---
        with torch.no_grad():
            for img_left, img_right in tqdm(loader, desc=f"View {i+1}"):
                img_left = img_left.to(device)
                img_right = img_right.to(device)
                
                # Fold Ensemble (5つのモデルの平均をとる)
                fold_outputs = {'total': [], 'gdm': [], 'green': []}
                
                for model in models:
                    out_t, out_gdm, out_gr = model(img_left, img_right)
                    fold_outputs['total'].append(out_t)
                    fold_outputs['gdm'].append(out_gdm)
                    fold_outputs['green'].append(out_gr)
                
                # Fold平均
                avg_t = torch.mean(torch.stack(fold_outputs['total']), dim=0)
                avg_gdm = torch.mean(torch.stack(fold_outputs['gdm']), dim=0)
                avg_gr = torch.mean(torch.stack(fold_outputs['green']), dim=0)
                
                # 結果をリストに追加 (CPUへ)
                view_preds['total'].extend(avg_t.cpu().numpy().flatten())
                view_preds['gdm'].extend(avg_gdm.cpu().numpy().flatten())
                view_preds['green'].extend(avg_gr.cpu().numpy().flatten())
        
        # NumPy配列化
        view_preds = {k: np.array(v) for k, v in view_preds.items()}
        all_view_preds.append(view_preds)

    # ------------------------------------------------------------------
    # 3. アンサンブル (TTA Averaging) & 後処理
    # ------------------------------------------------------------------
    print("Aggregating TTA results...")
    final_preds = {
        'total': np.mean([p['total'] for p in all_view_preds], axis=0),
        'gdm':   np.mean([p['gdm']   for p in all_view_preds], axis=0),
        'green': np.mean([p['green'] for p in all_view_preds], axis=0)
    }
    
    # 提出ファイルの作成
    processor = PredictionProcessor(all_target_cols=cfg.inference.all_target_cols)
    
    # test_df(全行)を渡して、sample_idとマージしてもらう
    submission = processor.create_submission(final_preds, test_df_unique)
    
    save_dir = Path(cfg.dir.output_dir) / cfg.exp_name
    
    # 2. ディレクトリが存在しない場合は作成する (重要)
    save_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_dir, index=False)
    print(f"Saved submission to {save_dir}")

if __name__ == "__main__":
    main()