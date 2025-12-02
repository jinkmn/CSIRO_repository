# src/data/postprocessing.py
import pandas as pd
import numpy as np

class PredictionProcessor:
    """
    モデルの出力(3変数)から残りの変数を計算し、提出用DataFrameを作成するクラス
    """
    def __init__(self, all_target_cols: list):
        self.all_target_cols = all_target_cols

    def create_submission(
        self, 
        predictions: dict, 
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Args:
            predictions: {'total': np.array, 'gdm': np.array, 'green': np.array}
            test_df: テストデータのDataFrame (sample_id, image_path等を含む)
        """
        # 1. 予測結果の取得
        pred_total = predictions['total']
        pred_gdm = predictions['gdm']
        pred_green = predictions['green']

        # 2. 残りの変数を計算 (マイナス値は0にクリップ)
        # Clover = GDM - Green
        pred_clover = np.maximum(0, pred_gdm - pred_green)
        # Dead = Total - GDM
        pred_dead = np.maximum(0, pred_total - pred_gdm)

        # 3. ワイド形式のDataFrame作成 (画像1枚につき1行)
        # test_dfは重複排除済みであることを想定
        preds_wide = pd.DataFrame({
            'image_path': test_df['image_path'],
            'Dry_Green_g': pred_green,
            'Dry_Dead_g': pred_dead,
            'Dry_Clover_g': pred_clover,
            'GDM_g': pred_gdm,
            'Dry_Total_g': pred_total
        })

        # 4. ロング形式に変換 (melt)
        preds_long = preds_wide.melt(
            id_vars=['image_path'],
            value_vars=self.all_target_cols,
            var_name='target_name',
            value_name='target'
        )
        
        # 5. 元のtest.csv (sample_idを持つ) とマージして順序を復元
        # test_dfに全行が含まれている場合、ここでマージが必要
        # 注意: 呼び出し元で test_df がユニークか全行かによって処理が変わりますが、
        # ここでは「全行持っているtest_df」とマージする安全な方法をとります。
        
        # sample_idが必要なので、image_pathとtarget_nameをキーにして結合したいが、
        # test_dfの形状が不明なため、汎用的に sample_id を構築してマージする手法をとります。
        
        # sample_id = "image_pathのファイル名__target_name" の構造を利用
        preds_long['filename'] = preds_long['image_path'].apply(lambda x: Path(x).name)
        preds_long['sample_id'] = preds_long['filename'] + '__' + preds_long['target_name']
        
        # 必要なカラムだけ返す
        submission = preds_long[['sample_id', 'target']].sort_values('sample_id').reset_index(drop=True)
        
        return submission