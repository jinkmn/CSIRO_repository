# src/data/preprocessing.py
import pandas as pd
import os
import numpy as np

# 定数として定義しておくと便利
TARGET_COLUMNS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
TARGET_MAPPING = {col: i for i, col in enumerate(TARGET_COLUMNS)}

def get_unique_image_paths(df: pd.DataFrame, root_dir: str) -> list:
    """
    DataFrameから重複のない画像パスのリスト(フルパス)を生成して返す
    特徴抽出(Feature Extraction)の前に使用する
    """
    unique_df = df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    full_paths = [os.path.join(root_dir, p) for p in unique_df['image_path']]
    return full_paths

def prepare_train_xy(df: pd.DataFrame, path_to_embed: dict) -> dict:
    """
    学習用DataFrameと「画像パス->特徴量」の辞書を受け取り、
    5つのターゲットそれぞれに対する (X, y) のセットを作成して返す。
    
    Returns:
        targets_data (dict): 
        {
            0: {'X': [embed, ...], 'y': [val, ...]}, # Dry_Clover_g用
            1: {'X': [embed, ...], 'y': [val, ...]}, # Dry_Dead_g用
            ...
        }
    """
    # 5つのターゲット用に空箱を用意
    targets_data = {i: {'X': [], 'y': [], 'image_path':[]} for i in range(5)}

    for _, row in df.iterrows():
        # --- ターゲット名の特定 ---
        if 'target_name' in row:
            t_name = row['target_name']
        else:
            # sample_id = "image_name__target_name" の形式から取得
            try:
                t_name = row['sample_id'].split('__')[1]
            except IndexError:
                continue

        # --- 画像パスの特定 ---
        if 'image_path' in row:
            img_path = row['image_path']
        else:
            # sample_id から復元する場合
            img_path = row['sample_id'].split('__')[0]
        
        # --- データの格納 ---
        if t_name in TARGET_MAPPING:
            t_idx = TARGET_MAPPING[t_name]
            
            # 【修正2】row['image_path'] ではなく、上で確保した img_path 変数を使う
            # (辞書のキーと形式が一致している前提)
            embed = path_to_embed.get(img_path)
            
            if embed is None:
                continue 

            targets_data[t_idx]['X'].append(embed)
            targets_data[t_idx]['y'].append(row['target'])
            # ここで保存
            targets_data[t_idx]['image_path'].append(img_path)
            
    return targets_data

def parse_test_row(row):
    """テストデータの1行から推論に必要な情報を取り出すヘルパー関数"""
    sample_id = row['sample_id']
    img_path = row['image_path']
    # sample_id = "img_name__target_name"
    target_name = sample_id.split('__')[1]
    target_idx = TARGET_MAPPING.get(target_name)
    
    return sample_id, img_path, target_idx