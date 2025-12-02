# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd
import albumentations as A

class DualStreamDataset(Dataset):
    """
    画像を読み込み、左右に分割して (img_left, img_right) を返すデータセット
    """
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        transform: A.Compose,
        return_target: bool = False,
        target_cols: list = None
    ):
        self.df = df
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_target = return_target
        self.target_cols = target_cols
        
        # 画像パスのリスト
        self.image_paths = df['image_path'].values
        
        # ターゲットがある場合は保持
        if return_target and target_cols:
            self.targets = df[target_cols].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # 画像パスの構築
        img_rel_path = self.image_paths[idx]
        full_path = self.root_dir / Path(img_rel_path).name # フラットな構造に対応
        
        # 1. 画像読み込み (OpenCV)
        image = cv2.imread(str(full_path))
        
        # エラーハンドリング: 画像がない場合は黒画像で埋める
        if image is None:
            # 元画像サイズを仮定 (H=1000, W=2000)
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 左右分割ロジック
        height, width = image.shape[:2]
        mid_point = width // 2
        
        img_left = image[:, :mid_point]
        img_right = image[:, mid_point:]

        # 3. Transform適用 (左右それぞれに同じ変換をかける)
        # Transformは {'image': tensor} の辞書を返すため ['image'] で取り出す
        img_left_tensor = self.transform(image=img_left)['image']
        img_right_tensor = self.transform(image=img_right)['image']

        # 4. ターゲットを返すかどうか
        if self.return_target:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img_left_tensor, img_right_tensor, target
        else:
            return img_left_tensor, img_right_tensor