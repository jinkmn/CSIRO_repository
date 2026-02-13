# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd
import albumentations as A
from src.utils.utils import *

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
        
        self.image_paths = df['image_path'].values 
        
        if return_target and target_cols:
            self.targets = df[target_cols].fillna(0).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_rel_path = self.image_paths[idx]
        full_path = self.root_dir / img_rel_path
        
        # 1. 画像読み込み (OpenCV)
        image = cv2.imread(str(full_path))
        
        # エラーハンドリング
        if image is None:
            image = np.zeros((512, 1024, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 左右分割
        height, width = image.shape[:2]
        mid_point = width // 2
        
        img_left = image[:, :mid_point]
        img_right = image[:, mid_point:]

        # 3. Transform適用
        # Albumentationsは {'image': ...} を返すので ['image'] を取り出す
        if self.transform:
            img_left_tensor = additional_transform(img_left)
            img_left_tensor = self.transform(image=img_left_tensor)['image']
            img_right_tensor = additional_transform(img_right)
            img_right_tensor = self.transform(image=img_right_tensor)['image']
        else:
            # Transformがない場合のフォールバック
            img_left_tensor = torch.from_numpy(img_left.transpose(2, 0, 1)).float()
            img_right_tensor = torch.from_numpy(img_right.transpose(2, 0, 1)).float()

        # 4. ターゲットを返すかどうか
        if self.return_target and hasattr(self, 'targets'):
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return (img_left_tensor, img_right_tensor), target
        else:
            return (img_left_tensor, img_right_tensor)
        
