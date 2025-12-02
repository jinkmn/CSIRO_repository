# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TransformFactory:
    """
    学習および推論（TTA）で使用するTransformを生成するクラス
    """
    def __init__(self, img_size: int = 1000):
        self.img_size = img_size
        
        # ImageNetの平均・分散での正規化とTensor変換（共通処理）
        self.base_transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]

    def get_train_transforms(self) -> A.Compose:
        """学習用: ここにAugmentationを追加できます"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # 必要に応じてここにShiftScaleRotateなどを追加
            *self.base_transforms
        ])

    def get_valid_transforms(self) -> A.Compose:
        """検証用: リサイズのみ"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

    def get_tta_transforms(self) -> list[A.Compose]:
        """
        推論用TTA: 3つのビュー（オリジナル、左右反転、上下反転）を返す
        """
        # 1. Original
        original = A.Compose([
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        # 2. Horizontal Flip
        hflip = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        # 3. Vertical Flip
        vflip = A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        return [original, hflip, vflip]