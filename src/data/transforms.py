# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class TransformFactory:
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

class BiomassTransformFactory:
    def __init__(self, img_size: int):
        self.img_size = img_size

    def get_train_transforms(self):
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 2.0), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def get_valid_transforms(self):
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

class BiomassTransformFactory2:
    def __init__(self, img_size: int):
        self.img_size = img_size

    def get_train_transforms(args):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.75
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3), 
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=0.75
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            A.Resize(args.img_size, args.img_size),
            ToTensorV2()
        ])

    def get_valid_transforms(self):
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])