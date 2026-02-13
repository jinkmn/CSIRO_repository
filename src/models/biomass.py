# src/models/biomass.py
import torch
import torch.nn as nn
import timm
from collections import OrderedDict

class Convnext(nn.Module):
    """
    左右の画像を入力とし、3つの目的変数を予測するDual-Streamモデル
    """
    def __init__(self, model_name: str, pretrained: bool = False, img_size: int = 512):
        """
        Args:
            model_name: timmのモデル名 (例: 'convnext_small')
            pretrained: ImageNet等の事前学習済み重みを使うか
                        (推論時は独自の重みをロードするためFalseでOK)
        """
        super().__init__()

        # 1. バックボーン (共通の特徴抽出器)
        # num_classes=0, global_pool='avg' にすることで、
        # 最終層の分類ヘッドを取り除き、特徴量ベクトルだけを取り出せるようにします
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
            img_size=img_size 
        )

        # 特徴量の次元数 (ConvNeXt-Smallなら768など)
        self.n_features = self.backbone.num_features
        
        # 左右の画像を結合するので、入力次元は2倍になる
        self.n_combined = self.n_features * 2

        # 2. 予測ヘッド (3つのターゲットに対し、それぞれ専用の層を用意)
        self.head_total = self._create_head() # Dry_Total_g 用
        self.head_gdm = self._create_head()   # GDM_g 用
        self.head_green = self._create_head() # Dry_Green_g 用

    def _create_head(self) -> nn.Sequential:
        """MLP (多層パーセプトロン) ヘッドの作成ヘルパー"""
        return nn.Sequential(
            nn.Linear(self.n_combined, self.n_combined // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.n_combined // 2, 1) # 出力は1つの実数値
        )

    def forward(self, img_left: torch.Tensor, img_right: torch.Tensor):
        """
        順伝播処理
        Args:
            img_left: 左画像のバッチ [B, C, H, W]
            img_right: 右画像のバッチ [B, C, H, W]
        """
        # 左右それぞれバックボーンに通す (重みは共有)
        feat_left = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        
        # 特徴量を結合 [B, n_features] x 2 -> [B, n_features * 2]
        combined = torch.cat([feat_left, feat_right], dim=1)

        # 各ヘッドで予測
        out_total = self.head_total(combined)
        out_gdm = self.head_gdm(combined)
        out_green = self.head_green(combined)

        return out_total, out_gdm, out_green

    def load_weights(self, weight_path: str, device: str):
        """
        学習済み重み(.pth)を安全にロードするメソッド
        DataParallelで保存された重み('module.'が付いている)にも対応
        """
        state_dict = torch.load(weight_path, map_location=device)
        
        # 'module.' プレフィックスの除去処理
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') # module.fc.weight -> fc.weight
            new_state_dict[name] = v
            
        self.load_state_dict(new_state_dict)
        print(f"Weights loaded from: {weight_path}")

class BiomassModel(nn.Module):
    """
    左右の画像を入力とし、3つの目的変数を予測するDual-Streamモデル
    """
    def __init__(
        self, 
        model_name: str, 
        pretrained: bool = True, 
        out_dim: int = 3,  
        drop_rate: float = 0.3,
        freeze: bool = True
    ):
        """
        Args:
            model_name: timmのモデル名
            pretrained: ImageNet重みを使うか
            out_dim: 出力次元数 (train.pyとの互換性用、内部では3つ固定で実装)
            drop_rate: ドロップアウト率
        """
        super().__init__()

        # 1. バックボーン
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        # 特徴量の次元数
        self.n_features = self.backbone.num_features
        # 左右結合後の次元数
        self.n_combined = self.n_features * 2

        # 2. 予測ヘッド (改良版)
        # 3つのターゲットに対し、それぞれ専用の層を用意
        self.head_total = self._create_head_simple(self.n_combined, 1, drop_rate)
        self.head_gdm   = self._create_head_simple(self.n_combined, 1, drop_rate)
        self.head_green = self._create_head_simple(self.n_combined, 1, drop_rate)
        self.head_dead = self._create_head_simple(self.n_combined, 1, drop_rate)
        self.head_clover = self._create_head_simple(self.n_combined, 1, drop_rate)

    def _create_head(self, in_dim, out_dim, drop_rate) -> nn.Sequential:
        """
        MLPヘッドの作成
        Linear -> BN -> GELU -> Dropout -> Linear
        """
        hidden_dim = 512 # 隠れ層のサイズ (経験的に512程度がバランス良し)
        
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),     # 学習安定化のため追加
            nn.GELU(),                      # ReLUより現代的な活性化関数
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def _create_head_simple(self, in_dim, out_dim, drop_rate) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=drop_rate), 
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, img_left: torch.Tensor, img_right: torch.Tensor):
        """
        順伝播処理
        Return: (Batch, 3) のTensor
        """
        # 1. Backbone (重み共有)
        feat_left = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        
        # 2. 特徴量結合 [B, F] x 2 -> [B, 2F]
        combined = torch.cat([feat_left, feat_right], dim=1)

        # 3. 各ヘッドで予測 [B, 1]
        out_total = self.head_total(combined)
        out_gdm   = self.head_gdm(combined)
        out_green = self.head_green(combined)
        out_dead = self.head_dead(combined)
        out_clover = self.head_clover(combined)

        # 4. 結合して返す [B, 3]
        return torch.cat([out_clover, out_dead, out_green, out_total, out_gdm], dim=1)

    def load_weights(self, weight_path: str, device: str):
        """
        推論用: 重みロードヘルパー
        """
        if not weight_path:
            return
            
        state_dict = torch.load(weight_path, map_location=device)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            
        self.load_state_dict(new_state_dict)
        print(f"Weights loaded from: {weight_path}")


class BiomassModel2(nn.Module):
    """
    左右の画像を入力とし、3つの目的変数を予測するDual-Streamモデル
    """
    def __init__(
        self, 
        model_name: str, 
        pretrained: bool = True, 
        out_dim: int = 3,  
        drop_rate: float = 0.3,
        freeze: bool = True,
        img_size: int = 512
    ):
        """
        Args:
            model_name: timmのモデル名
            pretrained: ImageNet重みを使うか
            out_dim: 出力次元数 (train.pyとの互換性用、内部では3つ固定で実装)
            drop_rate: ドロップアウト率
        """
        super().__init__()

        # 1. バックボーン
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size
        )

        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        # 特徴量の次元数
        self.n_features = self.backbone.num_features
        # 左右結合後の次元数
        self.n_combined = self.n_features * 2

        # 2. 予測ヘッド (改良版)
        # 3つのターゲットに対し、それぞれ専用の層を用意
        self.head_total = self._create_head_simple(self.n_combined, 1, drop_rate)
        self.head_gdm   = self._create_head_simple(self.n_combined, 1, drop_rate)
        self.head_green = self._create_head_simple(self.n_combined, 1, drop_rate)
        self.head_dead = self._create_head_simple(self.n_combined, 1, drop_rate)
        self.head_clover = self._create_head_simple(self.n_combined, 1, drop_rate)

    def _create_head(self, in_dim, out_dim, drop_rate) -> nn.Sequential:
        """
        MLPヘッドの作成
        Linear -> BN -> GELU -> Dropout -> Linear
        """
        hidden_dim = 512 # 隠れ層のサイズ (経験的に512程度がバランス良し)
        
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),     # 学習安定化のため追加
            nn.GELU(),                      # ReLUより現代的な活性化関数
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def _create_head_simple(self, in_dim, out_dim, drop_rate) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=drop_rate), 
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, img_left: torch.Tensor, img_right: torch.Tensor):
        """
        順伝播処理
        Return: (Batch, 3) のTensor
        """
        # 1. Backbone (重み共有)
        feat_left = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        
        # 2. 特徴量結合 [B, F] x 2 -> [B, 2F]
        combined = torch.cat([feat_left, feat_right], dim=1)

        # 3. 各ヘッドで予測 [B, 1]
        out_total = self.head_total(combined)
        out_gdm   = self.head_gdm(combined)
        out_green = self.head_green(combined)
        out_dead = self.head_dead(combined)
        out_clover = self.head_clover(combined)

        # 4. 結合して返す [B, 3]
        return torch.cat([out_clover, out_dead, out_green, out_total, out_gdm], dim=1)

    def load_weights(self, weight_path: str, device: str):
        """
        推論用: 重みロードヘルパー
        """
        if not weight_path:
            return
            
        state_dict = torch.load(weight_path, map_location=device)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            
        self.load_state_dict(new_state_dict)
        print(f"Weights loaded from: {weight_path}")