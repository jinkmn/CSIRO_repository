# src/features/siglip.py
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
import numpy as np

class SigLIPExtractor:
    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cuda"):
        """
        Args:
            model_name (str): HuggingFaceのモデル名 (例: google/siglip-so400m-patch14-384)
            batch_size (int): バッチサイズ
            device (str): 実行デバイス
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f"Loading SigLIP Extractor: {model_name} on {self.device} with batch_size={batch_size}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except:
            # Processorがないモデル（Vision専用など）のフォールバック
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, image_paths: list) -> np.ndarray:
        all_embeds = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="SigLIP Extraction"):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_images = []
            
            # 画像読み込み
            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        batch_images.append(img.convert('RGB'))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # エラー時は黒画像などで埋める（サイズ合わせ）
                    batch_images.append(Image.new('RGB', (224, 224)))

            if not batch_images:
                continue

            try:
                # 前処理
                inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embeds = self.model.get_image_features(**inputs)

                # CPUへ移動してリストに追加
                all_embeds.append(embeds.cpu())

            except Exception as e:
                print(f"Error in batch inference: {e}")

        if not all_embeds:
            return np.array([])
        
        concatenated_tensor = torch.cat(all_embeds, dim=0)
        
        # 2. NumPyに変換する
        embeds_np = concatenated_tensor.numpy()

        # 3. 形状を確認（printはreturnと分ける）
        print(f"Extracted features shape: {embeds_np.shape}")

        # 4. 配列だけを返す
        return embeds_np