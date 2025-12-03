# src/features/dino.py
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
import numpy as np

class DinoExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading Feature Extractor: {model_name} on {self.device}...")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, image_paths: list) -> np.ndarray:
        embeds = []
        # 高速化のためバッチ処理に変更したい場合はここを書き換える
        for img_path in tqdm(image_paths, desc="Extracting features"):
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeds.append(outputs.pooler_output.cpu())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # エラー時はゼロ埋めで埋める
                embeds.append(torch.zeros((1, self.model.config.hidden_size)))

        return np.array(torch.cat(embeds))