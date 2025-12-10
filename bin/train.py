import sys
import os
import gc
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from timm.utils import ModelEmaV2
from tqdm import tqdm
import wandb
from sklearn.metrics import mean_squared_error, r2_score

# srcパスの追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.utils import *

# ==============================================================================
# Helper Functions (Modified for Multi-Task)
# ==============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, model_ema=None, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train", dynamic_ncols=True)
    # labels は (Batch, 3) を想定
    for i, (images, labels) in enumerate(pbar):
        if isinstance(images, (list, tuple)):
            img_left = images[0].to(device)
            img_right = images[1].to(device)
            images_in = (img_left, img_right)
        else:
            images_in = images.to(device)
            
        labels = labels.to(device).float() # (Batch, 3)
        
        if isinstance(images_in, tuple):
            outputs = model(images_in[0], images_in[1])
        else:
            outputs = model(images_in)
            
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps

        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad() 
            
            if model_ema is not None:
                model_ema.update(model)

        running_loss += (loss.item() * accumulation_steps) * labels.size(0)
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    trues = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Valid", dynamic_ncols=True)
        for images, labels in pbar:
            if isinstance(images, (list, tuple)):
                img_left = images[0].to(device)
                img_right = images[1].to(device)
                outputs = model(img_left, img_right)
            else:
                images = images.to(device)
                outputs = model(images)

            labels = labels.to(device).float()
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            preds.append(outputs.cpu().numpy())
            trues.append(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    preds = np.concatenate(preds) # (N, 3)
    trues = np.concatenate(trues) # (N, 3)
    
    return epoch_loss, preds, trues

# ==============================================================================
# Main Training Script
# ==============================================================================

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.get("seed", 42))
    wandb.init(
        project='CSIRO_Competition', 
        name=cfg.exp_name,       
        config=OmegaConf.to_container(cfg, resolve=True), 
        reinit=True,
        group=f'{cfg.model._target_.split(".")[-1]}',  
        job_type="train",
        mode=cfg.get("wandb_mode", "online")
    )

    print(f"=== Experiment: {cfg.exp_name} ===")
    ROOT = cfg.dir.data_dir
    OUTPUT_DIR = Path(cfg.dir.output_dir) / cfg.exp_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. データ読み込みと前処理
    raw_train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    train_df = preprocess_train_data(raw_train_df, n_splits=5)
    
    limit = cfg.dir.data_limit
    if limit is not None:
        print(f"⚠️ Debug Mode: Limiting data to {limit} samples.")
        unique_paths = train_df['image_path'].unique()[:limit]
        train_df = train_df[train_df['image_path'].isin(unique_paths)].reset_index(drop=True)

    # 2. ターゲットの設定
    train_targets_cols = ['Dry_Total_g','GDM_g','Dry_Green_g']   
    all_target_columns = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g' ]   
    print(f"Targets to train (Multi-task): {train_targets_cols}")
    
    num_targets = len(train_targets_cols) # 3
    
    oof_preds_dict = {col: np.zeros(len(train_df)) for col in train_targets_cols}
    
    metrics_log = {}
    
    fold_scores = []

    tf_factory = hydra.utils.instantiate(cfg.transform)
    train_transform = tf_factory.get_train_transforms()
    val_transform = tf_factory.get_valid_transforms()
    
    for fold in range(5):
        print(f"\n{'='*10} Fold {fold} {' ='*10}")
        train_fold = train_df[train_df['fold'] != fold].reset_index(drop=True)
        val_fold = train_df[train_df['fold'] == fold].reset_index(drop=True)
        
        
        train_dataset = hydra.utils.instantiate(
            cfg.dataset, 
            df=train_fold, 
            root_dir=ROOT, 
            transform=train_transform,      
            return_target=True,              
            target_cols=all_target_columns
        )
        val_dataset = hydra.utils.instantiate(
            cfg.dataset, 
            df=val_fold, 
            root_dir=ROOT, 
            transform=val_transform,      
            return_target=True,              
            target_cols=all_target_columns
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.training.batch_size, 
            shuffle=True, 
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.training.batch_size, 
            shuffle=False, 
            num_workers=cfg.training.num_workers,
            pin_memory=True
        )
        

        model = hydra.utils.instantiate(cfg.model, out_dim=num_targets)
        model.to(device)

        model_ema = ModelEmaV2(model, decay=0.999, device=device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        warmup_epochs = cfg.training.get("warmup_epochs", 1)
        total_epochs = cfg.training.epochs
        
        scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=0.001, 
        end_factor=1.0,    
        total_iters=warmup_epochs
        )

        scheduler_cosine = CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs - warmup_epochs, 
        eta_min=cfg.training.get("min_lr", 1e-6)
        )
    
        scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs]
        )

        criterion = BiomassWeightedMSELoss(weights=[0.1, 0.1, 0.1, 0.5, 0.2], device=device)
        best_val_loss = float('inf')
        best_preds = None # (N_val, 3)
        
        for epoch in range(cfg.training.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, model_ema=model_ema, accumulation_steps=cfg.training.get("accumulation_steps",1))
            if scheduler is not None:
                scheduler.step()
            val_loss, val_preds, val_trues = validate(model_ema.module, val_loader, criterion, device)
            

            rmse_per_target = []
            for i, target_name in enumerate(train_targets_cols):
                rmse = np.sqrt(mean_squared_error(val_trues[:, i], val_preds[:, i]))
                rmse_per_target.append(rmse)
                wandb.log({f"fold{fold}/{target_name}_rmse": rmse})
            
            mean_rmse = np.mean(rmse_per_target)

            print(f"Epoch {epoch+1}/{cfg.training.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean RMSE: {mean_rmse:.4f}")
            
            wandb.log({
                f"fold{fold}/train_loss": train_loss,
                f"fold{fold}/val_loss": val_loss,
                f"fold{fold}/mean_rmse": mean_rmse
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = val_preds
                
                save_path = OUTPUT_DIR / f"best_model_fold{fold}.pth"
                torch.save(model_ema.module.state_dict(), save_path)
                
        print(f"✨ Best Val Loss: {best_val_loss:.4f}")
        fold_scores.append(best_val_loss)
        
        val_indices = train_df[train_df['fold'] == fold].index
        
        for i, target_name in enumerate(train_targets_cols):
            oof_preds_dict[target_name][val_indices] = best_preds[:, i].flatten()
        
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nAverage Fold Loss: {np.mean(fold_scores):.4f}")

    print("\nCalculating derived targets and overall metrics...")
    
    oof_df = train_df.copy()
    all_target_columns = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # 学習した予測値を代入
    for col in train_targets_cols:
        oof_df[f"pred_{col}"] = oof_preds_dict[col].clip(min=0)
    
    # ダミー実装: 学習していないカラムは0埋め
    for col in all_target_columns:
        if f"pred_{col}" not in oof_df.columns:
             oof_df[f"pred_{col}"] = 0.0

    # Metric計算
    y_true = train_df[all_target_columns].values
    y_pred = oof_df[[f"pred_{col}" for col in all_target_columns]].values
    valid_weights = [0.1, 0.1, 0.1, 0.5, 0.2] 
    
    rmse, r2 = calc_weighted_metrics(y_true.T, y_pred.T, valid_weights)
    
    print(f"\n=== Overall CV Score ===")
    print(f"Weighted RMSE: {rmse:.5f}")
    print(f"Weighted R2:   {r2:.5f}")
    
    metrics_log["overall_weighted_rmse"] = rmse
    metrics_log["overall_weighted_r2"] = r2
    wandb.log(metrics_log)

    oof_save_path = OUTPUT_DIR / f'oof_{cfg.exp_name}.csv'
    oof_df.to_csv(oof_save_path, index=False)
    print(f"OOF saved to {oof_save_path}")

    wandb.finish()

if __name__ == "__main__":
    main()