
import os
import argparse
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

from ribline.data.dataset import RibDataset
from ribline.model.unet.dataset import RibDLDataset
from ribline.model.unet.model import RibUNet
from ribline.model.unet.loss import DiceBCELoss


def train(config_path, resume_path=None):
    """Trains the lightweight 3D U-Net.

    Recognised config keys (defaults reproduce the released ``rib_unet_3d.pth``):

    ``data_config``: ``image_directory``, ``centerline_directory``, ``split_file``,
    ``augmentation`` (bool, default ``false``; random left-right flips and
    +-10 % intensity scaling of training patches -- the released checkpoint was
    trained WITHOUT augmentation).

    ``model_config``: ``base_filters`` (16).

    ``training_config``: ``output_directory``, ``samples_per_volume`` (8),
    ``epochs`` (20), ``batch_size`` (2), ``learning_rate`` (1e-3),
    ``num_workers`` (4), ``mask_cache_dir`` (``<output_directory>/mask_cache``),
    ``torch_threads`` (optional), ``seed`` (optional).

    The effective recipe is written to ``<output_directory>/models/train_recipe.json``.
    Checkpoints: ``unet_latest.pth`` (weights, every epoch), ``unet_ep{N}.pth``
    every 5 epochs, plus ``*_resume.pth`` with optimizer state.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Setup Data
    data_cfg = config['data_config']
    model_cfg = config.get('model_config', {})
    train_cfg = config['training_config']
    train_split = json.load(open(data_cfg['split_file']))['train']

    seed = train_cfg.get('seed', None)
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    if train_cfg.get('torch_threads'):
        torch.set_num_threads(int(train_cfg['torch_threads']))
    
    # Base RibDataset
    raw_ds_train = RibDataset(
        data_cfg['image_directory'], 
        data_cfg['centerline_directory'], 
        selected_files=train_split
    )
    output_dir = train_cfg['output_directory']
    cache_dir = train_cfg.get('mask_cache_dir') or os.path.join(output_dir, "mask_cache")

    samples_per_volume = int(train_cfg.get('samples_per_volume', 8))
    num_epochs = int(train_cfg.get('epochs', 20))
    batch_size = int(train_cfg.get('batch_size', 2))
    lr = float(train_cfg.get('learning_rate', 1e-3))
    num_workers = int(train_cfg.get('num_workers', 4))
    augmentation = bool(data_cfg.get('augmentation', False))
    base_filters = int(model_cfg.get('base_filters', 16))
    
    train_ds = RibDLDataset(
        raw_ds_train, 
        cache_dir, 
        samples_per_volume=samples_per_volume,
        training=True,
        augmentation=augmentation,
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Model
    model = RibUNet(in_channels=1, out_channels=1, base_filters=base_filters)
    
    # Optimization: Adam + 0.5 * soft Dice + 0.5 * BCE-with-logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()
    
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        state = torch.load(resume_path)
        # Check if state is full ckp or just weights
        if 'model_state_dict' in state:
             model.load_state_dict(state['model_state_dict'])
             optimizer.load_state_dict(state['optimizer_state_dict'])
             start_epoch = state['epoch'] + 1
             print(f"Resumed at epoch {start_epoch}")
        else:
             # Just weights
             model.load_state_dict(state)
             print("Resumed weights only (no epoch info found in checkpoint)")

    # Directories
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    recipe = {
        "config_path": os.path.abspath(config_path),
        "num_train_subjects": len(train_split),
        "samples_per_volume": samples_per_volume,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": "Adam",
        "loss": "0.5*soft_dice + 0.5*BCEWithLogits",
        "augmentation": augmentation,
        "base_filters": base_filters,
        "patch_size": [64, 64, 64],
        "input_normalization": "per-volume z-score of raw stored intensities (no HU standardization)",
        "architecture": "plain 3-level 3D U-Net (Conv-GN-ReLU x2 per level, no residual connections)",
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "seed": seed,
        "resume_path": resume_path,
    }
    with open(os.path.join(model_dir, "train_recipe.json"), "w") as f:
        json.dump(recipe, f, indent=2)
    
    print(f"Starting Training on {len(train_split)} subjects (CPU Mode).")
    print(f"Mask Cache: {cache_dir}")
    print(f"Recipe: {json.dumps(recipe)}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images = batch['image']
            masks = batch['mask']
            
            # Forward
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / max(steps, 1)
        print(f"Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.4f}")
        
        # Save Checkpoint (Full State)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, os.path.join(model_dir, "unet_latest_resumable.pth"))
        
        # Backward compatibility for inference scripts that expect simple weights
        torch.save(model.state_dict(), os.path.join(model_dir, "unet_latest.pth"))
        
        if (epoch+1) % 5 == 0:
             torch.save(model.state_dict(), os.path.join(model_dir, f"unet_ep{epoch+1}.pth"))
             # Also save resumable
             torch.save(checkpoint, os.path.join(model_dir, f"unet_ep{epoch+1}_resume.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train(args.config, args.resume)
