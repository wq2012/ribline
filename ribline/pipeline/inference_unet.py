
import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
import json
from tqdm import tqdm

from ribline.data.image_reader import load_image, resolve_subject_image_path
from ribline.model.unet.model import RibUNet

def sliding_window_inference(model, image_arr, patch_size=(64, 64, 64), stride=(48, 48, 48), batch_size=2):
    """
    Performs sliding window inference on a 3D volume.
    Automatically zero-pads volumes smaller than ``patch_size`` along any axis
    and crops the output probability map back to the original shape.
    """
    model.eval()

    orig_z, orig_y, orig_x = image_arr.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    # Normalize Image before optional padding
    mean = np.mean(image_arr)
    std = np.std(image_arr)
    image_norm = ((image_arr - mean) / (std + 1e-8)).astype(np.float32)

    pad_z = max(0, pz - orig_z)
    pad_y = max(0, py - orig_y)
    pad_x = max(0, px - orig_x)
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        image_norm = np.pad(
            image_norm,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
        )

    z_dim, y_dim, x_dim = image_norm.shape

    # Output buffer (Probability accumulation)
    # We use a count buffer for averaging overlapping patches
    prob_map = torch.zeros(image_norm.shape, dtype=torch.float32)
    count_map = torch.zeros(image_norm.shape, dtype=torch.float32)

    # Generate coordinates
    z_starts = list(range(0, z_dim - pz + sz, sz))
    y_starts = list(range(0, y_dim - py + sy, sy))
    x_starts = list(range(0, x_dim - px + sx, sx))

    # Clip last starts to ensure we cover the edge
    z_starts = [min(z, z_dim - pz) for z in z_starts]
    y_starts = [min(y, y_dim - py) for y in y_starts]
    x_starts = [min(x, x_dim - px) for x in x_starts]

    # Unique/Sort
    z_starts = sorted(list(set(z_starts)))
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    patches = []
    coords = []

    image_tensor = torch.from_numpy(image_norm)

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = image_tensor[z:z+pz, y:y+py, x:x+px]
                patches.append(patch.unsqueeze(0)) # Add channel dim
                coords.append((z, y, x))

    # Batch Processing
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            batch_coords = coords[i:i+batch_size]

            # Stack: (B, 1, D, H, W)
            input_tensor = torch.stack(batch_patches)

            # Forward
            logits = model(input_tensor)
            probs = torch.sigmoid(logits) # (B, 1, D, H, W)

            for j, (z, y, x) in enumerate(batch_coords):
                prob_patch = probs[j, 0, :, :, :]
                prob_map[z:z+pz, y:y+py, x:x+px] += prob_patch
                count_map[z:z+pz, y:y+py, x:x+px] += 1.0

    # Average and crop back to original shape
    avg_prob = prob_map / (count_map + 1e-8)
    return avg_prob[:orig_z, :orig_y, :orig_x].numpy()

def run_inference(config_path, checkpoint_path, output_dir=None, split="val",
                  subjects=None, skip_existing=False):
    """Runs sliding-window 3D U-Net inference and writes ``<subject>_prob.nii.gz`` maps.

    Note on preprocessing: the network input is a per-volume z-score of the *raw*
    stored intensities (no Hounsfield standardization), exactly as in
    ``ribline.model.unet.dataset.RibDLDataset`` used for training.

    Args:
        config_path: Training/inference JSON config (uses ``data_config``).
        checkpoint_path: ``state_dict`` checkpoint (e.g. ``rib_unet_3d.pth``).
        output_dir: Output directory (default: ``<output_directory>/predictions``).
        split: Which ``split.json`` partition to run on: ``val`` (default),
            ``train`` or ``all``. Ignored when ``subjects`` is given.
        subjects: Optional explicit list of subject IDs.
        skip_existing: Skip subjects whose probability map already exists.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    data_cfg = config['data_config']
    
    splits = json.load(open(data_cfg['split_file']))
    if subjects is None:
        if split == "all":
            subjects = list(splits.get("train", [])) + list(splits.get("val", []))
        else:
            subjects = list(splits[split])
    image_dir = data_cfg['image_directory']
    
    if output_dir is None:
        output_dir = os.path.join(config['training_config']['output_directory'], "predictions")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model
    model = RibUNet(in_channels=1, out_channels=1, base_filters=16)
    # Checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    print(f"Running Inference on {len(subjects)} subjects (split={split})...")
    
    for subject in tqdm(subjects):
        save_path = os.path.join(output_dir, f"{subject}_prob.nii.gz")
        if skip_existing and os.path.exists(save_path):
            continue
        img_path = resolve_subject_image_path(image_dir, subject)
        if not os.path.exists(img_path):
             continue
             
        # Load
        image = load_image(img_path)
        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
        # Inference
        prob_arr = sliding_window_inference(model, img_arr)
        
        # Save
        prob_img = sitk.GetImageFromArray(prob_arr)
        prob_img.CopyInformation(image)
        
        sitk.WriteImage(prob_img, save_path)
        
    print("Inference Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"],
                        help="split.json partition to run inference on")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Optional text file with one subject ID per line")
    parser.add_argument("--skip-existing", action="store_true")
    
    args = parser.parse_args()
    subj = None
    if args.subjects:
        subj = [l.strip() for l in open(args.subjects) if l.strip()]
    
    run_inference(args.config, args.checkpoint, args.output, split=args.split,
                  subjects=subj, skip_existing=args.skip_existing)
