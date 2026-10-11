
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
from ...data.dataset import RibDataset

class RibDLDataset(Dataset):
    def __init__(self, rib_dataset: RibDataset, cache_dir: str, patch_size=(64, 64, 64), 
                 samples_per_volume=4, training=True, augmentation=False):
        """
        Deep Learning Wrapper for RibDataset.
        Handles:
        1. Mask Generation (cached to disk)
        2. Normalization: per-volume z-score of the RAW stored intensities. NOTE:
           no Hounsfield standardization is applied here (the released U-Net was
           trained this way; `standardize_hu` is only used by the forest/Stage-3 code).
        3. Patch Sampling (Foreground/Background)
        4. Optional augmentation (`augmentation=True`): random left-right flip of
           the patch and a random intensity scale in [0.9, 1.1]. The released
           `rib_unet_3d.pth` was trained with `augmentation=False`.
        """
        self.rib_dataset = rib_dataset
        self.cache_dir = cache_dir
        self.patch_size = np.array(patch_size)
        self.samples_per_volume = samples_per_volume
        self.training = training
        self.augmentation = bool(augmentation)
        
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        # We define length as (num_subjects * samples_per_volume)
        # This allows the DataLoader to iterate "samples"
        return len(self.rib_dataset) * self.samples_per_volume

    def _load_volume_and_mask(self, subject_idx):
        """
        Loads image and mask. Generates/Caches mask if missing.
        """
        # Load Raw Data
        image, centerlines, subject_id = self.rib_dataset.load_item(subject_idx)
        
        # Check Cache
        mask_path = os.path.join(self.cache_dir, f"{subject_id}_mask.nii.gz")
        
        if os.path.exists(mask_path):
            mask_img = sitk.ReadImage(mask_path)
            # Ensure orientation matches? We assume RibDataset doesn't change it.
            # But just in case, we might want to be safe.
            # Usually OK if it's just saved/loaded.
        else:
            # Generate Mask
            # Use existing RibDataset logic or custom? dataset.py has generate_mask
            mask_img = self.rib_dataset.generate_mask(image, centerlines)
            
            # Save to Cache
            # Set same meta data
            mask_img.SetOrigin(image.GetOrigin())
            mask_img.SetSpacing(image.GetSpacing())
            mask_img.SetDirection(image.GetDirection())
            
            sitk.WriteImage(mask_img, mask_path)
            
        return image, mask_img

    def __getitem__(self, idx):
        # Map linear index to subject
        subject_idx = idx // self.samples_per_volume
        
        image_sitk, mask_sitk = self._load_volume_and_mask(subject_idx)
        
        # Convert to Numpy
        img_arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32) # (Z, Y, X)
        mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(np.float32) # (Z, Y, X)
        
        # Normalization (Z-score per volume)
        mean = np.mean(img_arr)
        std = np.std(img_arr)
        img_arr = (img_arr - mean) / (std + 1e-8)
        
        # Patch Sampling
        if self.training:
            patch = self._sample_patch_random(img_arr, mask_arr)
            if self.augmentation:
                patch = self._augment(patch)
        else:
            # Deterministic sampling for validation? 
            # Or just center crop? Let's do random for now, effectively "Stochastic Validation"
            # Or use a seeded random generator based on idx.
            rng = np.random.RandomState(idx)
            patch = self._sample_patch_random(img_arr, mask_arr, rng=rng)
            
        return patch

    @staticmethod
    def _augment(patch, rng=np.random):
        """Random left-right flip (X axis) and intensity scaling in [0.9, 1.1]."""
        img, mask = patch['image'], patch['mask']
        if rng.rand() < 0.5:
            img = torch.flip(img, dims=[3])
            mask = torch.flip(mask, dims=[3])
        scale = float(rng.uniform(0.9, 1.1))
        img = img * scale
        return {'image': img.contiguous(), 'mask': mask.contiguous()}

    def _sample_patch_random(self, img, mask, rng=np.random):
        """
        Samples a 3D patch.
        Strategy: 
        - 70% chance to center on a random foreground voxel.
        - 30% chance to be completely random.
        """
        z_dim, y_dim, x_dim = img.shape
        pz, py, px = self.patch_size
        
        # Padding if volume is smaller than patch
        # (Assuming volumes are large enough for now, otherwise crash)
        
        # Decide center
        is_foreground = rng.rand() < 0.7
        foreground_indices = np.argwhere(mask > 0)
        
        if is_foreground and len(foreground_indices) > 0:
            center_idx = rng.choice(len(foreground_indices))
            cz, cy, cx = foreground_indices[center_idx]
            
            # Add random jitter
            dz = rng.randint(-pz//4, pz//4)
            dy = rng.randint(-py//4, py//4)
            dx = rng.randint(-px//4, px//4)
            
            sz = int(np.clip(cz + dz - pz//2, 0, z_dim - pz))
            sy = int(np.clip(cy + dy - py//2, 0, y_dim - py))
            sx = int(np.clip(cx + dx - px//2, 0, x_dim - px))
        else:
            # Random crop
            sz = rng.randint(0, z_dim - pz + 1)
            sy = rng.randint(0, y_dim - py + 1)
            sx = rng.randint(0, x_dim - px + 1)
            
        # Crop
        img_patch = img[sz:sz+pz, sy:sy+py, sx:sx+px]
        mask_patch = mask[sz:sz+pz, sy:sy+py, sx:sx+px]
        
        # Add channel dim
        return {
            'image': torch.from_numpy(img_patch).unsqueeze(0), # (1, D, H, W)
            'mask': torch.from_numpy(mask_patch).unsqueeze(0)  # (1, D, H, W)
        }
