import os
import glob
import numpy as np
from .image_reader import load_image, resolve_subject_image_path, strip_image_extension
from .centerline_parser import parse_rib_xls
import SimpleITK as sitk

class RibDataset:
    def __init__(self, image_dir, centerline_dir=None, selected_files=None, radius=2.0):
        """
        Args:
            image_dir (str): Directory containing 3D volume files (.nii.gz, .nii, .img/.hdr, .mhd, .nrrd).
            centerline_dir (str, optional): Directory containing .xls centerline files.
            selected_files (list, optional): List of filenames (without extension) to include.
            radius (float): Radius around centerline to consider as positive class.
        """
        self.image_dir = image_dir
        self.centerline_dir = centerline_dir
        self.radius = radius
        self.subjects = self._discover_subjects(selected_files)
        
    def _discover_subjects(self, selected_files):
        """
        Matches images with centerline files across supported 3D volume extensions.
        Returns a sorted list of unique subject IDs (filenames without extension).
        """
        if not self.image_dir or not os.path.isdir(self.image_dir):
            return []

        candidate_ids = []
        seen = set()
        for fname in sorted(os.listdir(self.image_dir)):
            if fname.endswith(("_prob.nii.gz", "_prob.hdr", "_prob.img", "_mask.nii.gz")):
                continue
            subject_id, ext = strip_image_extension(fname)
            if not subject_id or subject_id in seen:
                continue
            if ext == ".hdr" and not os.path.exists(os.path.join(self.image_dir, f"{subject_id}.img")):
                continue
            seen.add(subject_id)
            candidate_ids.append(subject_id)

        subjects = []
        for subject_id in candidate_ids:
            if selected_files is not None and subject_id not in selected_files:
                continue
            if self.centerline_dir is None:
                subjects.append(subject_id)
                continue

            gt_path = os.path.join(self.centerline_dir, f"{subject_id}_interp.xls")
            if os.path.exists(gt_path) and os.path.getsize(gt_path) > 0:
                subjects.append(subject_id)
            else:
                gt_path_exact = os.path.join(self.centerline_dir, f"{subject_id}.xls")
                if os.path.exists(gt_path_exact) and os.path.getsize(gt_path_exact) > 0:
                    subjects.append(subject_id)

        return sorted(subjects)

    @staticmethod
    def load_centerlines(gt_path):
        """Loads ground-truth rib centerlines from a tabular .xls file."""
        return parse_rib_xls(gt_path)

    def __len__(self):
        return len(self.subjects)

    def load_item(self, idx):
        """
        Loads image and centerlines for a given index.
        Returns:
            image (sitk.Image): The 3D image.
            centerlines (dict): Dictionary of centerline coordinates.
            subject_id (str): Subject identifier.
        """
        subject_id = self.subjects[idx]
        img_path = resolve_subject_image_path(self.image_dir, subject_id)
        image = load_image(img_path)

        centerlines = {}
        if self.centerline_dir is not None:
            gt_path = os.path.join(self.centerline_dir, f"{subject_id}_interp.xls")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(self.centerline_dir, f"{subject_id}.xls")
            if os.path.exists(gt_path):
                centerlines = parse_rib_xls(gt_path)

        return image, centerlines, subject_id

    def generate_mask(self, image, centerlines):
        """
        Generates a binary mask where 1 indicates rib centerline vicinity.
        """
        # Create empty mask with same size as image
        mask_arr = np.zeros(image.GetSize()[::-1], dtype=np.uint8) # SITK: x,y,z -> Numpy: z,y,x
        
        spacing = image.GetSpacing() # sx, sy, sz
        origin = image.GetOrigin()
        
        for rib_name, points in centerlines.items():
            for pt in points:
                phys_pt = pt
                idx_x = int((phys_pt[0] - origin[0]) / spacing[0])
                idx_y = int((phys_pt[1] - origin[1]) / spacing[1])
                idx_z = int((phys_pt[2] - origin[2]) / spacing[2])
                
                avg_spacing = np.mean(spacing)
                radius_vox = int(np.ceil(self.radius / avg_spacing))
                
                z_dim, y_dim, x_dim = mask_arr.shape
                
                for z in range(max(0, idx_z - radius_vox), min(z_dim, idx_z + radius_vox + 1)):
                    for y in range(max(0, idx_y - radius_vox), min(y_dim, idx_y + radius_vox + 1)):
                        for x in range(max(0, idx_x - radius_vox), min(x_dim, idx_x + radius_vox + 1)):
                            dist_sq = (x - idx_x)**2 + (y - idx_y)**2 + (z - idx_z)**2
                            if dist_sq <= radius_vox**2:
                                mask_arr[z, y, x] = 1
                                
        mask_img = sitk.GetImageFromArray(mask_arr)
        mask_img.CopyInformation(image)
        return mask_img

        
    def augment_item(self, image, centerlines):
        """
        Applies random rigid transformation (Rotation + Translation) to image and centerlines.
        """
        # Random Parameters
        angle_z = np.radians(np.random.uniform(-15, 15))
        tx = np.random.uniform(-10, 10) # mm
        ty = np.random.uniform(-10, 10)
        tz = np.random.uniform(-10, 10)
        
        # Transform
        transform = sitk.Euler3DTransform()
        
        # Center of rotation (Image Center)
        size = image.GetSize()
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        center_idx = [sz/2.0 for sz in size]
        # center_phys = origin + center_idx * spacing (element-wise)
        center_phys = [origin[d] + center_idx[d]*spacing[d] for d in range(3)]
        
        transform.SetCenter(center_phys)
        transform.SetRotation(0, 0, angle_z)
        transform.SetTranslation((tx, ty, tz))
        
        # Resample Image
        # Use bilinear or bspline? Linear is fast.
        aug_image = sitk.Resample(image, image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())
        
        # Transform Centerlines
        aug_centerlines = {}
        for rib, pts in centerlines.items():
            new_pts = []
            for pt in pts:
                # pt is (x,y,z) physical
                # The transform maps Output Point -> Input Point for resampling.
                # But for points, we want T(p).
                # sitk.Resample(ref, transform) means: output(x) = input(transform(x)).
                # So the transform describes "Where to look in input".
                # This is the INVERSE of the coordinate transform if we think "Move the object".
                # If we use transform.TransformPoint(pt), we are applying the "Sampling Transform".
                # Wait.
                # Resample: I_new(x) = I_old(T(x)).
                # T maps New Space -> Old Space.
                # If T is "Rotation +15 deg", then T(x) rotates x by +15.
                # So I_new(x) gets value from I_old(x rotated).
                # This means the content moves by -15 deg ??
                # Let's verify.
                # If I_new(0) = I_old(T(0)).
                # This is standard "Pull back".
                
                # For centerlines (Points), we want the points to match the NEW image content.
                # The new image content at `x` corresponds to old content at `T(x)`.
                # So if a feature was at `p_old` in old image, where is it in `p_new`?
                # We want `p_old = T(p_new)`.
                # So `p_new = T_inv(p_old)`.
                
                # So we need the INVERSE transform for points.
                
                p_inv = transform.GetInverse().TransformPoint(pt)
                new_pts.append(p_inv)
            aug_centerlines[rib] = np.array(new_pts)
            
        return aug_image, aug_centerlines

    def get_samples_from_data(self, image, centerlines, num_samples=1000):
        """
        Samples points (indices) from the provided image/centerlines.
        Used by get_training_samples and external callers.
        """
        img_arr = sitk.GetArrayFromImage(image) # View might be risky if we modify? GetArray is copy.
        # Actually we just need shape.
        z_dim, y_dim, x_dim = img_arr.shape
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        
        # 1. Positive Samples
        pos_samples = []
        all_rib_points = []
        for r, pts in centerlines.items():
            all_rib_points.extend(pts)
            
        if not all_rib_points:
            return [], []

        # Sample random points from centerline
        selected_indices = np.random.choice(len(all_rib_points), size=num_samples // 2, replace=True)
        
        for i in selected_indices:
            pt = all_rib_points[i]
            
            # Physical -> Index
            # We must use proper rounding or simple cast
            idx_x = int((pt[0] - origin[0]) / spacing[0])
            idx_y = int((pt[1] - origin[1]) / spacing[1])
            idx_z = int((pt[2] - origin[2]) / spacing[2])
            
            # Check bounds
            if (0 <= idx_z < z_dim and 
                0 <= idx_y < y_dim and 
                0 <= idx_x < x_dim):
                pos_samples.append(((idx_x, idx_y, idx_z), 1))
                
        # 2. Negative Samples
        neg_samples = []
        
        count = 0
        while count < num_samples // 2:
            rz = np.random.randint(0, z_dim)
            ry = np.random.randint(0, y_dim)
            rx = np.random.randint(0, x_dim)
            
            # TODO: Check collision with positive?
            neg_samples.append(((rx, ry, rz), 0))
            count += 1
            
        return pos_samples, neg_samples

    def get_training_samples(self, idx, num_samples=1000):
        """
        Wrapper to load item and get samples (No augmentation by default).
        """
        image, centerlines, _ = self.load_item(idx)
        return self.get_samples_from_data(image, centerlines, num_samples)
