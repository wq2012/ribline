"""Multi-scale 3D differential-geometric, anatomical, and RSID context features."""

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from ribline.model.anatomy import (
    extract_thoracic_frame,
    phys_to_canonical,
    sample_lung_distance,
    standardize_hu,
)


class FeatureExtractor:
    """Extracts per-voxel multi-scale differential, anatomical and RSID features.

    Per scale sigma in ``scales`` (mm): Gaussian-smoothed intensity, gradient
    magnitude, the six second-order Gaussian derivatives (Ixx, Iyy, Izz, Ixy,
    Ixz, Iyz) and the Laplacian (9 features), plus the raw intensity (1).

    Random Shift Intensity Difference (RSID) features: ``num_pairs`` pairs of
    integer offsets drawn uniformly in ``[-max_offset, max_offset]^3`` **in
    voxels** (so +-20 voxels = +-40 mm on the 2 mm SCR grid); the feature is
    ``I(x + u1) - I(x + u2)`` on the (standardised) intensity image.

    Anatomically-anchored mode (``use_anatomical_features=True``) additionally
    standardises Hounsfield units and appends, per scale, a cheap *tubularity
    proxy* built only from the diagonal second derivatives,
    ``(-min(0,Ixx) - min(0,Iyy) - min(0,Izz)) * r / (r + 50)`` with
    ``r = sqrt(Ixx^2 + Iyy^2 + Izz^2)`` (no Hessian eigen-decomposition and no
    Frangi/Sato filter is computed), and five anatomical channels: canonical
    coordinates |u|, u, v, w of ``phys_to_canonical`` and the 3D distance to
    the lung mask ``d_lung``.
    """

    def __init__(
        self,
        scales=None,
        rsid_params=None,
        use_anatomical_features: bool = False,
        use_standard_hu: bool = False,
    ):
        if scales is None:
            scales = [1.0, 2.0, 4.0]
        self.scales = list(scales)
        self.rsid_params = rsid_params
        self.use_anatomical_features = bool(use_anatomical_features)
        self.use_standard_hu = bool(
            use_standard_hu or use_anatomical_features
        )
        self.rsid_offsets = None

        if self.rsid_params:
            self._init_rsid_offsets()

    def _init_rsid_offsets(self):
        seed = self.rsid_params.get("seed", 42)
        num_pairs = self.rsid_params.get("num_pairs", 50)
        max_offset = self.rsid_params.get("max_offset", 10)
        rng = np.random.RandomState(seed)
        self.rsid_offsets = rng.randint(
            -max_offset, max_offset + 1, size=(num_pairs, 2, 3)
        )

    def _prepare_sitk_image(self, image: sitk.Image) -> sitk.Image:
        if not self.use_standard_hu:
            return image
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        # Only standardize if this is a CT intensity image (not a [0, 1] probability map)
        if float(np.max(arr)) > 5.0:
            arr = standardize_hu(arr)
            out = sitk.GetImageFromArray(arr)
            out.CopyInformation(image)
            return out
        return image

    def compute_features(self, image: sitk.Image, frame: dict = None) -> list:
        """Computes dense 3D feature volumes for the given SimpleITK image."""
        proc_img = self._prepare_sitk_image(image)
        features = [proc_img]

        for sigma in self.scales:
            smoothed = sitk.SmoothingRecursiveGaussian(proc_img, sigma)
            features.append(smoothed)

            grad = sitk.GradientMagnitudeRecursiveGaussian(proc_img, sigma)
            features.append(grad)

            def get_gaussian_derivative(input_img, orders, sig):
                temp = input_img
                for dim, order in enumerate(orders):
                    filt = sitk.RecursiveGaussianImageFilter()
                    filt.SetSigma(sig)
                    filt.SetDirection(dim)
                    filt.SetOrder(order)
                    temp = filt.Execute(temp)
                return temp

            gxx = get_gaussian_derivative(proc_img, (2, 0, 0), sigma)
            gyy = get_gaussian_derivative(proc_img, (0, 2, 0), sigma)
            gzz = get_gaussian_derivative(proc_img, (0, 0, 2), sigma)
            gxy = get_gaussian_derivative(proc_img, (1, 1, 0), sigma)
            gxz = get_gaussian_derivative(proc_img, (1, 0, 1), sigma)
            gyz = get_gaussian_derivative(proc_img, (0, 1, 1), sigma)

            features.extend([gxx, gyy, gzz, gxy, gxz, gyz])

            laplacian = sitk.LaplacianRecursiveGaussian(proc_img, sigma)
            features.append(laplacian)

            if self.use_anatomical_features:
                # Add 3D ridge tubularity proxy from diagonal curvature at scale sigma
                hxx = sitk.GetArrayViewFromImage(gxx)
                hyy = sitk.GetArrayViewFromImage(gyy)
                hzz = sitk.GetArrayViewFromImage(gzz)
                # For a bright tubular rib in 3D, two cross-sectional curvatures are strongly negative
                neg_curv = np.minimum(0.0, hxx) + np.minimum(0.0, hyy) + np.minimum(0.0, hzz)
                ridge_strength = np.sqrt(hxx**2 + hyy**2 + hzz**2 + 1e-6)
                tubularity = (-neg_curv) * (ridge_strength / (ridge_strength + 50.0))
                tub_img = sitk.GetImageFromArray(tubularity.astype(np.float32))
                tub_img.CopyInformation(proc_img)
                features.append(tub_img)

        if self.use_anatomical_features:
            arr = sitk.GetArrayFromImage(proc_img).astype(np.float32)
            spacing = proc_img.GetSpacing()
            origin = proc_img.GetOrigin()
            if frame is None and float(np.max(arr)) > 5.0:
                frame = extract_thoracic_frame(arr, spacing=spacing, origin=origin)
            if frame is not None:
                nz, ny, nx = arr.shape
                sx, sy, sz = spacing
                ox, oy, oz = origin
                z_phys = np.arange(nz, dtype=np.float32) * sz + oz
                y_phys = np.arange(ny, dtype=np.float32) * sy + oy
                x_phys = np.arange(nx, dtype=np.float32) * sx + ox

                u_1d = (x_phys - frame["x_spine"]) / frame["w_thorax"]
                v_1d = (y_phys - frame["y_spine"]) / frame["d_thorax"]
                w_1d = (z_phys - frame["z_apex"]) / 240.0

                u_vol = np.broadcast_to(u_1d[None, None, :], (nz, ny, nx)).astype(np.float32)
                abs_u_vol = np.abs(u_vol)
                v_vol = np.broadcast_to(v_1d[None, :, None], (nz, ny, nx)).astype(np.float32)
                w_vol = np.broadcast_to(w_1d[:, None, None], (nz, ny, nx)).astype(np.float32)

                dist_sub = frame["dist_lung_sub"]
                nz_s, ny_s, nx_s = dist_sub.shape
                iz_s = np.clip(np.arange(nz) // 2, 0, nz_s - 1)
                iy_s = np.clip(np.arange(ny) // 2, 0, ny_s - 1)
                ix_s = np.clip(np.arange(nx) // 2, 0, nx_s - 1)
                dl_vol = dist_sub[
                    iz_s[:, None, None], iy_s[None, :, None], ix_s[None, None, :]
                ].astype(np.float32)

                for vol_arr in (abs_u_vol, u_vol, v_vol, w_vol, dl_vol):
                    f_img = sitk.GetImageFromArray(vol_arr)
                    f_img.CopyInformation(proc_img)
                    features.append(f_img)

        return features

    def extract_at_points(self, image: sitk.Image, points, feature_images=None) -> np.ndarray:
        """Extracts feature vectors at specified (x, y, z) voxel indices (vectorized)."""
        if points is None or len(points) == 0:
            return np.empty((0, 0), dtype=np.float32)

        pts_arr = np.asarray(points, dtype=np.int64)  # (N, 3) -> (x, y, z)
        n_points = len(pts_arr)
        feature_parts = []

        if feature_images:
            n_dense = len(feature_images)
            X_dense = np.zeros((n_points, n_dense), dtype=np.float32)
            z_idx = pts_arr[:, 2]
            y_idx = pts_arr[:, 1]
            x_idx = pts_arr[:, 0]
            for i, feat_img in enumerate(feature_images):
                arr = sitk.GetArrayViewFromImage(feat_img)
                X_dense[:, i] = arr[z_idx, y_idx, x_idx]
            feature_parts.append(X_dense)

        if self.rsid_offsets is not None:
            proc_img = self._prepare_sitk_image(image)
            img_arr = sitk.GetArrayViewFromImage(proc_img)  # (z, y, x)
            shape = img_arr.shape
            pts_zyx = pts_arr[:, [2, 1, 0]]  # (N, 3)
            off1 = self.rsid_offsets[:, 0, :]  # (P, 3)
            off2 = self.rsid_offsets[:, 1, :]  # (P, 3)

            def get_vals(offsets):
                coords = pts_zyx[None, :, :] + offsets[:, None, :]
                z_tgt = np.clip(coords[..., 0], 0, shape[0] - 1)
                y_tgt = np.clip(coords[..., 1], 0, shape[1] - 1)
                x_tgt = np.clip(coords[..., 2], 0, shape[2] - 1)
                return img_arr[z_tgt, y_tgt, x_tgt].T.astype(np.float32)

            val1 = get_vals(off1)
            val2 = get_vals(off2)
            X_rsid = val1 - val2
            feature_parts.append(X_rsid)

        if not feature_parts:
            return np.zeros((n_points, 0), dtype=np.float32)

        return np.concatenate(feature_parts, axis=1)

    def get_feature_config(self) -> dict:
        """Returns serializable feature extractor configuration."""
        cfg = {
            "scales": self.scales,
            "rsid_params": self.rsid_params,
        }
        if self.use_anatomical_features:
            cfg["use_anatomical_features"] = True
        if self.use_standard_hu:
            cfg["use_standard_hu"] = True
        return cfg

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            scales=config.get("scales", [1.0, 2.0, 4.0]),
            rsid_params=config.get("rsid_params", None),
            use_anatomical_features=config.get("use_anatomical_features", False),
            use_standard_hu=config.get("use_standard_hu", False),
        )

    def extract_slice(self, image: sitk.Image, slice_idx: int, feature_images=None) -> np.ndarray:
        """Extracts features for an entire axial slice z = slice_idx (vectorized)."""
        size = image.GetSize()  # (x, y, z)
        width, height = size[0], size[1]
        n_pixels = width * height
        feature_parts = []

        if feature_images:
            n_dense = len(feature_images)
            X_dense = np.zeros((n_pixels, n_dense), dtype=np.float32)
            for i, feat_img in enumerate(feature_images):
                arr = sitk.GetArrayViewFromImage(feat_img)
                slice_arr = arr[slice_idx, :, :]
                X_dense[:, i] = slice_arr.flatten()
            feature_parts.append(X_dense)

        if self.rsid_offsets is not None:
            proc_img = self._prepare_sitk_image(image)
            img_arr = sitk.GetArrayViewFromImage(proc_img)
            shape = img_arr.shape
            y_range = np.arange(height)
            x_range = np.arange(width)
            y_grid, x_grid = np.meshgrid(y_range, x_range, indexing="ij")
            y_flat = y_grid.flatten()
            x_flat = x_grid.flatten()

            off1 = self.rsid_offsets[:, 0, :]
            off2 = self.rsid_offsets[:, 1, :]

            def get_vals_slice(offsets):
                z_tgt = np.clip(slice_idx + offsets[:, 0], 0, shape[0] - 1)
                y_tgt = np.clip(y_flat[None, :] + offsets[:, 1][:, None], 0, shape[1] - 1)
                x_tgt = np.clip(x_flat[None, :] + offsets[:, 2][:, None], 0, shape[2] - 1)
                return img_arr[z_tgt[:, None], y_tgt, x_tgt].T.astype(np.float32)

            val1 = get_vals_slice(off1)
            val2 = get_vals_slice(off2)
            X_rsid = val1 - val2
            feature_parts.append(X_rsid)

        if not feature_parts:
            return np.zeros((n_pixels, 0), dtype=np.float32)

        return np.concatenate(feature_parts, axis=1)
