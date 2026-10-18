"""Anatomical coordinate frame and statistical rib cage atlas for 3D CT."""

import os
import joblib
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree


def standardize_hu(arr: np.ndarray) -> np.ndarray:
    """Standardizes raw CT intensity values to calibrated Hounsfield Units (HU).

    Handles both unsigned 12-bit shifted CT scans (0 = -1024 HU, water = 1024)
    and already-signed Hounsfield Unit scans (air ~ -1000 HU, water = 0 HU).
    """
    out = arr.astype(np.float32, copy=True)
    if float(np.min(out)) >= -100.0:
        out -= 1024.0
    return out


def resample_curve(pts: np.ndarray, num_points: int = 80) -> np.ndarray:
    """Resamples a 3D polyline uniformly by arc length."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts[:1], num_points, axis=0).astype(np.float32)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total_len = float(cum[-1])
    if total_len < 1e-3:
        return np.repeat(pts[:1], num_points, axis=0).astype(np.float32)
    s_new = np.linspace(0.0, total_len, num_points)
    return np.column_stack(
        [np.interp(s_new, cum, pts[:, j]) for j in range(3)]
    ).astype(np.float32)


def curve_length(pts: np.ndarray) -> float:
    """Computes total Euclidean arc length of a 3D curve in mm."""
    if pts is None or len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def extract_thoracic_frame(
    hu_arr: np.ndarray,
    spacing: tuple = (2.0, 2.0, 2.0),
    origin: tuple = (0.0, 0.0, 0.0),
    compute_full_distances: bool = False,
) -> dict:
    """Extracts the canonical thoracic coordinate frame from a 3D CT volume.

    On a 2x downsampled grid: torso = HU > -350 (slice-wise hole filling);
    internal air = torso & HU < -380, opened and reduced to the <= 4 largest
    components (> 800 sub-voxels) = lungs; ``z_apex`` = max of the per-side
    99.5th-percentile Z of the lung mask; the spine (X_spine, Y_spine) is the
    peak of the smoothed bone projection (HU > 150) in the posterior 40 % of
    the AP depth, refined by a slice-wise median and a Gaussian (sigma = 12
    slices) to give ``x_spine_z`` / ``y_spine_z``; thoracic half-width
    ``w_thorax`` and AP depth ``d_thorax`` come from bone-voxel percentiles.
    Measured on the 59 valid SCR scans (``ribline.scripts.polar_angle_stats``):
    the posterior end of R01/L01 lies +11.9 +- 9.3 mm above ``z_apex`` (n = 117)
    and the midline error |X_spine(Z) - mid(R,L attachments)| is 2.5 +- 2.4 mm.

    Args:
        hu_arr: 3D numpy array (Z, Y, X) in raw or standardized HU.
        spacing: Voxel spacing (sx, sy, sz) in mm.
        origin: Physical origin (ox, oy, oz) in mm.
        compute_full_distances: If True, also returns full-resolution 3D
            distance fields (d_lung, d_bone) for voxel feature extraction.

    Returns:
        Dictionary describing the patient's thoracic coordinate frame.
    """
    hu = standardize_hu(hu_arr)
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    nz, ny, nx = hu.shape

    # 2x downsampled grid for fast morphological lung & spine localization
    hu_sub = hu[::2, ::2, ::2]
    nz_s, ny_s, nx_s = hu_sub.shape
    sx_s, sy_s, sz_s = sx * 2.0, sy * 2.0, sz * 2.0

    torso = hu_sub > -350.0
    torso_filled = np.zeros_like(torso)
    for z in range(nz_s):
        torso_filled[z] = ndi.binary_fill_holes(torso[z])

    air_internal = torso_filled & (~torso) & (hu_sub < -380.0)
    air_open = ndi.binary_opening(air_internal, structure=np.ones((2, 2, 2), dtype=bool))
    lbl, n_comp = ndi.label(air_open)
    lung_mask_sub = np.zeros_like(air_open)
    if n_comp > 0:
        sizes = ndi.sum(air_open, lbl, range(1, n_comp + 1))
        for idx in np.argsort(sizes)[::-1][:4]:
            if sizes[idx] > 800:
                lung_mask_sub |= lbl == (idx + 1)
        if not np.any(lung_mask_sub):
            lung_mask_sub = lbl == (int(np.argmax(sizes)) + 1)
    else:
        lung_mask_sub = air_internal

    zz_l, yy_l, xx_l = np.nonzero(lung_mask_sub)
    if len(zz_l) > 50:
        x_mid_lung = float(np.median(xx_l) * sx_s + ox)
        y_mid_lung = float(np.median(yy_l) * sy_s + oy)
        r_mask = (xx_l * sx_s + ox) < x_mid_lung
        l_mask = (xx_l * sx_s + ox) >= x_mid_lung
        z_apex_r = (
            float(np.percentile(zz_l[r_mask], 99.5) * sz_s + oz)
            if np.any(r_mask)
            else float(np.percentile(zz_l, 99.5) * sz_s + oz)
        )
        z_apex_l = (
            float(np.percentile(zz_l[l_mask], 99.5) * sz_s + oz)
            if np.any(l_mask)
            else float(np.percentile(zz_l, 99.5) * sz_s + oz)
        )
        z_apex = max(z_apex_r, z_apex_l)
        z_base = float(np.percentile(zz_l, 2.0) * sz_s + oz)
    else:
        x_mid_lung = ox + 0.5 * nx * sx
        y_mid_lung = oy + 0.5 * ny * sy
        z_apex = oz + 0.85 * nz * sz
        z_apex_r = z_apex_l = z_apex
        z_base = oz + 0.15 * nz * sz

    # Locate posterior vertebral column (spine) via vertical bone projection
    bone_sub = hu_sub > 150.0
    zz_b, yy_b, xx_b = np.nonzero(bone_sub)
    if len(zz_b) > 50:
        x_mid_init = 0.5 * (
            float(np.percentile(xx_b * sx_s + ox, 5.0))
            + float(np.percentile(xx_b * sx_s + ox, 95.0))
        )
        y_ant_init = float(np.percentile(yy_b * sy_s + oy, 5.0))
        y_post_init = float(np.percentile(yy_b * sy_s + oy, 95.0))
        # Vertebral column lies in the posterior 40% of the thoracic AP depth
        y_spine_min = y_ant_init + 0.60 * (y_post_init - y_ant_init)
    else:
        x_mid_init = x_mid_lung
        y_spine_min = y_mid_lung + 20.0

    z_top_idx = min(nz_s, max(1, int((z_apex - oz) / sz_s) + 2))
    z_bot_idx = max(0, min(z_top_idx - 1, int((z_apex - 220.0 - oz) / sz_s)))
    bone_str = np.clip(hu_sub[z_bot_idx:z_top_idx] - 120.0, 0.0, 600.0)
    proj_xy = (
        np.mean(bone_str, axis=0)
        if bone_str.shape[0] > 0
        else np.mean(np.clip(hu_sub - 120.0, 0.0, 600.0), axis=0)
    )
    proj_smooth = ndi.gaussian_filter(proj_xy, sigma=3.0)
    yy_grid, xx_grid = np.meshgrid(
        np.arange(ny_s) * sy_s + oy,
        np.arange(nx_s) * sx_s + ox,
        indexing="ij",
    )
    mask_search = (np.abs(xx_grid - x_mid_init) < 50.0) & (yy_grid >= y_spine_min)
    if np.any(mask_search):
        iy, ix = np.unravel_index(
            np.argmax(np.where(mask_search, proj_smooth, -1.0)), proj_smooth.shape
        )
        x_spine = float(ix * sx_s + ox)
        y_spine = float(iy * sy_s + oy)
        # Refine x_spine via center of mass of posterior vertebral peak
        near_peak = (
            (np.abs(xx_grid - x_spine) < 28.0)
            & (np.abs(yy_grid - y_spine) < 28.0)
            & (proj_smooth > 0.35 * proj_smooth[iy, ix])
        )
        if np.any(near_peak):
            w_p = proj_smooth[near_peak]
            x_spine = float(np.sum(xx_grid[near_peak] * w_p) / np.sum(w_p))
            y_spine = float(np.sum(yy_grid[near_peak] * w_p) / np.sum(w_p))
    else:
        x_spine = x_mid_lung
        y_spine = y_mid_lung + 60.0

    # Slice-by-slice curved spine midline X_spine(Z), Y_spine(Z)
    x_spine_z = np.full(nz, x_spine, dtype=np.float32)
    y_spine_z = np.full(nz, y_spine, dtype=np.float32)
    for z_s in range(nz_s):
        sl = bone_sub[z_s]
        if np.any(sl):
            ys, xs = np.nonzero(sl)
            xs_p = xs * sx_s + ox
            ys_p = ys * sy_s + oy
            near_sp = (np.abs(xs_p - x_spine) < 32.0) & (np.abs(ys_p - y_spine) < 36.0)
            if np.sum(near_sp) >= 5:
                z0 = min(nz - 1, z_s * 2)
                z1 = min(nz - 1, z_s * 2 + 1)
                x_spine_z[z0] = x_spine_z[z1] = float(np.median(xs_p[near_sp]))
                y_spine_z[z0] = y_spine_z[z1] = float(np.median(ys_p[near_sp]))
    x_spine_z = ndi.gaussian_filter1d(x_spine_z, sigma=12.0, mode="nearest")
    y_spine_z = ndi.gaussian_filter1d(y_spine_z, sigma=12.0, mode="nearest")

    # Measure thoracic cage half-width and AP depth in thoracic region
    if len(zz_b) > 100:
        zb_p = zz_b * sz_s + oz
        yb_p = yy_b * sy_s + oy
        xb_p = xx_b * sx_s + ox
        in_thorax = (
            (zb_p <= z_apex - 10.0)
            & (zb_p >= z_apex - 210.0)
            & (np.abs(xb_p - x_spine) < 175.0)
        )
        if np.sum(in_thorax) > 100:
            w_thorax = float(np.percentile(np.abs(xb_p[in_thorax] - x_spine), 96.0))
            y_ant = float(np.percentile(yb_p[in_thorax], 5.0))
            y_post = float(np.percentile(yb_p[in_thorax], 95.0))
        else:
            w_thorax = 135.0
            y_ant = y_spine - 160.0
            y_post = y_spine + 10.0
    else:
        w_thorax = 135.0
        y_ant = y_spine - 160.0
        y_post = y_spine + 10.0

    dist_lung_sub = ndi.distance_transform_edt(
        ~lung_mask_sub, sampling=(sz_s, sy_s, sx_s)
    )

    z_top_margin_mm = float((oz + (nz - 1) * sz) - z_apex)
    frame = {
        "x_spine": x_spine,
        "y_spine": y_spine,
        "x_spine_z": x_spine_z,
        "y_spine_z": y_spine_z,
        "y_lung": y_mid_lung,
        "z_apex": z_apex,
        "z_apex_r": z_apex_r,
        "z_apex_l": z_apex_l,
        "z_base": z_base,
        "z_top_margin_mm": z_top_margin_mm,
        "fov_truncated_superior": bool(z_top_margin_mm < 15.0),
        "w_thorax": max(w_thorax, 90.0),
        "y_ant": y_ant,
        "y_post": y_post,
        "d_thorax": max(y_post - y_ant, 100.0),
        "spacing": (sx, sy, sz),
        "origin": (ox, oy, oz),
        "vol_shape": (nz, ny, nx),
        "dist_lung_sub": dist_lung_sub.astype(np.float32),
    }

    if compute_full_distances:
        hu_smooth = ndi.gaussian_filter(hu, sigma=1.2)
        bone_core = (hu > 120.0) | (hu_smooth > 80.0)
        d_bone = ndi.distance_transform_edt(~bone_core, sampling=(sz, sy, sx)).astype(
            np.float32
        )
        frame["d_bone"] = d_bone
        frame["hu_smooth"] = hu_smooth.astype(np.float32)

    return frame


def sample_lung_distance(pts: np.ndarray, frame: dict) -> np.ndarray:
    """Queries the 3D distance to lung cavity (in mm) at physical 3D points."""
    if len(pts) == 0:
        return np.empty((0,), dtype=np.float32)
    ox, oy, oz = frame["origin"]
    sx, sy, sz = frame["spacing"]
    dist_sub = frame["dist_lung_sub"]
    nz_s, ny_s, nx_s = dist_sub.shape
    iz = np.clip(np.round((pts[:, 2] - oz) / (sz * 2.0)).astype(int), 0, nz_s - 1)
    iy = np.clip(np.round((pts[:, 1] - oy) / (sy * 2.0)).astype(int), 0, ny_s - 1)
    ix = np.clip(np.round((pts[:, 0] - ox) / (sx * 2.0)).astype(int), 0, nx_s - 1)
    return dist_sub[iz, iy, ix]


def phys_to_canonical(pts: np.ndarray, frame: dict) -> np.ndarray:
    """Maps physical (X, Y, Z) points in mm to canonical (u, v, w) coordinates."""
    pts = np.asarray(pts, dtype=np.float32)
    u = (pts[:, 0] - frame["x_spine"]) / frame["w_thorax"]
    v = (pts[:, 1] - frame["y_spine"]) / frame["d_thorax"]
    w = (pts[:, 2] - frame["z_apex"]) / 240.0
    return np.column_stack([u, v, w]).astype(np.float32)


def canonical_to_phys(canon_pts: np.ndarray, frame: dict) -> np.ndarray:
    """Maps canonical (u, v, w) coordinates back to physical (X, Y, Z) in mm."""
    canon_pts = np.asarray(canon_pts, dtype=np.float32)
    x = canon_pts[:, 0] * frame["w_thorax"] + frame["x_spine"]
    y = canon_pts[:, 1] * frame["d_thorax"] + frame["y_spine"]
    z = canon_pts[:, 2] * 240.0 + frame["z_apex"]
    return np.column_stack([x, y, z]).astype(np.float32)


def get_polar_angle(pts: np.ndarray, frame: dict, unwrap: bool = True) -> np.ndarray:
    """Computes the azimuthal angle around a kyphosis-adjusted hemithorax focus.

    Implementation (this is exactly what is computed, see Eq. 2 of the paper)::

        dx    = |X - x_spine|                       (scalar x_spine of the frame)
        yc(Z) = (y_ant + 0.25 * d_thorax) + 0.16 * clip(z_apex - Z, 0, 220 mm)
        theta = atan2(-(Y - yc(Z)), dx - 8.0 mm)

    i.e. the focus sits 8 mm lateral of the spine midline and moves posteriorly
    towards the lung apex. Measured on the 1,402 annotated ribs of the 59 valid
    SCR scans (``python -m ribline.scripts.polar_angle_stats``): the unwrapped
    angle increases from the posterior to the anterior end for 100 % of ribs
    (theta_end > theta_start); it is *strictly* monotonic for 78.6 % of ribs
    (vs. 57.0 % for a naive torso-centre angle) and shows a local reversal
    larger than 0.1 rad on 0.6 % of ribs (vs. 7.6 %). It is therefore a robust
    ordering parameter, not a guarantee of strict monotonicity.
    """
    pts = np.asarray(pts, dtype=np.float32)
    dx = np.abs(pts[:, 0] - frame["x_spine"])
    y_ant = frame.get("y_ant", frame["y_lung"] - 0.5 * frame["d_thorax"])
    yc = (y_ant + 0.25 * frame["d_thorax"]) + 0.16 * np.clip(
        frame["z_apex"] - pts[:, 2], 0.0, 220.0
    )
    xc = 8.0
    ang = np.arctan2(-(pts[:, 1] - yc), dx - xc)
    if unwrap and len(ang) > 1:
        ang = np.unwrap(ang)
    return ang.astype(np.float32)


def get_anatomical_rib_mask(
    hu_arr: np.ndarray,
    frame: dict,
    max_bone_dist_mm: float = 3.5,
    suppress_intercostal_valleys: bool = True,
) -> np.ndarray:
    """Returns a 3D boolean mask of valid rib-cage candidate voxels.

    Excludes:
      - Non-bone background (> max_bone_dist_mm from cortical/trabecular bone),
      - Central vertebral body/spinous process corridor (|X - X_spine(Z)| < 10.5 mm),
      - Anterior sternal plate (|X - X_spine(Z)| < 31.0 mm in anterior thorax),
      - Extrathoracic shoulder bones (scapula, lateral clavicle, humerus) where
        d_lung > 26 mm in the upper thorax (Z > Z_apex - 140 mm),
      - Extrathoracic craniocaudal slices outside [Z_apex - 365 mm, Z_apex + 32 mm],
      - Vertical intercostal soft-tissue valleys between adjacent ribs.
    """
    hu = standardize_hu(hu_arr)
    nz, ny, nx = hu.shape
    sx, sy, sz = frame["spacing"]
    ox, oy, oz = frame["origin"]

    if "hu_smooth" in frame:
        hu_smooth = frame["hu_smooth"]
    else:
        hu_smooth = ndi.gaussian_filter(hu, sigma=1.2)

    if "d_bone" in frame:
        d_bone = frame["d_bone"]
    else:
        bone_core = (hu > 115.0) | (hu_smooth > 75.0)
        d_bone = ndi.distance_transform_edt(~bone_core, sampling=(sz, sy, sx))

    mask = d_bone <= max_bone_dist_mm

    # Z bounds relative to lung apex (lower ribs R11/R12 reach down to z_apex - 355 mm)
    z_phys = np.arange(nz, dtype=np.float32) * sz + oz
    valid_z = (z_phys >= frame["z_apex"] - 365.0) & (
        z_phys <= frame["z_apex"] + 32.0
    )
    mask &= valid_z[:, None, None]

    # X, Y grids per slice
    x_phys = np.arange(nx, dtype=np.float32) * sx + ox
    y_phys = np.arange(ny, dtype=np.float32) * sy + oy
    x_sp_z = frame["x_spine_z"][:, None]  # (nz, 1)
    dx_abs = np.abs(x_phys[None, :] - x_sp_z)  # (nz, nx)

    # Exclude central vertebral column (|X - X_spine(Z)| < 10.5 mm posteriorly)
    spine_corridor = (dx_abs[:, None, :] < 10.5) & (
        y_phys[None, :, None] > frame["y_lung"] - 20.0
    )
    mask &= ~spine_corridor

    # Exclude anterior sternal plate (|X - X_spine(Z)| < 31.0 mm anteriorly)
    sternum_corridor = (dx_abs[:, None, :] < 31.0) & (
        y_phys[None, :, None] < frame["y_lung"] - 20.0
    )
    mask &= ~sternum_corridor

    # Exclude extrathoracic upper-body bones (scapula, lateral clavicle) using lung distance
    dist_sub = frame["dist_lung_sub"]
    nz_s, ny_s, nx_s = dist_sub.shape
    iz_s = np.clip(np.arange(nz) // 2, 0, nz_s - 1)
    iy_s = np.clip(np.arange(ny) // 2, 0, ny_s - 1)
    ix_s = np.clip(np.arange(nx) // 2, 0, nx_s - 1)
    d_lung_full = dist_sub[iz_s[:, None, None], iy_s[None, :, None], ix_s[None, None, :]]

    upper_thorax = z_phys[:, None, None] > (frame["z_apex"] - 140.0)
    extrathoracic_shoulder = upper_thorax & (d_lung_full > 26.0)
    mask &= ~extrathoracic_shoulder

    if suppress_intercostal_valleys and nz >= 5:
        # Suppress soft-tissue vertical valleys between adjacent ribs
        hu_z_up = np.roll(hu_smooth, -1, axis=0)
        hu_z_dn = np.roll(hu_smooth, 1, axis=0)
        is_valley = (
            (hu_smooth < np.minimum(hu_z_up, hu_z_dn) - 12.0)
            & (hu < 135.0)
        )
        is_valley[0] = False
        is_valley[-1] = False
        mask &= ~is_valley

    return mask


class RibCageStatisticalAtlas:
    """Statistical 24-rib shape atlas and anatomical coordinate prior.

    Learned strictly from training volumes. Provides:
      1. Canonical 24-rib mean curves and variance profiles,
      2. Slant-corrected posterior-equivalent Z height estimation,
      3. Continuous expected rib index field k_atlas(u, v, w) in [1, 12],
      4. Unary & pairwise transition Gaussian priors for Bilateral Viterbi DP,
      5. Patient-conditioned deformable template completion across all 24 ribs.
    """

    def __init__(self, num_curve_points: int = 80):
        self.num_curve_points = num_curve_points
        self.mean_canon = {}
        self.std_canon = {}
        self.mu_f = {}
        self.sd_f = {}
        self.mu_s = {}
        self.sd_s = {}
        self.slant_pts = None
        self.slant_dz = None
        self.slant_k = None
        self._slant_tree = None
        self._atlas_full_tree = None
        self._atlas_full_pts = None
        self._atlas_full_k = None

    def fit(self, frames_and_gts: list) -> "RibCageStatisticalAtlas":
        """Fits the statistical rib cage atlas on a list of (frame, gt_ribs) tuples."""
        canon_ribs = {f"{s}{k:02d}": [] for s in ["R", "L"] for k in range(1, 13)}
        feats_k = {k: [] for k in range(1, 13)}
        steps_k = {k: [] for k in range(1, 12)}
        all_pts_sym = []
        all_dz_slant = []
        all_k_idx = []
        full_canon_pts = []
        full_canon_k = []

        for frame, gt in frames_and_gts:
            if not gt:
                continue
            for side in ["R", "L"]:
                z_p_map = {}
                for k in range(1, 13):
                    name = f"{side}{k:02d}"
                    if name not in gt or len(gt[name]) < 2:
                        continue
                    pts = gt[name]
                    # Ensure ordered posterior (closer to spine, larger Y) to anterior
                    if (
                        abs(pts[0, 0] - frame["x_spine"]) - 0.5 * pts[0, 1]
                        > abs(pts[-1, 0] - frame["x_spine"]) - 0.5 * pts[-1, 1]
                    ):
                        pts = pts[::-1]
                    c = resample_curve(pts, self.num_curve_points)
                    c_u = phys_to_canonical(c, frame)
                    canon_ribs[name].append(c_u)

                    # Symmetric (|u|, v, w) for slant correction
                    u_sym = np.column_stack([np.abs(c_u[:, 0]), c_u[:, 1], c_u[:, 2]])
                    all_pts_sym.append(u_sym)
                    all_dz_slant.append(c[:, 2] - c[0, 2])
                    all_k_idx.append(np.full(len(c), float(k), dtype=np.float32))

                    full_canon_pts.append(c_u)
                    full_canon_k.append(np.full(len(c), float(k), dtype=np.float32))

                    z_post_rel = float(pts[0, 2] - frame["z_apex"])
                    z_mean_rel = float(np.mean(pts[:, 2]) - frame["z_apex"])
                    w_rel = float(
                        np.max(np.abs(pts[:, 0] - frame["x_spine"])) / frame["w_thorax"]
                    )
                    l_rel = float(curve_length(pts) / frame["w_thorax"])
                    y_min_rel = float(
                        (np.min(pts[:, 1]) - frame["y_spine"]) / frame["d_thorax"]
                    )
                    feats_k[k].append([z_post_rel, z_mean_rel, w_rel, l_rel, y_min_rel])
                    z_p_map[k] = float(pts[0, 2])

                for k in range(1, 12):
                    if k in z_p_map and (k + 1) in z_p_map:
                        steps_k[k].append(z_p_map[k] - z_p_map[k + 1])

        for r, arrs in canon_ribs.items():
            if arrs:
                self.mean_canon[r] = np.mean(arrs, axis=0).astype(np.float32)
                self.std_canon[r] = (np.std(arrs, axis=0) + 1e-3).astype(np.float32)

        for k in range(1, 13):
            self.mu_f[k] = np.mean(feats_k[k], axis=0).astype(np.float32)
            self.sd_f[k] = (np.std(feats_k[k], axis=0) + 1e-3).astype(np.float32)

        for k in range(1, 12):
            self.mu_s[k] = float(np.mean(steps_k[k]))
            self.sd_s[k] = float(np.std(steps_k[k]) + 1.0)

        # Subsample slant lookup cloud to ~15,000 points for fast kNN queries
        pts_sym = np.vstack(all_pts_sym).astype(np.float32)
        dz_arr = np.concatenate(all_dz_slant).astype(np.float32)
        k_arr = np.concatenate(all_k_idx).astype(np.float32)
        if len(pts_sym) > 15000:
            rng = np.random.RandomState(42)
            sel = rng.choice(len(pts_sym), size=15000, replace=False)
            pts_sym = pts_sym[sel]
            dz_arr = dz_arr[sel]
            k_arr = k_arr[sel]

        self.slant_pts = pts_sym
        self.slant_dz = dz_arr
        self.slant_k = k_arr
        self._slant_tree = cKDTree(self.slant_pts)

        # Mean template point cloud for fast atlas distance queries
        mean_list = []
        mean_k_list = []
        for s in ["R", "L"]:
            for k in range(1, 13):
                r = f"{s}{k:02d}"
                if r in self.mean_canon:
                    mean_list.append(self.mean_canon[r])
                    mean_k_list.append(
                        np.full(len(self.mean_canon[r]), float(k), dtype=np.float32)
                    )
        self._atlas_full_pts = np.vstack(mean_list).astype(np.float32)
        self._atlas_full_k = np.concatenate(mean_k_list).astype(np.float32)
        self._atlas_full_tree = cKDTree(self._atlas_full_pts)
        return self

    def _ensure_trees(self):
        if self._slant_tree is None and self.slant_pts is not None:
            self._slant_tree = cKDTree(self.slant_pts)
        if self._atlas_full_tree is None and self._atlas_full_pts is not None:
            self._atlas_full_tree = cKDTree(self._atlas_full_pts)

    def get_z_post_equiv(self, pts: np.ndarray, frame: dict) -> np.ndarray:
        """Computes slant-corrected posterior-equivalent Z height (in mm)."""
        if len(pts) == 0:
            return np.empty((0,), dtype=np.float32)
        self._ensure_trees()
        c_u = phys_to_canonical(pts, frame)
        u_sym = np.column_stack([np.abs(c_u[:, 0]), c_u[:, 1], c_u[:, 2]])
        _, idx = self._slant_tree.query(u_sym, k=16)
        dz_hat = np.mean(self.slant_dz[idx], axis=1)
        return (pts[:, 2] - dz_hat).astype(np.float32)

    def query_atlas_prior(self, pts: np.ndarray, frame: dict) -> tuple:
        """Queries prior rib probability and expected continuous rib index k in [1, 12]."""
        if len(pts) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        self._ensure_trees()
        c_u = phys_to_canonical(pts, frame)
        dists, idx = self._atlas_full_tree.query(c_u, k=5)
        mean_d = np.mean(dists, axis=1)
        prior_prob = np.exp(-0.5 * (mean_d / 0.12) ** 2).astype(np.float32)
        k_hat = np.mean(self._atlas_full_k[idx], axis=1).astype(np.float32)
        return prior_prob, k_hat

    def extract_rib_feature_vector(
        self, pts: np.ndarray, frame: dict, full_span_hint: bool = False
    ) -> tuple:
        """Computes the 5D anatomical feature vector and slant-corrected Z_post."""
        pts = np.asarray(pts, dtype=np.float32)
        ang = get_polar_angle(pts, frame, unwrap=True)
        if ang[-1] < ang[0]:
            pts = pts[::-1]
            ang = ang[::-1]

        zp_eq_all = self.get_z_post_equiv(pts, frame)
        # If the posterior end is close to the spine, pts[0, 2] is directly Z_post;
        # otherwise use the median of the posterior-most quarter of slant-corrected Z.
        dx0 = abs(float(pts[0, 0]) - frame["x_spine"])
        n_q = max(1, len(pts) // 4)
        if dx0 < 48.0 and pts[0, 1] > frame["y_lung"] - 15.0:
            z_post_est = float(np.mean(pts[: min(3, len(pts)), 2]))
        else:
            z_post_est = float(np.median(zp_eq_all[:n_q]))

        z_post_rel = z_post_est - frame["z_apex"]
        z_mean_rel = float(np.mean(pts[:, 2]) - frame["z_apex"])
        w_rel = float(np.max(np.abs(pts[:, 0] - frame["x_spine"])) / frame["w_thorax"])
        l_rel = float(curve_length(pts) / frame["w_thorax"])
        y_min_rel = float((np.min(pts[:, 1]) - frame["y_spine"]) / frame["d_thorax"])
        f_vec = np.array(
            [z_post_rel, z_mean_rel, w_rel, l_rel, y_min_rel], dtype=np.float32
        )
        return f_vec, z_post_est

    def predict_patient_templates(
        self, frame: dict, observed_ribs: dict = None
    ) -> dict:
        """Predicts patient-specific 3D rib curves for all 24 ribs.

        Adapts the canonical 24-rib template to the patient's thoracic frame
        and smoothly interpolates observed 3D deformation offsets across rib
        index k in {1..12}, arc parameter s in [0, 1], and contralateral side.
        """
        base_phys = {
            r: canonical_to_phys(c_u, frame) for r, c_u in self.mean_canon.items()
        }
        if not observed_ribs:
            return base_phys

        # Estimate per-rib 3D displacement vectors delta(side, k, s) at Observed ribs
        deltas = {}
        weights = {}
        for r, obs_pts in observed_ribs.items():
            if r not in base_phys or len(obs_pts) < 5:
                continue
            tmpl = base_phys[r]
            # Project obs_pts onto template arc parameter s in [0, N-1]
            # Match each obs point to nearest template index
            dists = np.linalg.norm(obs_pts[:, None, :] - tmpl[None, :, :], axis=2)
            t_idx = np.argmin(dists, axis=1)
            min_d = np.min(dists, axis=1)
            valid = min_d < 32.0
            if np.sum(valid) < 5:
                continue
            t_v = t_idx[valid]
            obs_v = obs_pts[valid]
            # Compute smooth displacement profile along template indices 0..N-1
            N = len(tmpl)
            disp = np.zeros((N, 3), dtype=np.float32)
            conf = np.zeros(N, dtype=np.float32)
            for idx_s in range(N):
                w_s = np.exp(-0.5 * ((t_v - idx_s) / 6.0) ** 2)
                w_sum = float(np.sum(w_s))
                if w_sum > 0.1:
                    disp[idx_s] = np.sum(
                        w_s[:, None] * (obs_v - tmpl[t_v]), axis=0
                    ) / w_sum
                    conf[idx_s] = min(1.0, w_sum / 2.0)
            deltas[r] = disp
            weights[r] = conf

        if not deltas:
            return base_phys

        adapted = {}
        for side in ["R", "L"]:
            opp = "L" if side == "R" else "R"
            for k in range(1, 13):
                r = f"{side}{k:02d}"
                tmpl = base_phys[r].copy()
                N = len(tmpl)
                num_disp = np.zeros((N, 3), dtype=np.float32)
                den_w = np.full((N, 1), 0.15, dtype=np.float32)  # Prior shrinkage toward 0

                # Interpolate across same-side ribs kp = 1..12
                for kp in range(1, 13):
                    rp = f"{side}{kp:02d}"
                    if rp in deltas:
                        w_k = np.exp(-0.5 * ((k - kp) / 1.75) ** 2)
                        w_vec = (w_k * weights[rp])[:, None]
                        num_disp += w_vec * deltas[rp]
                        den_w += w_vec

                    # Contralateral rib coupling (mirror X displacement, preserve Y and Z displacement!)
                    rp_opp = f"{opp}{kp:02d}"
                    if rp_opp in deltas:
                        w_k_opp = 0.45 * np.exp(-0.5 * ((k - kp) / 1.5) ** 2)
                        w_vec = (w_k_opp * weights[rp_opp])[:, None]
                        mirrored = deltas[rp_opp].copy()
                        mirrored[:, 0] = -mirrored[:, 0]
                        num_disp += w_vec * mirrored
                        den_w += w_vec

                adapted[r] = tmpl + (num_disp / den_w)
        return adapted

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "num_curve_points": self.num_curve_points,
            "mean_canon": self.mean_canon,
            "std_canon": self.std_canon,
            "mu_f": self.mu_f,
            "sd_f": self.sd_f,
            "mu_s": self.mu_s,
            "sd_s": self.sd_s,
            "slant_pts": self.slant_pts,
            "slant_dz": self.slant_dz,
            "slant_k": self.slant_k,
            "_atlas_full_pts": self._atlas_full_pts,
            "_atlas_full_k": self._atlas_full_k,
        }
        joblib.dump(state, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "RibCageStatisticalAtlas":
        state = joblib.load(path)
        obj = cls(num_curve_points=state.get("num_curve_points", 80))
        for k, v in state.items():
            setattr(obj, k, v)
        obj._ensure_trees()
        return obj
