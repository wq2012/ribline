"""Patient-conditioned deformable template completion and sub-voxel medial refinement."""

import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import splev, splprep

from ribline.model.anatomy import (
    RibCageStatisticalAtlas,
    curve_length,
    get_polar_angle,
    resample_curve,
    standardize_hu,
)


class RibSnake:
    """Deformable active contour and template-guided refiner for 3D rib centerlines.

    Performs three anatomically constrained operations:
      1. Apical First-Rib (`R01`/`L01`) sternoclavicular/clavicular branch trimming
         using the canonical thoracic width invariant (W_1 / W_thorax <= 0.55).
      2. Patient-conditioned deformable template recovery of low-contrast apical
         (`R01`/`L01`) and thoracolumbar floating (`R12`/`L12`) ribs within the
         template's tubular Voronoi basin when supported by cortical bone and
         voxel probability evidence.
      3. Sub-voxel cross-sectional normal-plane centroid refinement and uniform
         arc-length cubic B-spline smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        beta: float = 0.45,
        sigma: float = 1.2,
        iterations: int = 5,
        spacing: tuple = (2.0, 2.0, 2.0),
        max_shift_mm: float = 1.6,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.iterations = int(iterations)
        self.spacing = np.asarray(spacing, dtype=np.float32)
        self.max_shift_mm = float(max_shift_mm)

    def refine(
        self,
        labeled_ribs: dict,
        image_np: np.ndarray,
        origin: tuple = (0.0, 0.0, 0.0),
        prob_np: np.ndarray = None,
        frame: dict = None,
        atlas: RibCageStatisticalAtlas = None,
    ) -> dict:
        """Refines and completes a dictionary of labeled 3D rib centerlines."""
        if not labeled_ribs:
            return {}

        hu = standardize_hu(image_np)
        if frame is not None and "hu_smooth" in frame:
            hu_smooth = frame["hu_smooth"]
        else:
            hu_smooth = ndimage.gaussian_filter(hu, sigma=1.0)

        work_ribs = {
            k: np.asarray(v, dtype=np.float32).copy()
            for k, v in labeled_ribs.items()
            if len(v) >= 2
        }

        # 1. Trim lateral clavicular branches on R01 / L01 using thoracic width invariant
        if frame is not None:
            for r_first in ("R01", "L01"):
                if r_first in work_ribs:
                    work_ribs[r_first] = self._trim_apical_clavicle_branch(
                        work_ribs[r_first], frame
                    )

        # 2. Patient-conditioned deformable template completion for missing apical/floating ribs
        if frame is not None and atlas is not None and prob_np is not None:
            work_ribs = self._recover_missing_extremity_ribs(
                work_ribs, hu_smooth, prob_np, frame, atlas, origin, hu_raw=hu
            )

        # 3. Cross-sectional normal-plane sub-voxel medial ridge refinement
        refined = {}
        for k, points in sorted(work_ribs.items()):
            if len(points) < 2:
                refined[k] = points
                continue
            refined[k] = self._refine_rib(
                points, hu_smooth, origin, prob_np=prob_np
            )

        return refined

    def _trim_apical_clavicle_branch(
        self, pts: np.ndarray, frame: dict
    ) -> np.ndarray:
        """Trims lateral clavicle extensions from R01/L01 beyond 0.55 * w_thorax."""
        if len(pts) < 6 or curve_length(pts) <= 135.0:
            return pts
        ang = get_polar_angle(pts, frame, unwrap=True)
        if ang[-1] < ang[0]:
            pts = pts[::-1].copy()
        dx = np.abs(pts[:, 0] - frame["x_spine"])
        max_allowed_dx = 0.55 * frame["w_thorax"]

        # Keep contiguous rib points before the curve escapes laterally into the clavicle
        valid_mask = dx <= max_allowed_dx
        if not np.all(valid_mask) and np.sum(valid_mask) >= 8:
            # Find first excursion beyond max_allowed_dx after the posterior neck
            bad_indices = np.where(~valid_mask)[0]
            # If the excursion is at the anterior/lateral end, truncate there
            if bad_indices[0] >= 6:
                trimmed = pts[: bad_indices[0]].copy()
                if curve_length(trimmed) >= 45.0:
                    pts = trimmed
            elif bad_indices[-1] < len(pts) - 6:
                trimmed = pts[bad_indices[-1] + 1 :].copy()
                if curve_length(trimmed) >= 45.0:
                    pts = trimmed

        # Also cap total arc length of R01/L01 at 140 mm from posterior start
        if curve_length(pts) > 140.0:
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(diffs)])
            keep_n = int(np.searchsorted(cum, 135.0))
            if keep_n >= 8:
                pts = pts[:keep_n].copy()
        return pts

    def _recover_missing_extremity_ribs(
        self,
        work_ribs: dict,
        hu_smooth: np.ndarray,
        prob_np: np.ndarray,
        frame: dict,
        atlas: RibCageStatisticalAtlas,
        origin: tuple,
        hu_raw: np.ndarray = None,
    ) -> dict:
        """Recovers missing R01/L01 or R12/L12 using patient-conditioned templates."""
        templates = atlas.predict_patient_templates(
            frame, observed_ribs=work_ribs
        )
        nz, ny, nx = hu_smooth.shape
        sx, sy, sz = self.spacing
        ox, oy, oz = origin
        z_min_fov = oz + 6.0
        z_max_fov = oz + (nz - 1) * sz - 4.0

        # First check bilateral floating 12th rib recovery below R11/L11
        hu_max3 = None
        if "R12" not in work_ribs or "L12" not in work_ribs:
            hu_src = hu_raw if hu_raw is not None else hu_smooth
            hu_max3 = ndimage.maximum_filter(hu_src, size=3)

            cand_12 = {}
            for side in ("R", "L"):
                r_key = f"{side}12"
                same_11 = f"{side}11"
                if r_key in work_ribs or same_11 not in work_ribs:
                    continue
                p11 = work_ribs[same_11]
                z11_0, y11_0, x11_0 = (
                    float(p11[0, 2]),
                    float(p11[0, 1]),
                    float(p11[0, 0]),
                )
                if z11_0 < z_min_fov + 28.0:
                    continue
                z_low = max(z_min_fov, z11_0 - 44.0)
                z_high = z11_0 - 22.0
                if z_high <= z_low + 4.0:
                    continue
                y_low, y_high = y11_0 - 12.0, y11_0 + 18.0
                if side == "R":
                    x_low = frame["x_spine"] - 56.0
                    x_high = min(x11_0 - 2.0, frame["x_spine"] - 24.0)
                else:
                    x_low = max(x11_0 + 2.0, frame["x_spine"] + 24.0)
                    x_high = frame["x_spine"] + 56.0
                if x_high <= x_low + 8.0:
                    continue

                iz0 = max(0, int(round((z_low - oz) / sz)))
                iz1 = min(nz - 1, int(round((z_high - oz) / sz)))
                iy0 = max(0, int(round((y_low - oy) / sy)))
                iy1 = min(ny - 1, int(round((y_high - oy) / sy)))
                ix0 = max(0, int(round((x_low - ox) / sx)))
                ix1 = min(nx - 1, int(round((x_high - ox) / sx)))

                sub_p = prob_np[iz0 : iz1 + 1, iy0 : iy1 + 1, ix0 : ix1 + 1]
                sub_hu = hu_max3[iz0 : iz1 + 1, iy0 : iy1 + 1, ix0 : ix1 + 1]
                mask = (sub_p >= 0.30) & (sub_hu >= 150.0)
                zz, yy, xx = np.where(mask)
                if len(xx) > 0:
                    pts_cand = np.column_stack(
                        [
                            (ix0 + xx) * sx + ox,
                            (iy0 + yy) * sy + oy,
                            (iz0 + zz) * sz + oz,
                        ]
                    )
                    d_to_11 = np.min(
                        np.linalg.norm(
                            pts_cand[:, None, :] - p11[None, :, :], axis=2
                        ),
                        axis=1,
                    )
                    keep_far = d_to_11 >= 18.0
                    zz, yy, xx = zz[keep_far], yy[keep_far], xx[keep_far]

                max_p = float(sub_p[zz, yy, xx].max()) if len(xx) > 0 else 0.0
                if len(xx) >= 12 and max_p >= 0.45:
                    xs_unique = sorted(np.unique(xx), reverse=(side == "R"))
                    if len(xs_unique) >= 6:
                        pts_12 = []
                        for x_u in xs_unique:
                            sel = xx == x_u
                            w = sub_p[zz[sel], yy[sel], xx[sel]]
                            px = (ix0 + float(x_u)) * sx + ox
                            py = (
                                iy0 + float(np.sum(yy[sel] * w) / np.sum(w))
                            ) * sy + oy
                            pz = (
                                iz0 + float(np.sum(zz[sel] * w) / np.sum(w))
                            ) * sz + oz
                            pts_12.append([px, py, pz])
                        rec_arr = ndimage.gaussian_filter1d(
                            np.asarray(pts_12, dtype=np.float32),
                            sigma=1.2,
                            axis=0,
                            mode="nearest",
                        )
                        if curve_length(rec_arr) >= 18.0:
                            cand_12[r_key] = rec_arr

            # Accept recovered 12th ribs if supported bilaterally (either contralateral 12th
            # rib was already extracted or both R12 and L12 corridors have valid ribs)
            for side in ("R", "L"):
                r_key = f"{side}12"
                opp_12 = f"{'L' if side == 'R' else 'R'}12"
                if r_key in cand_12 and (
                    opp_12 in work_ribs or opp_12 in cand_12
                ):
                    work_ribs[r_key] = cand_12[r_key]

        for side in ("R", "L"):
            opp_side = "L" if side == "R" else "R"
            # Check R01 / L01 and R12 / L12
            for k_num in (1, 12):
                r_key = f"{side}{k_num:02d}"
                if r_key in work_ribs:
                    continue
                if r_key not in templates:
                    continue

                if k_num == 12:
                    opp_12 = f"{opp_side}12"
                    same_11 = f"{side}11"
                    if opp_12 not in work_ribs:
                        if same_11 not in work_ribs:
                            continue
                        if float(np.min(work_ribs[same_11][:, 2])) < z_min_fov + 28.0:
                            continue

                tmpl = templates[r_key].copy()
                # If contralateral rib is observed, anchor template Z height to contralateral rib
                opp_key = f"{opp_side}{k_num:02d}"
                if opp_key in work_ribs:
                    dz_anchor = float(
                        work_ribs[opp_key][0, 2] - tmpl[0, 2]
                    )
                    if abs(dz_anchor) < 18.0:
                        tmpl[:, 2] += 0.75 * dz_anchor

                # Ensure template lies within scan FOV
                in_fov = (tmpl[:, 2] >= z_min_fov) & (tmpl[:, 2] <= z_max_fov)
                if np.sum(in_fov) < 12:
                    continue
                tmpl = tmpl[in_fov]

                # Avoid overlapping already-labeled adjacent rib (Rib 2 or Rib 11)
                adj_key = f"{side}{2 if k_num == 1 else 11:02d}"
                if adj_key in work_ribs:
                    adj_pts = work_ribs[adj_key]
                    d_to_adj = np.min(
                        np.linalg.norm(
                            tmpl[:, None, :] - adj_pts[None, :, :], axis=2
                        ),
                        axis=1,
                    )
                    if float(np.mean(d_to_adj)) < 11.0:
                        continue

                # Search tubular neighborhood around template for bone + probability ridge
                recovered_pts = []
                for pt in tmpl:
                    # Local 3D search box of radius 8 mm (4 voxels)
                    vx = (pt[0] - ox) / sx
                    vy = (pt[1] - oy) / sy
                    vz = (pt[2] - oz) / sz
                    ix0 = max(0, int(round(vx)) - 3)
                    ix1 = min(nx - 1, int(round(vx)) + 3)
                    iy0 = max(0, int(round(vy)) - 3)
                    iy1 = min(ny - 1, int(round(vy)) + 3)
                    iz0 = max(0, int(round(vz)) - 3)
                    iz1 = min(nz - 1, int(round(vz)) + 3)

                    sub_p = prob_np[iz0 : iz1 + 1, iy0 : iy1 + 1, ix0 : ix1 + 1]
                    sub_hu = hu_smooth[iz0 : iz1 + 1, iy0 : iy1 + 1, ix0 : ix1 + 1]
                    score = sub_p * np.clip((sub_hu - 60.0) / 300.0, 0.0, 1.5)
                    max_s = float(np.max(score)) if score.size > 0 else 0.0
                    if max_s >= 0.14:
                        mask_s = score >= 0.65 * max_s
                        zz, yy, xx = np.where(mask_s)
                        w = score[zz, yy, xx]
                        cx = (ix0 + float(np.sum(xx * w) / np.sum(w))) * sx + ox
                        cy = (iy0 + float(np.sum(yy * w) / np.sum(w))) * sy + oy
                        cz = (iz0 + float(np.sum(zz * w) / np.sum(w))) * sz + oz
                        recovered_pts.append([cx, cy, cz])

                if len(recovered_pts) >= 8:
                    rec_arr = np.asarray(recovered_pts, dtype=np.float32)
                    rec_arr = ndimage.gaussian_filter1d(
                        rec_arr, sigma=1.5, axis=0, mode="nearest"
                    )
                    if curve_length(rec_arr) >= 20.0:
                        work_ribs[r_key] = rec_arr

        return work_ribs


    def _refine_rib(
        self,
        points: np.ndarray,
        hu_smooth: np.ndarray,
        origin: tuple,
        prob_np: np.ndarray = None,
    ) -> np.ndarray:
        """Centers a 3D rib polyline along the cross-sectional medial bone ridge."""
        pts = np.asarray(points, dtype=np.float32).copy()
        c_len = curve_length(pts)
        if c_len < 6.0 or len(pts) < 4:
            return pts

        n_target = max(15, int(round(c_len / 2.0)))
        pts = resample_curve(pts, n_target)
        init_pts = pts.copy()

        sx, sy, sz = self.spacing
        ox, oy, oz = origin
        nz, ny, nx = hu_smooth.shape

        # Precompute normalized medialness field on-the-fly via local normal-plane disks
        # Disk offsets in normal plane: 5x5 grid in [-2.5, +2.5] mm
        grid_1d = np.linspace(-2.5, 2.5, 5, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(grid_1d, grid_1d)
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()
        disk_mask = (u_flat**2 + v_flat**2) <= 6.5
        u_flat = u_flat[disk_mask]
        v_flat = v_flat[disk_mask]

        for _ in range(self.iterations):
            # Compute unit tangents along the curve
            tangents = np.gradient(pts, axis=0)
            t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            tangents = tangents / np.maximum(t_norms, 1e-6)

            # Construct two orthogonal unit normals n1, n2 per point
            ref_vec = np.zeros_like(tangents)
            ref_vec[:, 2] = 1.0
            # Where tangent is nearly vertical, use X axis as reference
            vert_mask = np.abs(tangents[:, 2]) > 0.9
            ref_vec[vert_mask, 2] = 0.0
            ref_vec[vert_mask, 0] = 1.0

            n1 = np.cross(tangents, ref_vec)
            n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-6)
            n2 = np.cross(tangents, n1)
            n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-6)

            # Sample 2D normal-plane disk around each point: (N, K, 3)
            offsets = (
                u_flat[None, :, None] * n1[:, None, :]
                + v_flat[None, :, None] * n2[:, None, :]
            )
            sample_phys = pts[:, None, :] + offsets  # (N, K, 3)

            vx = np.clip((sample_phys[:, :, 0] - ox) / sx, 0.0, nx - 1.0)
            vy = np.clip((sample_phys[:, :, 1] - oy) / sy, 0.0, ny - 1.0)
            vz = np.clip((sample_phys[:, :, 2] - oz) / sz, 0.0, nz - 1.0)

            coords = np.array([vz.ravel(), vy.ravel(), vx.ravel()])
            hu_vals = ndimage.map_coordinates(
                hu_smooth, coords, order=1, mode="nearest"
            ).reshape(pts.shape[0], -1)
            weights = np.clip((hu_vals - 40.0) / 350.0, 0.0, 2.0)

            if prob_np is not None:
                p_vals = ndimage.map_coordinates(
                    prob_np, coords, order=1, mode="nearest"
                ).reshape(pts.shape[0], -1)
                weights = weights * (0.25 + 0.75 * np.clip(p_vals, 0.0, 1.0))

            w_sum = np.sum(weights, axis=1, keepdims=True)
            valid_w = (w_sum[:, 0] > 1e-3)
            delta_normal = np.zeros_like(pts)
            if np.any(valid_w):
                delta_normal[valid_w] = (
                    np.sum(
                        weights[valid_w, :, None] * offsets[valid_w], axis=1
                    )
                    / w_sum[valid_w]
                )

            # Update positions with normal-plane centroid force + tangential smoothing
            pts = pts + self.alpha * delta_normal
            smoothed = ndimage.gaussian_filter1d(
                pts, sigma=self.sigma, axis=0, mode="nearest"
            )
            pts = (1.0 - self.beta) * pts + self.beta * smoothed

            # Clamp total displacement from initial skeleton to max_shift_mm
            disp = pts - init_pts
            disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
            exceed = disp_norm > self.max_shift_mm
            if np.any(exceed):
                pts = np.where(
                    exceed,
                    init_pts + disp * (self.max_shift_mm / np.maximum(disp_norm, 1e-6)),
                    pts,
                )

        # Final smooth cubic B-spline parameterization
        try:
            if len(pts) >= 6:
                tck, _ = splprep(
                    [pts[:, 0], pts[:, 1], pts[:, 2]], s=len(pts) * 0.15, k=3
                )
                u_new = np.linspace(0.0, 1.0, n_target)
                pts = np.vstack(splev(u_new, tck)).T.astype(np.float32)
        except Exception:
            pass

        return pts


DeformableRibRefiner = RibSnake
