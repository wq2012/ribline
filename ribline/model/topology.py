"""Bilateral Viterbi dynamic programming and deformable template rib labeling."""

import os
import numpy as np

from ribline.model.anatomy import (
    RibCageStatisticalAtlas,
    curve_length,
    get_polar_angle,
    phys_to_canonical,
    resample_curve,
)


class RibTopologyMatcher:
    """Assigns anatomical rib labels (R01..R12, L01..R12) using Bilateral Viterbi DP.

    Combines:
      1. Canonical thoracic coordinates anchored at the 3D lung apex & spine midline,
      2. Patient-specific craniocaudal rib-spacing scale estimation (s_Z),
      3. Slant-corrected posterior-equivalent height estimation,
      4. 5D anatomical feature likelihoods + canonical 3D curve distance,
      5. Inter-rib vertebral spacing transition potentials,
      6. Contralateral Left-Right vertebral symmetry coupling.
    """

    def __init__(self, atlas: RibCageStatisticalAtlas = None, atlas_path: str = None):
        if atlas is not None:
            self.atlas = atlas
        else:
            if atlas_path is None:
                default_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "configs",
                    "rib_atlas.joblib",
                )
                if os.path.exists(default_path):
                    atlas_path = default_path
            if atlas_path and os.path.exists(atlas_path):
                self.atlas = RibCageStatisticalAtlas.load(atlas_path)
            else:
                self.atlas = None

    def _estimate_fallback_frame(self, centerlines: list) -> dict:
        """Estimates a thoracic coordinate frame from unlabeled curves if no CT is given."""
        all_pts = np.vstack([c for c in centerlines if len(c) > 0])
        x_mid = 0.5 * (
            float(np.percentile(all_pts[:, 0], 5.0))
            + float(np.percentile(all_pts[:, 0], 95.0))
        )
        y_ant = float(np.percentile(all_pts[:, 1], 5.0))
        y_post = float(np.percentile(all_pts[:, 1], 95.0))
        y_lung = 0.5 * (y_ant + y_post)
        w_thorax = max(
            90.0, float(np.percentile(np.abs(all_pts[:, 0] - x_mid), 96.0))
        )
        z_max = float(np.percentile(all_pts[:, 2], 99.0))
        z_min = float(np.percentile(all_pts[:, 2], 1.0))
        return {
            "x_spine": x_mid,
            "y_spine": y_post,
            "y_lung": y_lung,
            "z_apex": z_max - 12.0,
            "z_apex_r": z_max - 12.0,
            "z_apex_l": z_max - 12.0,
            "z_base": z_min,
            "w_thorax": w_thorax,
            "y_ant": y_ant,
            "y_post": y_post,
            "d_thorax": max(100.0, y_post - y_ant),
            "spacing": (2.0, 2.0, 2.0),
            "origin": (0.0, 0.0, 0.0),
        }

    def _refine_frame_from_curves(self, valid_lines: list, frame: dict) -> dict:
        """Refines x_spine and y_spine using the bilateral symmetry of extracted ribs."""
        frame = dict(frame)
        long_lines = [c for c in valid_lines if curve_length(c) >= 45.0]
        if len(long_lines) < 4:
            return frame

        all_pts = np.vstack(long_lines)
        x_sym = 0.5 * (
            float(np.percentile(all_pts[:, 0], 3.0))
            + float(np.percentile(all_pts[:, 0], 97.0))
        )
        if abs(x_sym - frame["x_spine"]) > 12.0:
            frame["x_spine"] = 0.35 * frame["x_spine"] + 0.65 * x_sym

        # Also check posterior endpoints of R and L curves
        r_posts, l_posts, y_posts = [], [], []
        for c in long_lines:
            side = "R" if float(np.mean(c[:, 0])) < frame["x_spine"] else "L"
            # Posterior endpoint is the endpoint with larger Y (and closer to x_spine)
            s0 = float(c[0, 1]) - 0.5 * abs(float(c[0, 0]) - frame["x_spine"])
            s1 = float(c[-1, 1]) - 0.5 * abs(float(c[-1, 0]) - frame["x_spine"])
            p_post = c[0] if s0 >= s1 else c[-1]
            if abs(float(p_post[0]) - frame["x_spine"]) < 55.0:
                if side == "R":
                    r_posts.append(float(p_post[0]))
                else:
                    l_posts.append(float(p_post[0]))
                y_posts.append(float(p_post[1]))

        if len(r_posts) >= 3 and len(l_posts) >= 3:
            frame["x_spine"] = 0.5 * (
                float(np.median(r_posts)) + float(np.median(l_posts))
            )
        if len(y_posts) >= 6:
            frame["y_spine"] = float(np.percentile(y_posts, 75.0))

        w_est = float(np.percentile(np.abs(all_pts[:, 0] - frame["x_spine"]), 97.0))
        if w_est > 85.0:
            frame["w_thorax"] = 0.5 * frame["w_thorax"] + 0.5 * w_est
        return frame

    def match(
        self,
        centerlines: list,
        z_max: float = None,
        frame: dict = None,
    ) -> dict:
        """Matches unlabeled 3D rib curves to anatomical labels L01..L12, R01..R12."""
        valid_lines = [
            np.asarray(c, dtype=np.float32) for c in centerlines if len(c) >= 2
        ]
        if not valid_lines:
            return {}

        if frame is None:
            frame = self._estimate_fallback_frame(valid_lines)
        frame = self._refine_frame_from_curves(valid_lines, frame)

        if self.atlas is None:
            return self._heuristic_fallback_match(valid_lines, frame)

        # 1. Separate into Right (X < x_spine) and Left (X >= x_spine) and orient posterior->anterior
        hemi_raw = {"R": [], "L": []}
        for pts in valid_lines:
            if curve_length(pts) < 22.0:
                continue
            ang = get_polar_angle(pts, frame, unwrap=True)
            if ang[-1] < ang[0]:
                pts = pts[::-1].copy()
            side = "R" if float(np.mean(pts[:, 0])) < frame["x_spine"] else "L"
            hemi_raw[side].append(pts)

        # 2. Consolidate disjoint azimuthal fragments and remove overlapping duplicates
        hemi_cand = {}
        for side in ("R", "L"):
            hemi_cand[side] = self._consolidate_hemithorax_candidates(
                hemi_raw[side], frame, side
            )

        # 3. Estimate patient-specific vertical rib-cage scale s_Z from inter-rib steps
        s_z = self._estimate_patient_vertical_scale(hemi_cand["R"], hemi_cand["L"])

        # 4. Compute normalized unary costs U(i, k) for each candidate i and rib label k in 1..12
        unaries = {"R": [], "L": []}
        for side in ("R", "L"):
            for cand in hemi_cand[side]:
                u_cost = self._compute_unary_costs(cand, frame, side, s_z=s_z)
                unaries[side].append(u_cost)

        # 5. Contralateral Left-Right vertebral symmetry coupling:
        cands_r, cands_l = hemi_cand["R"], hemi_cand["L"]
        for ir, cr in enumerate(cands_r):
            for il, cl in enumerate(cands_l):
                dz_lr = abs(cr["z_sort"] - cl["z_sort"])
                if dz_lr < 7.5:
                    w_couple = 0.45 * np.exp(-0.5 * (dz_lr / 4.0) ** 2)
                    ur_old = unaries["R"][ir].copy()
                    ul_old = unaries["L"][il].copy()
                    unaries["R"][ir] = (1.0 - w_couple) * ur_old + w_couple * ul_old
                    unaries["L"][il] = (1.0 - w_couple) * ul_old + w_couple * ur_old

        # 6. Solve Outlier-Tolerant Viterbi DP per hemithorax, then refine with bilateral alignment
        labeled_r = self._solve_viterbi_hemithorax(cands_r, unaries["R"], "R", s_z=s_z)
        labeled_l = self._solve_viterbi_hemithorax(cands_l, unaries["L"], "L", s_z=s_z)

        if labeled_r and labeled_l:
            z_map_r = {int(k[1:]): v["z_sort"] for k, v in labeled_r.items()}
            z_map_l = {int(k[1:]): v["z_sort"] for k, v in labeled_l.items()}

            unaries_r2 = []
            for ir, cr in enumerate(cands_r):
                u = unaries["R"][ir].copy()
                for k in range(1, 13):
                    if k in z_map_l:
                        dz = abs(cr["z_sort"] - z_map_l[k])
                        if dz < 8.5:
                            u[k] -= 3.0 * np.exp(-0.5 * (dz / 4.0) ** 2)
                unaries_r2.append(u)

            unaries_l2 = []
            for il, cl in enumerate(cands_l):
                u = unaries["L"][il].copy()
                for k in range(1, 13):
                    if k in z_map_r:
                        dz = abs(cl["z_sort"] - z_map_r[k])
                        if dz < 8.5:
                            u[k] -= 3.0 * np.exp(-0.5 * (dz / 4.0) ** 2)
                unaries_l2.append(u)

            labeled_r = self._solve_viterbi_hemithorax(
                cands_r, unaries_r2, "R", s_z=s_z
            )
            labeled_l = self._solve_viterbi_hemithorax(
                cands_l, unaries_l2, "L", s_z=s_z
            )

        result = {}
        for k, v in labeled_r.items():
            result[k] = v["pts"]
        for k, v in labeled_l.items():
            result[k] = v["pts"]
        return result

    def match_zsort(self, centerlines: list, frame: dict = None) -> dict:
        """Naive craniocaudal-counting baseline ("Z-sort").

        Uses exactly the same hemithorax split and fragment consolidation as
        :meth:`match`, but replaces the Viterbi dynamic program by sorting the
        consolidated candidates of each side superior-to-inferior by their
        slant-corrected posterior height ``z_sort`` and labelling the first 12 as
        ribs 1..12. Any additional candidates are discarded. This is the
        "+ Z-Sort" baseline reported in the paper's ablation table.
        """
        valid_lines = [
            np.asarray(c, dtype=np.float32) for c in centerlines if len(c) >= 2
        ]
        if not valid_lines:
            return {}
        if frame is None:
            frame = self._estimate_fallback_frame(valid_lines)
        frame = self._refine_frame_from_curves(valid_lines, frame)
        if self.atlas is None:
            return self._heuristic_fallback_match(valid_lines, frame)
        hemi_raw = {"R": [], "L": []}
        for pts in valid_lines:
            if curve_length(pts) < 22.0:
                continue
            ang = get_polar_angle(pts, frame, unwrap=True)
            if ang[-1] < ang[0]:
                pts = pts[::-1].copy()
            side = "R" if float(np.mean(pts[:, 0])) < frame["x_spine"] else "L"
            hemi_raw[side].append(pts)
        out = {}
        for side in ("R", "L"):
            cands = self._consolidate_hemithorax_candidates(hemi_raw[side], frame, side)
            cands = sorted(cands, key=lambda c: c["z_sort"], reverse=True)
            for idx, c in enumerate(cands[:12]):
                out[f"{side}{idx + 1:02d}"] = c["pts"]
        return out

    def _estimate_patient_vertical_scale(
        self, cands_r: list, cands_l: list
    ) -> float:
        """Estimates patient-specific craniocaudal rib-spacing scale s_Z in [0.80, 1.25]."""
        steps = []
        span_estimates = []
        for cands in (cands_r, cands_l):
            long_c = [c for c in cands if c["len"] >= 50.0]
            for i in range(len(long_c) - 1):
                dz = long_c[i]["z_sort"] - long_c[i + 1]["z_sort"]
                if 9.5 <= dz <= 40.0:
                    steps.append(dz)
            if 9 <= len(long_c) <= 13:
                n_ribs = min(12, len(long_c))
                exp_span = float(self.atlas.mu_f[1][0] - self.atlas.mu_f[n_ribs][0])
                obs_span = float(long_c[0]["z_sort"] - long_c[n_ribs - 1]["z_sort"])
                if exp_span > 100.0 and obs_span > 100.0:
                    span_estimates.append(obs_span / exp_span)

        s_step = float(np.median(steps)) / 21.8 if len(steps) >= 4 else 1.0
        if span_estimates:
            s_span = float(np.mean(span_estimates))
            s_z = 0.45 * s_step + 0.55 * s_span
        else:
            s_z = s_step
        return float(np.clip(s_z, 0.80, 1.25))

    def _consolidate_hemithorax_candidates(
        self, raw_list: list, frame: dict, side: str
    ) -> list:
        """Merges disjoint same-rib fragments and removes genuinely overlapping duplicates."""
        if not raw_list:
            return []

        items = []
        for pts in raw_list:
            ang = get_polar_angle(pts, frame, unwrap=True)
            if ang[-1] < ang[0]:
                pts = pts[::-1].copy()
                ang = ang[::-1]
            f_vec, z_sort = self.atlas.extract_rib_feature_vector(pts, frame)
            items.append(
                {
                    "pts": pts,
                    "f_vec": f_vec,
                    "z_sort": z_sort,
                    "ang_min": float(ang[0]),
                    "ang_max": float(ang[-1]),
                    "len": curve_length(pts),
                }
            )

        # Pre-filter upward-sloping anterior/clavicular fragments before merging so they
        # cannot bridge onto legitimate apical ribs (e.g. L01/L02).
        pre_filtered = []
        for it in items:
            pts = it["pts"]
            dx0 = abs(float(pts[0, 0]) - frame["x_spine"])
            dz_end = float(pts[-1, 2] - pts[0, 2])
            if it["len"] < 120.0 and ((dx0 > 30.0 and dz_end > 4.0) or dz_end > 10.0):
                continue
            pre_filtered.append(it)
        if len(pre_filtered) >= 6:
            items = pre_filtered

        # Merge disjoint-angle fragments whose z_sort is within 11.5 mm
        changed = True
        while changed and len(items) > 1:
            changed = False
            best_pair = None
            best_dz = 11.5
            for i in range(len(items)):
                for j in range(len(items)):
                    if i == j:
                        continue
                    a, b = items[i], items[j]
                    if b["ang_min"] < a["ang_max"] - 0.12:
                        continue
                    # Do not jump upward in Z across a fragment gap or attach distant spurs to a complete rib
                    if float(b["pts"][0, 2] - a["pts"][-1, 2]) > 6.0:
                        continue
                    gap = float(np.linalg.norm(b["pts"][0] - a["pts"][-1]))
                    if a["len"] > 240.0 and (b["len"] < 45.0 or gap > 25.0):
                        continue
                    dz = abs(a["z_sort"] - b["z_sort"])
                    if dz < best_dz and gap < 75.0:
                        best_dz = dz
                        best_pair = (i, j, gap)
            if best_pair is not None:
                i, j, gap = best_pair
                a, b = items[i], items[j]
                n_br = max(2, int(round(gap / 2.0)))
                bridge = np.column_stack(
                    [
                        np.linspace(a["pts"][-1, d], b["pts"][0, d], n_br + 2)[1:-1]
                        for d in range(3)
                    ]
                ).astype(np.float32)
                merged_pts = np.vstack([a["pts"], bridge, b["pts"]])
                ang = get_polar_angle(merged_pts, frame, unwrap=True)
                f_vec, z_sort = self.atlas.extract_rib_feature_vector(merged_pts, frame)
                new_item = {
                    "pts": merged_pts,
                    "f_vec": f_vec,
                    "z_sort": z_sort,
                    "ang_min": float(ang[0]),
                    "ang_max": float(ang[-1]),
                    "len": curve_length(merged_pts),
                }
                items = [
                    it for idx, it in enumerate(items) if idx not in (i, j)
                ] + [new_item]
                changed = True

        # Filter out short non-posteriorly-rooted clutter fragments:
        # 1) Anterior upward-sloping clavicle/sternoclavicular fragments (z_end > z_start + 3 mm and dx_start > 30 mm)
        # 2) Short isolated lateral/anterior fragments that start far from the spine (len < 85 mm and dx_start > 66 mm)
        rooted_items = []
        for it in items:
            pts = it["pts"]
            dx0 = abs(float(pts[0, 0]) - frame["x_spine"])
            dz_end = float(pts[-1, 2] - pts[0, 2])
            if dx0 > 30.0 and dz_end > 3.0 and it["len"] < 140.0:
                continue
            if it["len"] < 85.0 and (dx0 > 66.0 or it["ang_min"] > 0.25):
                continue
            rooted_items.append(it)
        if len(rooted_items) >= 6:
            items = rooted_items

        # Sort superior to inferior by z_sort
        items.sort(key=lambda x: x["z_sort"], reverse=True)

        # Suppress genuinely overlapping duplicate spurs (3D distance < 6.5 mm or short fragment at same level)
        keep = [True] * len(items)
        for i in range(len(items)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(items)):
                if not keep[j]:
                    continue
                if abs(items[i]["z_sort"] - items[j]["z_sort"]) > 8.5:
                    continue
                if items[i]["len"] >= items[j]["len"]:
                    long_it, short_it, short_idx = items[i], items[j], j
                else:
                    long_it, short_it, short_idx = items[j], items[i], i
                p_short = resample_curve(short_it["pts"], 25)
                p_long = resample_curve(long_it["pts"], 60)
                d_s2l = float(
                    np.mean(
                        np.min(
                            np.linalg.norm(
                                p_short[:, None, :] - p_long[None, :, :], axis=2
                            ),
                            axis=1,
                        )
                    )
                )
                if d_s2l < 6.5 or (
                    abs(items[i]["z_sort"] - items[j]["z_sort"]) < 5.5
                    and short_it["len"] < 150.0
                    and long_it["len"] > max(100.0, 1.8 * short_it["len"])
                ):
                    keep[short_idx] = False

        return [it for idx, it in enumerate(items) if keep[idx]]


    def _compute_unary_costs(
        self, cand: dict, frame: dict, side: str, s_z: float = 1.0
    ) -> np.ndarray:
        """Computes relative unary assignment cost U(cand, k) for k in 1..12."""
        costs = np.full(13, 1e6, dtype=np.float64)
        pts = cand["pts"]
        f_vec = cand["f_vec"].copy()
        # Normalize vertical features by patient vertical scale s_z (relative to Rib 1 height +15 mm)
        f_vec[0] = (f_vec[0] - 15.0) / s_z + 15.0
        f_vec[1] = (f_vec[1] - 15.0) / s_z + 15.0

        c_u = phys_to_canonical(resample_curve(pts, 40), frame)
        c_u[:, 2] = (c_u[:, 2] - (15.0 / 240.0)) / s_z + (15.0 / 240.0)

        ang_span = cand["ang_max"] - cand["ang_min"]
        is_partial = ang_span < 1.35

        for k in range(1, 13):
            mu = self.atlas.mu_f[k]
            sd = self.atlas.sd_f[k]
            z_err = (f_vec - mu) / sd

            if is_partial and k not in (1, 12):
                w_feat = np.array([1.2, 0.6, 0.30, 0.20, 0.25], dtype=np.float64)
            else:
                w_feat = np.array([1.1, 0.7, 1.15, 1.05, 0.45], dtype=np.float64)

            mahal = 0.5 * float(np.sum(w_feat * (z_err**2)))

            tmpl = self.atlas.mean_canon[f"{side}{k:02d}"]
            d_c2t = np.min(
                np.linalg.norm(c_u[:, None, :] - tmpl[None, :, :], axis=2), axis=1
            )
            tmpl_cost = 10.0 * float(np.mean(d_c2t))
            costs[k] = mahal + tmpl_cost

        # Subtract minimum unary cost across k in 1..12 so real ribs are never rejected as clutter
        min_u = float(np.min(costs[1:13]))
        costs[1:13] = costs[1:13] - min_u
        if cand["len"] < 45.0 and min_u > 8.0:
            costs[1:13] += min_u - 8.0

        return costs

    def _solve_viterbi_hemithorax(
        self, cands: list, unaries: list, side: str = "R", s_z: float = 1.0
    ) -> dict:
        """Finds optimal strictly-increasing rib label sequence with outlier skipping."""
        M = len(cands)
        if M == 0:
            return {}

        dp = np.full((M, 13), 1e9, dtype=np.float64)
        back_i = np.full((M, 13), -1, dtype=int)
        back_k = np.full((M, 13), -1, dtype=int)

        # Long curves (>= 115 mm) are guaranteed to be real ribs (never clutter)
        clutter_cost = np.zeros(M, dtype=np.float64)
        for idx, c in enumerate(cands):
            if c["len"] >= 115.0:
                clutter_cost[idx] = 65.0
            elif c["len"] >= 55.0 and c["ang_min"] < -0.45:
                clutter_cost[idx] = 22.0
            else:
                clutter_cost[idx] = 2.5 + 8.0 * min(
                    1.0, max(0.0, (c["len"] - 25.0) / 50.0)
                )

        def sum_clutter(idx_start: int, idx_end: int) -> float:
            if idx_start >= idx_end:
                return 0.0
            return float(np.sum(clutter_cost[idx_start:idx_end]))

        for i in range(M):
            for k in range(1, 13):
                # Starting sequence at label k (missing top k-1 ribs)
                skip_top_ribs_pen = (k - 1) * 3.0
                base_cost = sum_clutter(0, i) + unaries[i][k] + skip_top_ribs_pen
                dp[i, k] = base_cost

                for ip in range(i):
                    clut_between = sum_clutter(ip + 1, i)
                    dz = cands[ip]["z_sort"] - cands[i]["z_sort"]
                    if dz < 4.5:
                        continue
                    for kp in range(1, k):
                        if k - kp > 3:
                            continue
                        exp_dz = s_z * sum(self.atlas.mu_s[m] for m in range(kp, k))
                        var_dz = (s_z**2) * sum(
                            self.atlas.sd_s[m] ** 2 for m in range(kp, k)
                        ) + 6.0
                        single_step = s_z * self.atlas.mu_s[kp]
                        # Only allow skipping an intermediate rib (k > kp + 1) without heavy penalty
                        # if the physical gap dz is genuinely >= 1.55x a single rib step!
                        if k > kp + 1 and dz < 1.55 * single_step:
                            skip_mid_pen = (k - kp - 1) * 18.0
                        else:
                            skip_mid_pen = (k - kp - 1) * 2.5

                        trans_cost = (
                            0.5 * ((dz - exp_dz) ** 2) / var_dz + skip_mid_pen
                        )
                        cand_cost = (
                            dp[ip, kp] + clut_between + unaries[i][k] + trans_cost
                        )
                        if cand_cost < dp[i, k]:
                            dp[i, k] = cand_cost
                            back_i[i, k] = ip
                            back_k[i, k] = kp

        best_total = 1e9
        best_state = (-1, -1)
        for i in range(M):
            rem_clut = sum_clutter(i + 1, M)
            for k in range(1, 13):
                skip_bot_pen = (12 - k) * 1.5
                tot = dp[i, k] + rem_clut + skip_bot_pen
                if tot < best_total:
                    best_total = tot
                    best_state = (i, k)

        if best_state[0] < 0:
            return {}

        labeled = {}
        curr_i, curr_k = best_state
        while curr_i >= 0 and curr_k >= 1:
            labeled[curr_k] = cands[curr_i]
            prev_i = back_i[curr_i, curr_k]
            prev_k = back_k[curr_i, curr_k]
            curr_i, curr_k = prev_i, prev_k

        return {f"{side}{k:02d}": v for k, v in sorted(labeled.items())}

    def _heuristic_fallback_match(self, valid_lines: list, frame: dict) -> dict:
        hemi = {"R": [], "L": []}
        for pts in valid_lines:
            side = "R" if float(np.mean(pts[:, 0])) < frame["x_spine"] else "L"
            hemi[side].append(pts)
        out = {}
        for side in ("R", "L"):
            hemi[side].sort(key=lambda p: float(np.mean(p[:, 2])), reverse=True)
            for idx, pts in enumerate(hemi[side][:12]):
                out[f"{side}{idx + 1:02d}"] = pts
        return out


BilateralViterbiLabeler = RibTopologyMatcher
