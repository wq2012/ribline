"""3D medial-axis skeletonization, junction splitting, and azimuthal rib chaining."""

import networkx as nx
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize

from ribline.model.anatomy import (
    curve_length,
    get_anatomical_rib_mask,
    get_polar_angle,
)


class CenterlineExtractor:
    """Extracts 3D rib centerline candidate curves from a probability volume.

    Unifies anatomical rib-cage masking, 3D topological skeletonization,
    junction decomposition (splitting inter-rib bridges by tangent collinearity),
    hairpin U-turn splitting, and azimuthal fragment chaining in cylindrical
    thoracic coordinates.
    """

    def __init__(self, threshold: float = 0.45, prune_spur_mm: float = 8.0):
        self.threshold = float(threshold)
        self.prune_spur_mm = float(prune_spur_mm)

    def extract(
        self,
        prob_map: np.ndarray,
        point_converter=None,
        spacing: tuple = (2.0, 2.0, 2.0),
        origin: tuple = (0.0, 0.0, 0.0),
        min_length: float = 28.0,
        hu_arr: np.ndarray = None,
        frame: dict = None,
        atlas=None,
    ) -> list:
        """Extracts unlabeled 3D rib centerline polylines in physical coordinates (mm).

        Args:
            prob_map: 3D probability array (Z, Y, X) in [0, 1].
            point_converter: Optional callable (x_idx, y_idx, z_idx) -> (X, Y, Z).
            spacing: Physical voxel spacing (sx, sy, sz) in mm.
            origin: Physical origin (ox, oy, oz) in mm.
            min_length: Minimum curve length in mm.
            hu_arr: Optional 3D CT intensity array (Z, Y, X) for anatomical masking.
            frame: Optional thoracic coordinate frame dict from extract_thoracic_frame.
            atlas: Optional RibCageStatisticalAtlas for slant-corrected height chaining.

        Returns:
            List of (N_i, 3) float32 numpy arrays in physical (X, Y, Z) mm coordinates.
        """
        if isinstance(spacing, (int, float)):
            spacing = (float(spacing), float(spacing), float(spacing))
        elif callable(point_converter) and spacing == (1.0, 1.0, 1.0):
            # Derive true physical spacing from point_converter when default (1,1,1) was left
            p0 = np.array(point_converter(0, 0, 0), dtype=np.float64)
            p1 = np.array(point_converter(1, 1, 1), dtype=np.float64)
            sp = np.abs(p1 - p0)
            if np.all(sp > 0):
                spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
                origin = (float(p0[0]), float(p0[1]), float(p0[2]))

        p_work = np.asarray(prob_map, dtype=np.float32).copy()
        if hu_arr is not None and frame is not None:
            anat_mask = get_anatomical_rib_mask(
                hu_arr, frame, max_bone_dist_mm=3.5, suppress_intercostal_valleys=True
            )
            p_work *= anat_mask.astype(np.float32)
            # Mild smoothing inside anatomical mask to heal 1-voxel quantization gaps
            p_work = ndi.gaussian_filter(p_work, sigma=0.55) * anat_mask.astype(
                np.float32
            )
        elif frame is not None:
            # Even without hu_arr, split central spine and sternum corridors
            nz, ny, nx_d = p_work.shape
            sx, sy, sz = spacing
            ox, oy, oz = origin
            x_phys = np.arange(nx_d, dtype=np.float32) * sx + ox
            y_phys = np.arange(ny, dtype=np.float32) * sy + oy
            dx_abs = np.abs(x_phys - frame["x_spine"])
            p_work[:, :, dx_abs < 10.5] = 0.0
            ant_mask = y_phys < (frame["y_lung"] - 20.0)
            p_work[:, ant_mask[:, None] & (dx_abs[None, :] < 31.0)] = 0.0

        binary_map = p_work > self.threshold
        if not np.any(binary_map):
            return []

        skeleton = skeletonize(binary_map)
        raw_chains = self._decompose_skeleton(
            skeleton, point_converter, spacing, origin
        )
        if not raw_chains:
            return []

        # Split any chain that makes a hairpin U-turn or steep vertical jump between ribs
        split_chains = []
        for pts in raw_chains:
            split_chains.extend(self._split_hairpins_and_jumps(pts, frame, atlas))

        # Chain collinear fragments along each rib arc
        if frame is not None:
            chained = self._chain_fragments(
                split_chains, frame, atlas, prob_map=p_work, spacing=spacing, origin=origin
            )
        else:
            chained = split_chains

        out = [c for c in chained if curve_length(c) >= min_length]
        return out

    def _idx_to_phys(
        self, z_idx: int, y_idx: int, x_idx: int, point_converter, spacing, origin
    ) -> list:
        if point_converter is not None:
            pt = point_converter(int(x_idx), int(y_idx), int(z_idx))
            return [float(pt[0]), float(pt[1]), float(pt[2])]
        return [
            float(x_idx * spacing[0] + origin[0]),
            float(y_idx * spacing[1] + origin[1]),
            float(z_idx * spacing[2] + origin[2]),
        ]

    def _decompose_skeleton(
        self, skeleton: np.ndarray, point_converter, spacing: tuple, origin: tuple
    ) -> list:
        """Decomposes a 3D voxel skeleton into unbranched chains, resolving junctions."""
        z_inds, y_inds, x_inds = np.nonzero(skeleton)
        if len(z_inds) < 2:
            return []

        G = nx.Graph()
        nodes_set = set(
            zip(z_inds.tolist(), y_inds.tolist(), x_inds.tolist())
        )
        for n in nodes_set:
            G.add_node(n)

        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
        for z, y, x in nodes_set:
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        if (dz, dy, dx) < (0, 0, 0):
                            continue
                        nb = (z + dz, y + dy, x + dx)
                        if nb in nodes_set:
                            dist = float(
                                np.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2)
                            )
                            G.add_edge((z, y, x), nb, weight=dist)

        # 1. Iteratively prune short leaf spurs attached to junctions
        changed = True
        while changed:
            changed = False
            leaves = [n for n in G.nodes() if G.degree(n) == 1]
            for leaf in leaves:
                if leaf not in G or G.degree(leaf) != 1:
                    continue
                path = [leaf]
                curr = leaf
                b_len = 0.0
                while True:
                    nbs = list(G.neighbors(curr))
                    if not nbs:
                        break
                    if G.degree(curr) > 2:
                        if b_len < self.prune_spur_mm:
                            G.remove_nodes_from(path[:-1])
                            changed = True
                        break
                    if G.degree(curr) == 1 and curr != leaf:
                        break
                    nxt = nbs[0] if curr == leaf else (nbs[1] if nbs[0] == path[-2] else nbs[0])
                    b_len += G[curr][nxt]["weight"]
                    if b_len >= self.prune_spur_mm:
                        break
                    path.append(nxt)
                    curr = nxt

        # 2. Collapse adjacent junction voxels (degree >= 3) and extract elementary chains
        junc_nodes = {n for n in G.nodes() if G.degree(n) >= 3}
        if not junc_nodes:
            # All components are simple paths or cycles
            chains = []
            for comp in nx.connected_components(G):
                if len(comp) < 3:
                    continue
                sub = G.subgraph(comp)
                ends = [n for n in sub.nodes() if sub.degree(n) == 1]
                if len(ends) >= 2:
                    p_nodes = nx.shortest_path(sub, ends[0], ends[1], weight="weight")
                else:
                    start = next(iter(comp))
                    d1 = nx.single_source_dijkstra_path_length(sub, start, weight="weight")
                    u = max(d1, key=d1.get)
                    d2 = nx.single_source_dijkstra_path_length(sub, u, weight="weight")
                    v = max(d2, key=d2.get)
                    p_nodes = nx.shortest_path(sub, u, v, weight="weight")
                coords = np.array(
                    [
                        self._idx_to_phys(z, y, x, point_converter, spacing, origin)
                        for (z, y, x) in p_nodes
                    ],
                    dtype=np.float32,
                )
                chains.append(coords)
            return chains

        # Remove junction nodes temporarily to get clean degree-2 interior chains
        G_no_junc = G.copy()
        G_no_junc.remove_nodes_from(junc_nodes)
        elem_chains = []
        for comp in nx.connected_components(G_no_junc):
            if len(comp) < 2:
                continue
            sub = G_no_junc.subgraph(comp)
            ends = [n for n in sub.nodes() if sub.degree(n) <= 1]
            if len(ends) < 2:
                continue
            p_nodes = nx.shortest_path(sub, ends[0], ends[1], weight="weight")
            # Attach neighboring junction voxel at each end if present
            nb0 = [n for n in G.neighbors(p_nodes[0]) if n in junc_nodes]
            nb1 = [n for n in G.neighbors(p_nodes[-1]) if n in junc_nodes]
            if nb0:
                p_nodes = [nb0[0]] + p_nodes
            if nb1:
                p_nodes = p_nodes + [nb1[0]]
            coords = np.array(
                [
                    self._idx_to_phys(z, y, x, point_converter, spacing, origin)
                    for (z, y, x) in p_nodes
                ],
                dtype=np.float32,
            )
            # Discard short steep vertical bridge segments between adjacent ribs
            elen = curve_length(coords)
            if elen < 3.0:
                continue
            dz_span = abs(float(coords[-1, 2] - coords[0, 2]))
            if elen < 28.0 and dz_span / max(elen, 1e-3) > 0.72:
                continue
            elem_chains.append(coords)

        return elem_chains

    def _split_hairpins_and_jumps(
        self, pts: np.ndarray, frame: dict = None, atlas=None
    ) -> list:
        """Splits a chain if it contains an inter-rib hairpin turn or vertical jump."""
        if len(pts) < 12 or frame is None:
            return [pts]

        ang = get_polar_angle(pts, frame, unwrap=True)
        # Smooth angle profile to detect genuine reversals (> 0.20 rad reversal)
        ang_s = ndi.gaussian_filter1d(ang, sigma=2.5, mode="nearest")
        dang = np.diff(ang_s)
        # Check if there is a major sign reversal where both sides are substantial
        pos_run = np.sum(dang > 0.005)
        neg_run = np.sum(dang < -0.005)
        if pos_run >= 6 and neg_run >= 6:
            # Split at the extremum of ang_s
            idx_max = int(np.argmax(ang_s))
            idx_min = int(np.argmin(ang_s))
            for idx_sp in sorted([idx_min, idx_max]):
                if 5 <= idx_sp <= len(pts) - 6:
                    c1 = pts[:idx_sp]
                    c2 = pts[idx_sp + 1 :]
                    res = []
                    if curve_length(c1) >= 10.0:
                        res.extend(self._split_hairpins_and_jumps(c1, frame, atlas))
                    if curve_length(c2) >= 10.0:
                        res.extend(self._split_hairpins_and_jumps(c2, frame, atlas))
                    if res:
                        return res

        # Also check for sudden jump in slant-corrected Z height (> 15 mm over 8 points)
        if atlas is not None and len(pts) >= 16:
            zp_eq = atlas.get_z_post_equiv(pts, frame)
            zp_s = ndi.gaussian_filter1d(zp_eq, sigma=2.0, mode="nearest")
            # Local gradient of slant-corrected Z
            dz_local = np.abs(zp_s[6:] - zp_s[:-6])
            max_idx = int(np.argmax(dz_local)) + 3
            if dz_local[max_idx - 3] > 16.0 and 6 <= max_idx <= len(pts) - 7:
                c1 = pts[: max_idx - 1]
                c2 = pts[max_idx + 2 :]
                res = []
                if curve_length(c1) >= 12.0:
                    res.append(c1)
                if curve_length(c2) >= 12.0:
                    res.append(c2)
                if len(res) == 2:
                    return res

        return [pts]

    def _chain_fragments(
        self,
        chains: list,
        frame: dict,
        atlas=None,
        prob_map: np.ndarray = None,
        spacing: tuple = (2.0, 2.0, 2.0),
        origin: tuple = (0.0, 0.0, 0.0),
    ) -> list:
        """Chains collinear rib segments in each hemithorax from posterior to anterior."""
        if not chains:
            return []

        # Separate by hemithorax (R: X < x_spine, L: X > x_spine) and orient posterior -> anterior
        hemi = {"R": [], "L": []}
        for c in chains:
            if len(c) < 2 or curve_length(c) < 8.0:
                continue
            ang = get_polar_angle(c, frame, unwrap=True)
            if ang[-1] < ang[0]:
                c = c[::-1].copy()
            side = "R" if np.mean(c[:, 0]) < frame["x_spine"] else "L"
            hemi[side].append(c)

        merged_all = []
        for side in ("R", "L"):
            segs = hemi[side]
            changed = True
            while changed and len(segs) > 1:
                changed = False
                best_score = 1e9
                best_pair = None

                # Precompute angle bounds and slant-corrected Z for each segment
                meta = []
                for s in segs:
                    ang = get_polar_angle(s, frame, unwrap=True)
                    if atlas is not None:
                        zp = atlas.get_z_post_equiv(s, frame)
                    else:
                        zp = s[:, 2]
                    # End tangent at tail and head
                    k_step = min(6, len(s) - 1)
                    t_end = s[-1] - s[-1 - k_step]
                    t_end = t_end / (np.linalg.norm(t_end) + 1e-6)
                    t_start = s[k_step] - s[0]
                    t_start = t_start / (np.linalg.norm(t_start) + 1e-6)
                    meta.append(
                        {
                            "ang_min": float(ang[0]),
                            "ang_max": float(ang[-1]),
                            "zp_med": float(np.median(zp)),
                            "zp_end": float( np.mean(zp[-k_step:]) ),
                            "zp_start": float( np.mean(zp[:k_step]) ),
                            "t_end": t_end,
                            "t_start": t_start,
                        }
                    )

                for i in range(len(segs)):
                    for j in range(len(segs)):
                        if i == j:
                            continue
                        # Check if segment j continues segment i (anterior to i)
                        mi, mj = meta[i], meta[j]
                        if mj["ang_min"] < mi["ang_max"] - 0.12:
                            continue
                        gap_vec = segs[j][0] - segs[i][-1]
                        gap_dist = float(np.linalg.norm(gap_vec))
                        if gap_dist > 44.0:
                            continue
                        dz_p = abs(mi["zp_end"] - mj["zp_start"])
                        dz_med = abs(mi["zp_med"] - mj["zp_med"])
                        if dz_p > 12.0 or dz_med > 14.0:
                            continue

                        if gap_dist < 4.5:
                            # Almost touching junction: check tangent collinearity
                            cos_t = float(np.dot(mi["t_end"], mj["t_start"]))
                            if cos_t < 0.72:
                                continue
                            score = gap_dist + 1.5 * dz_p + 10.0 * (1.0 - cos_t)
                        else:
                            chord = gap_vec / gap_dist
                            cos1 = float(np.dot(mi["t_end"], chord))
                            cos2 = float(np.dot(mj["t_start"], chord))
                            if cos1 < 0.62 or cos2 < 0.62:
                                continue
                            score = (
                                gap_dist
                                + 1.5 * dz_p
                                + 0.5 * dz_med
                                + 12.0 * (2.0 - cos1 - cos2)
                            )

                        if score < best_score:
                            best_score = score
                            best_pair = (i, j, gap_dist)

                if best_pair is not None:
                    i, j, gap_dist = best_pair
                    si, sj = segs[i], segs[j]
                    if gap_dist > 3.0:
                        n_bridge = max(2, int(round(gap_dist / 2.0)))
                        bridge = np.column_stack(
                            [
                                np.linspace(si[-1, d], sj[0, d], n_bridge + 2)[1:-1]
                                for d in range(3)
                            ]
                        ).astype(np.float32)
                        new_seg = np.vstack([si, bridge, sj])
                    else:
                        new_seg = np.vstack([si, sj])
                    segs = [
                        s for idx, s in enumerate(segs) if idx not in (i, j)
                    ] + [new_seg]
                    changed = True

            merged_all.extend(segs)

        return merged_all
