"""Unit tests for anatomical coordinate frame, topology matching, refinement, and metrics."""

import json
import os
import tempfile
import unittest
import jsonschema
import numpy as np

from ribline.model.anatomy import (
    RibCageStatisticalAtlas,
    canonical_to_phys,
    curve_length,
    extract_thoracic_frame,
    get_anatomical_rib_mask,
    get_polar_angle,
    phys_to_canonical,
    resample_curve,
    standardize_hu,
)
from ribline.model.metrics import (
    centerline_dice,
    chamfer_distance,
    dilated_dice,
    directional_chamfer,
    evaluate_subject_ribs,
    hausdorff_distance_95,
)
from ribline.model.reconstruction import CenterlineExtractor
from ribline.model.refinement import RibSnake
from ribline.model.topology import RibTopologyMatcher


class TestAnatomyAndTopology(unittest.TestCase):

    def _make_synthetic_thorax(self):
        """Creates a synthetic 3D thoracic CT volume (Z=140, Y=60, X=70) and 24 rib curves."""
        nz, ny, nx = 140, 60, 70
        spacing = (2.0, 2.0, 2.0)
        origin = (0.0, 0.0, 0.0)
        ct = np.full((nz, ny, nx), -950.0, dtype=np.float32)

        # Torso soft tissue cylinder
        zz, yy, xx = np.ogrid[:nz, :ny, :nx]
        torso = (((xx - 35) / 28.0) ** 2 + ((yy - 30) / 22.0) ** 2) <= 1.0
        ct[np.broadcast_to(torso, ct.shape)] = 40.0

        # Bilateral lung air cavities
        lung_r = (
            (((xx - 22) / 9.0) ** 2 + ((yy - 28) / 12.0) ** 2 <= 1.0)
            & (zz >= 30)
            & (zz <= 125)
        )
        lung_l = (
            (((xx - 48) / 9.0) ** 2 + ((yy - 28) / 12.0) ** 2 <= 1.0)
            & (zz >= 30)
            & (zz <= 125)
        )
        ct[lung_r | lung_l] = -850.0

        # Spine column at (x=35, y=44)
        spine = ((xx - 35) ** 2 + (yy - 44) ** 2) <= 9.0
        ct[np.broadcast_to(spine, ct.shape)] = 450.0

        # Generate 12 pairs of synthetic ribs in physical coordinates (mm) with 18 mm spacing
        ribs = {}
        prob_vol = np.zeros_like(ct, dtype=np.float32)
        for k in range(1, 13):
            z_post = 250.0 - (k - 1) * 18.0
            reach_scale = 0.44 if k == 1 else (0.45 if k == 12 else 0.78)
            ang_end = 0.85 if k < 11 else -0.25
            th = np.linspace(-1.35, ang_end, 50)
            dx = 15.0 + reach_scale * 48.0 * np.cos(th * 0.7)
            y = 60.0 - 30.0 * np.sin(th)
            z = z_post - 14.0 * (th - th[0]) / max(1e-3, th[-1] - th[0])

            for side, sign in (("R", -1.0), ("L", 1.0)):
                x = 70.0 + sign * dx
                pts = np.column_stack([x, y, z]).astype(np.float32)
                ribs[f"{side}{k:02d}"] = pts
                vx = np.clip(np.round(x / 2.0).astype(int), 0, nx - 1)
                vy = np.clip(np.round(y / 2.0).astype(int), 0, ny - 1)
                vz = np.clip(np.round(z / 2.0).astype(int), 0, nz - 1)
                rib_vox = np.zeros_like(ct, dtype=bool)
                rib_vox[vz, vy, vx] = True
                import scipy.ndimage as ndi

                rib_tube = ndi.binary_dilation(rib_vox, iterations=1)
                ct[rib_tube] = 550.0
                prob_vol[rib_tube] = 0.95

        return ct, prob_vol, ribs, spacing, origin

    def test_hu_standardization(self):
        raw_shifted = np.array([0.0, 200.0, 1024.0, 1500.0], dtype=np.float32)
        hu = standardize_hu(raw_shifted)
        self.assertTrue(np.allclose(hu, [-1024.0, -824.0, 0.0, 476.0]))

    def test_frame_and_canonical_roundtrip(self):
        ct, _, ribs, spacing, origin = self._make_synthetic_thorax()
        frame = extract_thoracic_frame(ct, spacing=spacing, origin=origin)
        self.assertAlmostEqual(frame["x_spine"], 70.0, delta=6.0)
        self.assertGreater(frame["z_apex"], 200.0)

        pts = ribs["R05"]
        canon = phys_to_canonical(pts, frame)
        recovered = canonical_to_phys(canon, frame)
        self.assertTrue(np.allclose(pts, recovered, atol=1e-3))

        ang = get_polar_angle(pts, frame, unwrap=True)
        self.assertGreater(ang[-1], ang[0])

        mask = get_anatomical_rib_mask(ct, frame)
        self.assertEqual(mask.shape, ct.shape)

    def test_metrics_precedence_and_accuracy(self):
        _, _, ribs, spacing, _ = self._make_synthetic_thorax()
        # Identical curves must have 0 Chamfer, 0 HD95, 1.0 clDice, 1.0 label accuracy
        self.assertAlmostEqual(chamfer_distance(ribs["R01"], ribs["R01"]), 0.0, places=5)
        self.assertAlmostEqual(
            hausdorff_distance_95(ribs["R01"], ribs["R01"]), 0.0, places=5
        )
        self.assertAlmostEqual(
            centerline_dice(ribs["R01"], ribs["R01"], tolerance_mm=3.0), 1.0, places=5
        )

        # Known offset of 2.0 mm along Z on a horizontal line: symmetric Chamfer must be 2.0 mm (not 3.0 mm)
        horiz = np.column_stack(
            [np.linspace(0.0, 50.0, 26), np.zeros(26), np.zeros(26)]
        ).astype(np.float32)
        shifted = horiz + np.array([0.0, 0.0, 2.0], dtype=np.float32)
        p2g, g2p, sym = directional_chamfer(horiz, shifted)
        self.assertAlmostEqual(p2g, 2.0, places=4)
        self.assertAlmostEqual(g2p, 2.0, places=4)
        self.assertAlmostEqual(sym, 2.0, places=4)
        self.assertAlmostEqual(
            chamfer_distance(horiz, shifted), 2.0, places=4
        )

        eval_res = evaluate_subject_ribs(gt_ribs=ribs, pred_ribs=ribs, spacing=spacing)
        self.assertAlmostEqual(eval_res["labeled_chamfer_mm"], 0.0, places=5)
        self.assertAlmostEqual(eval_res["label_acc_all"], 1.0, places=5)
        self.assertAlmostEqual(eval_res["label_acc_first"], 1.0, places=5)
        self.assertAlmostEqual(eval_res["label_acc_intermediate"], 1.0, places=5)
        self.assertAlmostEqual(eval_res["label_acc_twelfth"], 1.0, places=5)

    def test_extractor_matcher_and_refiner(self):
        ct, prob_vol, ribs, spacing, origin = self._make_synthetic_thorax()
        frame = extract_thoracic_frame(ct, spacing=spacing, origin=origin)
        atlas = RibCageStatisticalAtlas().fit([(frame, ribs)])

        extractor = CenterlineExtractor(threshold=0.45)
        curves = extractor.extract(
            prob_vol,
            spacing=spacing,
            origin=origin,
            min_length=15.0,
            hu_arr=ct,
            frame=frame,
            atlas=atlas,
        )
        self.assertGreaterEqual(len(curves), 20)

        matcher = RibTopologyMatcher(atlas=atlas)
        labeled = matcher.match(curves, frame=frame)
        self.assertGreaterEqual(len(labeled), 16)

        snake = RibSnake(spacing=spacing, iterations=2)
        refined = snake.refine(
            labeled, ct, origin=origin, prob_np=prob_vol, frame=frame, atlas=atlas
        )
        self.assertGreaterEqual(len(refined), 16)

    def test_config_schema_validation(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "config_schema.json",
        )
        with open(schema_path, "r") as f:
            schema = json.load(f)
        valid_cfg = {
            "data_config": {"image_directory": "/tmp/vol2mm"},
            "training_config": {"output_directory": "/tmp/out"},
        }
        jsonschema.validate(instance=valid_cfg, schema=schema)

    def test_markdown_no_latex_math(self):
        import re

        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        md_paths = [
            os.path.join(repo_root, "README.md"),
            os.path.join(repo_root, "pretrained_models", "README.md"),
        ]
        for md_path in md_paths:
            if not os.path.exists(md_path):
                continue
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("$", content, f"Found '$' in {md_path}")
            for macro in (
                r"\pm",
                r"\times",
                r"\downarrow",
                r"\uparrow",
                r"\theta",
                r"\text{",
                r"\le",
                r"\ge",
            ):
                self.assertNotIn(macro, content, f"Found {macro} in {md_path}")
            # Verify no unescaped underscores outside YAML frontmatter, fenced code blocks, inline code spans, or link targets
            stripped = re.sub(r"\A---.*?---", "", content, flags=re.DOTALL)
            stripped = re.sub(r"```.*?```", "", stripped, flags=re.DOTALL)
            stripped = re.sub(r"`[^`]*`", "", stripped)
            stripped = re.sub(r"\]\([^)]*\)", "]()", stripped)
            stripped = re.sub(r"https?://[^\s)\]]+", "", stripped)
            self.assertNotIn(
                "_",
                stripped,
                f"Found unescaped underscore outside code/URLs in {md_path}",
            )


if __name__ == "__main__":
    unittest.main()
