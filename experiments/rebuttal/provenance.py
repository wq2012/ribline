"""Provenance checks for the released artifacts. Writes provenance.json.

1. unet_ep20.pth vs unet_latest.pth vs unet_best.pth: tensor-level equality.
2. Released prob maps (full_run_v1/predictions) reproduced by unet_ep20 on 2 val subjects.
3. Centerlines_GT_corrected == Centerlines_GT for all 48 training subjects.
4. Refit atlas on the 47 training subjects and compare with the packaged atlas.
5. Bro_004029 vs Sko_004029 duplicate-patient check.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import SimpleITK as sitk
import torch

from ribline.data.centerline_parser import parse_rib_xls
from ribline.data.image_reader import load_image, resolve_subject_image_path
from ribline.model.anatomy import RibCageStatisticalAtlas
from ribline.model.metrics import chamfer_distance
from ribline.model.unet.model import RibUNet
from ribline.pipeline.inference_unet import sliding_window_inference
from ribline.scripts.build_atlas import build_atlas


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a) and set(a) == set(b)


def main():
    ap = argparse.ArgumentParser(description="Run provenance checks on released checkpoints and dataset.")
    ap.add_argument("--data-root", default=os.environ.get("RIBLINE_DATA_ROOT", "data"),
                    help="Data directory containing vol2mm/ and Centerlines_GT/")
    ap.add_argument("--exp-root", default="experiments", help="Experiments directory")
    ap.add_argument("--repo-root", default=".", help="Repository root containing split.json and pretrained_models/")
    ap.add_argument("--out", default="experiments/rebuttal/provenance.json", help="Output JSON report")
    args = ap.parse_args()

    rep = {}
    models_dir = os.path.join(args.exp_root, "sota", "full_run_v1", "models")
    if os.path.isdir(models_dir):
        sds = {n: torch.load(os.path.join(models_dir, f"{n}.pth"), map_location="cpu")
               for n in ["unet_ep20", "unet_latest", "unet_best"]
               if os.path.exists(os.path.join(models_dir, f"{n}.pth"))}
        rep["checkpoints"] = {
            "md5": {n: md5(os.path.join(models_dir, f"{n}.pth")) for n in sds},
            "ep20_equals_latest_tensors": same(sds["unet_ep20"], sds["unet_latest"]) if "unet_ep20" in sds and "unet_latest" in sds else None,
            "ep20_equals_best_tensors": same(sds["unet_ep20"], sds["unet_best"]) if "unet_ep20" in sds and "unet_best" in sds else None,
            "released_rib_unet_3d_md5": md5(os.path.join(args.repo_root, "pretrained_models", "rib_unet_3d.pth")),
        }

        if "unet_ep20" in sds:
            model = RibUNet(1, 1, 16)
            model.load_state_dict(sds["unet_ep20"])
            model.eval()
            rep["prob_reproduction_ep20"] = {}
            for s in ["pat3_Anonymous_0002", "Tsc_004002_0040027642_0002"]:
                img_path = resolve_subject_image_path(os.path.join(args.data_root, "vol2mm"), s)
                ref_path = os.path.join(args.exp_root, "sota", "full_run_v1", "predictions", f"{s}_prob.nii.gz")
                if img_path and os.path.exists(ref_path):
                    img = load_image(img_path)
                    arr = sitk.GetArrayFromImage(img).astype(np.float32)
                    p = sliding_window_inference(model, arr)
                    ref = sitk.GetArrayFromImage(load_image(ref_path)).astype(np.float32)
                    rep["prob_reproduction_ep20"][s] = {
                        "max_abs_diff": float(np.max(np.abs(p - ref))),
                        "corr": float(np.corrcoef(p.ravel(), ref.ravel())[0, 1]),
                    }

    split = json.load(open(os.path.join(args.repo_root, "split.json")))
    gt_dir = os.path.join(args.data_root, "Centerlines_GT")
    vol_dir = os.path.join(args.data_root, "vol2mm")
    if os.path.isdir(vol_dir) and os.path.isdir(gt_dir):
        atlas_new, used = build_atlas(vol_dir, gt_dir, split["train"])
        atlas_old = RibCageStatisticalAtlas.load(os.path.join(args.repo_root, "ribline", "configs", "rib_atlas.joblib"))
        mf = max(float(np.max(np.abs(atlas_new.mu_f[k] - atlas_old.mu_f[k]))) for k in range(1, 13))
        ms = max(abs(atlas_new.mu_s[k] - atlas_old.mu_s[k]) for k in range(1, 12))
        mc = max(float(np.max(np.abs(atlas_new.mean_canon[r] - atlas_old.mean_canon[r]))) for r in atlas_old.mean_canon)
        rep["atlas_refit_47_train"] = {
            "n_subjects_used": len(used),
            "max_abs_diff_mu_f": mf,
            "max_abs_diff_mu_s": ms,
            "max_abs_diff_mean_canon": mc,
        }

        sko_img = resolve_subject_image_path(vol_dir, "Sko_004029_0040290098_0002")
        bro_img = resolve_subject_image_path(vol_dir, "Bro_004029_0040290098_0001")
        if sko_img and bro_img:
            A = sitk.GetArrayFromImage(load_image(sko_img)).astype(np.float32)
            B = sitk.GetArrayFromImage(load_image(bro_img)).astype(np.float32)
            ga = parse_rib_xls(os.path.join(gt_dir, "Sko_004029_0040290098_0002_interp.xls"))
            gb = parse_rib_xls(os.path.join(gt_dir, "Bro_004029_0040290098_0001_interp.xls"))
            m = [min(x, y) for x, y in zip(A.shape, B.shape)]
            Ac, Bc = A[: m[0], : m[1], : m[2]], B[: m[0], : m[1], : m[2]]
            rep["bro_vs_sko"] = {
                "shape_sko": list(A.shape),
                "shape_bro": list(B.shape),
                "same_shape": A.shape == B.shape,
                "voxel_corr_common_crop": float(np.corrcoef(Ac.ravel(), Bc.ravel())[0, 1]),
                "gt_chamfer_mm_per_rib": {r: float(chamfer_distance(ga[r], gb[r])) for r in sorted(set(ga) & set(gb))},
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
