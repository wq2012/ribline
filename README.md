# ribline: 3D Rib Centerline Tracing and Anatomical Labeling in Thoracic CT

[![Python Package](https://github.com/wq2012/ribline/actions/workflows/python-package.yml/badge.svg)](https://github.com/wq2012/ribline/actions/workflows/python-package.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2610.16047-b31b1b.svg)](https://arxiv.org/abs/2610.16047)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/wq2012/ribline)
[![PyPI Version](https://img.shields.io/pypi/v/ribline.svg)](https://pypi.python.org/pypi/ribline)
[![Python Versions](https://img.shields.io/pypi/pyversions/ribline.svg)](https://pypi.org/project/ribline)

**ribline** is an open-source Python library for automatic 3D rib centerline extraction, anatomical rib numbering (`R01..R12, L01..L12`) and sub-voxel refinement in thoracic CT ([Wang, arXiv:2610.16047](#citation)). It combines a lung-apex / vertebral-midline thoracic coordinate frame, a hybrid voxel detector (anatomically anchored Semantic Context Forest + lightweight 3D U-Net), and a global labeling stage (bilateral Viterbi dynamic programming + `RibSnake` refinement). It re-implements and extends ideas from learning-based deformable template matching ([Wu et al., CVPR 2012](#citation)) and Semantic Context Forests ([Wang et al., MICCAI MCV 2013](#citation)). [[arXiv](https://arxiv.org/abs/2610.16047)] [[PDF](https://arxiv.org/pdf/2610.16047)] [[Hugging Face collection](https://huggingface.co/collections/wq2012/ribline)]

> **Version 0.3.0 (revision).** This release fixes the metric definitions (symmetric 10 mm labeling criterion, 50 mm penalty for missed ribs, precision / false positives, `nan` instead of a default 1.0 for undefined groups), ships the U-Net checkpoint that actually produced the reported probability maps, makes the training scripts config-faithful, adds reproducible atlas / ablation / statistics scripts, fixes one merged-rib annotation in the public ground truth, and documents the evaluation-protocol caveats below. All numbers in this README were produced by the scripts named next to them.

| Component | **ribline** |
| :--- | :--- |
| **Voxel detector** | Anatomically anchored SCF (150 trees, 286 features) max-fused with a plain 3-level 3D U-Net (340k parameters) |
| **Thoracic frame** | Automatic lung apex `Z_apex`, vertebral midline `X_spine(Z)`, kyphosis-aware azimuthal angle (ordering parameter) |
| **Labeling** | Fragment consolidation + patient-scaled bilateral Viterbi DP with contralateral coupling |
| **Refinement** | Apical clavicle trimming, terminal-rib template recovery, normal-plane centring (`RibSnake`) |
| **Labeled Chamfer distance (12 eval scans, 286 ribs)** | 2.05 +/- 0.48 mm, 95 % bootstrap CI [1.80, 2.33], median 1.89 mm |
| **Labeled clDice 3 mm / 4 mm** | 0.886 / 0.945 |
| **Labeling accuracy (symmetric 10 mm)** | 97.2 % (278 / 286 ribs; Wilson 95 % CI 94.6 - 98.6 %) |
| **Recall / precision** | 97.2 % / 100.0 % (0 false-positive curves out of 286) |

---

## Table of Contents
1. [Installation](#installation)
2. [Dataset, Split and Ground-Truth Fix](#dataset-split-and-ground-truth-fix)
3. [Pretrained Models](#pretrained-models)
4. [Usage](#usage)
5. [Algorithm (as implemented)](#algorithm-as-implemented)
6. [Evaluation Protocol](#evaluation-protocol)
7. [Results](#results)
8. [Limitations and Known Issues](#limitations-and-known-issues)
9. [Reproducing Every Number](#reproducing-every-number)
10. [Citation](#citation)

---

## Installation

```bash
pip install ribline
```

or for development with PyTorch support:

```bash
git clone https://github.com/wq2012/ribline.git
cd ribline
pip install -e ".[torch]"
python run_tests.py
```

---

## Dataset, Split and Ground-Truth Fix

The package supports the **Siemens Corporate Research (SCR) 3D thoracic CT rib centerline benchmark** (60 scans, 2.0 mm isotropic, Analyze-style `.hdr/.img` with 3-line ASCII headers, manual B-spline interpolated centerlines `*_interp.xls`) as well as NIfTI / MetaImage volumes via SimpleITK.

- **Split (`split.json`)**: 48 training scans (47 with annotations, 1,116 ribs; `Gist_p2_19_Anonymous_0015_interp.xls` is empty and skipped) and 12 evaluation scans. **The split is not random**: the 12 evaluation scans are the case-sensitive alphabetical tail of the file list. See [Limitations](#limitations-and-known-issues).
- **Coordinates**: centerline points are in physical millimetres `(X, Y, Z) = (i_x s_x, i_y s_y, i_z s_z)` with origin (0, 0, 0); `X` right-to-left, `Y` anterior-to-posterior, `Z` inferior-to-superior.
- **Intensities**: about half of the scans are stored as signed HU and half as unsigned shifted values. `ribline.model.anatomy.standardize_hu` subtracts 1024 when the volume minimum is >= -100 HU. This standardisation is used by the forests and by Stage 3; the U-Net is fed a per-volume z-score of the *raw* stored intensities (see below).
- **Ground-truth fix**: in `RIB_007_20091210_bone_1.5_0.6_12_38_interp.xls` the label `R07` contains two ribs (640 points with a 220 mm jump; the second curve is the eighth right rib). `python -m ribline.scripts.fix_gt_annotations --gt-dir Data/Centerlines_GT --out-dir Data/Centerlines_GT_fixed` splits it into `R07` / `R08`, touches no other file, and writes `fix_log.json`. The fixed set has 286 evaluation ribs; results on the unmodified ground truth are reported as well.

---

## Pretrained Models

All pretrained checkpoints were trained on the 47 annotated training scans only and are published on Hugging Face in the **[ribline collection](https://huggingface.co/collections/wq2012/ribline)**:

| Hugging Face Repository | Model Variant | 12-Scan LCD (mm) | 12-Scan Acc (%) | 5-Fold CV LCD (mm) | 5-Fold CV Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [`wq2012/ribline-3d-ct-centerline`](https://huggingface.co/wq2012/ribline-3d-ct-centerline) | **Hybrid SCF + 3D U-Net + Stage 3 (Full Suite)** | **2.05 +/- 0.48** | **97.2 %** | **3.64 +/- 3.90** | **91.8 %** |
| [`wq2012/ribline-anatomical-scf`](https://huggingface.co/wq2012/ribline-anatomical-scf) | **Anatomically Anchored SCF + Stage 3 (CPU-only)** | 2.88 +/- 1.32 | 95.1 % | 5.28 +/- 4.62 | 86.9 % |
| [`wq2012/ribline-3d-unet`](https://huggingface.co/wq2012/ribline-3d-unet) | **Lightweight 3D U-Net + Stage 3** | 3.40 +/- 1.99 | 92.0 % | 7.44 +/- 8.93 | 78.2 % |
| [`wq2012/ribline-scf-pass2-autocontext`](https://huggingface.co/wq2012/ribline-scf-pass2-autocontext) | **2-Pass Auto-Context SCF Baseline + Stage 3** | 9.36 +/- 10.57 | 71.0 % | -- | -- |

### Checkpoint Files

| File | Component | What it actually is |
| :--- | :--- | :--- |
| `rib_forest_pass1.joblib` + `feature_config_pass1.json` | Pass-1 forest | 120 trees, unlimited depth, 228 features: raw intensity, per scale (1, 2, 4 mm) smoothed intensity, gradient magnitude, 6 second derivatives, Laplacian (28), plus 200 RSID pairs with offsets in [-20, 20] voxels (+/-40 mm). Trained with random rigid augmentation. |
| `rib_forest_pass2.joblib` + `feature_config_img.json` + `feature_config_ctx.json` | Pass-2 auto-context forest | 100 trees, 306 features = the 228 Pass-1 features + 28 multi-scale features of the Pass-1 probability map + 50 random-shift probability differences with offsets in [-40, 40] voxels (+/-80 mm). |
| `rib_forest_anatomical.joblib` + `feature_config_anatomical.json` | Anatomically anchored SCF | 150 trees, depth 24, 286 features: HU-standardised input, the 28 differential features, a per-scale tubularity proxy from the diagonal second derivatives (no Hessian eigen-decomposition / Frangi filter), canonical coordinates `|u|, u, v, w`, lung distance, and 250 RSID pairs in [-12, 12] voxels (+/-24 mm). |
| `rib_unet_3d.pth` | 3D U-Net | Plain 3-level U-Net (16 -> 32 -> 64, Conv-GroupNorm-ReLU x2 per level, **no residual blocks**), 339,889 parameters, 64^3 patches, loss 0.5 soft-Dice + 0.5 BCE, Adam 1e-3, batch 2, 8 patches per scan per epoch, 20 epochs, no augmentation. **This file is the epoch-20 checkpoint** (`md5 9e456ba9ff81c402a58436c666f152ef`) that produced every probability map reported below (re-inference reproduces them with max abs. difference 0). Earlier releases shipped an unrelated mid-run snapshot by mistake. Training recipe: `pretrained_models/train_config_unet.json`. |
| `rib_atlas.joblib` | 24-rib statistical atlas | Mean canonical curves (80 points), 5-D descriptor statistics, inter-rib spacing statistics and a 15,000-point slant-correction cloud. Re-fitting with `python -m ribline.scripts.build_atlas` on the 47 training scans reproduces this file exactly. |
| `benchmark_val12_fixed_gt.json`, `benchmark_val12_original_gt.json`, `benchmark_train47.json`, `benchmark_cv5.json` | Evaluation reports | Per-rib / per-subject metrics of all ablation rows on the 12 eval scans (fixed and unmodified GT), the 47 training scans, and the 59-scan 5-fold CV (corrected metrics, bootstrap CIs, paired tests). |
| `polar_stats_all59.json`, `gt_fix_log.json` | Statistics / provenance | Monotonicity and frame statistics on all 59 scans; log of the ground-truth fix. |

---

## Usage

### Python API

```python
import numpy as np
import SimpleITK as sitk
from ribline import (
    load_image, parse_rib_xls, extract_thoracic_frame, RibCageStatisticalAtlas,
    CenterlineExtractor, RibTopologyMatcher, RibSnake, evaluate_subject_ribs,
)

ct_img = load_image("Data/vol2mm/pet_ct_bc__eh_Anonymous_0004.hdr")
ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
spacing, origin = ct_img.GetSpacing(), ct_img.GetOrigin()

unet_prob = sitk.GetArrayFromImage(load_image("predictions_unet/pet_ct_bc__eh_Anonymous_0004_prob.nii.gz"))
scf_prob = sitk.GetArrayFromImage(load_image("predictions_scf/pet_ct_bc__eh_Anonymous_0004_prob.nii.gz"))
hybrid_prob = np.maximum(unet_prob, 0.75 * scf_prob)

frame = extract_thoracic_frame(ct_arr, spacing=spacing, origin=origin)
atlas = RibCageStatisticalAtlas.load("ribline/configs/rib_atlas.joblib")
curves = CenterlineExtractor(threshold=0.45).extract(
    hybrid_prob, spacing=spacing, origin=origin, min_length=22.0, hu_arr=ct_arr, frame=frame, atlas=atlas)
labeled = RibTopologyMatcher(atlas=atlas).match(curves, frame=frame)
refined = RibSnake(spacing=spacing).refine(labeled, ct_arr, origin=origin, prob_np=hybrid_prob, frame=frame, atlas=atlas)

gt = parse_rib_xls("Data/Centerlines_GT_fixed/pet_ct_bc__eh_Anonymous_0004_interp.xls")
m = evaluate_subject_ribs(gt, refined)   # symmetric 10 mm criterion, 50 mm missing penalty
print(m["labeled_chamfer_mm"], m["label_acc_all"], m["precision"], m["num_false_positive_curves"])
```

### Command line

```bash
# Detector probability maps (any split: val | train | all)
python -m ribline.pipeline.inference_unet --config experiments/sota/full_config.json \
    --checkpoint pretrained_models/rib_unet_3d.pth --output probs_unet --split val
python -m ribline.pipeline.inference --config <anatomical_scf_config.json> --split val --output-dir probs_scf --prob-only

# Stage 3 (hybrid, Viterbi, RibSnake); --labeling zsort / --no-refine give the ablation baselines
ribline-sota --prob-dir probs_unet --rf-prob-dir probs_scf --image-dir Data/vol2mm \
    --output-dir centerlines --gt-dir Data/Centerlines_GT_fixed

# Scoring with CIs and paired tests
python -m ribline.scripts.benchmark_eval --gt-dir Data/Centerlines_GT_fixed \
    --method ours=centerlines --out benchmark.json

# Training
python -m ribline.pipeline.train --config <forest_config.json>
python -m ribline.pipeline.train_pass2 --config <pass2_config.json>
python -m ribline.pipeline.train_unet --config experiments/sota/full_config.json
python -m ribline.scripts.build_atlas --image-dir Data/vol2mm --gt-dir Data/Centerlines_GT_fixed --split split.json --out rib_atlas.joblib
```

---

## Algorithm (as implemented)

1. **Thoracic frame** (`ribline/model/anatomy.py`): HU standardisation; on a 2x down-sampled grid, torso (> -350 HU), internal air (< -380 HU, opened, <= 4 largest components) = lungs; `Z_apex` = max of per-side 99.5th-percentile lung Z; spine from the bone projection in the posterior 40 % of the AP depth, per-slice median, Gaussian-smoothed. Kyphosis-aware angle `theta = atan2(-(Y - y_c(Z)), |X - x_spine| - 8 mm)` with `y_c(Z) = Y_ant + 0.25 D + 0.16 clip(Z_apex - Z, 0, 220)`. Measured on 1,402 GT ribs of the 59 scans (`python -m ribline.scripts.polar_angle_stats`): `theta_end > theta_start` on 100 %, strictly monotonic on 78.6 % (vs 57.0 % for a torso-centre angle), local reversal > 0.1 rad on 0.6 % (vs 7.6 %). Posterior end of rib 1 lies +11.9 +/- 9.3 mm above `Z_apex`; midline error 2.5 +/- 2.4 mm.
2. **Detectors** (`ribline/model/features.py`, `ribline/model/unet/`): see the table above. The forests are evaluated on all non-air voxels (`gaussian(HU, 1.2) > -20`), a speed optimisation only; forest training negatives are sampled uniformly over the whole volume.
3. **Stage 3** (`reconstruction.py`, `topology.py`, `refinement.py`): anatomical mask (bone band, vertebral / sternal corridors, shoulder exclusion, intercostal valley suppression), threshold 0.45, 3D skeleton, graph decomposition, hairpin / height-jump splitting, azimuthal chaining; per-hemithorax candidate consolidation; patient scale `s_Z`; relative Mahalanobis + template unaries; bilateral Viterbi DP with clutter costs and skip penalties, two passes with contralateral coupling; `RibSnake` (apical trimming at 0.55 W, terminal-rib recovery, 5 iterations of normal-plane centring clamped to 1.6 mm, cubic smoothing spline). Output curves have 15 to ~200 points (median ~150).

---

## Evaluation Protocol

Implemented in `ribline.model.metrics.evaluate_subject_ribs` (defaults = protocol below):

| Metric | Definition |
| :--- | :--- |
| **Labeled Chamfer distance (LCD)** | mean over *all* GT ribs of `0.5 (d_p2g + d_g2p)` between a GT rib and the prediction with the same label; a GT rib without prediction counts **50 mm** |
| **Labeled clDice (3 / 4 mm)** | per-rib centerline Dice, missing ribs = 0 |
| **Labeling accuracy** | GT rib correct iff same-label prediction exists, **`d_p2g <= 10 mm` and `d_g2p <= 10 mm`**, and the GT rib is the closest one to that prediction; pooled over ribs (all / first / intermediate 2-11 / twelfth); undefined groups give `nan` |
| **Recall** | GT rib detected iff some predicted curve satisfies the symmetric criterion |
| **Precision / FP** | predicted curve is a true positive iff `d_p2g <= 10 mm` to some GT rib |
| **NLD** | normalised length difference, missing = 1 |
| **Uncertainty** | subject-level mean +/- SD with 95 % bootstrap CI (10,000 resamples); Wilson 95 % CI for pooled rates; two-sided Wilcoxon signed-rank tests between methods |

The previous release used a one-sided 8 mm criterion, dropped missed ribs from the LCD mean, defaulted 12th-rib accuracy to 1.0 when no 12th ribs existed and reported unlabeled clDice; the legacy quantities are still returned under `*_onesided` / `*_matched_only` keys for comparison.

**Protocol caveat (please read).** The ~60 constants of Stage 3 were tuned by repeatedly running the pipeline and inspecting the metrics on the 12 evaluation scans (the development history is in the project notes). The three learned detectors and the atlas were fitted on the 47 training scans only. Numbers on the 12 scans are therefore in-sample for the Stage-3 constants. The cross-validation below quantifies the effect.

---

## Results

### Ablation on the 12 evaluation scans (286 GT ribs, fixed GT)

Source: `experiments/rebuttal/ablation_val12/benchmark.json` (= `pretrained_models/benchmark_val12_fixed_gt.json`), produced by `python -m ribline.scripts.run_ablation --spec experiments/rebuttal/ablation_val12_spec.json` from the stored probability maps with the released Stage-3 code. "Z-sort" = same extraction and consolidation, candidates of each side labeled 1..12 by decreasing height, no RibSnake.

| Method | LCD (mm) mean +/- SD | 95 % CI | UCD | lab. clDice 3 / 4 mm | Acc all | Acc 1st | Acc 2-11 | Acc 12th | Recall | Precision (FP) | NLD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF Pass-1 + Z-sort | 14.10 +/- 11.10 | [7.96, 20.46] | 3.31 | 0.548 / 0.606 | 64.3 % | 79.2 % | 62.9 % | 63.6 % | 91.3 % | 95.5 % (13) | 0.264 |
| SCF Pass-2 + Z-sort | 21.30 +/- 13.33 | [13.79, 28.67] | 4.16 | 0.416 / 0.452 | 43.0 % | 45.8 % | 44.2 % | 27.3 % | 76.2 % | 100.0 % (0) | 0.254 |
| SCF Pass-2 + Stage 3 | 9.36 +/- 10.57 | [4.24, 16.19] | 3.73 | 0.683 / 0.720 | 71.0 % | 54.2 % | 73.8 % | 59.1 % | 78.0 % | 99.6 % (1) | 0.226 |
| Anatomical SCF + Stage 3 | 2.88 +/- 1.32 | [2.23, 3.73] | 2.41 | 0.845 / 0.915 | 95.1 % | 91.7 % | 95.0 % | 100.0 % | 95.1 % | 99.7 % (1) | 0.088 |
| 3D U-Net + Z-sort | 7.57 +/- 7.43 | [3.97, 12.17] | 2.75 | 0.710 / 0.813 | 83.6 % | 70.8 % | 86.7 % | 63.6 % | 92.0 % | 100.0 % (0) | 0.147 |
| 3D U-Net + Stage 3 | 3.40 +/- 1.99 | [2.42, 4.64] | 2.30 | 0.835 / 0.902 | 92.0 % | 66.7 % | 95.4 % | 81.8 % | 93.0 % | 100.0 % (0) | 0.121 |
| Hybrid + Viterbi (no RibSnake) | 2.85 +/- 1.70 | [2.07, 3.95] | 2.11 | 0.872 / 0.933 | 95.8 % | 83.3 % | 97.5 % | 90.9 % | 95.8 % | 99.6 % (1) | 0.147 |
| **Hybrid + full Stage 3 (ours)** | **2.05 +/- 0.48** | [1.80, 2.33] | **1.91** | **0.886 / 0.945** | **97.2 %** | **91.7 %** | **97.5 %** | **100.0 %** | **97.2 %** | **100.0 % (0)** | **0.068** |

Paired Wilcoxon tests, ours vs. "Hybrid + Viterbi": LCD p = 0.009, NLD p = 0.0005, labeled clDice@3 mm p = 0.20, accuracy p = 0.50 (not significant on 12 subjects). Ours vs. "3D U-Net + Stage 3": LCD p = 0.001, accuracy p = 0.008.

### Per-subject results of the full pipeline

| Evaluation scan | GT ribs | Pred. curves | LCD (mm) | UCD (mm) | lab. clDice 3 / 4 mm | Acc all | Recall | FP | NLD | ribs failing the symmetric criterion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `R3_anon_0002713245_0002` | 24 | 24 | 2.12 | 2.13 | 0.864 / 0.960 | 100.0 % | 100.0 % | 0 | 0.057 | -- |
| `RIB_007_20091210_bone_1.5_0.6_12_38` | 24 | 24 | 1.53 | 1.45 | 0.934 / 0.960 | 100.0 % | 100.0 % | 0 | 0.057 | -- |
| `Sko_004029_0040290098_0002` | 24 | 24 | 2.50 | 1.92 | 0.889 / 0.942 | 95.8 % | 95.8 % | 0 | 0.079 | L07 (traced 93 of 291 points) |
| `Tsc_004002_0040027642_0002` | 22 | 22 | 2.70 | 2.64 | 0.718 / 0.864 | 100.0 % | 100.0 % | 0 | 0.079 | -- |
| `Zer_004061_0040610659_0001` | 24 | 24 | 2.98 | 2.74 | 0.899 / 0.924 | 79.2 % | 79.2 % | 0 | 0.114 | L04-L08 (anterior third missing) |
| `pat1_Anonymous_0002` | 24 | 24 | 2.48 | 2.28 | 0.817 / 0.935 | 91.7 % | 91.7 % | 0 | 0.087 | R01, L01 (partial) |
| `pat3_Anonymous_0002` | 24 | 24 | 1.87 | 1.68 | 0.891 / 0.948 | 100.0 % | 100.0 % | 0 | 0.064 | -- |
| `pat8_Anonymous_0001` | 24 | 24 | 1.72 | 1.65 | 0.902 / 0.953 | 100.0 % | 100.0 % | 0 | 0.038 | -- |
| `pet_ct_bc__eh_Anonymous_0004` | 24 | 24 | 1.44 | 1.37 | 0.946 / 0.973 | 100.0 % | 100.0 % | 0 | 0.047 | -- |
| `pet_ct_bc__mk_Anonymous_0004` | 24 | 24 | 1.68 | 1.54 | 0.946 / 0.959 | 100.0 % | 100.0 % | 0 | 0.075 | -- |
| `pet_ct_bc__sd_Anonymous_0005` | 24 | 24 | 1.70 | 1.64 | 0.927 / 0.961 | 100.0 % | 100.0 % | 0 | 0.040 | -- |
| `t2_08_20100323_bone_1.5_0.6_8_136` | 24 | 24 | 1.92 | 1.93 | 0.895 / 0.956 | 100.0 % | 100.0 % | 0 | 0.078 | -- |
| **Mean (12 scans, 286 ribs)** | 286 | 286 | **2.05 +/- 0.48** | **1.91 +/- 0.43** | **0.886 / 0.945** | **97.2 %** | **97.2 %** | **0** | **0.068** | 8 partially traced ribs |

All 8 failures are partial traces (`d_p2g <= 4.2 mm`, `d_g2p` 10.6 - 34.8 mm), not label swaps; under the legacy one-sided criterion they count as correct. 25 / 286 ribs have HD95 > 10 mm (8 above 20 mm); 52 have clDice@3 mm < 0.8 (5 below 0.5).

Sensitivity: unmodified GT (285 ribs) 2.08 +/- 0.46 mm, 96.8 % (276/285); unmodified GT with the merged `R07` excluded (284 ribs) 2.05 +/- 0.48 mm, 97.2 %; without `Sko_004029` (possible duplicate of a training patient) 2.01 +/- 0.48 mm [1.75, 2.31], 97.3 % (255/262). Sources: `benchmark_gt_original.json`, `benchmark_gt_original_excl_rib007_R07.json`, `benchmark_noSko.json` in `experiments/rebuttal/ablation_val12/`.

### Cross-validation with frozen Stage-3 constants and evaluation on the training scans

**(a) Frozen pipeline on the 47 annotated training scans** (`experiments/rebuttal/run_train47.py` -> `experiments/rebuttal/ablation_train47/benchmark.json`; detectors in-sample, Stage-3 constants never inspected on these scans, 1,116 GT ribs):

| Method | LCD (mm) | 95 % CI | lab. clDice 3 / 4 mm | Acc all | Acc 1st | Acc 2-11 | Acc 12th | Recall | Precision (FP) | NLD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Anatomical SCF + Stage 3 | 3.62 +/- 3.97 | [2.67, 4.91] | 0.830 / 0.895 | 92.4 % | 89.2 % | 92.4 % | 95.7 % | 94.8 % | 97.4 % (29) | 0.094 |
| 3D U-Net + Stage 3 | 5.84 +/- 7.20 | [3.95, 8.03] | 0.785 / 0.826 | 83.9 % | 69.9 % | 86.0 % | 76.3 % | 91.6 % | 97.8 % (25) | 0.142 |
| Hybrid + Viterbi (no RibSnake) | 2.98 +/- 2.11 | [2.42, 3.63] | 0.875 / 0.930 | 95.6 % | 88.2 % | 96.1 % | 97.8 % | 96.1 % | 98.5 % (17) | 0.155 |
| **Hybrid + full Stage 3** | **2.63 +/- 1.91** | [2.14, 3.23] | **0.885 / 0.932** | 95.4 % (1065/1116) | 86.0 % | 96.1 % | 97.8 % | 95.6 % | 98.2 % (20) | **0.083** |

Median LCD 1.89 mm (same as on the 12 scans), 28 / 47 scans labeled perfectly, 4 scans with LCD > 5 mm (worst `HP_485533` 10.3 mm, `03_Anonymous_0002` 9.9 mm, label shifts on one side). The gap to the 12-scan figures (2.05 mm / 97.2 %) is the price of tuning Stage 3 on those scans plus the larger anatomical variety of the training cohort; treat the 12-scan numbers as an optimistic upper bound.

**(b) 5-fold cross-validation** over all 59 annotated scans (`experiments/rebuttal/cv_fold.py {0..4}` -> `experiments/rebuttal/benchmark_cv5.json`, shipped as `pretrained_models/benchmark_cv5.json`; per-fold re-fitted atlas, Anatomical SCF and 3D U-Net, frozen Stage-3 constants, 1,402 GT ribs):

| Method | LCD (mm) | 95 % CI | Median LCD | UCD | lab. clDice 3 / 4 mm | Acc all | Acc 1st | Acc 2-11 | Acc 12th | Recall | Precision (FP) | NLD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Anatomical SCF + Stage 3 | 5.28 +/- 4.62 | [4.16, 6.56] | 3.08 | 3.44 | 0.769 / 0.840 | 86.9 % (1218/1402) | 72.6 % | 87.7 % | 93.0 % | 90.8 % | 95.3 % (67) | 0.126 |
| 3D U-Net + Stage 3 | 7.44 +/- 8.93 | [5.31, 9.85] | 2.99 | 3.30 | 0.745 / 0.785 | 78.2 % (1097/1402) | 61.5 % | 80.9 % | 67.8 % | 87.0 % | **98.1 % (26)** | 0.170 |
| Hybrid + Viterbi (no RibSnake) | 4.17 +/- 4.01 | [3.29, 5.28] | 2.78 | 3.07 | 0.807 / 0.879 | 91.4 % (1282/1402) | 81.2 % | **92.2 %** | 93.9 % | 94.3 % | 97.9 % (29) | 0.195 |
| **Hybrid + full Stage 3** | **3.64 +/- 3.90** | [2.79, 4.74] | **2.39** | **2.85** | **0.830 / 0.889** | **91.8 % (1287/1402)** | **82.9 %** | **92.2 %** | **96.5 %** | **94.6 %** | 97.7 % (33) | **0.097** |

Wilson 95 % CI for `Acc all` of `Hybrid + full Stage 3`: 90.2 - 93.1 %. Per-fold mean LCD across folds 0..4: `3.12, 4.80, 4.91, 2.84, 2.43 mm`. 30 / 59 scans achieve 100 % symmetric labeling accuracy; 8 scans have LCD > 5 mm (worst: `BC3_Anonymous_0001` 26.58 mm, `BC1_Anonymous_0001` 15.76 mm). Paired Wilcoxon signed-rank tests across all 59 subjects: `Hybrid + full Stage 3` vs. `Hybrid + Viterbi (no RibSnake)`: LCD p = 2.0e-9, labeled clDice@3mm p = 3.2e-9, NLD p = 4.4e-11; vs. `Anatomical SCF + Stage 3`: LCD p = 2.1e-7, Acc all p = 9.0e-5; vs. `3D U-Net + Stage 3`: LCD p = 1.9e-5, Acc all p = 8.5e-6.

### Published numbers (not directly comparable)

Wu et al. 2012 report 89.5 - 95.9 % labeling accuracy on an SCR cohort, Lenga et al. 2018 94.1 %, and RibSeg v2 (2024) an LCD of 3.15 mm / 96.5 % on 0.7 mm RibFrac CT. Different scans, resolutions, splits and matching criteria; none of these methods was re-run here, and **no claim of outperforming them is made**.

---

## Limitations and Known Issues

- **Small, single-source data set**: 60 scans (59 annotated) at 2 mm from one archive; no acquisition metadata; generalisation to other scanners / resolutions / pathologies is untested.
- **Non-random, tiny held-out set**: 12 scans (286 ribs), alphabetical tail of the file list. 8 errors in 286 ribs give a Wilson CI of 94.6 - 98.6 % for labeling accuracy; 0 subject-level failures in 12 scans only bound the per-scan failure rate below ~25 %.
- **Stage-3 constants were tuned with the evaluation scans visible.** There is no untouched test set for this data set. Treat the released constants as fitted to SCR; see the cross-validation.
- **Annotation errors / inconsistencies**: one merged-rib annotation (fixed, logged); 12th ribs annotated inconsistently (absent in `Tsc_004002`); some residual error is annotation noise.
- **Possible patient overlap** between `Bro_004029_0040290098_0001` (train) and `Sko_004029_0040290098_0002` (eval): different matrix / intensity encoding / FOV (voxel correlation 0.25) but rib-length fingerprints among the 1 % most similar of all scan pairs; cannot be resolved from anonymised data (`experiments/rebuttal/provenance.json`, `bro_sko_lengths.json`).
- **Dominant failure mode**: ribs traced only over their posterior part across a costal-cartilage gap (counted as errors by the symmetric criterion).
- **Pass-2 forest provenance**: the released Pass-2 forest was trained on Pass-1 maps from an earlier Pass-1 forest (`experiments/full_v4`), not from the released Pass-1 forest; it is kept as an ablation baseline only.
- `ribline/scripts/correct_gt.py` is a development-time experiment that produced a copy of the GT identical to the original (verified for all 60 files); it is not part of the protocol.

---

## Reproducing Every Number

| Quantity | Script / file |
| :--- | :--- |
| Ablation table, per-subject table, CIs, paired tests | `python -m ribline.scripts.run_ablation --spec experiments/rebuttal/ablation_val12_spec.json` -> `benchmark.json` |
| Alternative GT protocols | `experiments/rebuttal/score_val12.py` |
| Monotonicity, rib-1 offset, midline error | `python -m ribline.scripts.polar_angle_stats ...` -> `polar_stats_all59.json` |
| Ground-truth fix | `python -m ribline.scripts.fix_gt_annotations ...` -> `fix_log.json` |
| Checkpoint identity, `Centerlines_GT_corrected` == GT, atlas refit, Bro/Sko | `experiments/rebuttal/provenance.py` -> `provenance.json` |
| Atlas | `python -m ribline.scripts.build_atlas ...` |
| Frozen Stage 3 on the 47 training scans | `experiments/rebuttal/run_train47.py` |
| 5-fold cross-validation | `experiments/rebuttal/cv_fold.py {0..4}` |
| Unit tests (19, incl. corrected metrics) | `python run_tests.py` |

---

## Citation

If you find **ribline** or its pretrained models useful in your research, please cite:

- **Paper**: [https://arxiv.org/abs/2610.16047](https://arxiv.org/abs/2610.16047) ([PDF](https://arxiv.org/pdf/2610.16047))

```bibtex
@article{wang2026hybrid,
  title={Hybrid Context Forests and Bilateral Viterbi Matching for {3D} Rib Centerline Tracing in {CT}},
  author={Wang, Quan},
  journal={arXiv preprint arXiv:2610.16047},
  year={2026},
  url={https://arxiv.org/abs/2610.16047}
}

@inproceedings{wu2012learning,
  title={A learning based deformable template matching method for automatic rib centerline extraction and labeling in {CT} images},
  author={Wu, Dijia and Liu, David and Pusk{\'a}s, Zsolt and Lu, Caiyun and Wimmer, Andreas and Tietjen, Christian and Soza, Grzegorz and Zhou, S Kevin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={980--987},
  year={2012},
  organization={IEEE}
}

@inproceedings{wang2013semantic,
  title={Semantic context forests for learning-based knee cartilage segmentation in {3D} {MR} images},
  author={Wang, Quan and Wu, Dijia and Lu, Le and Liu, Meizhu and Boyer, Kim L and Zhou, Shaohua Kevin},
  booktitle={International MICCAI Workshop on Medical Computer Vision},
  pages={105--115},
  year={2013},
  organization={Springer}
}
```
