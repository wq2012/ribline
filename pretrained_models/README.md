---
license: mit
library_name: ribline
tags:
- medical-imaging
- ct
- 3d-centerline-extraction
- rib-labeling
- semantic-context-forest
- 3d-unet
- computer-vision
- arxiv:2610.16047
metrics:
- chamfer_distance
- cldice
- accuracy
---

# ribline-3d-ct-centerline: Pretrained Semantic Context Forests, 3D U-Net and Statistical Rib Cage Atlas

Official checkpoints for **[ribline](https://github.com/wq2012/ribline)** (v0.3.0), a hybrid machine-learning / deformable-topology pipeline for automatic 3D rib centerline extraction and anatomical numbering (`R01..R12, L01..L12`) in thoracic CT, described in **[Hybrid Context Forests and Bilateral Viterbi Matching for 3D Rib Centerline Tracing in CT](https://arxiv.org/abs/2610.16047)** ([PDF](https://arxiv.org/pdf/2610.16047)) and part of the **[ribline Hugging Face Collection](https://huggingface.co/collections/wq2012/ribline)**:

| Hugging Face Repository | Model Variant | 12-Scan LCD (mm) | 12-Scan Acc (%) | 5-Fold CV LCD (mm) | 5-Fold CV Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [`wq2012/ribline-3d-ct-centerline`](https://huggingface.co/wq2012/ribline-3d-ct-centerline) | **Hybrid SCF + 3D U-Net + Stage 3 (Full Suite)** | **2.05 +/- 0.48** | **97.2 %** | **3.64 +/- 3.90** | **91.8 %** |
| [`wq2012/ribline-anatomical-scf`](https://huggingface.co/wq2012/ribline-anatomical-scf) | **Anatomically Anchored SCF + Stage 3 (CPU-only)** | 2.88 +/- 1.32 | 95.1 % | 5.28 +/- 4.62 | 86.9 % |
| [`wq2012/ribline-3d-unet`](https://huggingface.co/wq2012/ribline-3d-unet) | **Lightweight 3D U-Net + Stage 3** | 3.40 +/- 1.99 | 92.0 % | 7.44 +/- 8.93 | 78.2 % |
| [`wq2012/ribline-scf-pass2-autocontext`](https://huggingface.co/wq2012/ribline-scf-pass2-autocontext) | **2-Pass Auto-Context SCF Baseline + Stage 3** | 9.36 +/- 10.57 | 71.0 % | -- | -- |

All models were trained on the **47 annotated training scans** (1,116 ribs) of the Siemens Corporate Research (SCR) 2 mm thoracic CT benchmark (`split.json`) and evaluated on the **12 evaluation scans** (286 ribs after the documented ground-truth fix). Please read the *Protocol caveat* and *Limitations* below before quoting any number.

> **What changed in this revision.** The previously shipped `rib_unet_3d.pth` was a mid-run snapshot (`md5 de4b158b...`) that did **not** produce the reported probability maps; it has been replaced by the epoch-20 checkpoint (`md5 9e456ba9ff81c402a58436c666f152ef`) that reproduces them exactly. The evaluation metrics were corrected (symmetric 10 mm criterion, 50 mm penalty for missed ribs, `nan` for undefined groups, precision added) and every number below was recomputed with the released code. Architecture and training descriptions were rewritten to match the code (plain U-Net, Dice + BCE loss, raw-intensity z-score input, no augmentation).

---

## Included files

| File | Component | Description (as implemented) |
| :--- | :--- | :--- |
| `rib_forest_pass1.joblib` + `feature_config_pass1.json` | Pass-1 forest | 120 trees, 228 features: raw intensity, 27 multi-scale Gaussian derivative features (sigma 1 / 2 / 4 mm: smoothed, gradient magnitude, 6 second derivatives, Laplacian), 200 RSID pairs with offsets in [-20, 20] voxels (+/-40 mm). |
| `rib_forest_pass2.joblib` + `feature_config_img.json` + `feature_config_ctx.json` | Pass-2 auto-context forest | 100 trees, 306 features = 228 Pass-1 features + 28 multi-scale features of the Pass-1 probability map + 50 random-shift probability differences in [-40, 40] voxels (+/-80 mm). Trained on Pass-1 maps of an earlier Pass-1 forest; ablation baseline only. |
| `rib_forest_anatomical.joblib` + `feature_config_anatomical.json` | Anatomically anchored SCF | 150 trees, max depth 24, 286 features: HU-standardized 28 differential features, a diagonal-curvature tubularity proxy per scale (no Hessian eigen-decomposition / Frangi filter), canonical coordinates `|u|, u, v, w`, lung distance, 250 RSID pairs in [-12, 12] voxels (+/-24 mm). |
| `rib_unet_3d.pth` | 3D U-Net | Plain 3-level U-Net (16 -> 32 -> 64, Conv-GroupNorm-ReLU x2 per level, no residual blocks), 339,889 parameters. Input: per-volume z-score of raw stored intensities. 64^3 patches, 0.5 soft-Dice + 0.5 BCE, Adam 1e-3, batch 2, 8 patches / scan / epoch, 20 epochs, no augmentation (`train_config_unet.json`). |
| `rib_atlas.joblib` | 24-rib statistical atlas | Mean canonical curves (80 points per rib), 5-D descriptor mean / SD, inter-rib spacing statistics, 15,000-point slant-correction cloud. `python -m ribline.scripts.build_atlas` on the 47 training scans reproduces it exactly. |
| `benchmark_val12_fixed_gt.json`, `benchmark_val12_original_gt.json`, `benchmark_gt_original_excl_rib007_R07.json`, `benchmark_noSko.json`, `benchmark_train47.json`, `benchmark_cv5.json` | Evaluation reports | Per-rib and per-subject metrics of all ablation rows on the 12 eval scans (including GT-sensitivity variants), the 47 training scans, and the 59-scan 5-fold CV with bootstrap CIs and paired tests. |
| `polar_stats_all59.json`, `gt_fix_log.json`, `train_config_unet.json` | Statistics / provenance | Frame statistics on all 59 scans; log of the single ground-truth fix; U-Net training recipe. |

---

## Evaluation protocol

`ribline.model.metrics.evaluate_subject_ribs` (defaults): labeled Chamfer distance (LCD) averaged over all GT ribs with missing labels counting 50 mm; labeled clDice at 3 / 4 mm; a GT rib is correctly labeled iff the same-label prediction has mean prediction-to-GT **and** GT-to-prediction distance <= 10 mm and the GT rib is the closest one to it; recall uses the same symmetric criterion over any predicted curve; precision counts predicted curves lying within 10 mm of some GT rib. Subject-level means are given with SD and 95 % bootstrap CI (10,000 resamples), pooled rates with Wilson intervals.

**Protocol caveat.** The constants of the Stage-3 labeling / refinement code were tuned while inspecting metrics on these 12 scans; the three forests, the U-Net and the atlas were fitted on the 47 training scans only. The numbers below are therefore in-sample for the Stage-3 constants. The split is not random (alphabetical tail of the file list).

## Results on the 12 evaluation scans (286 GT ribs, `RIB_007` annotation fixed)

Source: `benchmark_val12_fixed_gt.json`, produced by `python -m ribline.scripts.run_ablation` with the released code.

| Method | LCD (mm) | 95 % CI | UCD (mm) | lab. clDice 3 / 4 mm | Acc all | Acc 1st | Acc 2-11 | Acc 12th | Recall | Precision (FP) | NLD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF Pass-1 + Z-sort | 14.10 +/- 11.10 | [7.96, 20.46] | 3.31 | 0.548 / 0.606 | 64.3 % | 79.2 % | 62.9 % | 63.6 % | 91.3 % | 95.5 % (13) | 0.264 |
| SCF Pass-2 + Z-sort | 21.30 +/- 13.33 | [13.79, 28.67] | 4.16 | 0.416 / 0.452 | 43.0 % | 45.8 % | 44.2 % | 27.3 % | 76.2 % | 100.0 % (0) | 0.254 |
| SCF Pass-2 + Stage 3 | 9.36 +/- 10.57 | [4.24, 16.19] | 3.73 | 0.683 / 0.720 | 71.0 % | 54.2 % | 73.8 % | 59.1 % | 78.0 % | 99.6 % (1) | 0.226 |
| Anatomical SCF + Stage 3 | 2.88 +/- 1.32 | [2.23, 3.73] | 2.41 | 0.845 / 0.915 | 95.1 % | 91.7 % | 95.0 % | 100.0 % | 95.1 % | 99.7 % (1) | 0.088 |
| 3D U-Net + Z-sort | 7.57 +/- 7.43 | [3.97, 12.17] | 2.75 | 0.710 / 0.813 | 83.6 % | 70.8 % | 86.7 % | 63.6 % | 92.0 % | 100.0 % (0) | 0.147 |
| 3D U-Net + Stage 3 | 3.40 +/- 1.99 | [2.42, 4.64] | 2.30 | 0.835 / 0.902 | 92.0 % | 66.7 % | 95.4 % | 81.8 % | 93.0 % | 100.0 % (0) | 0.121 |
| Hybrid + Viterbi (no RibSnake) | 2.85 +/- 1.70 | [2.07, 3.95] | 2.11 | 0.872 / 0.933 | 95.8 % | 83.3 % | 97.5 % | 90.9 % | 95.8 % | 99.6 % (1) | 0.147 |
| **Hybrid + full Stage 3** | **2.05 +/- 0.48** | [1.80, 2.33] | **1.91** | **0.886 / 0.945** | **97.2 %** (278/286) | **91.7 %** | **97.5 %** | **100.0 %** | **97.2 %** | **100.0 % (0)** | **0.068** |

On the unmodified ground truth (285 ribs): 2.08 +/- 0.46 mm, 96.8 % (276/285). All 8 ribs failing the symmetric criterion are partial traces (prediction lies on the rib, but covers only part of it), not label swaps.

**Frozen pipeline on the 47 annotated training scans** (1,116 ribs; detectors in-sample, Stage-3 constants never inspected on these scans; `benchmark_train47.json`): LCD 2.63 +/- 1.91 mm (CI [2.14, 3.23], median 1.89), labeled clDice 0.885 / 0.932, labeling accuracy 95.4 % (1065/1116), recall 95.6 %, precision 98.2 % (20 FP curves). **5-fold cross-validation across all 59 scans** (1,402 ribs; per-fold re-fitted atlas, Anatomical SCF and 3D U-Net, frozen Stage-3 constants; `benchmark_cv5.json`): LCD **3.64 +/- 3.90 mm** (CI [2.79, 4.74], median 2.39 mm), labeled clDice 0.830 / 0.889, symmetric labeling accuracy **91.8 %** (1287/1402, Wilson 95 % CI 90.2 - 93.1 %), recall 94.6 %, precision 97.7 % (33 FP curves), NLD 0.097 (Wilcoxon p = 2.0e-9 for RibSnake on LCD and p = 3.2e-9 on labeled clDice@3mm).

## Limitations

- 60-scan single-source dataset at 2 mm; no acquisition metadata; untested outside SCR.
- 12-scan non-random evaluation set; Wilson CI for accuracy 94.6 - 98.6 %; subject-level failure rate only bounded below ~25 %.
- Stage-3 constants fitted with access to the evaluation scans (see caveat).
- Known annotation issues (merged `R07` in `RIB_007`, inconsistent 12th ribs); possible patient overlap `Bro_004029` (train) / `Sko_004029` (eval) cannot be excluded.
- Published numbers from other papers (Wu 2012, Lenga 2018, RibSeg v2) use other cohorts and criteria; no comparative claim is made.

---

## Quick start

```python
from ribline import load_pretrained_models

# Automatically downloads and instantiates the 24-rib atlas, Anatomical SCF, and 3D U-Net
models = load_pretrained_models("wq2012/ribline-3d-ct-centerline", device="cpu")
atlas = models["atlas"]
rf_anat = models["forest"]
unet = models["unet"]
```

Or load individual local checkpoint files directly:

```python
import torch
from ribline import RibCageStatisticalAtlas, RibForest
from ribline.model.unet.model import RibUNet3D

atlas = RibCageStatisticalAtlas.load("rib_atlas.joblib")
rf_anat = RibForest.load("rib_forest_anatomical.joblib")
unet = RibUNet3D(in_channels=1, out_channels=1, base_filters=16)
unet.load_state_dict(torch.load("rib_unet_3d.pth", map_location="cpu"))
unet.eval()
```

See the GitHub README for the full inference / Stage-3 / evaluation commands.

---

## Citation

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
