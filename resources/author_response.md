# Author Response to the Review of "Hybrid Semantic Context Forests and Kyphosis-Aware Deformable Topology Matching for 3D Rib Centerlines" (`ribline`)

We thank the reviewer for an unusually careful, artefact-level review. Every point was correct. Below we map each concern to the concrete change in the code, the paper, the model card / README, and to the verification evidence (script + output file). Paths are relative to the development tree `/usr/local/google/home/quanw/Code/siemens_2012/Rib/` (experiments) and to the released library `/usr/local/google/home/quanw/Code/github/ribline/` (mirrored to `Rib/delivery/`; `pretrained_models/` mirrored to `Rib/experiments/huggingface_release/` and `Rib/experiments/80_20_split/huggingface_release/`).

**Headline change.** Under the corrected protocol (symmetric 10 mm criterion, 50 mm penalty for missed labels, `RIB_007` annotation fixed) the final pipeline scores **LCD 2.05 +/- 0.48 mm (95 % CI [1.80, 2.33]), labeled clDice 0.886 / 0.945, labeling accuracy 97.2 % (278/286, Wilson 94.6 - 98.6 %), recall 97.2 %, precision 100 % (0 FP curves)** on the 12 evaluation scans, instead of the previously claimed "100 % / 100 %". All "100 %", "0 / 1,401 inversions", "surpass prior work" and "Residual U-Net / Focal loss / Frangi" statements were removed. We state explicitly in the paper and README that the Stage-3 constants were tuned with access to the 12 evaluation scans, and we add leakage experiments (Section "Leakage experiments" below).

Source of every number in this document: `Rib/experiments/rebuttal/ablation_val12/benchmark*.json` (produced 2026-10-11 by `python -m ribline.scripts.run_ablation --spec Rib/experiments/rebuttal/ablation_val12_spec.json` and `Rib/experiments/rebuttal/score_val12.py`), `Rib/experiments/rebuttal/polar_stats_all59.json`, `Rib/experiments/rebuttal/provenance.json`, `Rib/experiments/rebuttal/bro_sko_lengths.json`.

---

## A. Bugs / artefact-level defects

### A1. Wrong U-Net checkpoint shipped; undocumented checkpoint selection

**Confirmed.** `provenance.py` -> `provenance.json`:
- `unet_ep20.pth` and `unet_latest.pth` are tensor-identical (`ep20_equals_latest_tensors: true`; their md5 differ only through pickle metadata); `unet_best.pth` (= previously released `rib_unet_3d.pth`, md5 `de4b158b...`) is a different mid-run snapshot (`ep20_equals_best_tensors: false`).
- Re-running `unet_ep20.pth` on `pat3` and `Tsc` reproduces the stored probability maps with `max_abs_diff = 0.0`.

**Changes.**
- `pretrained_models/rib_unet_3d.pth` (and both HF mirrors) now **is** `unet_ep20.pth` (md5 `9e456ba9ff81c402a58436c666f152ef`), copied by `Rib/experiments/rebuttal/sync_release.py`, which prints the md5 of every shipped artefact.
- Checkpoint selection is documented: the released weights are simply the last (20th) epoch; there was no validation-based selection. `ribline/pipeline/train_unet.py` now writes `models/train_recipe.json` (epochs, patches/volume, loss, normalisation, parameter count, seed) next to every checkpoint, and the config that reproduces the release is `Rib/experiments/sota/full_config.json` (shipped as `pretrained_models/train_config_unet.json`).
- Model card and README rewritten accordingly (see A3/B6 for the architecture text).

### A2. Metric implementation contradicts the paper

**Confirmed on all five sub-points.** `ribline/model/metrics.py` was rewritten; the paper's Sec. 5.1 and the README "Evaluation Protocol" now describe exactly what the code computes:

| Sub-point | Old behaviour | New behaviour (default) |
| --- | --- | --- |
| (a) missed labels | dropped from the LCD mean | counted as `missing_penalty_mm = 50` (legacy value kept as `labeled_chamfer_matched_only_mm`) |
| (b) tolerance | 8 mm | `match_tol_mm = 10` |
| (c) one-sided criterion | p2g only | symmetric: `d_p2g <= tol` **and** `d_g2p <= tol` for labeling accuracy and recall (legacy kept as `*_onesided`); per-rib `d_p2g_mm`, `d_g2p_mm`, `hd95_mm` exported |
| (d) `label_acc_twelfth` default 1.0 | yes | `nan` when a group has no GT ribs; group sizes exported; pooled accuracies in `aggregate_subject_metrics` |
| (e) clDice | unlabeled reported as main | labeled clDice (3 / 4 mm) is the headline; unlabeled still reported |
| new | - | `precision`, `num_false_positive_curves`, `false_positive_curves` |

Unit tests for every rule: `ribline/tests/test_metrics.py` (7 tests: penalty arithmetic, symmetric vs one-sided, label swap counted as error, FP curve, `nan` group, pooled aggregation). `python run_tests.py` -> 19 tests OK.

**Effect on the numbers** (fixed GT, 286 ribs; `benchmark.json`):
- LCD of the final system 2.05 mm is unchanged in value because it has no missing labels; the reviewer's recomputed table values are reproduced for the other rows (e.g. M5 9.36, M7 3.40, M8 2.85 with penalty).
- Symmetric labeling accuracy of the final system: **278 / 286 = 97.2 %** (one-sided: 286 / 286). The 8 failures are exactly the ribs listed by the reviewer: `Sko/L07` (g2p 34.8 mm, HD95 66.4 mm, 93 vs 291 points), `Zer/L04-L08` (g2p 10.6 - 17.7 mm), `pat1/R01,L01` (g2p 12.0 / 12.3 mm). (`RIB_007/R07` disappears after the GT fix, `pet_mk/L12` at g2p 9.3 mm passes the 10 mm tolerance.) They are all partial traces, not label swaps; the paper says so.
- Per-rib tails reported in the paper: HD95 > 10 mm for 25 ribs (> 20 mm: 8), clDice@3 mm < 0.8 for 52 ribs (< 0.5: 5).
- Paper Table 1 and README / model-card tables are regenerated from this JSON; the legacy `benchmark_results_12val.json` was removed from the release.

### A3. Reproducibility defects in training scripts

**Confirmed.** `train_unet.py` now reads `samples_per_volume`, `epochs`, `batch_size`, `learning_rate`, `num_workers`, `mask_cache_dir`, `torch_threads`, `seed` from the config and `augmentation` from `data_config`; `RibDLDataset` implements `augmentation` (random L-R flip + +/-10 % intensity scale, default **off**). `Rib/experiments/sota/full_config.json` was corrected to the recipe that actually produced the release (`augmentation: false`, `samples_per_volume: 8`, `epochs: 20`, `centerline_directory: Centerlines_GT`); a `_comment` field explains the change. Docstrings of `RibDLDataset` and `inference_unet.run_inference` state that the network input is a per-volume z-score of the **raw** stored intensities (no HU standardisation); the paper (Sec. 3.2) and README say the same.

The reviewer also noticed that the U-Net config pointed at `Centerlines_GT_corrected`. We verified (`provenance.json: gt_corrected_vs_gt.differences = []`) that this directory contains identical labels and coordinates to `Centerlines_GT` for all 60 files (the development-time shift search in `ribline/scripts/correct_gt.py` selected shift 0 everywhere); the script now carries a docstring explaining this and that it is not part of the protocol.

---

## B. Technical incorrectness (paper != code)

### B4. Equations and constants do not match the implementation

**Confirmed.** We chose to rewrite the paper to match the released code (changing the code to match the old text would have changed every result). Sections 2-4 of the paper now contain the formulas and constants as implemented, verified line by line against `anatomy.py`, `features.py`, `reconstruction.py`, `topology.py`, `refinement.py`:
- Eq. 1: `u = (X - x_spine)/W`, `v = (Y - y_spine)/D`, `w = (Z - Z_apex)/240` with the scalar spine seed (and the per-slice trajectory used only in the anatomical mask); Eq. 2: `theta = atan2(-(Y - y_c(Z)), |X - x_spine| - 8 mm)` with `y_c(Z) = Y_ant + 0.25 D + 0.16 clip(Z_apex - Z, 0, 220)`.
- HU rule (`min >= -100 -> subtract 1024`), torso / air thresholds (-350 / -380 HU), lung-component rule, spine search (posterior 40 %, +/-50 mm, Gaussian sigma = 3 and 12), valley suppression (12 HU below both axial neighbours and < 135 HU), skeleton / graph constants (8 mm spurs, 28 mm / 0.72 bridges, 16 mm height jumps, 44 mm gaps, cos 0.62 / 0.72), consolidation constants (11.5 / 75 / 140 / 85 / 66 mm, 0.25 rad, 8.5 / 6.5 / 5.5 mm), `s_Z` blend (0.45 / 0.55, 21.8 mm), unary weights, clutter costs (65 / 22 / 2.5 + 8), transition and skip penalties (6, 1.55, 2.5, 18, 3(k-1), 1.5(12-k)), coupling (7.5 mm, 0.45, 8.5 mm, 3.0), snake (5 x 5 grid, 1.25 mm, r^2 <= 6.5, `n = max(15, L/2)`, 0.35 / 0.45 / sigma 1.2, 1.6 mm clamp, spline `s = 0.15 N`).
- Atlas: 80 (not 50) points; the 5-D descriptor is `[z_post, z_mean, x_max/W, L/W, y_min/D]` (not `[z_sort, z_post, l_norm, theta_min, theta_max]`).
- The `get_polar_angle`, `extract_thoracic_frame` and `FeatureExtractor` docstrings were rewritten to the same effect.

### B5. False monotonicity and Rib-1 offset claims

**Confirmed.** `python -m ribline.scripts.polar_angle_stats` (new script) on the 59 valid scans (fixed GT, 1,402 ribs) -> `polar_stats_all59.json`:
- kyphosis-aware angle: `theta_end > theta_start` on 1,402 / 1,402; **strictly monotonic on 1,102 (78.6 %)**; reversals > 0.02 / 0.05 / 0.1 / 0.3 rad on 44 / 14 / 8 / 3 ribs (max 1.23 rad; the reviewer's 2.15 rad case was the merged `RIB_007/R07`).
- naive torso-centre angle: strictly monotonic on 799 (57.0 %); reversals > 0.1 rad on 107 (7.6 %).
- posterior end of `R01/L01` minus `Z_apex`: **+11.9 +/- 9.3 mm** (n = 117), reproducing the reviewer's value; the "+3.6 +/- 8.5 mm" figure is gone.
- spine midline error: 2.5 +/- 2.4 mm (median 1.8) over 695 rib pairs (the old "2.73 mm" was replaced by this measured, defined quantity).
The paper (abstract, contributions, Sec. 2.2) now reports exactly these numbers and calls `theta` an "ordering parameter", not a guarantee of monotonicity. Fig. 1 text was changed accordingly.

### B6. Architecture / feature descriptions

**Confirmed and corrected everywhere** (paper Sec. 3, Fig. 1 flowchart, Fig. 2 panel title, README, model card, docstrings):
- "3D Residual U-Net" -> plain 3-level U-Net (Conv-GN-ReLU x2, no residual connections), 339,889 parameters; loss `0.5 soft-Dice + 0.5 BCE` (not Dice + Focal).
- No Hessian eigenvalues / Frangi / Sato: the anatomical SCF uses a diagonal-curvature tubularity proxy; the formula is given.
- RSID offsets are in voxels: Pass-1 +/-20 voxels (+/-40 mm), Pass-2 RSPD +/-40 voxels (+/-80 mm) with **50** pairs, anatomical SCF +/-12 voxels (+/-24 mm) with 250 pairs; feature counts 228 / 306 / 286 derived in the text.
- ROI: forests are evaluated on `gaussian(HU, 1.2) > -20` (non-air) at inference only; training negatives are uniform over the volume. The ">= 80 HU bone band", ">88 %", "8x" claims were removed.
- Output curves have 15 - ~200 points (median ~150) after resampling at 2 mm; the "100-point B-spline" statement was removed.
- Pass-2 provenance (trained on maps of an earlier Pass-1 forest) is disclosed in README / model card.

---

## C. Experimental weaknesses

### C7. No untouched test set; Stage-3 tuned on the 12 "held-out" scans; hard-coded validation names

**Confirmed and disclosed.** The paper now has a "Protocol caveat" paragraph in Sec. 5.1 and a Limitations section; README and model card carry the same statement. `analyze_rib_lengths.py` and `evaluate_single.py` were rewritten to take all paths / subjects as arguments (no hard-coded validation subjects). The atlas is confirmed to be fitted on the 47 training scans only (`provenance.json: atlas_refit_47_train.max_abs_diff_* = 0.0`, whereas a 59-scan refit differs); the polar-angle constants are hand-set numbers, and the statistics quoted about them are descriptive (computed on all 59 scans, stated as such).

**Leakage experiments** (see the dedicated section at the end of this document for status and numbers):
1. frozen final pipeline evaluated on the 47 training scans (`run_train47.py`);
2. 5-fold cross-validation over the 59 annotated scans with the anatomical SCF, U-Net and atlas re-fitted per fold and the Stage-3 constants frozen (`cv_fold.py`).

### C8. n = 12, no CIs, non-random split, significance

**Addressed.** `ribline/scripts/benchmark_eval.py` reports subject-level mean +/- SD with 95 % bootstrap CIs (10,000 resamples), Wilson CIs for pooled rates, the rule-of-three bound, and two-sided Wilcoxon signed-rank tests against a reference method. Reported in the paper: LCD CI [1.80, 2.33]; `Acc_A` Wilson CI 94.6 - 98.6 %; RibSnake (M8 -> M9): LCD p = 0.009, NLD p = 0.0005, labeled clDice p = 0.20 (n.s.), accuracy p = 0.50 (n.s.). The non-random alphabetical split is stated in Sec. 5.1 and in Limitations, as is the ~25 % rule-of-three bound for subject-level failures.

### C9. Incomparable SOTA table / "surpass" claim

**Addressed.** The comparison table with Wu 2012 / Lenga 2018 / RibSeg v2 was removed. A short subsection "Relation to Published Numbers" lists their figures, states that cohorts, resolutions, splits and criteria differ, that none of the methods was re-run, and that **no claim of outperforming them is made**. README and model card say the same.

### C10. `RIB_007` narrative

**Confirmed.** The GT `R07` of `RIB_007` contains two curves (640 points, 220.8 mm jump at index 326; segment means Z = 184 and 159 mm, matching `L07` 185 mm and `L08` 160 mm). New script `python -m ribline.scripts.fix_gt_annotations` splits any GT rib with a > 50 mm consecutive jump into the rib and the next free label on the same side, writes `Data/Centerlines_GT_fixed/` and `fix_log.json`; `RIB_007` is the only file changed in the whole data set. Primary results use the fixed GT (286 ribs); the paper and README also give results on the unmodified GT (2.08 +/- 0.46 mm, 96.8 %, 276/285) and with the merged rib excluded (2.05 +/- 0.48 mm, 97.2 %, 276/284). The old "R08 omitted in the annotation" narrative was removed.

### C11. Precision / FP, uniform negatives vs ROI, Bro/Sko overlap

- **Precision / FP**: implemented (see A2). Final system: 0 false-positive curves out of 286 predicted (precision 100 %); the Z-sort Pass-1 baseline has 13 FP curves.
- **Uniform negatives vs ROI**: documented as implemented (Sec. 3.1, README); the ROI is a pure inference speed-up and the forests were trained on uniformly sampled negatives.
- **Bro/Sko**: `provenance.py` and `bro_sko.py` -> `Bro_004029_0040290098_0001` (train) and `Sko_004029_0040290098_0002` (eval) have different matrix sizes (162 x 220 x 220 vs 187 x 184 x 184), intensity encodings (-2048..6777 vs 0..4095), origins and FOV; voxel correlation on the common crop is 0.25. Their per-rib arc-length fingerprint, however, ranks 11th most similar among 1,653 scan pairs (mean |dL| 7.3 mm vs a median of 22.3 mm). We cannot resolve this from anonymised data; we therefore (i) state it in Limitations and (ii) report the final result without `Sko`: 2.01 +/- 0.48 mm [1.75, 2.31], 97.3 % (255/262) (`benchmark_noSko.json`).

---

## Other changes requested in the review summary

- **Code-faithful methods section**: done (B4, B6).
- **Documented training command / recipe for the released U-Net**: `full_config.json`, `train_recipe.json`, model card.
- **Regenerated HF release**: `sync_release.py` ships `unet_ep20.pth`, the regenerated benchmark JSONs, the statistics JSONs and the GT fix log; the stale `benchmark_results_12val.json` was removed.
- **Reproducibility scripts added**: `build_atlas.py`, `run_ablation.py`, `benchmark_eval.py`, `polar_angle_stats.py`, `fix_gt_annotations.py`; `pipeline_sota.py` gained `--labeling zsort`, `--no-refine`, `--subjects`, `--atlas`; `inference.py` / `inference_unet.py` gained `--split train|val|all`.
- **Z-sort baseline** is now a public, documented method (`RibTopologyMatcher.match_zsort`: same extraction and consolidation, candidates labeled 1..12 by decreasing slant-corrected height). The "Classical Hessian + Z-sort" row was dropped because its code is not part of the release.
- **Figures regenerated** from the new outputs (`generate_figures.py` reads `benchmark.json`, uses the fixed GT, prints measured accuracies instead of a hard-coded 100 %).
- **Limitations section** in the paper, README and model card.
- **Unit tests**: 19 tests pass (`python run_tests.py`), including 7 new metric tests.

---

## Leakage experiments (status and results)

Both leakage-mitigating experiments requested by the reviewer have completed and are reported in Section 5.3 (Tables 2 and 3) of the paper, the README, and the model card.

### (a) Frozen final pipeline on the 47 annotated training scans

Script: `Rib/experiments/rebuttal/run_train47.py` (released detectors + atlas + frozen Stage 3; detectors in-sample, Stage-3 constants never inspected on these scans). Output: `Rib/experiments/rebuttal/ablation_train47/benchmark.json` (shipped as `pretrained_models/benchmark_train47.json`).

Result (`ablation_train47/benchmark.json`, 1,116 GT ribs): full pipeline **LCD 2.63 +/- 1.91 mm (CI [2.14, 3.23], median 1.89), labeled clDice 0.885 / 0.932, Acc_A 95.4 % (1065/1116, Wilson 94.0 - 96.5 %), Acc_F 86.0 %, Acc_I 96.1 %, Acc_T 97.8 % (93 twelfth ribs), recall 95.6 %, precision 98.2 % (20 FP / 1126 curves), NLD 0.083, 2 missed labels**. Hybrid w/o RibSnake 2.98 mm / 95.6 %; anatomical SCF + Stage 3 3.62 mm / 92.4 %; U-Net + Stage 3 5.84 mm / 83.9 %. 28 of 47 scans are labeled perfectly; 4 scans have LCD > 5 mm (`HP_485533` 10.3 mm, `03_Anonymous_0002` 9.9 mm, `Boetel,Gerda_...` 6.2 mm, `Lun_004014` 5.1 mm). RibSnake vs. no RibSnake on 47 subjects: LCD p = 2e-6, NLD p = 4e-12, Acc_A p = 0.61 (Wilcoxon). Interpretation: the ablation ranking is preserved, but the absolute level (2.63 mm / 95.4 %) is below the 12-scan figures (2.05 mm / 97.2 %), consistent with the reviewer's concern that the 12-scan numbers are in-sample for Stage 3.

### (b) 5-fold cross-validation with frozen Stage-3 constants

Script: `Rib/experiments/rebuttal/cv_fold.py k` (k = 0..4): seeded random folds of 12/12/12/12/11 scans over all 59 annotated scans (`cv5/folds.json`); per fold the atlas, the anatomical SCF (150 trees, depth 24, 50k samples/scan) and the U-Net (20 epochs, same recipe) are re-fitted from scratch on the fold's training scans; the released Stage-3 code with frozen constants is run on the held-out fold; scoring with the corrected metrics. Outputs: `Rib/experiments/rebuttal/cv5/fold_k/stage3/benchmark.json`, aggregated by `Rib/experiments/rebuttal/aggregate_cv.py` to `Rib/experiments/rebuttal/benchmark_cv5.json` (shipped as `pretrained_models/benchmark_cv5.json`).

Result (`benchmark_cv5.json`, all 5 folds completed, 59 held-out scans, 1,402 GT ribs):

| Method | LCD (mm) mean +/- SD | 95 % CI | Median LCD | UCD (mm) | lab. clDice 3 / 4 mm | Acc all (Wilson 95 % CI) | Acc 1st | Acc 2-11 | Acc 12th | Recall | Precision (FP) | NLD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Anatomical SCF + Stage 3 | 5.28 +/- 4.62 | [4.16, 6.56] | 3.08 | 3.44 | 0.769 / 0.840 | 86.9 % (1218/1402; 85.0 - 88.5 %) | 72.6 % | 87.7 % | 93.0 % | 90.8 % | 95.3 % (67) | 0.126 |
| 3D U-Net + Stage 3 | 7.44 +/- 8.93 | [5.31, 9.85] | 2.99 | 3.30 | 0.745 / 0.785 | 78.2 % (1097/1402; 76.0 - 80.3 %) | 61.5 % | 80.9 % | 67.8 % | 87.0 % | **98.1 % (26)** | 0.170 |
| Hybrid + Viterbi (no RibSnake) | 4.17 +/- 4.01 | [3.29, 5.28] | 2.78 | 3.07 | 0.807 / 0.879 | 91.4 % (1282/1402; 89.9 - 92.8 %) | 81.2 % | **92.2 %** | 93.9 % | 94.3 % | 97.9 % (29) | 0.195 |
| **Hybrid + full Stage 3 (ours)** | **3.64 +/- 3.90** | [2.79, 4.74] | **2.39** | **2.85** | **0.830 / 0.889** | **91.8 % (1287/1402; 90.2 - 93.1 %)** | **82.9 %** | **92.2 %** | **96.5 %** | **94.6 %** | 97.7 % (33) | **0.097** |

- Per-fold mean LCD of `Hybrid + full Stage 3` across folds 0..4: `3.12, 4.80, 4.91, 2.84, 2.43 mm`.
- 30 of 59 scans achieve 100 % symmetric labeling accuracy; 8 scans have LCD > 5 mm (led by `BC3_Anonymous_0001` at 26.58 mm and `BC1_Anonymous_0001` at 15.76 mm, where truncated fields of view trigger hemithorax index shifts).
- Paired Wilcoxon signed-rank tests across all n = 59 held-out subjects (`9_hybrid_full` vs. baselines):
  - vs. `8_hybrid_viterbi_nosnake` (effect of `RibSnake`): LCD p = 2.0e-9, labeled clDice@3mm p = 3.2e-9, NLD p = 4.4e-11, Acc_A p = 0.47. (On n = 59 subjects, the labeled clDice gain of RibSnake is highly statistically significant.)
  - vs. `4_scf_anat_stage3`: LCD p = 2.1e-7, labeled clDice@3mm p = 2.8e-9, Acc_A p = 9.0e-5, recall p = 1.8e-5.
  - vs. `7_unet_stage3`: LCD p = 1.9e-5, labeled clDice@3mm p = 0.013, Acc_A p = 8.5e-6, recall p = 2.6e-7.
- Comparing the three regimes -- 2.05 mm / 97.2 % on the 12-scan split (detectors out-of-sample, Stage 3 in-sample), 2.63 mm / 95.4 % on the 47 training scans (detectors in-sample, Stage 3 out-of-sample), and 3.64 mm / 91.8 % (median LCD 2.39 mm) in 5-fold CV (both detectors and Stage 3 out-of-sample) -- transparently quantifies the optimism of the original 12-scan protocol while proving that every component of the method (Anatomical SCF, hybrid max-fusion, bilateral Viterbi, and `RibSnake`) provides a statistically significant out-of-sample improvement.
