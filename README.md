# Optical Degradation and Corruption-Aware Training of Convolutional Neural Networks for Diabetic Retinopathy Grading

This repository contains the code and results of a study on how simulated optical degradations of fundus photographs affect a five-grade diabetic retinopathy (DR) classifier, and whether training with degraded images protects it.

The network is an ImageNet-pretrained EfficientNet-B1 trained on the APTOS 2019 dataset. The clean-image pipeline follows Ghadai et al. (2026). Each fundus image is degraded with one of three physical models, each at three severity levels:

- defocus (Gaussian point spread function)
- scattering haze (single-scattering transmission model)
- sensor noise (additive Gaussian noise)

Two training regimes are compared under the same folds, seeds and settings.

## Main findings

- **Clean images.** The clean-trained model (Regime A) and the corruption-aware model (Regime B) perform the same. Out-of-fold QWK is 0.905 and 0.902 (paired difference −0.004, 95% CI −0.011 to 0.004).
- **Defocus and haze.** Both make the clean-trained model grade too low, so it misses referable disease.
  - Defocus at σ = 3 px: QWK 0.442, referral sensitivity 0.523.
  - Haze at t = 0.4: QWK 0.658, referral sensitivity 0.710.
- **Sensor noise.** Even mild noise has the opposite effect: most images are pushed into the Severe grade.
  - At s = 0.03, QWK falls to 0.204.
  - At s = 0.06, 92–96% of images of every grade are predicted Severe, and referral specificity drops to 0.025.
- **Temperature scaling.** It lowers the overconfidence of the clean-trained model (ECE 0.645 → 0.243 at s = 0.10). It cannot fix the errors, because it never changes the predicted grade.
- **Corruption-aware training.** Regime B keeps QWK between 0.80 and 0.90, referral specificity at 0.93 or higher, and ECE at or below 0.055 under all ten test conditions.

## Method

### Processing chain

```
crop retina -> pad to square -> resize 320 x 320 -> [optical degradation] -> CLAHE (L* channel) -> augmentation -> normalisation
```

The degradation is applied before CLAHE. This follows the physical order: the image is acquired first, then processed.

### Optical degradation model

The acquired image `g` is modelled as a degraded version of the ideal image `f`:

```
g = T[h * f] + n
```

Each degradation changes only one term:

| Degradation | Model | Severity 1 | Severity 2 | Severity 3 |
|---|---|---|---|---|
| Defocus | Gaussian PSF, σ in px at 224-px scale (scaled by 320/224) | σ = 1 | σ = 2 | σ = 3 |
| Haze | g = f·t + A(1 − t), A = 0.8 | t = 0.8 | t = 0.6 | t = 0.4 |
| Sensor noise | g = clip(f + n), n ~ N(0, s²) | s = 0.03 | s = 0.06 | s = 0.10 |

Test images are degraded with a fixed seed for each image and condition (`EVAL_SEED = 20260101`). Every model therefore sees exactly the same pixels.

### Network and training

| Setting | Value |
|---|---|
| Backbone | EfficientNet-B1, ImageNet-1K weights (torchvision) |
| Head | Dropout 0.2 → FC 512 + ReLU → Dropout 0.2 → FC 5 |
| Loss | Focal loss (γ = 1), class weights ∝ n_c^−1/2 (mean 1), label smoothing 0.05 |
| Augmentation | Flips, rotation ±20°, shift 5%, scale 0.9–1.1, brightness/contrast ±15%, Mixup (α = 0.4, p = 0.25) |
| Optimiser | AdamW, lr 2e-4 (backbone) / 1e-3 (head), weight decay 1e-4, one-cycle schedule (10% warm-up) |
| Training | 30 epochs, batch 16, EMA 0.998, gradient clipping 1.0, mixed precision |
| Checkpoint | Best QWK + macro-F1 of the EMA model, epochs 8–30 |
| Validation | Stratified 5-fold CV on 3296 development images (seed 42 + fold); untouched test set of 366 images |

### Training regimes

- **Regime A (clean training):** clean images only; checkpoint chosen on the clean held-out fold.
- **Regime B (corruption-aware training):** each training image is degraded with probability 0.5, with random type and severity. The checkpoint is chosen on a fixed held-out fold in which half the images are degraded (`VALMIX_SEED = 777`).

Two additional clean-image runs (with and without CLAHE, each with and without 5-view test-time augmentation) reproduce the base pipeline for comparison.

### Evaluation

- **Test conditions:** 10 in total (clean, plus 3 degradations × 3 severities).
- **Data:**
  - pooled out-of-fold predictions (n = 3296);
  - five-model ensemble on the test set (n = 366).
- **Metrics:**
  - quadratic weighted kappa (QWK)
  - accuracy and macro-F1
  - referral sensitivity and specificity (grade ≥ 2)
  - under- and over-grading rates
  - expected calibration error (ECE), before and after cross-fitted temperature scaling
- **Statistics:**
  - stratified bootstrap (1000 resamples)
  - paired bootstrap for Regime B − Regime A
  - Nadeau–Bengio corrected intervals for fold means

## Results

### Clean images

| Configuration | Test accuracy, % | Test macro-F1 | Test QWK | OOF QWK (95% CI) |
|---|---|---|---|---|
| Ghadai et al. (2026), CLAHE + TTA | 84.97 | 0.663 | – | – |
| No-CLAHE run | 85.79 | 0.694 | 0.912 | 0.912 (0.902–0.921) |
| CLAHE run + TTA | 84.97 | 0.695 | 0.910 | 0.911 (0.901–0.920) |
| Regime A (clean training) | 84.97 | 0.692 | 0.906 | 0.905 (0.896–0.915) |
| Regime B (corruption-aware) | 84.97 | 0.672 | 0.891 | 0.902 (0.890–0.912) |

### Robustness (out-of-fold QWK)

| Condition | Regime A | Regime B | Δ (B − A), 95% CI |
|---|---|---|---|
| Clean | 0.905 | 0.902 | −0.004 (−0.011 to 0.004) |
| Defocus, σ = 1 / 2 / 3 | 0.871 / 0.725 / 0.442 | 0.887 / 0.876 / 0.852 | up to 0.410 (0.381–0.441) |
| Haze, t = 0.8 / 0.6 / 0.4 | 0.895 / 0.851 / 0.658 | 0.901 / 0.899 / 0.892 | up to 0.234 (0.211–0.257) |
| Noise, s = 0.03 / 0.06 / 0.10 | 0.204 / 0.005 / 0.001 | 0.885 / 0.850 / 0.799 | up to 0.846 (0.829–0.861) |

## Requirements

- Python 3.13, PyTorch 2.11, torchvision
- OpenCV, NumPy, pandas, SciPy, scikit-learn, matplotlib
- One NVIDIA GPU (experiments were run on a Kaggle Tesla T4; one fold takes about 11–14 min)

```bash
pip install torch torchvision opencv-python numpy pandas scipy scikit-learn matplotlib
```

## Data

The study uses the public [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset: 3662 graded fundus photographs, with 1805 No DR, 370 Mild, 999 Moderate, 193 Severe and 295 Proliferative DR. The images are not included in this repository. Download them from Kaggle and set the data paths in the configuration (`ROOT`, `TRAIN_DIR`, `TEST_DIR`, `TRAIN_CSV`, `TEST_CSV`).

## Usage

All settings live in one configuration block at the top of the experiment script or notebook.

1. Set the data paths.
2. Choose the regime:
   - `REGIME = "A"` for clean training;
   - `REGIME = "B"` for corruption-aware training (`P_CORRUPT = 0.5`).
3. Run the script. It trains the five folds, evaluates every model under the ten conditions and writes:
   - `v4_robust_<REGIME>_results.csv`: all metrics (overall, per class, confusion matrices, calibration bins, bootstrap intervals, fold summaries) in long format;
   - `v4_robust_<REGIME>_probs.pkl`: predicted probabilities for every image, model and condition, for paired analyses;
   - figures of the degradation grid and the pipeline stages.

The degradation parameters (`DEFOCUS_SIGMA`, `HAZE_T`, `HAZE_A`, `NOISE_STD`) and seeds (`SEED`, `EVAL_SEED`, `VALMIX_SEED`) are stored with every result file, so each number can be traced to its configuration. With `RESUME = True`, an interrupted run continues from the last finished fold.

## Limitations

- The degradations are synthetic. Real defocus, cataract and sensor noise are more complex.
- Only one dataset and one architecture were tested. The test set contains only 17 Severe images.
- Regime B was tested on the same degradation types it saw during training.

## Citation

If you use this code or these results, please cite the paper (details will be added after publication):

```bibtex
@article{optical_degradation_dr_2026,
  title   = {Optical Degradation and Corruption-Aware Training of Convolutional Neural Networks for Diabetic Retinopathy Grading},
  author  = {[Authors]},
  journal = {Optical Memory and Neural Networks},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the base pipeline:

> Ghadai, K.K., Nayak, S.K., and Senapati, B.R., A clinically interpretable deep learning pipeline for diabetic retinopathy classification using EfficientNet, advanced data augmentation and GradCAM, *BMC Ophthalmology*, 2026, vol. 26, p. 406. https://doi.org/10.1186/s12886-026-04897-4

## License

[Add a license, for example MIT for the code. The APTOS 2019 images remain under the Kaggle competition terms.]
