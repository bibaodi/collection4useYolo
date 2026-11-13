# Thyroid nodule benign/malign classification (EfficientNet)

Image-level and nodule-level benign/malign classification of thyroid
ultrasound images using an EfficientNet-based CNN, with OOF
(out-of-fold) suspicious-sample analysis and active learning.

Originally `ml_task4efficientNet` (`etonb/task4efficientNet`).

## Pipeline overview

1. **Ultrasound-region detection** — YOLO detector crops the ultrasound
   region from the original image (`us_image`).
2. **Nodule detection/segmentation** — YOLO segmentation is preferred; the
   detector is the fallback. Boxes above threshold are expanded by 2W×2H to
   produce `nodule_image`.
3. **Nodule classification** — EfficientNet classifies `nodule_image` as
   benign/malign.
4. **Nodule-level aggregation** — transverse and longitudinal views are
   combined with equal weights to produce the nodule-level probability:
   `P_positive = (1 - p_neg_view1) * 0.5 + p_pos_view2 * 0.5`.

## Layout

| Path | Purpose |
|---|---|
| `srcs/train_nodule_feature_cnn_model_v75.py` | Main EfficientNet training script |
| `srcs/trainWithCaseAsOneImg.py` | Alternative training (one case = one image) |
| `srcs/create_sop4_training_images_backup.py` | Build cropped training images |
| `srcs/create_sop4_verify_images.py` | Build cropped independent-verification images |
| `srcs/OOF_analysis_sop7_all_match_tr13_0927.py` | OOF analysis with TR1-3 support |
| `srcs/cleanData_sop5_OOF_0928.py` | OOF-based suspicious-sample cleaning |
| `srcs/inference_verify.py` | Inference on the independent verification set |
| `srcs/predict_and_human_confirm.py` | Prediction + human confirmation |
| `srcs/active_learning_utils.py` | Active-learning utilities |
| `srcs/annotation_interface_simple.py` | CLI review interface for uncertain samples |
| `srcs/improved_logger.py` | Logging helper |
| `srcs/yolo_dataset_utils.py` | Save a DataFrame as a YOLO classification dataset |
| `srcs/demo_temperatureScaling.py` | Temperature-scaling demo |
| `srcs/image_subset_creator.py` | Build image subsets from a CSV list |
| `scripts/` | Dataset analysis and environment helpers |
| `models/config.yaml` | Thresholds, feature mapping, model/architecture config |
| `models/nodule_feature_cnn_v75/` | Training/verification reports |
| `dataset/` | Dataset tables (CSV/JSON) used by the scripts |
| `notebooks/` | Experiment notebooks (FIR, decision tree, pandas) |
| `docs/` | Active-learning reports and YOLO-format notes |
| `environment.yml` | Conda environment |

Large results and logs moved to `../data/nodule-classification/` (git-ignored).

## Environment

```bash
conda env create -f environment.yml
conda activate <env>
```

## Usage

Run from this directory so the relative `dataset/` and `models/` paths
resolve:

```bash
cd nodule-classification

# Train the classifier
python srcs/train_nodule_feature_cnn_model_v75.py

# OOF / suspicious-sample analysis
python srcs/OOF_analysis_sop7_all_match_tr13_0927.py
python srcs/cleanData_sop5_OOF_0928.py

# Independent verification-set inference
python srcs/inference_verify.py

# Dataset analysis
python scripts/count_type_values.py
python scripts/detailed_type_analysis.py
```

Several scripts contain absolute paths from the author's machines
(`/Users/mouxiaoyong/...`, `/home/eton/job-trainThyUS/...`,
`/train/data/...`); adjust the `CONFIG` block at the top before running.
`project_root` is now derived from `__file__`, so the repository-relative
`dataset/` and `models/` paths work as-is.

## Dataset notes (from the original Chinese readme)

- **dataset_images** — `images/` original, `us_images/` after ultrasound
  cropping, `nodule_images/` after segmentation+detection, all keyed by
  `access_no/sop_uid.jpg`.
- **dataset tables** — `dataset/all_verify_sop_with_predictions.csv` is the
  independent verification set; `dataset/0809v3*.csv` are 0809-series
  subsets; `dataset/all_features_mapping_numer_v4.json` maps categorical
  features.
- **0809 series versions** — v3 is the gold standard; v4 is the 0809 subset
  inside the training table; `v4_subset` removes OOF < 0.2 samples.
- **Training table** — remove `02.202408`/`02.202409` access numbers (used
  for validation) and `p_true < 0.2` (OOF-suspected) before training.

See `README.zh.md` for the original full Chinese description.

## Cleanup performed during the merge

- Removed `patchs/` (temporary memory-fix scripts) and
  `SYNTAX_FIX_SUMMARY.md`.
- Removed `readme.md.bak251019`.
- Consolidated four duplicate YOLO-format documents into
  `docs/yolo_dataset_utils.md`.
- Moved `someresults/` (~113 MB) and `logs/` to
  `../data/nodule-classification/`.
- Moved notebooks from `srcs/` to `notebooks/`.
- Fixed `project_root` to derive from `__file__` instead of a teammate's
  hardcoded `/Users/mouxiaoyong/...` path.
