# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this repository is

A consolidated monorepo for medical-ultrasound deep learning. It combines
three previously separate repositories (a fourth, `ml_dl2linkNet`, is pending):

| Top-level directory | Origin | Purpose |
|---|---|---|
| `yolo/` | `collection4useYolo` (this repo) | Ultralytics YOLO training, prediction, dataset splitting |
| `thyroid-data/` | `ml_thyroid.01_etonTrainCodes` | Thyroid annotation/data preparation (LabelMe, PACS, DICOM) |
| `nodule-classification/` | `ml_task4efficientNet` | EfficientNet benign/malign nodule classification |
| `scripts/` | shared | Cross-project dataset utilities |
| `demos/` | `22-demos` | Self-contained, unrelated demos |
| `data/` | shared | **Git-ignored** local datasets/results (see `data/README.md`) |
| `docs/` | shared | Design notes; `docs/history/` archives old notebooks/scripts |

There is no package, build system, test suite, or linter config. Each entry
point is run directly with `python` (or `bash`). The original repositories
remain available on their remotes; this monorepo is a snapshot merge.

## Environment

Python from conda base at `/mnt/datas/miniconda3`:

- `ultralytics 8.3.250`, `torch`, `opencv-python`, `onnxruntime`, `tqdm`,
  `Pillow`, `matplotlib`
- `pandas`, `scikit-learn`, `openpyxl` for data preparation
- `pydicom`, `itk` for DICOM/ultrasound preprocessing

`nodule-classification/environment.yml` pins that project's additional
dependencies. Paths are hardcoded for the author's machines
(`/mnt/xin1data/...`, `/home/eton/...`, `/mnt/datas/...`) and must be adjusted
locally.

## Commands

### YOLO (`yolo/`)

Most training scripts take **no CLI arguments**; configure them by editing the
dataset constants at the top (`datasetHome`, `datasetName`, `trainMessage`),
then run:

```bash
python yolo/trainSegment_v01.py          # segmentation
python yolo/trainDetectThyNodu_v01.py    # detection
python yolo/trainClassifyBoM_v01.py      # classification
```

Scripts that take arguments:

```bash
# Batch-classify every image in a folder -> CSV
python yolo/predict_classify.py --model runs/classify/train/weights/best.pt \
    --images /path/to/imgs --output out.csv [--path-style full|basename|stem]

# Validate a classification checkpoint
python yolo/validateModel_Classify.py <model.pt>

# Batch segment + crop ultrasound images
python yolo/useSegModelCropUSImg2NoduleOnlyRegionImg.py -m best.pt -d <img-dir> -o <out-dir> \
    [--save-types all|isolated|contour|crop|enlarged-crop] [--enlarge-percent 100]

# Split datasets (classification: per-class dirs; segmentation: images/ + labels/ pairs)
python yolo/splitClsDataset3.py <origin-dir> 0.8 <output-name-or-dir>
python yolo/splitSegDataset-symOp.py <input-dir> 0.8 <output-dir>

# Raw ONNX / .pt inference reproducer for segmentation
python yolo/predict_segNoCuttingToBox.py --model best.onnx --source image.jpg [--conf 0.55 --iou 0.1]
```

### Thyroid data (`thyroid-data/`)

Scripts live under `thyroid-data/srcs/`. Many rely on relative imports
(`../utils`), so run them from their own directory or via `python <path>`:

```bash
# LabelMe -> YOLO (detection or segmentation)
python thyroid-data/srcs/annotationFormatConverter/convertLbmeFmt_2Yolo.py

# PACS 301PX -> LabelMe
python thyroid-data/srcs/annotationFormatConverter/convert301PX_2Lbmeformat.py <cases-dir>

# Auto-annotate with an existing YOLO model
python thyroid-data/srcs/autoAnnotate/annotateByMLModel.py \
    --model_file model.pt --model_type segmentation \
    --input_folder in/ --output_folder out/

# Dataset statistics / spreadsheet matching
python thyroid-data/srcs/statistics_train_data.py
python thyroid-data/srcs/dataProcess/matchCsvWithSpreadsheet.py --help
```

`42scripts/` holds the operation/shell scripts. `tests/` contains
logger smoke tests. Notebooks were archived to
`docs/history/thyroid-notebooks/`.

### Nodule classification (`nodule-classification/`)

Run from the `nodule-classification/` directory so relative `dataset/` and
`models/` paths resolve:

```bash
cd nodule-classification

# Train the EfficientNet nodule classifier
python srcs/train_nodule_feature_cnn_model_v75.py

# OOF / suspicious-sample analysis
python srcs/OOF_analysis_sop7_all_match_tr13_0927.py
python srcs/cleanData_sop5_OOF_0928.py

# Independent verification-set inference
python srcs/inference_verify.py

# Dataset utilities
python scripts/count_type_values.py
python scripts/detailed_type_analysis.py
```

### Shared scripts (`scripts/`)

Take the dataset/label directory as `$1`; most prompt `y/n` before modifying
files:

```bash
scripts/findFieldsNotCorrect.sh <label-dir>
scripts/sendThynodulesDataSetToS3.sh <dataset.zip>
python scripts/filterTestPairsFromTrainDataset.py \
    --test-dir <test> --train-dir <train> --outdir ./reports --dry-run
```

## Architecture and conventions

### Configuration by editing (`yolo/`)
Training scripts define `datasetHome`, `datasetName`, `trainMessage`, then
reassign `datasetName`/`trainMessage` several times; the **last assignment
wins**. `trainMessage` is a free-text run-history note — keep that convention
when adding a run.

### Datasets live outside the repo
Dataset YAMLs (e.g. `segThy3ObjGlandNoduCarotid_v01.yaml`) resolve as
`datasetHome + datasetName`, typically under
`/mnt/xin1data/yolo-ultralytics-250101/datasets/` or `../datasets/`. Label
formats:

- detection: 5 fields — `class x_center y_center width height`
- segmentation: `class` + polygon coordinates (≥ 3 points, i.e. ≥ 7 fields)
- classification: one class directory per label under `train/` and `val/`

### Training pipeline (`yolo/`)
`YOLO(pretrained .pt)` -> `model.train(...)` -> `model.val()` ->
`model.predict(sample)` -> optional `model.export("onnx")`. Artifacts land in
`runs/<task>/<name>/` (git-ignored). `imgsz` is 96 for
detection/classification and 224 for segmentation.

### Segmentation output: cropped vs. uncropped
`model.predict()` crops masks to their bounding boxes.
`yolo/predict_segNoCuttingToBox.py` re-implements postprocessing
(`scale_masks`, `filter_mask_in_boxes`, `YOLOSeg`) so masks keep their full
contour for both ONNX Runtime and `.pt` inputs; it loads the checkpoint's
`ema`/`model` manually and calls
`ultralytics.utils.ops.non_max_suppression` / `scale_boxes` and
`engine.results.Results` directly. It is sensitive to the installed
Ultralytics version.

### Thyroid annotation pipeline (`thyroid-data/`)
`srcs/annotationFormatConverter/` converts between delivery formats and
LabelMe/YOLO; `srcs/dataProcess/` matches spreadsheets and builds classification
datasets; `srcs/autoAnnotate/` runs model-assisted labeling; `srcs/utils/`
holds the shared logger (`glog`) and helpers. Several scripts use
`sys.path.append(.../'utils')`, so preserve the `srcs/<subdir>/` layout.

### Nodule classification pipeline (`nodule-classification/`)
`srcs/create_sop4_*.py` crop ultrasound/nodule images using YOLO detectors;
`srcs/train_nodule_feature_cnn_model_v75.py` trains the EfficientNet model;
`srcs/OOF_analysis_*` and `srcs/cleanData_*` perform out-of-fold suspicious-
sample analysis; `srcs/inference_verify.py` and
`srcs/predict_and_human_confirm.py` evaluate on independent sets;
`srcs/active_learning_utils.py` + `annotation_interface_simple.py` implement
active learning. See `nodule-classification/docs/`.

### `data/`
Large or generated artifacts moved out of git during the merge
(spreadsheets, result CSVs, logs). Everything under `data/` is ignored except
its `README.md`.

### `demos/`
Unrelated, self-contained demos (AVX2/CUDA, JAX, MCP/skills, Docker, PID,
endianness). Do not couple them to the ML projects.

## Verification

There are no tests. Validate changes by running the relevant script and
inspecting its `runs/` output, generated CSV/report, or timestamped `.log` file
in the working directory. For Python syntax, run
`python -m py_compile <file>`.

## Merge roadmap

- **Step 1 (done):** `ml_thyroid.01_etonTrainCodes` -> `thyroid-data/`;
  `ml_task4efficientNet` -> `nodule-classification/`.
- **Step 2 (pending):** merge `ml_dl2linkNet` (LinkNet/UNet/SegFormer package,
  YOLO26 segmentation/classification, evaluation tools, deployment utilities).
  See the root `README.md` for the open decisions.
