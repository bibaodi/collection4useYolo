# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A flat collection of standalone scripts for training, validating, and running **Ultralytics YOLO** models on medical ultrasound datasets (thyroid gland/nodule, carotid, esophagus, plus benign/malign and thyroid-property classifiers). There is no package, build system, test suite, or linter config: each top-level `*.py` is an entry point run directly with `python`.

Environment: Python from conda base at `/mnt/datas/miniconda3`, with `ultralytics 8.3.250`, `torch`, `opencv-python`, `onnxruntime`, `tqdm`, `Pillow`, `matplotlib`. `predict_segNoCuttingToBox.py` depends on Ultralytics internals, so it is sensitive to the installed version.

## Commands

Most training scripts take **no CLI arguments**. Configure them by editing the dataset constants at the top (see "Configuration by editing"), then run:

```bash
python trainSegment_v01.py          # segmentation
python trainDetectThyNodu_v01.py    # detection
python trainClassifyBoM_v01.py      # classification
```

Scripts that do take arguments:

```bash
# Batch-classify every image in a folder -> CSV
python predict_classify.py --model runs/classify/train/weights/best.pt \
    --images /path/to/imgs --output out.csv [--path-style full|basename|stem]

# Validate a classification checkpoint (dataset path is set in the __main__ block; prompts y/n)
python validateModel_Classify.py <model.pt>

# Batch segment + crop ultrasound images
python useSegModelCropUSImg2NoduleOnlyRegionImg.py -m best.pt -d <img-dir> -o <out-dir> \
    [--save-types all|isolated|contour|crop|enlarged-crop] [--enlarge-percent 100]

# Split datasets (classification: per-class dirs; segmentation: images/ + labels/ pairs)
python splitClsDataset3.py <origin-dir> 0.8 <output-name-or-dir>
python splitSegDataset-symOp.py <input-dir> 0.8 <output-dir>

# Raw ONNX / .pt inference reproducer for segmentation
python predict_segNoCuttingToBox.py --model best.onnx --source image.jpg [--conf 0.55 --iou 0.1]
```

Dataset-curation helpers in `scripts/` take the dataset or label directory as `$1` (e.g. `scripts/checkImageFormatSuffixMatch.sh <dir>`, `scripts/showLabel_ClassID_Not0.sh <label-dir>`, `scripts/checkYoloSeg_DSLabelCorrect.py <dataset-dir>`). Most prompt for `y/n` before moving or modifying files.

## Architecture and conventions

### Configuration by editing
Training and quick-predict scripts define `datasetHome`, `datasetName`, and `trainMessage`, then reassign `datasetName`/`trainMessage` several times; the **last assignment wins**. Changing an experiment means editing and commenting/uncommenting those lines. `trainMessage` is a free-text run-history note (e.g. "10th make train&val data 50-50...") — keep that convention when adding a run. Paths are hardcoded for the author's machine (`/mnt/xin1data/...`, `/home/eton/...`) and must be adjusted locally.

### Datasets live outside the repo
Dataset YAMLs (e.g. `segThy3ObjGlandNoduCarotid_v01.yaml`) are resolved as `datasetHome + datasetName`, typically under `/mnt/xin1data/yolo-ultralytics-250101/datasets/` or `../datasets/`. Standard YOLO layout: `<dataset>/images/{train,val}`, `<dataset>/labels/{train,val}`, plus a YAML with `path`/`train`/`val`/`names`. Label formats:
- detection: 5 fields — `class x_center y_center width height`
- segmentation: `class` + polygon coordinates (at least 3 points, i.e. >= 7 fields)

### Training pipeline
Each `train*.py` follows the same shape: `YOLO(pretrained .pt)` -> `model.train(data=..., epochs=..., imgsz=...)` -> `model.val()` -> `model.predict(sample)` -> optional `model.export("onnx")`. Artifacts land in `runs/<task>/<name>/` and are gitignored (`runs/`, `outputs/`, `*.pt`, `checkpoints/`). `imgsz` is 96 for detection/classification and 224 for segmentation.

### Segmentation output: cropped vs. uncropped
`model.predict()` crops masks to their bounding boxes. `predict_segNoCuttingToBox.py` re-implements postprocessing (`scale_masks`, `filter_mask_in_boxes`, `YOLOSeg`) so masks keep their full contour for both ONNX Runtime and `.pt` inputs; it loads the checkpoint's `ema`/`model` manually and calls `ultralytics.utils.ops.non_max_suppression` / `scale_boxes` and `engine.results.Results` directly.

### Dataset curation
`scripts/` contains the data-prep and cleaning tools: class merging (`mergeBoMLabels2OneThyNodu.sh` collapses BoM classes 1 -> 0), train/val splits (copy or symlink), label validation (moves invalid label/image pairs into error folders), file-count/statistics helpers, S3 upload, and model packaging (`createModelPackage.sh`). Several operate on YOLO datasets recursively and follow symlinks with `find -L`.

### `22-demos/`
Unrelated, self-contained demos (AVX2/CUDA, JAX, MCP, Docker, PID, endianness). Not part of the YOLO workflow; do not run or modify them as part of it.

### Verification
There are no tests. Validate changes by running the relevant train/predict script and inspecting its `runs/` output and log files (several scripts write timestamped `.log` files in the working directory).
