# Medical Ultrasound ML Toolkit

A consolidated monorepo for medical-ultrasound deep-learning work: YOLO
training/inference, thyroid annotation-data preparation, and EfficientNet
nodule classification.

> **Merge status (2026-11-13)**
>
> | Source repository | Status | Location |
> |---|---|---|
> | `collection4useYolo` (this repo) | merged | `yolo/`, `scripts/`, `demos/` |
> | `thyroid_train01` (`ml_thyroid.01_etonTrainCodes`) | merged (snapshot) | `thyroid-data/` |
> | `task4efficientNet` (`ml_task4efficientNet`) | merged (snapshot) | `nodule-classification/` |
> | `dl2_linknet` (`ml_dl2linkNet`) | **pending — step 2** | — |
>
> Snapshots contain the latest committed state of each source repository.
> Per-project git history was intentionally not preserved; the originals remain
> on their remotes (`bibaodi/thyroid_train01`, `etonb/task4efficientNet`, …).

## Repository layout

| Directory | What it contains |
|---|---|
| [`yolo/`](yolo/README.md) | Ultralytics YOLO train / predict / split entry points for ultrasound segmentation, detection and classification datasets. |
| [`thyroid-data/`](thyroid-data/README.md) | Thyroid annotation-data pipeline: LabelMe/PACS/Xiaobai format conversion, DICOM extraction, auto-annotation, spreadsheet matching and dataset statistics. |
| [`nodule-classification/`](nodule-classification/README.md) | EfficientNet-based benign/malign thyroid-nodule classification: dataset creation, training, OOF analysis, active learning and independent-set inference. |
| [`scripts/`](scripts/README.md) | Cross-project dataset utilities (leak filtering, label validation, S3 upload). |
| [`demos/`](demos/README.md) | Self-contained demos (AVX2/CUDA, JAX, MCP/skills, Docker, PID, endianness, misc). |
| `data/` | **Local-only, git-ignored** large datasets/results moved out of history. See [`data/README.md`](data/README.md). |
| `docs/` | Design notes and run reports; `docs/history/` holds archived notebooks and operation scripts. |

Each top-level project directory has its own `README.md` with commands and
conventions. `CLAUDE.md` is the short orientation guide for AI assistants.

## Environment

The common environment is a conda install at `/mnt/datas/miniconda3` with:

- Python 3.10+
- `ultralytics 8.3.250` (the `yolo/` scripts depend on this version)
- `torch`, `opencv-python`, `onnxruntime`, `tqdm`, `Pillow`, `matplotlib`
- `pandas`, `scikit-learn`, `openpyxl` for the data-preparation projects
- `pydicom`, `itk` for DICOM/ultrasound preprocessing

`nodule-classification/environment.yml` and the `requirements*.txt` files in
each project pin additional, project-specific dependencies. Paths in the
scripts are hardcoded for the author's machines (`/mnt/xin1data/...`,
`/home/eton/...`, `/mnt/datas/...`) and must be adjusted locally.

## Dataset conventions

Datasets live **outside** the repository (e.g. under
`/mnt/xin1data/yolo-ultralytics-250101/datasets/`). Standard YOLO layout:

```
<dataset>/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── <dataset>.yaml          # path / train / val / names
```

Label formats:

- **Detection** — 5 fields: `class x_center y_center width height`
- **Segmentation** — `class` + polygon coordinates (≥ 3 points, i.e. ≥ 7 fields)
- **Classification** — one class directory per label under `train/` and `val/`

## Quick start per project

```bash
# YOLO segmentation / detection / classification training
python yolo/trainSegment_v01.py
python yolo/trainDetectThyNodu_v01.py
python yolo/trainClassifyBoM_v01.py

# Batch classify a folder into CSV
python yolo/predict_classify.py --model runs/classify/train/weights/best.pt \
    --images /path/to/imgs --output out.csv

# Convert LabelMe annotations to YOLO
python thyroid-data/srcs/annotationFormatConverter/convertLbmeFmt_2Yolo.py

# Train the nodule classifier (run from nodule-classification/)
cd nodule-classification
python srcs/train_nodule_feature_cnn_model_v75.py
```

## Cleanup performed during the merge

The following were removed or relocated because they were generated,
machine-specific, superseded, or too large for git:

| Removed / relocated | Reason |
|---|---|
| `demos/avx2-demos/build/`, `__pycache__/`, `*.pyc` | Build/runtime artifacts |
| `.DS_Store`, `.marscode/`, `readme.md.bak251019` | Machine-specific / backup |
| `nodule-classification/patchs/`, `SYNTAX_FIX_SUMMARY.md` | Temporary AI working notes |
| `nodule-classification/QUICK_REFERENCE-YOLO_FORMAT_SOLUTION.md`, `YOLO_FORMAT_SOLUTION.md`, `yolo_structure_example.txt`, `srcs/README_yolo_utils.md` | Duplicated; consolidated into `nodule-classification/docs/yolo_dataset_utils.md` |
| `thyroid-data/srcs/annotationFormatConverter/convert301PACS_formatOriginScreenshot.py` | Superseded by `convert301PX_2Lbmeformat.py` |
| `thyroid-data/srcs/annotationFormatConverter/adjustJsonCoordinates_basedCropedUsImgRectangle.py` | Superseded by `cropScreenshotAndAdjustJson.py` |
| `thyroid-data/srcs/annotationFormatConverter/convertXiaobaiNodulesFormartTolabelme.py` | Superseded by `srcs/convert_ThyNodulesJson.py` |
| `scripts/rm_top700.sh`, `split.sh`, `get_top_700_lines.sh`, `moveHeadN2NewFolder.sh` | One-off destructive / superseded |
| `yolo/splitClsDataset.py`, `yolo/splitClsDataset2.py` | Superseded by `splitClsDataset3.py` |
| `demos/demo01.py`, `demos/demo02.py` | Superseded by `demo03-thynodu.py` |
| `thyroid-data/42scripts/findFieldsNotCorrect.sh`, `sendThynodulesDataSetToS3.sh` | Duplicated in top-level `scripts/` |
| `nodule-classification/someresults/`, `logs/` | Moved to git-ignored `data/` |
| `thyroid-data/screened_tirads_bethesda_records.csv`, `thyroidNodulesStatistics-*.xlsx`, `srcs/processImgs/output-*` | Moved to git-ignored `data/` |
| `thyroid-data/jupyterNotebooks/`, `operation.history/`, `ref_srcs/` | Archived under `docs/history/` |

## Step 2 — merging `ml_dl2linkNet` (not started)

`ml_dl2linkNet` is a 13 GB, 380-commit repository containing the
LinkNet/UNet/SegFormer PyTorch package, YOLO26 segmentation/classification
pipelines, evaluation tools and deployment utilities. Before merging it will
need the same treatment:

1. Decide whether its history should be preserved (its `.git` is 610 MB).
2. Keep `runs/` (9.5 GB), `inference_results*/` (1.3 GB) and other generated
   outputs out of git; they are already ignored by its `.gitignore`.
3. Map its top-level packages (`linknet-torch/`, `yolo26_seg/`, `yolo26_sem/`,
   `yolo26_classify/`, `uisbp/`, `model-evaluate/`, `tools/`, `scripts/`) into
   a `segmentation/` (or similar) top-level directory, or keep it as
   `dl2linknet/` if preserving internal layout matters more.
4. Reconcile its `model-evaluate/` and `tools/` with the existing `yolo/` and
   `scripts/` utilities, which overlap (YOLO evaluation, DICOM handling,
   label conversion).

## Notes

- `yolo/` keeps the original flat entry-point style: experiment settings are
  edited at the top of each `train*.py` (`datasetHome`, `datasetName`,
  `trainMessage`); the last assignment wins.
- Several scripts write timestamped `.log` files in the working directory and
  prompt `y/n` before modifying files.
- There is no test suite; validate changes by running the relevant script and
  inspecting `runs/` or the log/report it writes.
