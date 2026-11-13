# YOLO training & inference

Ultralytics YOLO entry points for medical-ultrasound datasets (thyroid
gland/nodule, carotid, esophagus; benign/malign and thyroid-property
classifiers).

## Entry points

| Script | Task | Notes |
|---|---|---|
| `trainSegment_v01.py` | instance segmentation | `imgsz=224` |
| `trainDetectThyNodu_v01.py` | detection | `imgsz=96` |
| `trainDetectGeneralObj.py` | detection (generic) | |
| `trainClassifyBoM_v01.py` | classification | benign/malign, `imgsz=96` |
| `predict_seg.py` | segmentation inference | |
| `predict_segNoCuttingToBox.py` | segmentation inference, uncropped masks | ONNX Runtime + `.pt`; depends on Ultralytics internals |
| `predict_classify.py` | batch classification -> CSV | `--model --images --output` |
| `validateModel_Classify.py` | classification validation | prompts y/n |
| `useSegModelCropUSImg2NoduleOnlyRegionImg.py` | batch segment + crop | `-m -d -o --save-types` |
| `analyseModelBasicStructureUsingPred.py` | model introspection | |
| `splitClsDataset3.py` | classification train/val split | per-class dirs |
| `splitSegDataset-copyOp.py` / `splitSegDataset-symOp.py` | segmentation train/val split | copy / symlink |

Older split variants (`splitClsDataset.py`, `splitClsDataset2.py`) are kept
for reference; prefer `splitClsDataset3.py`.

## Configuration by editing

Most training scripts take **no CLI arguments**. Edit the constants at the top
of the script:

```python
datasetHome = "/mnt/xin1data/yolo-ultralytics-250101/datasets/"
datasetName = "..."
trainMessage = "10th make train&val data 50-50..."
```

`datasetName` and `trainMessage` are reassigned several times; the **last
assignment wins**. `trainMessage` is a free-text run-history note — keep that
convention when adding a run.

Dataset YAMLs resolve as `datasetHome + datasetName`. Standard YOLO layout:

```
<dataset>/
├── images/{train,val}/
├── labels/{train,val}/
└── <dataset>.yaml
```

- detection labels: `class x_center y_center width height`
- segmentation labels: `class` + polygon coordinates (≥ 3 points, ≥ 7 fields)
- classification: one class directory per label under `train/` and `val/`

## Training pipeline

`YOLO(pretrained .pt)` -> `model.train(...)` -> `model.val()` ->
`model.predict(sample)` -> optional `model.export("onnx")`. Artifacts land in
`runs/<task>/<name>/` (git-ignored).

## Segmentation output: cropped vs. uncropped

`model.predict()` crops masks to their bounding boxes.
`predict_segNoCuttingToBox.py` re-implements postprocessing
(`scale_masks`, `filter_mask_in_boxes`, `YOLOSeg`) so masks keep their full
contour for both ONNX Runtime and `.pt` inputs; it loads the checkpoint's
`ema`/`model` manually and calls
`ultralytics.utils.ops.non_max_suppression` / `scale_boxes` and
`engine.results.Results` directly. It is sensitive to the installed
Ultralytics version (`8.3.250`).

## Helper scripts (`yolo/scripts/`)

Dataset-curation helpers take the dataset or label directory as `$1`; most
prompt `y/n` before moving or modifying files:

- `checkImageFormatSuffixMatch.sh <dir>` — image extension vs. real format
- `showLabel_ClassID_Not0.sh <label-dir>` — class-ID report
- `checkYoloSeg_DSLabelCorrect.py <dataset-dir>` / `checkYoloDetect_DSLabelCorrect.py` — label validation
- `mergeBoMLabels2OneThyNodu.sh` — collapse BoM classes 1 -> 0
- `createModelPackage.sh` — package a model for delivery
- `createPreTrainedModelLinkToTrainDir.sh` — link pretrained weights
- `sendTrainedModelToS3.sh` — upload a trained model
- `watchFileFlagToStartTrain.sh` — wait-for-flag training trigger
- `addNotCroppedImagesToDataset.sh`, `appendFileToDataset.sh`,
  `del_LonglyImgfiles.py`, `deleteAllBrokenSymbolicLinks.sh`,
  `rename_file_to_dicom-index.sh`, `cpFixexNumfiles.sh`, `rmFixedNumfiles.sh`,
  `statis-files-count.sh`, `checkImgSz4Classify.sh` — dataset maintenance

Shared utilities live in the repository-level `scripts/` directory.
