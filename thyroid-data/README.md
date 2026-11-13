# Thyroid annotation & data preparation

Data-preparation pipeline for thyroid ultrasound datasets: convert delivery
formats (PACS 301PX, Xiaobai/XB JSON, LabelMe) into YOLO labels, extract DICOM
frames, auto-annotate with existing models, match spreadsheets, and build
classification/train-val datasets.

Originally `ml_thyroid.01_etonTrainCodes` (`bibaodi/thyroid_train01`).

## Layout

| Path | Purpose |
|---|---|
| `srcs/annotationFormatConverter/` | Format converters (PACS/XB/LabelMe -> LabelMe/YOLO) |
| `srcs/autoAnnotate/` | Model-assisted annotation (`annotateByMLModel.py`) |
| `srcs/dataProcess/` | Spreadsheet matching, dataset generators, TIRADS/Bethesda screening |
| `srcs/processImgs/` | Ultrasound-region cropping (`cropUSimgRectByGradientPhase.py`) |
| `srcs/utils/` | Shared logger (`glog`), helpers, task-type definitions |
| `srcs/dicom_tags_reader/` | DICOM tag reader (Python + Java app) |
| `srcs/kewei-db-presentation/` | Spreadsheet matching / report generation |
| `42scripts/` | Operation shell scripts (auto-label loops, DICOM renaming); shared S3/label-validation helpers live in `../scripts/` |
| `config/` | Training YAML for the thyroid-nodule classifier |
| `tests/` | Logger smoke tests |
| `notebooks/processImgs/` | Ultrasound-cropping experiments (moved from `srcs/processImgs/`) |

Archived material (old notebooks, operation history, reference training code)
lives under `../docs/history/`.

## Important imports

Several modules add `srcs/utils` (or the `srcs/` directory) to `sys.path`, so
preserve the `srcs/<subdir>/` layout. `srcs/autoAnnotate/annotateByMLModel.py`
expects `srcs/` on the path and imports `utils.glog` and
`annotationFormatConverter.LabelmeJson`.

## Main converters

| Script | Input | Output |
|---|---|---|
| `convert301PX_2Lbmeformat.py` | 301PX PACS cases | LabelMe JSON (multiple nodules per case) |
| `convert301PX_xbfmt_2Lbmefmt.py` | XB JSON + image pair | LabelMe JSON |
| `convertImgWithVersionedxbfmt_2Lbme.py` | Image folder with versioned XB JSON | LabelMe JSON (latest version) |
| `convert_ThyNodulesJson.py` | Xiaobai multi-frame JSON | LabelMe JSON per frame |
| `convertLbmeFmt_2Yolo.py` | LabelMe | YOLO detection or segmentation labels |
| `generateClassifyDatasetFromLabelmeFormat.py` | LabelMe | YOLO classification dataset |
| `cropScreenshotAndAdjustJson.py` | Ultrasound screenshot | Cropped image + adjusted JSON |
| `findLabelmeJsonPointLessthenFour.py` | LabelMe | Report of < 4-point shapes |
| `markCasesUsed4TrainToSpreadsheet.py` | Case directory + spreadsheet | Marked spreadsheet |

Superseded converters removed during the merge:

- `convert301PACS_formatOriginScreenshot.py` -> use `convert301PX_2Lbmeformat.py`
- `adjustJsonCoordinates_basedCropedUsImgRectangle.py` -> use `cropScreenshotAndAdjustJson.py`
- `convertXiaobaiNodulesFormartTolabelme.py` -> use `../convert_ThyNodulesJson.py`

## Usage

```bash
# PACS -> LabelMe
python thyroid-data/srcs/annotationFormatConverter/convert301PX_2Lbmeformat.py <cases-dir>

# LabelMe -> YOLO
python thyroid-data/srcs/annotationFormatConverter/convertLbmeFmt_2Yolo.py

# Auto-annotate with an existing YOLO model
python thyroid-data/srcs/autoAnnotate/annotateByMLModel.py \
    --model_file model.pt --model_type segmentation \
    --input_folder in/ --output_folder out/ --jobs 2

# Operation scripts
bash thyroid-data/42scripts/02-loop-autolabel-thyroidGland.sh --help
```

Most scripts use absolute paths from the author's machines
(`/mnt/f/...`, `/mnt/d/...`); adjust them locally. Several prompt `y/n`
before modifying files.

## Dataset notes

`../data/thyroid-data/` (git-ignored) holds the large result spreadsheets and
gradient-phase experiment outputs that were previously tracked:

- `screened_tirads_bethesda_records.csv`
- `thyroidNodulesStatistics-v3_etonAddSelectMar.xlsx`
- `processImgs/output-*.csv`, `output-phase.ecsv`, `res_of_gradents.xlsx`

## History

- 21/08/20 first edition.
- 22/01/29 thyroid-nodule model requested.
- eton@250310 add training script and YAML config from Daniel.
- 2026-11-13 merged into the monorepo (snapshot).
