# Cleanup manifest — step 1

Every entry below was either deleted or relocated during the
`ml_thyroid.01_etonTrainCodes` + `ml_task4efficientNet` merge. Source
repositories are untouched, so anything deleted here is still recoverable from
its remote.

## Deleted — build/runtime/machine-specific

| Path | Reason |
|---|---|
| `demos/avx2-demos/build/` | CMake/compiler build output (untracked) |
| `**/__pycache__/`, `*.pyc` | Python bytecode |
| `.DS_Store` (incl. `nodule-classification/logs/.DS_Store`) | macOS metadata |
| `nodule-classification/srcs/.marscode/` | Editor/machine metadata |
| `nodule-classification/readme.md.bak251019` | Backup copy |

## Deleted — temporary working notes

| Path | Reason |
|---|---|
| `nodule-classification/patchs/` | One-off memory-overflow patches |
| `nodule-classification/SYNTAX_FIX_SUMMARY.md` | Transient fix report |
| `nodule-classification/QUICK_REFERENCE-YOLO_FORMAT_SOLUTION.md` | Duplicate |
| `nodule-classification/YOLO_FORMAT_SOLUTION.md` | Duplicate |
| `nodule-classification/yolo_structure_example.txt` | Duplicate |
| `nodule-classification/srcs/README_yolo_utils.md` | Consolidated into `docs/yolo_dataset_utils.md` |
| `nodule-classification/ACTIVE_LEARNING_*.md` | Moved to `nodule-classification/docs/` (content kept) |

## Deleted — superseded scripts

| Path | Superseded by |
|---|---|
| `thyroid-data/srcs/annotationFormatConverter/convert301PACS_formatOriginScreenshot.py` | `convert301PX_2Lbmeformat.py` (multi-nodule, improved logging) |
| `thyroid-data/srcs/annotationFormatConverter/adjustJsonCoordinates_basedCropedUsImgRectangle.py` | `cropScreenshotAndAdjustJson.py` (combined crop + adjust) |
| `thyroid-data/srcs/annotationFormatConverter/convertXiaobaiNodulesFormartTolabelme.py` | `thyroid-data/srcs/convert_ThyNodulesJson.py` (configurable label, image lookup) |
| `yolo/splitClsDataset.py`, `yolo/splitClsDataset2.py` | `yolo/splitClsDataset3.py` (strict superset) |
| `demos/demo01.py`, `demos/demo02.py` | `demos/demo03-thynodu.py` |
| `scripts/rm_top700.sh` | One-off destructive |
| `scripts/split.sh` | Superseded by `splitClsDataset3.py` / `splitSegDataset-*.py` |
| `scripts/get_top_700_lines.sh` | Trivial one-off |
| `scripts/moveHeadN2NewFolder.sh` | Duplicate of `addNotCroppedImagesToDataset.sh` |
| `thyroid-data/42scripts/findFieldsNotCorrect.sh` | Canonical copy in top-level `scripts/` |
| `thyroid-data/42scripts/sendThynodulesDataSetToS3.sh` | Canonical copy in top-level `scripts/` |

## Relocated — large local-only data (git-ignored)

| Original path | New path |
|---|---|
| `ml_task4efficientNet/someresults/` | `data/nodule-classification/someresults/` |
| `ml_task4efficientNet/logs/` | `data/nodule-classification/logs/` |
| `ml_thyroid.../screened_tirads_bethesda_records.csv` | `data/thyroid-data/` |
| `ml_thyroid.../thyroidNodulesStatistics-v3_etonAddSelectMar.xlsx` | `data/thyroid-data/` |
| `ml_thyroid.../srcs/processImgs/output-gradX.csv` | `data/thyroid-data/processImgs/` |
| `ml_thyroid.../srcs/processImgs/output-gradY.csv` | `data/thyroid-data/processImgs/` |
| `ml_thyroid.../srcs/processImgs/output-magnitude.csv` | `data/thyroid-data/processImgs/` |
| `ml_thyroid.../srcs/processImgs/output-phase.ecsv` | `data/thyroid-data/processImgs/` |
| `ml_thyroid.../srcs/processImgs/res_of_gradents.xlsx` | `data/thyroid-data/processImgs/` |

## Relocated — archived under `docs/history/`

| Original path | New path |
|---|---|
| `ml_thyroid.../jupyterNotebooks/` | `docs/history/thyroid-notebooks/jupyterNotebooks/` |
| `ml_thyroid.../operation.history/` | `docs/history/thyroid-operation-history/operation.history/` |
| `ml_thyroid.../ref_srcs/` | `docs/history/thyroid-ref-srcs/ref_srcs/` |
| `ml_thyroid.../srcs/annotationFormatConverter/*.ipynb` | `docs/history/thyroid-notebooks/annotationFormatConverter/` |
| `ml_thyroid.../srcs/demos/` | `demos/thyroid/` |
| `ml_thyroid.../srcs/opencvTrackbarpos.py` | `demos/thyroid/` |
| `ml_thyroid.../srcs/processImgs/*.ipynb` | `thyroid-data/notebooks/processImgs/` |
| `ml_task4efficientNet/srcs/*.ipynb` | `nodule-classification/notebooks/` |
| `ml_task4efficientNet/srcs/ssh2ec2.sh` | `nodule-classification/scripts/ssh2ec2.sh` |
| `ml_task4efficientNet/srcs/streamlit-01-app.py` | `demos/streamlit-01-app.py` |
| `ml_task4efficientNet/readme.md` | `nodule-classification/README.zh.md` |
| `ml_task4efficientNet/ACTIVE_LEARNING_*.md` | `nodule-classification/docs/` |

## Deliberately kept (not duplicates)

| Path | Why |
|---|---|
| `yolo/splitSegDataset-copyOp.py` vs `-symOp.py` | Copy vs symlink modes |
| `nodule-classification/srcs/cleanData_sop5_OOF_0928.py` vs `OOF_analysis_sop7_all_match_tr13_0927.py` | Different dataset configs (SOP5 vs TR13) |
| `nodule-classification/srcs/create_sop4_training_images_backup.py` vs `create_sop4_verify_images.py` | Training vs verification image creation |
| `nodule-classification/srcs/trainWithCaseAsOneImg.py` vs `train_nodule_feature_cnn_model_v75.py` | Alternative training modes |
| `thyroid-data/srcs/processImgs/cropUSimgRectByGradientPhase.py` vs `annotationFormatConverter/cropScreenshotAndAdjustJson.py` | Different cropping algorithms |
