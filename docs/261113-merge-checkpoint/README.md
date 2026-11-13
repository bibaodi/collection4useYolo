# Repository consolidation — checkpoint

- **Date:** 2026-11-13
- **Branch:** `merge/repo-consolidation`
- **HEAD:** `325de37`
- **Working tree:** clean
- **Status:** **Step 1 complete. Step 2 (`ml_dl2linkNet`) not started.**
- **Origin remote:** `git@github.com:bibaodi/collection4useYolo.git` (nothing pushed)

This memo is the resume point. Read it together with
[`cleanup-manifest.md`](cleanup-manifest.md) and
[`step2-ml_dl2linkNet-plan.md`](step2-ml_dl2linkNet-plan.md).

## 1. Goal

Consolidate four independent repositories into one home repository
(`ml_collections4yolo`):

| # | Source repository | Target | Status |
|---|---|---|---|
| 1 | `ml_collections4yolo` (home) | `yolo/`, `scripts/`, `demos/` | done |
| 2 | `ml_thyroid.01_etonTrainCodes` (`bibaodi/thyroid_train01`) | `thyroid-data/` | done |
| 3 | `ml_task4efficientNet` (`etonb/task4efficientNet`) | `nodule-classification/` | done |
| 4 | `ml_dl2linkNet` (`bibaodi/dl2_linknet`) | TBD (`dl2linknet/` or `segmentation/`) | **pending** |

Agreed constraints:

- Layout: **by source project** (top-level directory per repository).
- History: **latest snapshot only** — per-project git history was not preserved.
  Originals remain on their remotes.
- Cleanup: remove build artifacts, temporary reports, duplicates and superseded
  scripts; move large result dumps out of git.
- Step 1 first, then decide how/when to merge `ml_dl2linkNet`.

## 2. Repository layout at this checkpoint

```
ml_collections4yolo/
├── README.md                       # master index, cleanup report, step-2 plan
├── CLAUDE.md                       # AI-assistant orientation guide
├── .gitignore                      # consolidated ignore rules
├── yolo/                           # YOLO train/predict/split entry points
│   ├── README.md
│   ├── trainSegment_v01.py
│   ├── trainDetectThyNodu_v01.py
│   ├── trainDetectGeneralObj.py
│   ├── trainClassifyBoM_v01.py
│   ├── predict_seg.py
│   ├── predict_segNoCuttingToBox.py
│   ├── predict_classify.py
│   ├── validateModel_Classify.py
│   ├── useSegModelCropUSImg2NoduleOnlyRegionImg.py
│   ├── analyseModelBasicStructureUsingPred.py
│   ├── splitClsDataset3.py
│   ├── splitSegDataset-copyOp.py
│   ├── splitSegDataset-symOp.py
│   └── scripts/                    # YOLO dataset helpers (18 files)
├── thyroid-data/                   # from ml_thyroid.01_etonTrainCodes
│   ├── README.md
│   ├── 42scripts/
│   ├── config/
│   ├── srcs/
│   │   ├── annotationFormatConverter/
│   │   ├── autoAnnotate/
│   │   ├── dataProcess/
│   │   ├── processImgs/
│   │   ├── utils/
│   │   ├── dicom_tags_reader/
│   │   ├── kewei-db-presentation/
│   │   ├── convert_ThyNodulesJson.py
│   │   ├── extract_all_imgs_from_dicom.py
│   │   ├── groupImagesByCsv.sh
│   │   ├── removeBlankImagesOutofContinusLabelsRange.py
│   │   └── statistics_train_data.py
│   ├── tests/
│   └── notebooks/processImgs/
├── nodule-classification/          # from ml_task4efficientNet
│   ├── README.md
│   ├── README.zh.md                # original Chinese readme
│   ├── environment.yml
│   ├── srcs/
│   ├── scripts/
│   ├── models/
│   ├── dataset/
│   ├── notebooks/
│   └── docs/
├── scripts/                        # shared utilities
│   ├── README.md
│   ├── filterTestPairsFromTrainDataset.py
│   ├── findFieldsNotCorrect.sh
│   └── sendThynodulesDataSetToS3.sh
├── demos/                          # 22-demos + thyroid DICOM demos
│   └── README.md
├── docs/
│   ├── 261112-filter-test-pairs-from-train-objectives-and-work-done.md
│   ├── 261113-merge-checkpoint/    # this checkpoint
│   └── history/                    # archived notebooks / operation history / ref_srcs
└── data/                           # git-ignored local-only dumps (119 MB)
    ├── README.md
    ├── thyroid-data/
    └── nodule-classification/
```

Tracked files: **235**. Snapshot of the full list in
[`tracked-files.txt`](tracked-files.txt); git state in
[`git-state.txt`](git-state.txt).

## 3. Commits

```
325de37 Remove remaining superseded script duplicates
a9cd039 Consolidate thyroid_train01 and task4efficientNet into monorepo
f266d0f Add test-pair filter tool and run notes        # pre-existing branch work
```

## 4. Verification performed

| Check | Result |
|---|---|
| `python -m py_compile` on every `*.py` | pass |
| `bash -n` on every `*.sh` | pass |
| Markdown relative links in top-level/project READMEs | 0 broken |
| Reconciliation of every source tracked file | 0 lost |
| `git status` | clean |
| Source repositories (`ml_thyroid...`, `ml_task4efficientNet`) | untouched (clean) |

The four Chinese-named missing-image logs are present under
`data/nodule-classification/logs/{train,verify}/`.

## 5. Cleanup performed

See [`cleanup-manifest.md`](cleanup-manifest.md) for the exact list. Summary:

- Deleted build/runtime junk (`build/`, `__pycache__`, `.DS_Store`,
  `.marscode/`, backups).
- Removed temporary AI reports and patch scripts; consolidated four duplicate
  YOLO-format documents into `nodule-classification/docs/yolo_dataset_utils.md`.
- Removed superseded converters and split/demo variants.
- Moved `someresults/` (~113 MB), `logs/`, thyroid spreadsheets and
  gradient-phase CSVs to git-ignored `data/`.
- Archived notebooks, operation history and reference training code under
  `docs/history/`.

## 6. Known path adjustments

- `project_root` in `nodule-classification/srcs/{create_sop4_verify_images,
  create_sop4_training_images_backup,cleanData_sop5_OOF_0928,
  OOF_analysis_sop7_all_match_tr13_0927}.py` now derives from `__file__`
  (was a teammate's `/Users/mouxiaoyong/...`).
- `thyroid-data/tests/*.py` import `../srcs/utils`.
- `thyroid-data/42scripts/02-loop-autolabel-thyroidGland.sh` resolves
  `annotateByMLModel.py` relative to the script.
- Remaining hardcoded machine paths (`/mnt/xin1data`, `/home/eton`,
  `/mnt/datas`, `/Users/Shared/tars`, …) still need local adjustment.

## 7. Environment notes

- `yolo/` targets **ultralytics 8.3.250**.
- `nodule-classification/` uses `efficientnet_env` from
  `environment.yml` (torch 2.9, scikit-learn 1.7.2, ultralytics 8.3.218).
- `ml_dl2linkNet` (step 2) requires **ultralytics 8.4.60** and
  `trainLinkNet18_env`. This version conflict must be resolved or documented
  per project.

## 8. How to resume

```bash
cd /home/eton/00-srcs/ml_collections4yolo
git checkout merge/repo-consolidation
git log --oneline -3          # expect 325de37 at the top
git status                    # expect clean

# Read the decision material before touching ml_dl2linkNet:
less docs/261113-merge-checkpoint/step2-ml_dl2linkNet-plan.md
```

Step 2 has **not** been started; no files from `ml_dl2linkNet` have been
copied. The open decisions are recorded in
[`step2-ml_dl2linkNet-plan.md`](step2-ml_dl2linkNet-plan.md).
