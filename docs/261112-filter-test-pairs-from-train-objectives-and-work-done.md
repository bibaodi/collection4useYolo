# Filter Test-Set Pairs out of the Train Dataset — Objectives and Work Done

- **Date:** 2026-11-12
- **Author:** eton (with Claude Code)
- **Script:** `scripts/filterTestPairsFromTrainDataset.py`
- **Status:** applied; train dataset cleaned

## 1. Objective

Prevent test-set leakage by removing every image/JSON pair from the **train**
dataset that also exists in the **test** dataset, and leave an auditable CSV +
log report of what was found and removed.

Inputs:

| Role | Path |
|---|---|
| Test set | `/mnt/datas/42workspace/34-pj_ML_data_models_UltrasoundIntelligence/10-datas4ML/10-testDatasets/Testset_hhus_basilicVein_v1.0.261102` |
| Train set | `/mnt/datas/42workspace/34-pj_ML_data_models_UltrasoundIntelligence/10-datas4ML/16-basilicVein4UGVA/37-stage-annotationReviewed_ready4train/basilicVein_v03.260924` |

Deliverables:

1. A Python application that performs the filtering.
2. A CSV with one row per **test** image, recording whether it was found in
   train and what was removed.
3. A log recording the inputs and the result summary.

## 2. Dataset analysis

Both datasets use one folder per series, each containing image/JSON pairs with
the same stem.

| | Pairs | Series | Naming |
|---|---:|---:|---|
| Test | 471 | 15 | `frm_<frame>.jpg` |
| Train | 4,376 | 21 | `frm_<frame>.jpg` (14 real v03 dirs) + `frame_<frame>.jpg` (7 symlinked v02 series) |

Findings that drove the design:

- **Names are not directly comparable.** Test
  `hhus-basilicVein-91.1.1/frm_0409.jpg` corresponds to train
  `UGVA2607-91.1.1/frame_0409.jpg`.
- **Frame numbers repeat across series**, so matching on the stem alone is
  wrong; the series must be part of the identity.
- **Byte identity is not guaranteed.** 350/471 test images are MD5-identical
  to train; 65 are re-encoded copies (max per-channel pixel diff ≤ 7); 56 are
  genuinely new frames absent from train.
- **7 train series are symlinks** (`UGVA2607-*`) into
  `basilicVein-v02.260916_UGVA_hhus_basilicVein_N11`, so removals mutate that
  dataset as well.

## 3. Matching design

Identity is decided by `--match-mode`:

| Mode | Rule | Catches |
|---|---|---|
| `name` | Strip series prefixes (`hhus-basilicVein-`, `UGVA2607-`) and frame prefixes (`frm_`, `frame_`), then compare `<series>/<frame>` | Re-encoded copies |
| `content` | MD5 of the image bytes | Renamed copies |
| `both` (default) | Union of the two | Everything above |

Name-only matches are confirmed pixel-by-pixel (`--verify-pixel`, tolerance 12)
so a coincidental frame-number collision is not deleted. Content matches are
byte-identical and skip the check.

## 4. Application

`scripts/filterTestPairsFromTrainDataset.py`

- Discovers image/JSON pairs under both roots (follows symlinked series by
  default; `--no-symlinks` to disable).
- Builds train lookup tables by normalised name and (optionally) MD5.
- Removes image + JSON for every match, de-duplicating across test rows.
- Safety options: `--dry-run` (report only), `--trash-dir` (move instead of
  unlink), warnings for symlinked train series.
- Console/log stages: `START`, `INPUT`, `PROGRESS`, `RESULT`, `END`; file log
  is DEBUG-level, console INFO-level.
- Output file names get a `-<YYMMDDTHHMMSS>` suffix.

CSV columns:

`image-name` (basename), `series`, `json-name` (basename), `found_in_train`,
`match_type`, `train_image_matches`, `train_json_matches`, `images_removed`,
`jsons_removed`, `status`, `note`.

## 5. Results (applied 2026-11-12 17:34:52)

| Series | Test | Found | Removed | Not found |
|---|---:|---:|---:|---:|
| hhus-basilicVein-722.13.1.1 | 30 | 30 | 30 | 0 |
| hhus-basilicVein-722.16.2.1 | 22 | 22 | 22 | 0 |
| hhus-basilicVein-722.2.3.3 | 30 | 30 | 30 | 0 |
| hhus-basilicVein-722.20.1.1 | 29 | 29 | 29 | 0 |
| hhus-basilicVein-722.22.1.1 | 17 | 17 | 17 | 0 |
| hhus-basilicVein-722.23.1.1 | 30 | 30 | 30 | 0 |
| hhus-basilicVein-722.25.1.1 | 21 | 21 | 21 | 0 |
| hhus-basilicVein-722.4.1.1 | 30 | 30 | 30 | 0 |
| hhus-basilicVein-722.7.2.1 | 30 | 30 | 30 | 0 |
| hhus-basilicVein-722.8.1.1 | 30 | 30 | 30 | 0 |
| hhus-basilicVein-91.1.1 | 50 | 30 | 30 | 20 |
| hhus-basilicVein-91.2.1 | 30 | 30 | 30 | 0 |
| hhus-basilicVein-92.1.1 | 49 | 16 | 16 | 33 |
| hhus-basilicVein-92.2.1 | 43 | 40 | 47 | 3 |
| hhus-basilicVein-95.1.2 | 30 | 30 | 30 | 0 |
| **Total** | **471** | **415** | **422** | **56** |

- 415 test images found (350 content + 65 name).
- 422 unique train images and 422 JSONs removed (7 test images matched
  duplicated train copies, hence 422 > 415).
- Train pair count: 4,376 → **3,954**.
- Post-run check: 0 matched train files remain; test set untouched (471 pairs).

## 6. Artifacts

| Kind | Path |
|---|---|
| Application | `scripts/filterTestPairsFromTrainDataset.py` |
| CSV report | `basilicVein_v03.260924/test_pair_filter_report-261112.csv` |
| Run log | `basilicVein_v03.260924/filterTestPairs-261112T173452.log` |

## 7. Usage

```bash
python scripts/filterTestPairsFromTrainDataset.py \
  --test-dir  '/mnt/.../Testset_hhus_basilicVein_v1.0.261102' \
  --train-dir '/mnt/.../basilicVein_v03.260924' \
  --outdir ./filter-reports --dry-run
```

Drop `--dry-run` to apply; add `--trash-dir <dir>` to make removal
reversible. Use `--match-mode name|content|both`, `--series-prefix`,
`--frame-prefix`, `--pixel-tolerance`, `--limit` to adapt to other deliveries.

## 8. Verification

- `python3 -m py_compile` passes.
- Dry run on the real data reproduced the expected 415/56 split.
- Synthetic end-to-end tests covered: exact content copy, re-encoded copy,
  same-frame-number different content (rejected), cross-series duplicate,
  missing JSON, and `--trash-dir` moves.
- Post-apply checks confirmed the train pair count dropped by 422 and no
  matched file remains.

## 9. Caveats / follow-ups

- The train `README.md` metadata is now stale: `image_count: 3077` should be
  2,808 for the real v03 series (the symlinked v02 series went from 1,299 to
  1,146).
- The run log still records the CSV's pre-rename path
  (`test_pair_filter_report.csv`); the delivered file was renamed to
  `test_pair_filter_report-261112.csv`.
- 7 removals affect files inside symlinked v02 series, not just the v03 folder.
- The pixel tolerance (12) fits the observed re-encode delta (≤ 7); raise it
  only if a future delivery is re-encoded more aggressively.
