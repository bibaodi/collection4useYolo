# Step 2 — merging `ml_dl2linkNet`

**Status: not started.** No files have been copied. This document records the
survey and the decisions that must be made before merging.

Source: `../ml_dl2linkNet` (`bibaodi/dl2_linknet`)
Target: `/home/eton/00-srcs/ml_collections4yolo`

## 1. Size survey

| Item | Size / count |
|---|---|
| Working tree (excl. `.git`) | ~13 GB |
| `.git` directory | ~610 MB |
| Commits | 380 |
| `runs/` (training outputs) | ~9.5 GB |
| `inference_results*/` | ~1.3 GB |
| Remaining source/docs | tens of MB |

The size is almost entirely generated output. A snapshot (tracked files only)
avoids all of it.

## 2. Top-level contents

| Path | What it is | Overlaps with |
|---|---|---|
| `linknet-torch/` | LinkNet / UNet / SegFormer PyTorch package | — |
| `yolo26_seg/` | YOLO26 segmentation pipeline | `yolo/` |
| `yolo26_sem/` | YOLO26 semantic-segmentation pipeline | `yolo/` |
| `yolo26_classify/` | YOLO26 classification pipeline | `yolo/trainClassifyBoM_v01.py` |
| `uisbp/` | Ultrasound image-processing / deployment utilities | — |
| `model-evaluate/` | Evaluation tooling | `yolo/`, `scripts/` |
| `tools/` | DICOM / dataset / label tools | `scripts/`, `thyroid-data/` |
| `scripts/` | Project shell/python helpers | top-level `scripts/` |
| `config/` | Experiment configs | per-project `config/` |
| `history-scripts/` | Legacy scripts | `docs/history/` |
| `docs/` | Design notes | `docs/` |
| `nbs/` | Notebooks | `docs/history/`, `*/notebooks/` |

## 3. Decisions to make

### D1 — Layout

| Option | Result | Trade-off |
|---|---|---|
| **A. By source project** (consistent with step 1) | `dl2linknet/` | Matches the existing convention (one top-level dir per merged repo); minimal path rewriting; duplicates remain visible. |
| **B. By function** | `segmentation/` (+ fold `yolo26_seg/`, `yolo26_sem/` into it) | Cleaner end state, but requires rewriting imports/configs and merging with existing `yolo/`; high regression risk. |

**Recommendation: A.** Step 1 already established source-project layout; B can
be a later refactor once everything is in one repo.

### D2 — History

**Recommendation: snapshot (tracked files only), same as step 1.** The 610 MB
`.git` and 380 commits stay on the `dl2_linknet` remote.

### D3 — Cleanup scope

Proposed, mirroring step 1:

- **Exclude from the copy:** `runs/`, `inference_results*/`, `__pycache__/`,
  `*.pt`/`*.pth`/`*.onnx`, `.DS_Store`, editor dirs.
- **Move to git-ignored `data/dl2linknet/`:** large result dumps that are
  actually needed for reference.
- **Archive to `docs/history/`:** `history-scripts/`.
- **Delete duplicates:** reconcile `model-evaluate/` and `tools/` against
  `yolo/` and `scripts/` (keep one canonical copy, note supersessions).
- **Keep:** `linknet-torch/`, `uisbp/`, `yolo26_*/`, `config/`, `docs/`,
  `nbs/` (notebooks may move under the project's `notebooks/`).

### D4 — Environment conflict

`ml_dl2linkNet` requires **`trainLinkNet18_env` / ultralytics 8.4.60**;
`yolo/` is pinned to **8.3.250** and `predict_segNoCuttingToBox.py` depends on
Ultralytics internals.

Options:

1. Document the per-project environment (add `environment.yml` per project and
   note in each README) — lowest risk.
2. Upgrade `yolo/` to 8.4.60 and re-verify `predict_segNoCuttingToBox.py`,
   `scale_masks`, `filter_mask_in_boxes`, `YOLOSeg` — needed only if a single
   environment is required.

**Recommendation: 1 for the merge, 2 as a separate tracked task.**

## 4. Proposed execution sequence

1. Confirm D1–D4.
2. Create branch `merge/dl2linknet` off `merge/repo-consolidation` (or
   continue on it, per preference).
3. Snapshot tracked files:
   ```bash
   cd ../ml_dl2linkNet
   git archive HEAD | tar -x -C ../ml_collections4yolo/dl2linknet
   ```
4. Run the same cleanup pass (build junk, temp reports, duplicates, large
   dumps -> `data/dl2linknet/`).
5. Fix `sys.path` / `project_root` / absolute-path assumptions; add
   `dl2linknet/README.md` and `environment.yml`.
6. Verify: `py_compile` all `*.py`, `bash -n` all `*.sh`, check Markdown links,
   reconcile source tree against `git ls-tree HEAD`.
7. Update root `README.md`, `CLAUDE.md`, `.gitignore`, `data/README.md`.
8. Commit and write `docs/261113-merge-checkpoint/` successor or amend this
   memo.

## 5. Risks

- **Import breakage** — several packages assume their repo root is on
  `sys.path`; moving them under `dl2linknet/` keeps that intact only if the
  copy preserves internal layout (option A does).
- **Ultralytics version** — see D4.
- **Large generated files** — do not commit `runs/` or `inference_results*/`;
  they are already in `.gitignore`.
- **Duplicate tools** — `model-evaluate/` and `tools/` need a human decision
  on which copy is canonical before deleting anything.
