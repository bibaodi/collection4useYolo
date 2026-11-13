# Shared scripts

Cross-project dataset utilities. Most take the dataset/label directory as the
first positional argument and prompt `y/n` before modifying files.

| Script | Purpose |
|---|---|
| `filterTestPairsFromTrainDataset.py` | Remove test-set image/JSON pairs from a train dataset (leak prevention); writes CSV + log |
| `findFieldsNotCorrect.sh` | Find YOLO detection labels that do not have 5 fields |
| `sendThynodulesDataSetToS3.sh` | Upload a dataset ZIP to S3 |

`filterTestPairsFromTrainDataset.py` options:

```bash
python scripts/filterTestPairsFromTrainDataset.py \
    --test-dir  /path/to/test \
    --train-dir /path/to/train \
    --outdir ./reports \
    --match-mode name|content|both \
    --dry-run
```

Use `--trash-dir <dir>` to make removal reversible. See
`../docs/261112-filter-test-pairs-from-train-objectives-and-work-done.md` for
the full design and the 2026-11-12 run report.

YOLO-specific dataset helpers live in [`../yolo/scripts/`](../yolo/scripts/).
