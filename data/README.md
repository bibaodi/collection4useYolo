# Local data (git-ignored)

This directory holds **large or generated artifacts** that were moved out of
git during the repository consolidation. Nothing here is committed except this
README.

```
data/
├── thyroid-data/
│   ├── screened_tirads_bethesda_records.csv
│   ├── thyroidNodulesStatistics-v3_etonAddSelectMar.xlsx
│   └── processImgs/
│       ├── output-gradX.csv
│       ├── output-gradY.csv
│       ├── output-magnitude.csv
│       ├── output-phase.ecsv
│       └── res_of_gradents.xlsx
└── nodule-classification/
    ├── logs/               # missing-image logs from 2025-10-16
    └── someresults/        # OOF / type-column result dumps
```

These files are inputs or outputs of the corresponding projects:

- Thyroid spreadsheets and gradient-phase outputs were produced/consumed by
  `thyroid-data/srcs/` scripts.
- Nodule logs and `someresults/` document the 2025-10-22 OOF/type analysis.

They remain available locally; delete them if you no longer need the history.
To re-track a file, add a negation to `.gitignore` (for example
`!data/path/to/file.csv`).
