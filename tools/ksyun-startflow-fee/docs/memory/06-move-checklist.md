# 06 — Moving the project

## What to copy

```
ksyun-starflow/
├── main.py
├── ksyun_starflow/        (all .py)
├── tests/                 (all .py)
├── docs/memory/           (these notes)
├── requirements.txt
├── pyproject.toml
├── config.example.yaml
└── README.md
```

## What NOT to copy

| Path | Why |
|---|---|
| `accesskey.txt` | **live AK/SK** — re-download from the console at the new location |
| `config.yaml` (if present) | may contain AK/SK |
| `__pycache__/`, `.pytest_cache/` | stale bytecode / gvfs-incompatible cache |
| `output/`, `output02/`, `output03/` | scratch runs from before the account fix |
| `result-output/` | regenerable; keep only if you need the 2026-11-16 snapshot |
| `.venv/` | rebuild it |

A safe copy:

```bash
rsync -av --exclude='accesskey.txt' --exclude='config.yaml' \
      --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.venv' \
      --exclude='output*' --exclude='result-output' \
      ksyun-starflow/ /new/location/ksyun-starflow/
```

## Re-setup

```bash
cd /new/location/ksyun-starflow
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # requests>=2.31, PyYAML>=6.0
pip install pytest botocore              # optional: tests
```

There is no `pip install -e .` step required — `main.py` works from the
project root. (`pyproject.toml` declares the console script
`ksyun-starflow = "ksyun_starflow.cli:main"` if you prefer installing.)

## Credentials at the new location

```bash
export KSYUN_ACCESS_KEY='AK…'
export KSYUN_SECRET_KEY='SK…'
# or
python main.py --date 2026-11-16 --config /path/to/accesskey.txt ...
```

Never commit the credential file. Add it to `.gitignore` if the new location
is a repo.

## Verify after moving

```bash
# 1. tests — expect 33 passed
python3 -m pytest -q -p no:cacheprovider

# 2. plan only (no API calls)
python main.py --date 2026-11-16 --dry-run

# 3. real run
python main.py --date 2026-11-16 --config accesskey.txt --output-dir result-output
```

Expected from step 3 (numbers drift as today's bill accrues — the account
names and 5 keys must match):

* 5 API keys: `yryKey01`, `zhaiyfKey01`, `crsKey01`, `etonKey01`, `test-01`
* accounts: `ruiying.ye`, `yafan.zhai`, `ruisong.chen`, `eton`, `root`
* `api_key_daily.csv` maps `API-KEY-1319677108893802496` → `eton`
* `ram_account_daily.csv` has rows for every account × model
* no `sk-` string anywhere under the output dir:
  `grep -r 'sk-' result-output/ || echo clean`

## Sanity checks specific to the new host

* **Filesystem**: if it is not a gvfs SMB share, `pytest` no longer needs
  `-p no:cacheprovider` — but keeping the flag is harmless.
* **Python**: 3.10+ (uses `match`-free but modern typing). Verified on 3.13.
* **Network**: needs outbound HTTPS to `aicp.api.ksyun.com`,
  `bill-union.api.ksyun.com`, `iam.api.ksyun.com`,
  `resourcemanager.api.ksyun.com`.
* **Locale**: CSV output is UTF-8 with BOM, so Excel on a Chinese Windows
  host opens it correctly. Do not switch to `utf-8` without BOM.
* **Time zone**: day bounds are computed in local time; run the report on a
  host whose time zone matches the billing cycle (`Asia/Shanghai`).
