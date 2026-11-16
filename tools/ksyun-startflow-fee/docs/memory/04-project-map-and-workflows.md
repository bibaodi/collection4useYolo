# 04 — Project map, workflows, reports

## Layout

```
ksyun-starflow/
├── main.py                    # thin entry point → ksyun_starflow.cli.main()
├── requirements.txt           # requests>=2.31, PyYAML>=6.0
├── pyproject.toml             # name ksyun-starflow, console script ksyun-starflow
├── config.example.yaml        # copy to config.yaml
├── accesskey.txt              # LIVE AK/SK — never copy, never print
├── README.md                  # user-facing docs
├── docs/memory/               # these notes
├── ksyun_starflow/
│   ├── cli.py                 # argparse + orchestration + CSV writing
│   ├── client.py              # HTTP client, retries, pagination, error mapping
│   ├── sigv4.py               # dependency-free AWS SigV4
│   ├── endpoints.py           # one function per API action (+ XML ListUsers)
│   ├── normalize.py           # raw API dicts → flat row dicts
│   ├── pricing.py             # price extraction + cost computation
│   ├── reconcile.py           # estimated vs official reconciliation
│   ├── reports.py             # CSV column definitions + writers
│   └── config.py              # CLI/env/file credential resolution
├── tests/                     # 33 tests, all offline (mock transport)
│   ├── test_sigv4.py          # cross-checked against botocore
│   ├── test_normalize.py
│   ├── test_pricing.py
│   ├── test_reconcile.py
│   └── test_e2e.py            # full run against a mocked API
├── result-output/             # last real run (2026-11-16)
├── output/ output02/ output03/ # earlier scratch runs
└── docs/memory/
```

## Data flow

```
config.py ──► client.py (SigV4) ──► endpoints.py ──► normalize.py
                                                        │
                          pricing.py (GetModelDetail) ──┘
                                    │
                                    ▼
                            estimated token rows
                                    │
              endpoints.QueryItemBills / QueryProductConsume
                                    │
                                    ▼
                              reconcile.py
                                    │
                                    ▼
                              reports.py → CSV
```

`cli.py` is the only module that knows the whole pipeline:

| Function | Purpose |
|---|---|
| `discover_accounts()` | `ListAccounts` — managed sub-accounts (empty here) |
| `discover_users()` | `iam ListUsers` — IAM users (XML) |
| `build_account_directory(api_keys, users, managed_accounts, overrides, sub_account_ids=…)` | merge into one `account_id → record` directory |
| `resolve_key_accounts(api_keys, directory, overrides)` | KeyId → account record |
| `fetch_token_rows(..., key_accounts, group_by="model")` | per-key, per-reasoning-type, per-day usage + cost |
| `fetch_item_bills()` / `fetch_product_consume()` / `fetch_project_consume()` | official side |
| `propagate_accounts(token_rows, bill_rows, directory)` | attach `account_id/name/type` to bill rows |
| `reconcile(...)` | see [03](03-pricing-and-reconciliation.md) |
| `write_reports(..., accounts=…)` | all CSVs + `summary.json` |
| `run()` | orchestration, prints the "by RAM account" breakdown |

## Run

```bash
cd ksyun-starflow
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# single day, credentials from the console download
python main.py --date 2026-11-16 --config accesskey.txt --output-dir result-output

# a month
python main.py --start 2026-11-01 --end 2026-11-16 --output-dir result-output

# plan only, no API calls
python main.py --date 2026-11-16 --dry-run
```

Credentials resolution: `--ak/--sk` → `KSYUN_ACCESS_KEY`/`KSYUN_SECRET_KEY`
→ `--config` (yaml/json/accesskey.txt) → interactive `getpass`.

Useful flags: `--reasoning-type normal` (cuts request count),
`--group-by keyId`, `--no-account-discovery`, `--key-subaccount-map`,
`--price-overrides`, `--no-token` / `--no-official` / `--no-consume`,
`--log-level DEBUG`, `--max-pages`, `--bill-page-size`.

## Tests

```bash
python3 -m pytest -q -p no:cacheprovider      # 33 passed
# the -p no:cacheprovider is needed on the SMB share (see 05)
```

## Reports (`result-output/`)

| File | Grain | Notes |
|---|---|---|
| `ram_account_daily.csv` | date × account × model | **the answer**; `row_count` included |
| `account_model_summary.csv` | account × model | period roll-up |
| `model_daily.csv` | date × model | |
| `api_key_daily.csv` | date × account × API key | |
| `accounts.csv` | account | id, name, type, login, real name, email, main account |
| `api_keys.csv` | API key | owner, project, status — **no `KeyValue`** |
| `reconciled_daily.csv` | fact | estimated / official / allocated + `amount_type` |
| `token_usage.csv` | raw | per account / key / model / type |
| `item_bills.csv` | bill line | `QueryItemBills` |
| `product_consume.csv` | day | official 星流 total |
| `project_consume.csv` | day | project total |
| `summary.json` | run | counts, totals, `accounts`, `key_accounts`, `by_account`, redacted config |
| `run.log` | log | |

All CSVs are UTF-8 **with BOM** (`utf-8-sig`); money is 6 decimals
(`f"{x:.6f}"`).

## Other run directories

`output/`, `output02/`, `output03/` are earlier scratch runs (before the
account fix). `result-output/` is the current, correct one. All are
regenerable — do not copy them when moving the project.
