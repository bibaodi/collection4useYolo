# 星流 (MaaS) Daily Cost Report

Pulls Token usage and official bills from the Kingsoft Cloud (KSYUN) OpenAPI
and writes CSV reports — including **daily 星流 cost per RAM sub-account
(托管子账号)** — so you can answer *"which RAM account is incurring 星流 fees?"*
and keep processing the numbers in Excel / pandas / SQL.

The workflow follows `../星流 MaaS 模型与 API Key 日账单接口调用步骤.md`:

```
DescribeModels → GetModelDetail → DescribeApikeys → QueryTokenData
        → estimate cost → QueryItemBills → QueryProductConsume
        → QueryProjectConsume → reconcile → CSV
```

Working notes — API shapes verified against the live API, the key → account
attribution, the pricing/reconciliation rules, and a move checklist — are in
[`docs/memory/`](docs/memory/README.md). Read that first if you are picking
the project up somewhere else.

## Quick start

```bash
cd ksyun-starflow
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

export KSYUN_ACCESS_KEY='AK...'
export KSYUN_SECRET_KEY='SK...'

# Single day
python main.py --date 2026-07-31 --output-dir output

# A month, split per managed sub-account
python main.py --start 2026-07-01 --end 2026-07-31 \
    --sub-account-id 2031447841 \
    --sub-account-id 2031447842 \
    --output-dir output
```

Output lands in `output/` (see [Reports](#reports)).

## Credentials

Resolution order — first non-empty value wins:

1. `--ak` / `--sk` (convenience; visible in shell history, so avoid on shared hosts)
2. `KSYUN_ACCESS_KEY` / `KSYUN_SECRET_KEY` environment variables
3. `config.yaml` / `config.json` (copy `config.example.yaml`), or the console
   credential file `accesskey.txt` (`AccessKeyId：…` / `私有访问密钥：…`)
4. Interactive prompt (`getpass`), never written to disk

Region defaults to `cn-beijing-6`. Everything is overridable via CLI,
environment, or the config file.

```bash
# use the console's downloaded credential file directly
python main.py --date 2026-11-16 --config accesskey.txt --output-dir result-output
```

## Which RAM account / sub-account?

There are two different ideas of "account" in KSYUN:

| Concept | API | Field |
|---|---|---|
| IAM user (子用户) | `DescribeApikeys` / `iam.api.ksyun.com` `ListUsers` | `CreateUserId` → `CreateUserName` (e.g. `eton`) |
| Managed / member account (托管子账号) | `resourcemanager.api.ksyun.com` `ListAccounts` | `AccountId` (numeric, e.g. `2031447841`) |

**An API key belongs to the RAM user that created it.** `DescribeApikeys`
returns `CreateUserId` (opaque id) and `CreateUserName` (login name), and
that pair is what links `API-KEY-1319677108893802496` to account `eton`.
Every report therefore carries `account_id` / `account_name` /
`account_type`, resolved as:

1. `--key-subaccount-map map.json` / `--sub-account-id` (explicit override),
2. the key's `CreateUserId` / `CreateUserName` (`iam_user`, or `root` when
   the id is the numeric main account),
3. managed sub-accounts from `ListAccounts` (`managed_account`).

`QueryTokenData` itself has no account column, so usage is queried per API
key (as the workflow doc specifies: *API Key list × ReasoningType*) and the
key's owner is attached to every row.

`map.json` accepts either form:

```json
{
  "API-KEY-1319703773481033728": "2031447841",
  "API-KEY-1319703773481033729": "2031447842"
}
```

```json
[
  {"key_id": "API-KEY-1319703773481033728", "sub_account_id": "2031447841"}
]
```

If no owner can be resolved, the row lands in an empty-account
(`UNATTRIBUTED`) bucket; `api_key_daily.csv` / `model_daily.csv` remain
fully usable.

## Reports

| File | Granularity | Use |
|---|---|---|
| `ram_account_daily.csv` | date × **account × model** | **Primary answer: which RAM account spends how much on which model** |
| `account_model_summary.csv` | account × model (period total) | Account roll-up |
| `model_daily.csv` | date × model | Cost per model |
| `api_key_daily.csv` | date × account × API key | Trace a cost back to a key |
| `accounts.csv` | account list | `account_id` → `account_name`, login, real name, email |
| `api_keys.csv` | key list | Key owner, project, status |
| `reconciled_daily.csv` | fact table | Estimated / official / allocated per row (`ESTIMATED` / `ALLOCATED` / `FINAL`) |
| `token_usage.csv` | raw usage | Tokens and counts per account / key / model / type |
| `item_bills.csv` | official bills | `QueryItemBills` rows (amount, project, item, config/extra info) |
| `product_consume.csv` | official daily 星流 | `QueryProductConsume` |
| `project_consume.csv` | official daily project | `QueryProjectConsume` |
| `summary.json` | run metadata | Counts, totals, accounts, redacted config |
| `run.log` | log | Request / warning trail |

CSV files are written as UTF-8 **with BOM** (`utf-8-sig`) so Chinese text
opens correctly in Excel. Amounts use 6 decimal places.

## Cost formulas

Online (`normal` / `web`), prices per 1,000,000 tokens:

```
CacheMissToken      × OnlineCalculateInput     / 1e6
+ CacheToken        × OnlineCalculateRight     / 1e6
+ OutputToken       × OnlineCalculateOutput    / 1e6
+ CacheWrite5mToken × cache-write-5m price     / 1e6
+ CacheWrite1hToken × cache-write-1h price     / 1e6
+ WebSearch         × 30  / 1000
+ WebParser         × 15  / 1000
+ ImageCount        × image price
+ VideoDuration     × video price
+ VoiceWordCount    × voice price
```

Batch (`batch`):

```
CacheMissToken × BatchCalculateInput     / 1e6
+ CacheToken   × BatchCalculateCacheRight / 1e6
+ OutputToken  × BatchCalculateOutput     / 1e6
```

Cache-write prices, image/video/voice prices are extracted from
`GetModelDetail`. The live API reports them as
`ExtraFields.priceConfig.priceItem` records
(`{"label": "在线推理输入", "price": 2.0, "unit": "元/百万Token"}`), which are
mapped to the fields above with unit normalisation (`元/千Token` × 1000,
`元/千次` ÷ 1000). Flat `OnlineCalculate*` / `PricingRules` shapes are
still supported. Override them when the shape is unexpected:

```bash
python main.py --date 2026-07-31 --price-overrides prices.json
```

```json
{
  "deepseek-r1-0528": {
    "online_input": 0.004,
    "online_cache_right": 0.0008,
    "online_output": 0.016,
    "cache_write_5m": 0.005,
    "cache_write_1h": 0.01
  }
}
```

Reconciliation (workflow doc section 9), adapted to the account × model grain:

* official item bills are pooled per `(day, model)` — the finest grain the
  bill API exposes — and split across the estimated rows of that model
  pro-rata by estimated cost; a model used by a single account is `FINAL`,
  a shared model is `ALLOCATED`;
* the part of the official 星流 total that item bills do not cover
  (`QueryProductConsume − Σ item bills`, e.g. bills still accruing for
  today) is distributed across every estimated row by
  `official_amount × estimated_cost / total_estimated_cost`;
* rows with no official counterpart stay `ESTIMATED`;
* bills whose model never appears in token data become their own `FINAL`
  rows, so the official total is never lost.

`official_cost` is the item-bill amount attributable to the row;
`allocated_cost` additionally carries the residual — use `allocated_cost`
when you need the account totals to add up to the official 星流 bill.

## CLI reference

```
--date YYYY-MM-DD             single day
--start / --end               inclusive range
--month YYYYMM                override bill month
--ak / --sk / --region        credentials overrides
--config FILE                 config.yaml / config.json / accesskey.txt
--product-group-code CODE     星流 product group code (default: auto via ListProductGroups)
--product-group-name NAME     product group name to match (default: 星流)
--sub-account-id ID           managed sub-account; repeatable
--no-account-discovery        skip ListAccounts
--key-subaccount-map FILE     KeyId → SubAccountId JSON
--reasoning-type TYPE         normal / batch / web; repeatable (default: all)
--group-by FIELD              model / keyId / endpointId (default: model)
--price-overrides FILE        per-model price overrides
--pay-type N                  QueryItemBills pay type (0=prepaid, 1=realtime, 2=postpaid)
--output-dir DIR              default ./output
--timeout / --retries         HTTP tuning
--max-pages N                 pagination safety cap
--bill-page-size N            QueryItemBills page size (max 1000)
--no-token / --no-official / --no-consume   skip stages
--dry-run                     print plan and exit
--log-level LEVEL             DEBUG / INFO / WARNING / ERROR
```

## Endpoints and signing

| Service | Endpoint | SigV4 service |
|---|---|---|
| MaaS | `https://aicp.api.ksyun.com` | `aicp` |
| Bills | `https://bill-union.api.ksyun.com` | `bill-union` |
| IAM | `https://iam.api.ksyun.com` | `iam` |
| Resource Manager | `https://resourcemanager.api.ksyun.com` | `resourcemanager` |

Signing is AWS SigV4 (`AWS4-HMAC-SHA256`, headers
`content-type;host;x-amz-date`), implemented dependency-free in
`ksyun_starflow/sigv4.py` and cross-checked against `botocore` in the tests.
The `.md` uses the underscore host `bill_union.api.ksyun.com`; KSYUN's request
structure doc uses the hyphenated `bill-union.api.ksyun.com` (which is also the
SigV4 service name). Both resolve; the hyphenated form is the default and can
be changed via `KSYUN_BILL_ENDPOINT`.

## Tests

```bash
python -m pytest -q
# or, without pytest:
python -m unittest discover -s tests -v
```

## Security notes

* Secrets are read from env / config / prompt and never logged. `summary.json`
  contains only a redacted AK (`AKxx***`) and `secret_key: "***"`.
* The app only performs read-only `Describe*` / `Query*` calls.
* Keep `config.yaml` out of version control if it contains AK/SK.

## Caveats

* Official bill data is only available from month `202111` onward.
* `QueryItemBills` returns at most 1000 rows per page; pagination uses the
  `LastSortValue` cursor.
* `ListAccounts` is a best-effort discovery. If your account is not a
  financial-management main account, pass `--sub-account-id` explicitly.
* `estimated_cost` is `tokens × published price`; the official bill can
  differ (contract discounts, cached-token pricing, free quota, bills still
  accruing for the current day). `official_cost` / `allocated_cost` are the
  authoritative figures.
* QueryTokenData is called per API key × reasoning type × day (as the doc
  specifies), so a month with many keys means many requests. Use
  `--reasoning-type normal` to cut the call count.
* `DescribeApikeys` returns each key's secret `KeyValue`; it is deliberately
  **not** written to any report.
