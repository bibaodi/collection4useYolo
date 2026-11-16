# 05 — Pitfalls and decisions

Every item below cost real debugging time; keep them in mind when changing the
code.

## Credentials

* **The console credential file uses full-width Chinese colons**:
  `AccessKeyId：AK…` / `私有访问密钥：SK…`. A naive `split(":")` silently
  yields nothing, the AK/SK end up `None`, and the failure surfaces as
  `UnicodeEncodeError: 'latin-1' codec can't encode characters …` inside
  SigV4 signing, or `TypeError: can only concatenate str (not "NoneType")`
  in `signing_key`. `config.parse_credentials_text` handles `：`, `:`, `=`
  and the labels `AccessKeyId` / `私有访问密钥` / `secret`.
* `--config` accepts `config.yaml`, `config.json` **and** `accesskey.txt`;
  `load_config_file` falls back to credential parsing when YAML/JSON does not
  produce a mapping.
* **Never print or copy the credential file.** A tool policy blocked even a
  partial `awk`/`substr` dump with
  *"[Credential Materialization] … partial credential output still exposes a
  live AK/SK; inspect only line counts/lengths or use the credential directly
  in the consuming call without echoing any of its characters."*
  Use Python that reads the file without echoing values.
* `DescribeApikeys.KeyValue` is the **API key secret** (`sk-…`). It is
  deliberately dropped in `normalize.normalize_api_key_row` and never appears
  in any CSV. `summary.json` only has a redacted AK (`AKxx***`).

## API behaviour

* **Epoch milliseconds.** `QueryTokenData` `StartTimestamp` / `EndTimestamp`
  are ms. Passing seconds makes the range span 1970 → now and the API answers
  `查询时间范围过长`.
* **≤ 1 day per query.** Even in ms, a multi-day range fails with the same
  error. Loop day by day.
* **A 1-hour window returns 0 rows** — nothing is wrong, the API needs a
  full-day window.
* **`ListUsers` returns XML**, not JSON. `client.py` stores the body as
  `{"_raw": "<ListUsersResponse…>"}`. `endpoints._xml_items` parses
  `root.iter("Item")` and `IsTruncated`. A JSON-only client silently sees
  zero users.
* **`ListAccounts` is empty** for this account (`TotalCount 0`, no 托管子账号).
  Never rely on it for attribution.
* **`QueryTokenData` has no account column.** Attribution *must* go through
  the API key owner.
* **`QueryItemBills.ConfigInfo` is a list of strings**
  (`["model(千Token):24.8"]`), not an object. `_model_from_price_labels`
  extracts the text before `(`/`（`/`:`.
* **`ExtraFields.priceConfig.priceItem`** — flat `OnlineCalculate*` fields
  are `null`; without the nested parse every estimated cost comes out `0`.
* `GetModelDetail` prices are **published** prices; bill amounts carry
  contract discounts, so the two never match exactly.

## Reconciliation

* Residual spread had to include **every** row with `estimated_cost > 0`, not
  just the pooled ones. Otherwise an extra `UNALLOCATED` bucket appeared and
  `test_e2e` saw 2 rows where it expected 1.
* Renaming `sub_account_id` → `account_id`/`account_name`/`account_type`
  broke tests with `KeyError: 'sub_account_id'`; the rename is intentional
  (the value is an IAM UserId, not a managed account id).

## Environment

* The project lives on a **gvfs SMB share**. `pytest`'s cache dir fails with
  `Errno 95 Operation not supported`. Run
  `python3 -m pytest -q -p no:cacheprovider`, or
  `python3 -m unittest discover -s tests -v`.
* Python 3.13; only `requests` and `PyYAML` are runtime deps. `botocore` is
  an optional test-only cross-check for `test_sigv4.py`.

## Data semantics

* **"Today" is provisional.** The product-consume total for 2026-11-16 read
  `4.29` at 16:53 and `6.55` at 17:24. Re-running the same date gives
  different official numbers; that is the billing pipeline, not a bug.
* Official bill data exists only from month `202111`.
* `QueryItemBills` pages at ≤ 1000 rows via the `LastSortValue` cursor.
* Only `ReasoningType=normal` had data (`batch`/`web` empty; the model has
  `Batch: false`).

## Deliberate non-goals

* The app is **read-only** (`Describe*` / `Query*` only).
* No database, no daemon, no scheduling — CSV in, downstream tooling out.
* No per-API-key cost breakdown as the primary output. The user was explicit:
  *"i do not care statis by api-key, but by account and model name. the
  account indexed result is necessary."* `api_key_daily.csv` exists only as
  a trace.
