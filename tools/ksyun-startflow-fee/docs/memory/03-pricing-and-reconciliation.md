# 03 — Pricing and reconciliation

## Where the price actually is

`GetModelDetail?ModelId=<id>` returns a flat object whose
`OnlineCalculateInput` / `OnlineCalculateOutput` / `OnlineCalculateRight` /
`PricingRules` are **`null`**. The real numbers are nested in
`ExtraFields.priceConfig.priceItem[]`:

```json
"priceConfig": {
  "priceItem": [
    {"label": "在线推理输入",     "price": 2.0,  "unit": "元/百万Token"},
    {"label": "在线推理输出",     "price": 8.0,  "unit": "元/百万Token"},
    {"label": "在线推理缓存命中", "price": 0.04, "unit": "元/百万Token"}
  ],
  "pricingRules": []
}
```

`pricing.price_items(detail)` reads that list; `_classify_price_label(label)`
maps the Chinese label to a field by keyword:

| Label keyword | Field |
|---|---|
| 输入 (输入 / 推理输入) | `online_input` |
| 输出 | `online_output` |
| 缓存命中 / 命中 | `online_cache_right` |
| 缓存写入 5m / 5分钟 | `cache_write_5m` |
| 缓存写入 1h / 1小时 | `cache_write_1h` |
| 联网搜索 / 搜索 | `web_search` |
| 网页解析 / 解析 | `web_parser` |
| 图片 / 图像 | `image` |
| 视频 | `video` |
| 语音 / 声音 | `voice` |
| 批量输入 / 批量输出 / 批量缓存 | `batch_*` |

`parse_model_detail` prefers the flat `OnlineCalculate*` fields when they are
non-null, then `by_label` (the priceItem mapping), then `PricingRules`.

## Units

Every price is normalised to **元 per 1,000,000 tokens** for the money
fields, and **元 per call** for the per-call fields.

| Unit in API | Action |
|---|---|
| `元/百万Token` | as-is (base unit) |
| `元/千Token` | **× 1000** |
| `元/千次` | **÷ 1000** (per-call fields) |
| `元/次` | as-is (per-call fields) |

`MONEY_FIELDS` = input / output / cache-right / cache-write / batch;
`PER_CALL_FIELDS` = web search / web parser / image / video / voice.

`price-overrides FILE` (per model, same field names) wins over everything.

## Cost formulas

Online (`ReasoningType` `normal` / `web`), per 1,000,000 tokens:

```
CacheMissToken      × online_input     / 1e6
+ CacheToken        × online_cache_right / 1e6
+ OutputToken       × online_output    / 1e6
+ CacheWrite5mToken × cache_write_5m   / 1e6
+ CacheWrite1hToken × cache_write_1h   / 1e6
+ WebSearch         × web_search  (per call)
+ WebParser         × web_parser  (per call)
+ ImageCount        × image       (per call)
+ VideoDuration     × video       (per call)
+ VoiceWordCount    × voice       (per call)
```

Batch (`batch`):

```
CacheMissToken × batch_input       / 1e6
+ CacheToken   × batch_cache_right / 1e6
+ OutputToken  × batch_output      / 1e6
```

The web-search / parser defaults (30 and 15 per 1000 calls) are the published
rates; they are only used when the model detail does not carry a price.

## Reconciliation (`reconcile.py`)

Grain: **`(usage_date, model_name)`**. That is the finest grain the bill API
exposes — `QueryItemBills` has no RAM user and no API key.

Algorithm:

1. **Pool** official item bills per `(bill_date, model_name)` →
   `official_total(day, model)`.
2. **Split** each pool across the estimated rows of that model:
   * one row → `amount_type = FINAL`, `official_cost = pool`;
   * several rows → `amount_type = ALLOCATED`,
     `official_cost = pool × estimated_cost / Σ estimated_cost`.
3. **Residual** `official_total − Σ direct_total` (product consume minus
   itemised bills; positive when bills are still accruing) is spread across
   **every** row with `estimated_cost > 0` by estimated share, added to
   `allocated_cost`. If no row has an estimated cost, one
   `UNALLOCATED` row carries the residual so nothing is lost.
4. **Rows with no official counterpart** stay `ESTIMATED`
   (`official_cost = 0`, `allocated_cost = estimated_cost`).
5. **Bills with a model that never appears in token data** become their own
   `FINAL` rows, so the official total is never silently dropped.

`amount_type` therefore means:

| Value | Meaning |
|---|---|
| `ESTIMATED` | tokens × price only, no official bill found |
| `ALLOCATED` | official amount split pro-rata by estimated share |
| `FINAL` | official amount belongs to exactly one account |
| `UNALLOCATED` | official money with no estimated basis at all |

Use `estimated_cost` for usage-based showback, `official_cost` for
"what the bill says for this model", and `allocated_cost` when the account
totals must add up to the official 星流 bill.

## Observed gap (2026-11-16)

| | amount |
|---|---|
| estimated (tokens × price) | ~16.94 |
| itemised bills | 4.29 |
| product consume (official 星流) | 6.55 |

Why they differ:

* the estimated price is the **published** price; the bill carries
  contract discounts (`Discount 0.5`, `ContractItemPrice 0.001` vs
  `ItemPrice 0.002`);
* the bill only has itemised lines for `deepseek-v4.1-flash-1017` while
  token data covers 5 models;
* the current day is still accruing (4.29 at 16:53 → 6.55 at 17:24).
