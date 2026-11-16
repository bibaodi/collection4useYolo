# 01 — KSYUN OpenAPI notes (verified 2026-11-16)

All calls are SigV4 (`AWS4-HMAC-SHA256`) signed; headers
`content-type;host;x-amz-date`. Region `cn-beijing-6`.

| Service | Endpoint | SigV4 service | API version |
|---|---|---|---|
| MaaS / AICP | `https://aicp.api.ksyun.com` | `aicp` | `2024-06-12` |
| 费用中心 | `https://bill-union.api.ksyun.com` | `bill-union` | `2025-08-01` (item bills), `2022-12-22` (consume), `2020-01-01` (product groups) |
| IAM | `https://iam.api.ksyun.com` | `iam` | `2015-11-01` |
| Resource Manager | `https://resourcemanager.api.ksyun.com` | `resourcemanager` | `2021-03-20` |

The workflow doc uses the underscore host `bill_union.api.ksyun.com`; the
hyphenated form is the SigV4 service name and also resolves. Default is the
hyphenated form (overridable via `KSYUN_BILL_ENDPOINT`).

## `DescribeApikeys` (POST form)

```
POST /?Action=DescribeApikeys&Version=2024-06-12
Content-Type: application/x-www-form-urlencoded

Marker=1&MaxResults=100&DefaultKey=false
```

Response: `{Data: [...], TotalCount, Marker, MaxResults, RequestId}`.

Fields that matter (real values from this account):

| Field | Example | Notes |
|---|---|---|
| `KeyId` | `API-KEY-1319677108893802496` | report key |
| `Name` | `etonKey01` | user-chosen name |
| `KeyValue` | `sk-…` | **the API secret — never write to any report** |
| `CreateUserId` | `ARkgUxbRQWc3oWfiL0n8A` | IAM UserId of the owner (or numeric main account id) |
| `CreateUserName` | `eton` | owner login name |
| `ProjectId` / `ProjectName` | `127860` / `ksyun-trial` | |
| `Status` | `"1"` / `"2"` | `1` = enabled, `2` = disabled |
| `CreateTimestamp` | `1794802007000` | epoch ms |
| `AssociatedModelIds` | `[]` | empty ⇒ all models |
| `AllAssociatedModel` | `true` | |

Pagination is `Marker` (1-based) + `MaxResults`.

## `ListUsers` (GET) — returns **XML**

```
GET /?Action=ListUsers&Version=2015-11-01&Page=1&PageSize=100
```

The client's JSON decoder stores the body as `{"_raw": "<ListUsersResponse…>"}`;
parse with `xml.etree.ElementTree` (`endpoints._xml_items`). Structure:

```
ListUsersResponse/ListUserResult/Users/member/Item
  UserId, UserName, RealName, Email, Phone, Krn, CreateDate,
  ConsoleLogin, EnableMFA, PwdFreeze, Path, Remark, Id
ListUserResult/IsTruncated, Total
```

`Krn` = `krn:ksc:iam::2000192131:user/eton` — the `2000192131` is the
**main account id**. 7 users exist in this directory.

## `ListAccounts` (GET) — empty here

Returns XML with `<Accounts></Accounts>` and `TotalCount 0`: this account is
not a financial-management main account, so there are **no 托管子账号**.
Do not rely on it; fall back to IAM users / API-key owners.

## `QueryTokenData` (GET)

```
GET /?Action=QueryTokenData&Version=2024-06-12
    &StartTimestamp=1794787200000&EndTimestamp=1794873599999
    &GroupBy=model&MaxResults=10000[&KeyId=…][&ReasoningType=normal][&SubAccountId=…][&LastKey=…]
```

* Timestamps are **epoch milliseconds**. Passing seconds produces
  `查询时间范围过长` (it reads 1970 → now).
* Ranges longer than ~1 day also fail with `查询时间范围过长`; query day by
  day.
* A 1-hour range returned 0 rows — the API needs a full-day window.
* `GroupBy` = `model` | `keyId` | `endpointId`; the value comes back as
  `GroupByValue` (and also `Model` / the key id respectively).
* **No account field.** `SubAccountId` is accepted but is the managed
  sub-account filter, not the IAM user.
* Only `ReasoningType=normal` returned data (`batch`/`web` were empty —
  the model has `Batch: false`).

Response row fields used:

```
Model, GroupByValue, InputToken, OutputToken, TotalToken,
TotalCacheToken, TotalCacheMissToken, TotalWebSearch, TotalWebParser,
TotalImageCount, TotalVideoDuration, TotalVoiceWordCount,
TotalCreationCache5mToken, TotalCreationCache1hToken, …
```

Top-level `Sum*` fields carry period aggregates (ignored; rows are summed).
Pagination uses `LastKey`.

Observed 2026-11-16: 151,798,555 input / 516,482 output tokens across 5 models.

## `QueryItemBills` (POST JSON)

```
POST /?Action=QueryItemBills&Version=2025-08-01
{"CustomerBillMonth":202611,"PayType":1,"Size":1000,"LastSortValue":"","ProductGroupCode":"KSP"}
```

Response `Data.Results[]` + `Data.LastSortValue` (cursor).

Fields worth knowing:

| Field | Example | Notes |
|---|---|---|
| `BillStartTime` / `BillEndTime` | `2026-11-03 16:00:00` | hourly lines |
| `BillRealAmount` | `0.020000` | the official amount (after discount) |
| `Cash` / `Reward` / `Vouchers` | | split of the amount |
| `UserId` | `2000192131` | **main** account id — not a RAM user |
| `LoginName` / `UserName` | `xin1zp` / `新医智普（北京）科技有限公司` | |
| `ProjectId` / `ProjectName` | `0` / `默认项目` | |
| `ProductGroupName` | `金山云星流模型服务` | product group code resolves to `KSP` |
| `ProductTypeName` | `模型服务` | |
| `ItemName` | `deepseek-v4.1-flash-1017` | **model name** |
| `BillItemCode` | `DEEPSEEK-V4.1-FLASH-1017` | upper-cased model |
| `InstanceId` | `2000192131_<32-hex>` | `<main account>_<request hash>` |
| `ConfigInfo` | `["deepseek-v4.1-flash-1017(千Token):24.8340"]` | list of strings, **not** a JSON object |
| `PriceFactorInfo` | `["TOKEN类型:请求token-cache未命中", …]` | token type of the line |
| `ItemPrice` / `ItemPriceUnit` | `0.002` / `千Token/小时` | |
| `BillItemValue` / `BillItemUnit` | `24.834` / `千Token` | quantity |
| `Discount` / `ContractItemPrice` | `0.5` / `0.001` | |

`ConfigInfo`/`ExtraInfo`/`TagInfo` are searched recursively by
`normalize._identifiers_from_bill`; a plain-string list is parsed with
`_model_from_price_labels` (text before `(`/`（`/`:`).

## `QueryProductConsume` / `QueryProjectConsume` (GET)

```
GET /?Action=QueryProductConsume&Version=2022-12-22&StartDay=2026-11-16&EndDay=2026-11-16&Page=1&Size=200
GET /?Action=QueryProjectConsume&Version=2022-12-22&…
```

Product row: `billDay`, `productGroupId` `311`, `productGroupName`
`金山云星流模型服务`, `realtimeBillsAmount`, `billFinalAmount`, `cash`,
`reward`, `vouchers`, `cloudTicketDenomination`.

Project row: `billDay`, `projectId` `127860`, `projectName` `ksyun-trial`,
`realtimeBillsAmount`, `postPayBillsAmount`, `billFinalAmount`, ….

**The current day keeps accruing**: at 16:53 the 2026-11-16 product total was
`4.29`; at 17:24 the same day read `6.55`. Treat "today" as provisional.

## `DescribeModels` / `GetModelDetail`

`DescribeModels` = `Marker`/`MaxResults` paging, `ModelSet[]` of
`{ModelId, ModelName}`.

`GetModelDetail?ModelId=<id>` returns a flat object where
`OnlineCalculateInput` / `OnlineCalculateOutput` / `OnlineCalculateRight` /
`PricingRules` are **`null`**. The real prices are nested:

```json
"ExtraFields": {
  "priceConfig": {
    "priceItem": [
      {"label": "在线推理输入",     "price": 2.0,  "unit": "元/百万Token"},
      {"label": "在线推理输出",     "price": 8.0,  "unit": "元/百万Token"},
      {"label": "在线推理缓存命中", "price": 0.04, "unit": "元/百万Token"}
    ],
    "pricingRules": []
  }
}
```

## `ListProductGroups` (GET)

`{Data: [{Id, Code, Name, EnName}, …]}`; `星流` resolves to
`Code = "KSP"`, `Name = "金山云星流模型服务"`.
