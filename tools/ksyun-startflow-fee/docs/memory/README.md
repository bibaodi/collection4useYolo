# Memory / project notes — ksyun-starflow

Working notes for the 星流 (MaaS) daily cost report. **Read this first after
moving the project.** Everything here was verified against the live KSYUN
OpenAPI on 2026-11-16 (account `2000192131`, project `127860 ksyun-trial`,
bill month `202611`).

| File | Contents |
|---|---|
| [01-ksyun-api-notes.md](01-ksyun-api-notes.md) | Endpoints, versions, exact request/response shapes of every action used |
| [02-account-attribution.md](02-account-attribution.md) | How an API key is tied to a RAM account; observed key → account mapping |
| [03-pricing-and-reconciliation.md](03-pricing-and-reconciliation.md) | Where prices really live, unit handling, cost + reconcile formulas |
| [04-project-map-and-workflows.md](04-project-map-and-workflows.md) | Module map, how to run, tests, report inventory |
| [05-pitfalls-and-decisions.md](05-pitfalls-and-decisions.md) | Traps hit this session and why the code looks the way it does |
| [06-move-checklist.md](06-move-checklist.md) | What to copy / not copy, how to re-verify after moving |

## 30-second summary

* `DescribeApikeys` returns the key **owner** as `CreateUserId` /
  `CreateUserName` — that is the only reliable key → RAM account link.
* `QueryTokenData` has **no account column**; usage must be queried per API
  key and the owner attached client-side.
* Real model prices are **not** in `OnlineCalculate*` (those are `null`);
  they are in `ExtraFields.priceConfig.priceItem` with Chinese labels and
  units (`元/百万Token`, `元/千Token`, `元/千次`).
* Bill line items carry the model in `ItemName` (and in `ConfigInfo` strings
  like `"model(千Token):24.8"`), and the **main** account in `UserId`.
  There is no RAM user on a bill line — account split is therefore
  estimated-share based.
