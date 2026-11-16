# 02 — Account attribution (key → RAM account)

The original question: **which RAM account (子账号) is incurring 星流 fees?**

## The link

```
DescribeApikeys.KeyId
   → DescribeApikeys.CreateUserId   (opaque IAM UserId, e.g. ARkgUxbRQWc3oWfiL0n8A)
   → DescribeApikeys.CreateUserName (login, e.g. eton)
   → iam ListUsers (UserId → UserName, RealName, Email, Krn)
```

`QueryTokenData` is queried **per API key** and each row inherits the key's
owner. That is the whole attribution mechanism.

`CreateUserId` sometimes holds the numeric main account id (`2000192131`);
in that case the owner is `root` and `account_type = "root"`.

## Observed mapping (2026-11-16, main account 2000192131)

| KeyId | Key name | account_id | account_name | type |
|---|---|---|---|---|
| `API-KEY-1319707462627254272` | `yryKey01` | `pPpqS9YYS1ydNq9xvDwHw` | `ruiying.ye` | iam_user |
| `API-KEY-1319703773481033728` | `zhaiyfKey01` | `IiQ92K6gQVePrctkKWGKQ` | `yafan.zhai` | iam_user |
| `API-KEY-1319694563166937088` | `crsKey01` | `cOYG55pRNeJmdLU1Rvmg` | `ruisong.chen` | iam_user |
| `API-KEY-1319677108893802496` | `etonKey01` | `ARkgUxbRQWc3oWfiL0n8A` | `eton` | iam_user |
| `API-KEY-1315038504406061056` | `test-01` | `2000192131` | `root` | root |

This table was the bug report: `API-KEY-1319677108893802496` belongs to
`eton`, but the first version of the app left `sub_account_id` empty because
it only looked at `SubAccountId`/`AccountName`, which `DescribeApikeys` does
not return.

## Resolution precedence

1. Explicit overrides: `--key-subaccount-map FILE` (KeyId → account id) and
   `--sub-account-id ID` (repeatable).
2. The key's `CreateUserId` / `CreateUserName`.
3. Managed sub-accounts from `ListAccounts` (`managed_account`).
4. `ListUsers` (IAM) enriches names/login/real name/email in `accounts.csv`.

`account_type` values: `iam_user`, `root`, `managed_account`, `custom`
(override only), or empty (`UNATTRIBUTED`).

## Why bills cannot do the split

`QueryItemBills` lines only carry `UserId` = **main** account (`2000192131`)
and `LoginName` (`xin1zp`). There is no RAM user, no API key, and
`ConfigInfo` is just `["model(千Token):24.8"]`. Bills also exist only for a
single model in this data (`deepseek-v4.1-flash-1017`).

Consequence: the **estimated** cost knows the account; the **official**
amount does not. The account split of the official amount is therefore
pro-rata by estimated cost (see
[03-pricing-and-reconciliation.md](03-pricing-and-reconciliation.md)).

## Result, 2026-11-16 (single day)

| account | estimated | official | allocated |
|---|---|---|---|
| ruisong.chen | 11.132506 | 3.066208 | 4.529094 |
| eton | 4.399610 | 1.211777 | 1.789916 |
| ruiying.ye | 1.630978 | 0.002261 | 0.216582 |
| yafan.zhai | 0.035416 | 0.009754 | 0.014408 |
| **total** | **~16.94** | **4.29** | **6.55** |

`allocated` > `official` because product-consume (6.55, still accruing) was
larger than the itemised bills (4.29) at run time.
