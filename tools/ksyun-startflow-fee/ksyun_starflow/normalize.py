"""Normalise raw KSYUN API payloads into flat, CSV-friendly dicts.

Field names differ between API versions (``BillRealAmount`` vs
``bill_real_amount``, ``ModelName`` vs ``GroupByValue``...), so every
accessor is case-insensitive and accepts several aliases.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator, Mapping

MONEY = Decimal("0")


# ---------------------------------------------------------------------------
# Scalar coercion
# ---------------------------------------------------------------------------
def pick(mapping: Mapping[str, Any] | None, *names: str, default: Any = None) -> Any:
    """Case-insensitive field lookup across aliases."""
    if not isinstance(mapping, Mapping):
        return default
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return default


def to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return default


def to_decimal(value: Any, default: Decimal = MONEY) -> Decimal:
    if value in (None, ""):
        return default
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    text = str(value).strip().replace(",", "")
    for symbol in ("¥", "￥", "$", " "):
        text = text.replace(symbol, "")
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return default


def parse_jsonish(value: Any) -> Any:
    """Decode JSON strings; return dict/list as-is; otherwise ``{}``."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {"_raw": value}


def iter_dicts(obj: Any) -> Iterator[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def _deep_find(obj: Any, *names: str) -> Any:
    for node in iter_dicts(obj):
        value = pick(node, *names)
        if value not in (None, ""):
            return value
    return ""


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def normalise_date(value: Any, fallback: str = "") -> str:
    """Return ``YYYY-MM-DD`` from a variety of date formats."""
    if value in (None, ""):
        return fallback
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return fallback
    # Millisecond / second epoch
    if re.fullmatch(r"\d{13}", text):
        return datetime.fromtimestamp(int(text) / 1000, tz=timezone.utc).date().isoformat()
    if re.fullmatch(r"\d{10}", text):
        return datetime.fromtimestamp(int(text), tz=timezone.utc).date().isoformat()
    # 2026-07-31, 2026/07/31, 20260731, with optional time
    match = re.match(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return fallback


def month_of(day: str) -> str:
    return day[:7].replace("-", "")


# ---------------------------------------------------------------------------
# Model / API key
# ---------------------------------------------------------------------------
def normalize_model_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": str(pick(raw, "ModelId", "modelId", "Id", default="")),
        "model_name": str(pick(raw, "ModelName", "modelName", "Name", default="")),
        "provider": str(pick(raw, "Provider", "provider", default="")),
        "metering_mode": str(pick(raw, "MeteringMode", "meteringMode", default="")),
        "context_length": to_int(pick(raw, "ContextLength", "contextLength")),
        "free_total_quota": to_int(pick(raw, "FreeTotalQuota", "freeTotalQuota")),
        "free_used_quota": to_int(pick(raw, "FreeUsedQuota", "freeUsedQuota")),
        "batch": str(pick(raw, "Batch", "batch", default="")),
        "cache_context": str(pick(raw, "CacheContext", "cacheContext", default="")),
        "input_types": "|".join(map(str, pick(raw, "InputType", "inputType", default=[]) or [])),
        "output_types": "|".join(map(str, pick(raw, "OutputType", "outputType", default=[]) or [])),
        "active_status": str(pick(raw, "ActiveStatus", "activeStatus", default="")),
        "raw": raw,
    }


ACCOUNT_TYPE_IAM_USER = "iam_user"
ACCOUNT_TYPE_MANAGED = "managed_account"
ACCOUNT_TYPE_CUSTOM = "custom"
ACCOUNT_TYPE_ROOT = "root"
ACCOUNT_TYPE_UNATTRIBUTED = "unattributed"


def normalize_api_key_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one ``DescribeApikeys`` record.

    The API key's owner is returned as ``CreateUserId`` / ``CreateUserName``
    (an IAM user, e.g. ``eton``).  Managed / member sub-accounts, when they
    apply, surface as ``SubAccountId`` / ``SubAccountName``.  Both are
    normalised into ``account_id`` / ``account_name`` / ``account_type`` so
    every report can be indexed by RAM account.
    """
    associated_ids = pick(raw, "AssociatedModelIds", "associatedModelIds", default=[]) or []
    owner_user_id = str(pick(raw, "CreateUserId", "OwnerUserId", "UserId", default="") or "")
    owner_user_name = str(pick(raw, "CreateUserName", "OwnerUserName", "UserName", default="") or "")
    managed_id = str(pick(raw, "SubAccountId", "AccountId", default="") or "")
    managed_name = str(pick(raw, "SubAccountName", "AccountName", default="") or "")

    if managed_id:
        account_id, account_name, account_type = managed_id, managed_name, ACCOUNT_TYPE_MANAGED
    elif owner_user_id:
        # A numeric owner id is the main account (root), not an IAM sub-user.
        account_type = ACCOUNT_TYPE_ROOT if owner_user_id.isdigit() else ACCOUNT_TYPE_IAM_USER
        account_id, account_name = owner_user_id, owner_user_name
    else:
        account_id = account_name = account_type = ""

    return {
        "key_id": str(pick(raw, "KeyId", "keyId", "Id", default="")),
        "key_name": str(pick(raw, "Name", "KeyName", "keyName", default="")),
        "project_id": str(pick(raw, "ProjectId", "projectId", default="")),
        "project_name": str(pick(raw, "ProjectName", "projectName", default="")),
        "status": str(pick(raw, "Status", "status", default="")),
        "owner_user_id": owner_user_id,
        "owner_user_name": owner_user_name,
        "account_id": account_id,
        "account_name": account_name,
        "account_type": account_type,
        "associated_model_ids": "|".join(map(str, associated_ids)),
        "all_associated_model": str(pick(raw, "AllAssociatedModel", "allAssociatedModel", default="")),
        "create_timestamp": str(pick(raw, "CreateTimestamp", "createTimestamp", default="")),
        "raw": raw,
    }


def normalize_user_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one IAM ``ListUsers`` record (``UserId`` -> ``UserName``)."""
    krn = str(pick(raw, "Krn", "krn", default="") or "")
    main_account_id = ""
    match = re.search(r"::(\d+):", krn)
    if match:
        main_account_id = match.group(1)

    user_name = str(pick(raw, "UserName", "userName", default="") or "")
    return {
        "account_id": str(pick(raw, "UserId", "userId", default="") or ""),
        "account_name": user_name,
        "account_type": ACCOUNT_TYPE_IAM_USER,
        "login_name": user_name,
        "real_name": str(pick(raw, "RealName", "realName", default="") or ""),
        "email": str(pick(raw, "Email", "email", default="") or ""),
        "main_account_id": main_account_id,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------
def normalize_token_row(
    raw: Mapping[str, Any],
    *,
    usage_date: str,
    reasoning_type: str,
    key_id: str = "",
    account_id: str = "",
    account_name: str = "",
    account_type: str = "",
    group_by: str = "model",
) -> dict[str, Any]:
    group_value = pick(raw, "GroupByValue", "groupByValue", default="")
    if group_by.lower() in {"keyid", "key_id"}:
        key_id = str(group_value or key_id)
    model_name = pick(raw, "Model", "ModelName", "modelName", default="")
    if not model_name and group_by.lower() == "model":
        model_name = group_value

    return {
        "usage_date": normalise_date(
            pick(raw, "UsageDate", "usageDate", "BillDay", "Day", "Date"), fallback=usage_date
        ),
        "account_id": str(
            pick(raw, "SubAccountId", "subAccountId", "AccountId", "accountId", default=account_id)
        ),
        "account_name": str(
            pick(raw, "SubAccountName", "subAccountName", "AccountName", "accountName", default=account_name)
        ),
        "account_type": str(pick(raw, "AccountType", "accountType", default=account_type)),
        "project_id": str(pick(raw, "ProjectId", "projectId", default="")),
        "project_name": str(pick(raw, "ProjectName", "projectName", default="")),
        "key_id": str(pick(raw, "KeyId", "keyId", "ApiKeyId", "ApiKey", default=key_id)),
        "model_name": str(model_name),
        "reasoning_type": str(
            pick(raw, "ReasoningType", "reasoningType", "Reasoning", default=reasoning_type)
        ),
        "input_token": to_int(pick(raw, "InputToken", "inputToken", "TotalInputToken", "SumInputToken")),
        "output_token": to_int(pick(raw, "OutputToken", "outputToken", "TotalOutputToken", "SumOutputToken")),
        "total_token": to_int(pick(raw, "TotalToken", "totalToken")),
        "cache_token": to_int(
            pick(raw, "CacheToken", "cacheToken", "TotalCacheToken", "CacheHitToken", "CacheHit")
        ),
        "cache_miss_token": to_int(
            pick(raw, "CacheMissToken", "cacheMissToken", "TotalCacheMissToken", "CacheMiss")
        ),
        "cache_write_5m_token": to_int(
            pick(
                raw,
                "CacheWrite5mToken",
                "cacheWrite5mToken",
                "cache_write_5m_token",
                "CreationCache5mToken",
                "CacheCreation5mToken",
                "CacheWrite5MinToken",
                "TotalCreationCache5mToken",
            )
        ),
        "cache_write_1h_token": to_int(
            pick(
                raw,
                "CacheWrite1hToken",
                "cacheWrite1hToken",
                "cache_write_1h_token",
                "CreationCache1hToken",
                "CacheCreation1hToken",
                "CacheWrite1HToken",
                "TotalCreationCache1hToken",
            )
        ),
        "web_search": to_int(pick(raw, "WebSearch", "webSearch", "TotalWebSearch")),
        "web_parser": to_int(pick(raw, "WebParser", "webParser", "TotalWebParser")),
        "image_count": to_int(pick(raw, "ImageCount", "imageCount", "TotalImageCount")),
        "voice_word_count": to_int(
            pick(raw, "VoiceWordCount", "voiceWordCount", "TotalVoiceWordCount")
        ),
        "video_duration": to_int(
            pick(raw, "VideoDuration", "videoDuration", "TotalVideoDuration")
        ),
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Bills
# ---------------------------------------------------------------------------
def _model_from_price_labels(blob: Any) -> str:
    """Return the model name from ``["model(千Token):24.8"]`` style strings."""
    items = blob if isinstance(blob, list) else [blob]
    for item in items:
        if not isinstance(item, str):
            continue
        match = re.match(r"\s*([^()（）:：]+?)\s*[（(:：]", item)
        if match:
            return match.group(1)
    return ""


def _identifiers_from_bill(raw: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """Extract (model, key, project_id, project_name, account) from a bill.

    ``ConfigInfo`` / ``ExtraInfo`` are free-form JSON (or ``key=value``
    strings), so they are searched recursively for known identifiers.
    """
    config_info = parse_jsonish(pick(raw, "ConfigInfo", "configInfo"))
    blobs = [
        config_info,
        parse_jsonish(pick(raw, "ExtraInfo", "extraInfo")),
        parse_jsonish(pick(raw, "TagInfo", "tagInfo")),
    ]
    combined = [blob for blob in blobs if blob] + [raw]

    model_name = str(
        pick(raw, "ModelName", "modelName", "Model", default="")
        or _deep_find(combined, "ModelName", "modelName", "Model", "ModelId", "modelId")
        or pick(raw, "ItemName", "itemName", default="")
        or _model_from_price_labels(config_info)
        or ""
    )
    key_id = str(
        pick(raw, "KeyId", "keyId", "ApiKeyId", "ApiKey", default="")
        or _deep_find(combined, "KeyId", "keyId", "ApiKeyId", "ApiKey", "ApiKeyID")
    )
    project_id = str(
        pick(raw, "ProjectId", "projectId", default="")
        or _deep_find(combined, "ProjectId", "projectId")
    )
    project_name = str(
        pick(raw, "ProjectName", "projectName", default="")
        or _deep_find(combined, "ProjectName", "projectName")
    )
    account = str(
        pick(raw, "SubAccountId", "subAccountId", "AccountId", "accountId", default="")
        or _deep_find(combined, "SubAccountId", "subAccountId", "AccountId", "accountId", "MemberId")
    )
    return model_name, key_id, project_id, project_name, account


def normalize_bill_row(raw: Mapping[str, Any], *, bill_month: str = "") -> dict[str, Any]:
    bill_date = normalise_date(
        pick(raw, "BillStartTime", "billStartTime", "BillDay", "billDay", "BillDate", "bill_date")
    )
    model_name, key_id, project_id, project_name, account = _identifiers_from_bill(raw)
    return {
        "bill_month": str(bill_month or month_of(bill_date)),
        "bill_date": bill_date,
        "bill_start_time": str(pick(raw, "BillStartTime", "billStartTime", default="")),
        "bill_end_time": str(pick(raw, "BillEndTime", "billEndTime", default="")),
        "bill_real_amount": to_decimal(
            pick(raw, "BillRealAmount", "billRealAmount", "RealAmount", "BillAmount")
        ),
        "cash": to_decimal(pick(raw, "Cash", "cash")),
        "reward": to_decimal(pick(raw, "Reward", "reward")),
        "vouchers": to_decimal(pick(raw, "Vouchers", "vouchers")),
        "project_id": project_id,
        "project_name": project_name,
        "account_id": account,
        "account_name": str(
            pick(raw, "SubAccountName", "subAccountName", "AccountName", "accountName", default="")
        ),
        "billing_account_id": str(pick(raw, "UserId", "userId", default="")),
        "billing_account_name": str(pick(raw, "LoginName", "loginName", default="")),
        "product_group_code": str(
            pick(raw, "ProductGroupCode", "productGroupCode", default="")
        ),
        "product_group_name": str(
            pick(raw, "ProductGroupName", "productGroupName", default="")
        ),
        "product_type_name": str(
            pick(raw, "ProductTypeName", "productTypeName", default="")
        ),
        "bill_item_code": str(pick(raw, "BillItemCode", "billItemCode", default="")),
        "item_name": str(pick(raw, "ItemName", "itemName", default="")),
        "instance_id": str(pick(raw, "InstanceId", "instanceId", default="")),
        "instance_name": str(pick(raw, "InstanceName", "instanceName", default="")),
        "model_name": model_name,
        "key_id": key_id,
        "config_info": json.dumps(parse_jsonish(pick(raw, "ConfigInfo", "configInfo")), ensure_ascii=False)
        if pick(raw, "ConfigInfo", "configInfo") not in (None, "")
        else "",
        "extra_info": json.dumps(parse_jsonish(pick(raw, "ExtraInfo", "extraInfo")), ensure_ascii=False)
        if pick(raw, "ExtraInfo", "extraInfo") not in (None, "")
        else "",
        "raw": raw,
    }


def normalize_product_consume_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bill_day": normalise_date(pick(raw, "billDay", "BillDay", "bill_day")),
        "product_group_id": str(pick(raw, "productGroupId", "ProductGroupId", default="")),
        "product_group_name": str(pick(raw, "productGroupName", "ProductGroupName", default="")),
        "realtime_bills_amount": to_decimal(
            pick(raw, "realtimeBillsAmount", "RealtimeBillsAmount")
        ),
        "bill_final_amount": to_decimal(pick(raw, "billFinalAmount", "BillFinalAmount")),
        "cash": to_decimal(pick(raw, "cash", "Cash")),
        "reward": to_decimal(pick(raw, "reward", "Reward")),
        "vouchers": to_decimal(pick(raw, "vouchers", "Vouchers")),
        "cloud_ticket_denomination": to_decimal(
            pick(raw, "cloudTicketDenomination", "CloudTicketDenomination")
        ),
        "raw": raw,
    }


def normalize_project_consume_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bill_day": normalise_date(pick(raw, "billDay", "BillDay", "bill_day")),
        "project_id": str(pick(raw, "projectId", "ProjectId", default="")),
        "project_name": str(pick(raw, "projectName", "ProjectName", default="")),
        "realtime_bills_amount": to_decimal(
            pick(raw, "realtimeBillsAmount", "RealtimeBillsAmount")
        ),
        "postpay_bills_amount": to_decimal(
            pick(raw, "postPayBillsAmount", "PostPayBillsAmount")
        ),
        "bill_final_amount": to_decimal(pick(raw, "billFinalAmount", "BillFinalAmount")),
        "cash": to_decimal(pick(raw, "cash", "Cash")),
        "reward": to_decimal(pick(raw, "reward", "Reward")),
        "vouchers": to_decimal(pick(raw, "vouchers", "Vouchers")),
        "raw": raw,
    }


def normalize_account_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "account_id": str(pick(raw, "AccountId", "accountId", "Id", "UserId", default="")),
        "account_name": str(
            pick(raw, "AccountName", "accountName", "DisplayName", "UserName", "RealName", default="")
        ),
        "account_type": str(pick(raw, "Type", "type", default="")),
        "raw": raw,
    }
