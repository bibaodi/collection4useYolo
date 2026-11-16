"""Typed wrappers around the KSYUN OpenAPI actions used by this app.

Response envelopes vary between actions and API versions, so the extractors
below are deliberately tolerant: they search the payload for the most
specific list of records that contains a known identifier field.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, Mapping

from .client import KsyunClient

log = logging.getLogger(__name__)

AICP_VERSION = "2024-06-12"
BILL_ITEM_VERSION = "2025-08-01"
BILL_CONSUME_VERSION = "2022-12-22"
PRODUCT_GROUP_VERSION = "2020-01-01"
IAM_VERSION = "2015-11-01"
RM_VERSION = "2021-03-20"


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------
def iter_dicts(obj: Any) -> Iterator[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def find_records(payload: Any, *identifier_keys: str) -> list[dict[str, Any]]:
    """Return the longest list of dicts that contains any identifier key."""
    candidates: list[list[dict[str, Any]]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, list):
            if obj and all(isinstance(item, dict) for item in obj):
                if any(
                    key in item for item in obj for key in identifier_keys
                ):
                    candidates.append(obj)
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value)

    walk(payload)
    if not candidates:
        return []
    return max(candidates, key=len)


def find_dict(payload: Any, *identifier_keys: str) -> dict[str, Any]:
    for node in iter_dicts(payload):
        if any(key in node for key in identifier_keys):
            return node
    return payload if isinstance(payload, dict) else {}


def get_path(obj: Any, *path: str, default: Any = None) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _has_more(payload: Any, page_len: int, page_size: int, seen: int) -> bool:
    if page_len == 0:
        return False
    total = _first_not_none(
        get_path(payload, "TotalCount"),
        get_path(payload, "Total"),
        get_path(payload, "Data", "TotalCount"),
        get_path(payload, "Data", "Total"),
    )
    if total is not None:
        try:
            return seen < int(total)
        except (TypeError, ValueError):
            pass
    truncated = _first_not_none(
        get_path(payload, "IsTruncated"),
        get_path(payload, "Data", "IsTruncated"),
    )
    if truncated is not None:
        return bool(truncated)
    return page_len >= page_size


def _paginate_marker(
    fetch_page,
    *identifier_keys: str,
    page_size: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    """Page through a ``Marker``/``MaxResults`` endpoint (1-based)."""
    collected: list[dict[str, Any]] = []
    for marker in range(1, max_pages + 1):
        payload = fetch_page(marker)
        page = find_records(payload, *identifier_keys)
        if not page:
            break
        collected.extend(page)
        if not _has_more(payload, len(page), page_size, len(collected)):
            break
    return collected


# ---------------------------------------------------------------------------
# 星流 / MaaS (AICP)
# ---------------------------------------------------------------------------
def describe_models(client: KsyunClient, *, page_size: int = 100, max_pages: int = 1000) -> list[dict]:
    return _paginate_marker(
        lambda marker: client.post_form(
            {"Action": "DescribeModels", "Version": AICP_VERSION},
            {"Marker": marker, "MaxResults": page_size},
        ),
        "ModelId",
        "ModelName",
        page_size=page_size,
        max_pages=max_pages,
    )


def get_model_detail(client: KsyunClient, model_id: str) -> dict[str, Any]:
    payload = client.post_form(
        {"Action": "GetModelDetail", "Version": AICP_VERSION},
        {"ModelId": model_id},
    )
    return find_dict(payload, "ModelId")


def describe_apikeys(
    client: KsyunClient, *, page_size: int = 100, max_pages: int = 1000
) -> list[dict]:
    return _paginate_marker(
        lambda marker: client.post_form(
            {"Action": "DescribeApikeys", "Version": AICP_VERSION},
            {"Marker": marker, "MaxResults": page_size, "DefaultKey": "false"},
        ),
        "KeyId",
        page_size=page_size,
        max_pages=max_pages,
    )


def query_token_data(
    client: KsyunClient,
    *,
    start_ms: int,
    end_ms: int,
    key_id: str | None = None,
    group_by: str = "model",
    reasoning_type: str | None = None,
    sub_account_id: int | str | None = None,
    page_size: int = 10000,
    max_pages: int = 1000,
) -> list[dict[str, Any]]:
    """Fetch token-usage rows, following the ``LastKey`` cursor."""
    collected: list[dict[str, Any]] = []
    last_key: str | None = None

    for _ in range(max_pages):
        query: dict[str, Any] = {
            "Action": "QueryTokenData",
            "Version": AICP_VERSION,
            "StartTimestamp": start_ms,
            "EndTimestamp": end_ms,
            "GroupBy": group_by,
            "MaxResults": page_size,
        }
        if key_id:
            query["KeyId"] = key_id
        if reasoning_type:
            query["ReasoningType"] = reasoning_type
        if sub_account_id is not None:
            query["SubAccountId"] = sub_account_id
        if last_key:
            query["LastKey"] = last_key

        payload = client.get(query)
        page = find_records(payload, "GroupByValue", "KeyId", "ModelName", "UsageDate")
        collected.extend(page)

        next_key = _first_not_none(
            get_path(payload, "LastKey"),
            get_path(payload, "Data", "LastKey"),
        )
        if not page or not next_key or str(next_key) == str(last_key):
            break
        last_key = str(next_key)

    return collected


# ---------------------------------------------------------------------------
# 费用中心 (bill-union)
# ---------------------------------------------------------------------------
def query_item_bills(
    client: KsyunClient,
    *,
    bill_month: str | int,
    product_group_code: str | None = None,
    pay_type: int = 1,
    size: int = 1000,
    sub_account_id: int | str | None = None,
    max_pages: int = 1000,
) -> list[dict[str, Any]]:
    """Fetch itemised bills for one month, following ``LastSortValue``."""
    collected: list[dict[str, Any]] = []
    last_sort_value: str = ""

    for _ in range(max_pages):
        body: dict[str, Any] = {
            "CustomerBillMonth": int(bill_month),
            "PayType": pay_type,
            "Size": size,
            "LastSortValue": last_sort_value,
        }
        if product_group_code:
            body["ProductGroupCode"] = product_group_code
        if sub_account_id is not None:
            try:
                body["SubAccountId"] = int(sub_account_id)
            except (TypeError, ValueError):
                body["SubAccountId"] = sub_account_id

        payload = client.post_json(
            {"Action": "QueryItemBills", "Version": BILL_ITEM_VERSION}, body
        )
        page = find_records(payload, "BillItemCode", "BillRealAmount", "ItemName")
        collected.extend(page)

        next_sort = _first_not_none(
            get_path(payload, "Data", "LastSortValue"),
            get_path(payload, "LastSortValue"),
        )
        if not page or not next_sort or next_sort == last_sort_value:
            break
        last_sort_value = str(next_sort)

    return collected


def query_product_consume(
    client: KsyunClient,
    *,
    start_day: str,
    end_day: str,
    page_size: int = 200,
    max_pages: int = 1000,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get(
            {
                "Action": "QueryProductConsume",
                "Version": BILL_CONSUME_VERSION,
                "StartDay": start_day,
                "EndDay": end_day,
                "Page": page,
                "Size": page_size,
            }
        )
        rows = find_records(payload, "billDay", "productGroupId", "productGroupName")
        collected.extend(rows)
        if len(rows) < page_size:
            break
    return collected


def query_project_consume(
    client: KsyunClient,
    *,
    start_day: str,
    end_day: str,
    page_size: int = 200,
    max_pages: int = 1000,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get(
            {
                "Action": "QueryProjectConsume",
                "Version": BILL_CONSUME_VERSION,
                "StartDay": start_day,
                "EndDay": end_day,
                "Page": page,
                "Size": page_size,
            }
        )
        rows = find_records(payload, "billDay", "projectId", "projectName")
        collected.extend(rows)
        if len(rows) < page_size:
            break
    return collected


def list_product_groups(client: KsyunClient) -> list[dict[str, Any]]:
    payload = client.get(
        {"Action": "ListProductGroups", "Version": PRODUCT_GROUP_VERSION}
    )
    data = get_path(payload, "Data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return find_records(payload, "Code", "Name")


# ---------------------------------------------------------------------------
# IAM / Resource Manager (RAM sub-accounts)
# ---------------------------------------------------------------------------
def list_accounts(
    client: KsyunClient, *, page_size: int = 100, max_pages: int = 1000
) -> list[dict[str, Any]]:
    """List member accounts (成员账号) under the current resource directory."""
    collected: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get(
            {
                "Action": "ListAccounts",
                "Version": RM_VERSION,
                "PageNumber": page,
                "PageSize": page_size,
                "IsAll": "true",
            }
        )
        rows = find_records(payload, "AccountId", "AccountName")
        collected.extend(rows)
        if len(rows) < page_size:
            break
    return collected


def _xml_items(text: str) -> tuple[list[dict[str, Any]], bool | None]:
    """Parse KSYUN's XML responses (``<member><Item>…``) into dicts."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [], None

    records: list[dict[str, Any]] = []
    for item in root.iter("Item"):
        record = {child.tag: (child.text or "").strip() for child in item}
        if record:
            records.append(record)

    truncated = root.findtext(".//IsTruncated")
    return records, (truncated.lower() == "true") if truncated else None


def list_users(
    client: KsyunClient, *, page_size: int = 100, max_pages: int = 1000
) -> list[dict[str, Any]]:
    """List IAM users (子用户) — the ``CreateUserId`` of an API key.

    The IAM endpoint answers with XML rather than JSON, so both shapes are
    handled here.
    """
    collected: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get(
            {
                "Action": "ListUsers",
                "Version": IAM_VERSION,
                "Page": page,
                "PageSize": page_size,
            }
        )

        truncated: bool | None = None
        if isinstance(payload, dict) and isinstance(payload.get("_raw"), str):
            rows, truncated = _xml_items(payload["_raw"])
        else:
            rows = find_records(payload, "UserId", "UserName")

        collected.extend(rows)
        if truncated is False:
            break
        if len(rows) < page_size:
            break
    return collected
