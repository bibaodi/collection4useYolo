"""CSV / JSON report writers."""

from __future__ import annotations

import csv
import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

log = logging.getLogger(__name__)

TOKEN_USAGE_FIELDS = [
    "usage_date",
    "account_id",
    "account_name",
    "account_type",
    "project_id",
    "project_name",
    "key_id",
    "model_name",
    "reasoning_type",
    "input_token",
    "output_token",
    "total_token",
    "cache_token",
    "cache_miss_token",
    "cache_write_5m_token",
    "cache_write_1h_token",
    "web_search",
    "web_parser",
    "image_count",
    "voice_word_count",
    "video_duration",
    "estimated_cost",
]

API_KEY_FIELDS = [
    "key_id",
    "key_name",
    "account_id",
    "account_name",
    "account_type",
    "owner_user_id",
    "owner_user_name",
    "project_id",
    "project_name",
    "status",
    "associated_model_ids",
    "create_timestamp",
]

DAILY_COST_FIELDS = [
    "usage_date",
    "account_id",
    "account_name",
    "account_type",
    "project_id",
    "project_name",
    "key_id",
    "model_name",
    "estimated_cost",
    "official_cost",
    "allocated_cost",
    "amount_type",
    "bill_month",
]

ITEM_BILL_FIELDS = [
    "bill_month",
    "bill_date",
    "bill_start_time",
    "bill_end_time",
    "account_id",
    "account_name",
    "billing_account_id",
    "billing_account_name",
    "project_id",
    "project_name",
    "product_group_code",
    "product_group_name",
    "product_type_name",
    "bill_item_code",
    "item_name",
    "instance_id",
    "instance_name",
    "model_name",
    "key_id",
    "bill_real_amount",
    "cash",
    "reward",
    "vouchers",
    "config_info",
    "extra_info",
]

USER_FIELDS = [
    "account_id",
    "account_name",
    "account_type",
    "login_name",
    "real_name",
    "email",
    "main_account_id",
]

PRODUCT_CONSUME_FIELDS = [
    "bill_day",
    "product_group_id",
    "product_group_name",
    "realtime_bills_amount",
    "bill_final_amount",
    "cash",
    "reward",
    "vouchers",
    "cloud_ticket_denomination",
]

PROJECT_CONSUME_FIELDS = [
    "bill_day",
    "project_id",
    "project_name",
    "realtime_bills_amount",
    "postpay_bills_amount",
    "bill_final_amount",
    "cash",
    "reward",
    "vouchers",
]

RECONCILED_FIELDS = [
    "usage_date",
    "bill_date",
    "bill_month",
    "account_id",
    "account_name",
    "account_type",
    "project_id",
    "project_name",
    "key_id",
    "model_name",
    "estimated_cost",
    "official_cost",
    "allocated_cost",
    "amount_type",
    "source",
    "updated_at",
]

RAM_ACCOUNT_FIELDS = [
    "usage_date",
    "account_id",
    "account_name",
    "account_type",
    "model_name",
    "estimated_cost",
    "official_cost",
    "allocated_cost",
    "row_count",
]

ACCOUNT_SUMMARY_FIELDS = [
    "account_id",
    "account_name",
    "account_type",
    "estimated_cost",
    "official_cost",
    "allocated_cost",
    "row_count",
]


def format_money(value: Any) -> str:
    if value in (None, ""):
        return "0.000000"
    try:
        return f"{Decimal(str(value)):.6f}"
    except (InvalidOperation, ValueError):
        return str(value)


def format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return format_money(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # utf-8-sig keeps Chinese readable when opened in Excel.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_scalar(row.get(field)) for field in fieldnames})
            count += 1
    log.info("wrote %s (%d rows)", path, count)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    log.info("wrote %s", path)
    return path
