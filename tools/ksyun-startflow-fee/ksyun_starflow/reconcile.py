"""Reconcile estimated token costs with official 星流 bills.

Implements section 9 of the workflow doc, adapted so the result can be
indexed by **RAM account × model**:

* official item bills are pooled per (day, model) — the finest grain the
  bill API exposes — and allocated to the estimated rows of that model
  pro-rata by estimated cost (``FINAL`` when a model has a single account,
  ``ALLOCATED`` when it is shared);
* the part of the official 星流 total that item bills do not cover is
  allocated across the remaining estimated rows pro-rata by estimated cost;
* rows with no official counterpart stay ``ESTIMATED``;
* bills whose model never appears in token data become ``FINAL`` rows of
  their own, so the official total is never lost.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence

from .normalize import MONEY

log = logging.getLogger(__name__)

AMOUNT_ESTIMATED = "ESTIMATED"
AMOUNT_ALLOCATED = "ALLOCATED"
AMOUNT_FINAL = "FINAL"


@dataclass
class ReconciledRow:
    usage_date: str
    model_name: str
    key_id: str
    project_id: str
    estimated_cost: Decimal
    official_cost: Decimal
    allocated_cost: Decimal
    amount_type: str
    bill_month: str
    updated_at: str
    account_id: str = ""
    account_name: str = ""
    account_type: str = ""
    project_name: str = ""
    bill_date: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _amount(row: Mapping[str, Any]) -> Decimal:
    return Decimal(str(row.get("bill_real_amount") or 0))


def _estimated_amount(row: Mapping[str, Any]) -> Decimal:
    return Decimal(str(row.get("estimated_cost") or 0))


def _account_fields(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "account_id": str(row.get("account_id") or ""),
        "account_name": str(row.get("account_name") or ""),
        "account_type": str(row.get("account_type") or ""),
    }


def reconcile(
    estimated_rows: Sequence[Mapping[str, Any]],
    official_bills: Sequence[Mapping[str, Any]],
    *,
    bill_month: str = "",
    official_total: Decimal | None = None,
) -> list[ReconciledRow]:
    """Return reconciled rows.

    ``estimated_rows`` need ``usage_date``, ``model_name``, ``key_id``,
    ``project_id``, ``estimated_cost`` and optionally ``account_*``.
    ``official_bills`` are normalised bill rows (``normalize_bill_row``).
    ``official_total`` is the 星流 platform amount for the period; when
    omitted the sum of item bills is used.
    """
    # -- pool item bills by (day, model) --------------------------------
    bill_pool: dict[tuple[str, str], Decimal] = {}
    unmatched_bill_rows: list[Mapping[str, Any]] = []
    direct_total = MONEY
    for bill in official_bills:
        amount = _amount(bill)
        direct_total += amount
        model_name = str(bill.get("model_name") or "")
        bill_date = str(bill.get("bill_date") or "")
        if model_name:
            bill_pool[(bill_date, model_name)] = (
                bill_pool.get((bill_date, model_name), MONEY) + amount
            )
        else:
            unmatched_bill_rows.append(bill)

    # -- group estimated rows by (day, model) ---------------------------
    groups: dict[tuple[str, str], list[int]] = {}
    for index, estimated in enumerate(estimated_rows):
        key = (str(estimated.get("usage_date") or ""), str(estimated.get("model_name") or ""))
        groups.setdefault(key, []).append(index)

    rows: list[ReconciledRow | None] = [None] * len(estimated_rows)

    for (usage_date, model_name), indexes in groups.items():
        official = bill_pool.pop((usage_date, model_name), MONEY)
        estimated_total = sum(_estimated_amount(estimated_rows[i]) for i in indexes)
        single = len(indexes) == 1

        for index in indexes:
            estimated = estimated_rows[index]
            estimated_cost = _estimated_amount(estimated)

            if official > 0:
                if estimated_total > 0:
                    share = estimated_cost / estimated_total
                else:
                    share = Decimal(1) / Decimal(len(indexes))
                official_cost = official * share
                amount_type = AMOUNT_FINAL if single else AMOUNT_ALLOCATED
            else:
                official_cost = MONEY
                amount_type = AMOUNT_ESTIMATED

            rows[index] = ReconciledRow(
                usage_date=usage_date,
                model_name=model_name,
                key_id=str(estimated.get("key_id") or ""),
                project_id=str(estimated.get("project_id") or ""),
                project_name=str(estimated.get("project_name") or ""),
                estimated_cost=estimated_cost,
                official_cost=official_cost,
                allocated_cost=official_cost,
                amount_type=amount_type,
                bill_month=bill_month,
                updated_at=_now(),
                bill_date=usage_date,
                source="item_bill" if official > 0 else "token_data",
                **_account_fields(estimated),
            )

    # -- bills with no model identifier ---------------------------------
    for bill in unmatched_bill_rows:
        amount = _amount(bill)
        rows.append(
            ReconciledRow(
                usage_date=str(bill.get("bill_date") or ""),
                model_name="",
                key_id=str(bill.get("key_id") or ""),
                project_id=str(bill.get("project_id") or ""),
                project_name=str(bill.get("project_name") or ""),
                estimated_cost=MONEY,
                official_cost=amount,
                allocated_cost=amount,
                amount_type=AMOUNT_FINAL,
                bill_month=bill_month or str(bill.get("bill_month") or ""),
                updated_at=_now(),
                bill_date=str(bill.get("bill_date") or ""),
                source="item_bill_unmatched",
                **_account_fields(bill),
            )
        )

    # -- models billed but absent from token data ------------------------
    for (bill_date, model_name), amount in bill_pool.items():
        rows.append(
            ReconciledRow(
                usage_date=bill_date,
                model_name=model_name,
                key_id="",
                project_id="",
                project_name="",
                estimated_cost=MONEY,
                official_cost=amount,
                allocated_cost=amount,
                amount_type=AMOUNT_FINAL,
                bill_month=bill_month,
                updated_at=_now(),
                bill_date=bill_date,
                source="item_bill_unmatched",
            )
        )

    final_rows = [row for row in rows if row is not None]

    # -- residual official amount (product consume > item bills) --------
    if official_total is None:
        official_total = direct_total
    else:
        official_total = Decimal(str(official_total))

    remaining = official_total - direct_total
    if remaining < 0:
        log.warning(
            "Official 星流 total %s is less than matched item bills %s; "
            "allocation remainder clamped to 0",
            official_total,
            direct_total,
        )
        remaining = MONEY

    pending = [row for row in final_rows if row.estimated_cost > 0]
    pending_total = sum(row.estimated_cost for row in pending)

    if remaining > 0:
        if pending_total > 0:
            # Doc section 9: residual × estimated_cost / total_estimated_cost,
            # spread over every estimated row so the account totals add up.
            for row in pending:
                row.allocated_cost += remaining * row.estimated_cost / pending_total
                if row.amount_type == AMOUNT_ESTIMATED:
                    row.amount_type = AMOUNT_ALLOCATED
        else:
            final_rows.append(
                ReconciledRow(
                    usage_date="",
                    model_name="UNALLOCATED",
                    key_id="",
                    project_id="",
                    estimated_cost=MONEY,
                    official_cost=MONEY,
                    allocated_cost=remaining,
                    amount_type=AMOUNT_ALLOCATED,
                    bill_month=bill_month,
                    updated_at=_now(),
                    bill_date="",
                    source="product_consume_residual",
                )
            )

    return final_rows


def aggregate(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    value_fields: Sequence[str] = ("estimated_cost", "official_cost", "allocated_cost"),
) -> list[dict[str, Any]]:
    """Sum ``value_fields`` grouped by ``group_fields`` (order preserving)."""
    buckets: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        bucket = buckets.setdefault(
            key,
            {
                **{field: str(row.get(field, "")) for field in group_fields},
                **{field: MONEY for field in value_fields},
                "row_count": 0,
            },
        )
        for field in value_fields:
            bucket[field] += Decimal(str(row.get(field) or 0))
        bucket["row_count"] += 1
    return list(buckets.values())
