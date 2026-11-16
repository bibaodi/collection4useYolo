from __future__ import annotations

from decimal import Decimal

from ksyun_starflow.reconcile import (
    AMOUNT_ALLOCATED,
    AMOUNT_ESTIMATED,
    AMOUNT_FINAL,
    aggregate,
    reconcile,
)


def _estimated(day, key_id, model, cost, account_id="2031447841"):
    return {
        "usage_date": day,
        "key_id": key_id,
        "model_name": model,
        "project_id": "0",
        "project_name": "ksyun-trial",
        "estimated_cost": Decimal(str(cost)),
        "account_id": account_id,
        "account_name": "child",
        "account_type": "managed_account",
    }


def _bill(day, key_id, model, amount, account_id="2031447841"):
    return {
        "bill_date": day,
        "bill_month": "202607",
        "key_id": key_id,
        "model_name": model,
        "project_id": "0",
        "project_name": "ksyun-trial",
        "bill_real_amount": Decimal(str(amount)),
        "account_id": account_id,
        "account_name": "child",
    }


def test_direct_model_match_is_final():
    rows = reconcile(
        [_estimated("2026-07-31", "API-KEY-1", "m1", 10)],
        [_bill("2026-07-31", "API-KEY-1", "m1", 12)],
        bill_month="202607",
        official_total=Decimal("12"),
    )
    assert len(rows) == 1
    assert rows[0].amount_type == AMOUNT_FINAL
    assert rows[0].official_cost == Decimal("12")
    assert rows[0].allocated_cost == Decimal("12")
    assert rows[0].account_name == "child"


def test_model_pool_is_shared_between_accounts():
    """Two accounts on the same model split the official amount pro-rata."""
    estimated = [
        _estimated("2026-07-31", "API-KEY-1", "m1", 10, account_id="a"),
        _estimated("2026-07-31", "API-KEY-2", "m1", 30, account_id="b"),
    ]
    rows = reconcile(
        estimated,
        [_bill("2026-07-31", "", "m1", 8)],
        bill_month="202607",
        official_total=Decimal("8"),
    )
    by_account = {row.account_id: row for row in rows}
    assert by_account["a"].official_cost == Decimal("2")
    assert by_account["b"].official_cost == Decimal("6")
    assert {row.amount_type for row in rows} == {AMOUNT_ALLOCATED}
    assert sum(row.allocated_cost for row in rows) == Decimal("8")


def test_residual_official_amount_allocated_pro_rata():
    estimated = [
        _estimated("2026-07-31", "API-KEY-1", "m1", 10),
        _estimated("2026-07-31", "API-KEY-2", "m2", 30),
    ]
    bills = [_bill("2026-07-31", "API-KEY-1", "m1", 5)]
    rows = reconcile(
        estimated, bills, bill_month="202607", official_total=Decimal("20")
    )
    by_model = {row.model_name: row for row in rows}
    # m1: 5 direct + 15 * 10 / 40
    assert by_model["m1"].official_cost == Decimal("5")
    assert by_model["m1"].allocated_cost == Decimal("8.75")
    # m2: no direct bill, 15 * 30 / 40 residual
    assert by_model["m2"].amount_type == AMOUNT_ALLOCATED
    assert by_model["m2"].allocated_cost == Decimal("11.25")
    assert sum(row.allocated_cost for row in rows) == Decimal("20")


def test_unmatched_bill_becomes_final_row():
    rows = reconcile(
        [_estimated("2026-07-31", "API-KEY-1", "m1", 10)],
        [
            _bill("2026-07-31", "API-KEY-1", "m1", 10),
            _bill("2026-07-31", "API-KEY-9", "m9", 3),
        ],
        bill_month="202607",
        official_total=Decimal("13"),
    )
    assert sum(1 for row in rows if row.amount_type == AMOUNT_FINAL) == 2
    assert sum(row.allocated_cost for row in rows) == Decimal("13")


def test_no_official_data_leaves_estimated_rows():
    rows = reconcile(
        [_estimated("2026-07-31", "API-KEY-1", "m1", 10)],
        [],
        bill_month="202607",
        official_total=None,
    )
    assert rows[0].amount_type == AMOUNT_ESTIMATED
    assert rows[0].allocated_cost == Decimal("0")
    assert rows[0].estimated_cost == Decimal("10")


def test_residual_is_added_to_final_rows():
    rows = reconcile(
        [_estimated("2026-07-31", "API-KEY-1", "m1", 10)],
        [_bill("2026-07-31", "API-KEY-1", "m1", 10)],
        bill_month="202607",
        official_total=Decimal("12"),
    )
    assert len(rows) == 1
    assert rows[0].official_cost == Decimal("10")
    assert rows[0].allocated_cost == Decimal("12")
    assert rows[0].amount_type == AMOUNT_FINAL


def test_residual_without_estimated_cost_creates_unallocated_row():
    rows = reconcile(
        [_estimated("2026-07-31", "API-KEY-1", "m1", 0)],
        [],
        bill_month="202607",
        official_total=Decimal("12"),
    )
    residual = [row for row in rows if row.model_name == "UNALLOCATED"]
    assert len(residual) == 1
    assert residual[0].allocated_cost == Decimal("12")
    assert residual[0].amount_type == AMOUNT_ALLOCATED


def test_aggregate_sums_grouped_values():
    rows = [
        {"usage_date": "2026-07-31", "account_id": "1", "account_name": "a",
         "estimated_cost": Decimal("1"), "official_cost": Decimal("2"), "allocated_cost": Decimal("3")},
        {"usage_date": "2026-07-31", "account_id": "1", "account_name": "a",
         "estimated_cost": Decimal("4"), "official_cost": Decimal("5"), "allocated_cost": Decimal("6")},
        {"usage_date": "2026-07-31", "account_id": "2", "account_name": "b",
         "estimated_cost": Decimal("7"), "official_cost": Decimal("8"), "allocated_cost": Decimal("9")},
    ]
    result = aggregate(rows, group_fields=("account_id", "account_name"))
    assert len(result) == 2
    first = next(row for row in result if row["account_id"] == "1")
    assert first["estimated_cost"] == Decimal("5")
    assert first["official_cost"] == Decimal("7")
    assert first["allocated_cost"] == Decimal("9")
    assert first["row_count"] == 2
