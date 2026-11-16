"""Command-line entry point for the 星流 daily cost report."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import endpoints, normalize, pricing, reconcile, reports
from .client import KsyunAPIError, KsyunClient
from .config import AppConfig, resolve

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_logging(level: str, output_dir: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(output_dir / "run.log", encoding="utf-8"))
        except OSError as exc:  # pragma: no cover - filesystem dependent
            print(f"warning: cannot write log file: {exc}", file=sys.stderr)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_dates(args: argparse.Namespace) -> list[date]:
    try:
        if args.date:
            start = end = date.fromisoformat(args.date)
        elif args.start and args.end:
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
        else:
            raise SystemExit("provide --date YYYY-MM-DD, or both --start and --end")
    except ValueError as exc:
        raise SystemExit(f"invalid date: {exc}") from exc

    if end < start:
        raise SystemExit(f"--end {end} is before --start {start}")

    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def month_of_day(day: date) -> str:
    return f"{day.year:04d}{day.month:02d}"


def day_bounds_ms(day: date) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1) - timedelta(milliseconds=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def build_clients(cfg: AppConfig) -> tuple[KsyunClient, KsyunClient, KsyunClient, KsyunClient]:
    def make(service: str, endpoint: str) -> KsyunClient:
        return KsyunClient(
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            region=cfg.region,
            service=service,
            endpoint=endpoint,
            timeout=cfg.timeout,
            retries=cfg.retries,
        )

    return (
        make(cfg.aicp_service, cfg.aicp_endpoint),
        make(cfg.bill_service, cfg.bill_endpoint),
        make(cfg.iam_service, cfg.iam_endpoint),
        make(cfg.rm_service, cfg.rm_endpoint),
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_accounts(rm_client: KsyunClient, cfg: AppConfig) -> list[dict[str, str]]:
    """List managed / member sub-accounts (托管子账号) under this directory."""
    try:
        raw_accounts = endpoints.list_accounts(rm_client, max_pages=cfg.max_pages)
    except KsyunAPIError as exc:
        log.warning(
            "ListAccounts failed (%s); falling back to IAM users / API-key owners",
            exc,
        )
        return []

    accounts: list[dict[str, str]] = []
    for item in raw_accounts:
        normalised = normalize.normalize_account_row(item)
        if normalised["account_id"]:
            accounts.append(
                {
                    "account_id": normalised["account_id"],
                    "account_name": normalised["account_name"],
                    "account_type": normalize.ACCOUNT_TYPE_MANAGED,
                    "login_name": "",
                    "real_name": "",
                    "email": "",
                    "main_account_id": "",
                }
            )
    log.info("discovered %d managed sub-account(s) via ListAccounts", len(accounts))
    return accounts


def discover_users(iam_client: KsyunClient, cfg: AppConfig) -> list[dict[str, str]]:
    """List IAM users (子用户) — the ``CreateUserId`` owner of an API key."""
    try:
        raw_users = endpoints.list_users(iam_client, max_pages=cfg.max_pages)
    except KsyunAPIError as exc:
        log.warning("ListUsers failed (%s); API-key owners are still reported", exc)
        return []

    users: list[dict[str, str]] = []
    for item in raw_users:
        normalised = normalize.normalize_user_row(item)
        if normalised["account_id"]:
            users.append(
                {
                    "account_id": normalised["account_id"],
                    "account_name": normalised["account_name"],
                    "account_type": normalised["account_type"],
                    "login_name": normalised["login_name"],
                    "real_name": normalised["real_name"],
                    "email": normalised["email"],
                    "main_account_id": normalised["main_account_id"],
                }
            )
    log.info("discovered %d IAM user(s) via ListUsers", len(users))
    return users


def build_account_directory(
    api_keys: Sequence[Mapping[str, Any]],
    users: Sequence[Mapping[str, str]],
    managed_accounts: Sequence[Mapping[str, str]],
    overrides: Mapping[str, str],
    *,
    sub_account_ids: Sequence[str] = (),
) -> dict[str, dict[str, str]]:
    """Build ``account_id -> account`` from every source we have.

    Precedence: explicit overrides / ``--sub-account-id``, then managed
    sub-accounts, then IAM users, then the API-key owner fields.
    """
    directory: dict[str, dict[str, str]] = {}

    def add(account_id: str, account_name: str, account_type: str, **extra: str) -> None:
        account_id = str(account_id or "")
        if not account_id:
            return
        entry = directory.setdefault(
            account_id,
            {
                "account_id": account_id,
                "account_name": "",
                "account_type": "",
                "login_name": "",
                "real_name": "",
                "email": "",
                "main_account_id": "",
            },
        )
        if account_name:
            entry["account_name"] = account_name
        if account_type:
            entry["account_type"] = account_type
        for field_name, value in extra.items():
            if value:
                entry[field_name] = value

    for account_id in sub_account_ids:
        add(str(account_id), "", normalize.ACCOUNT_TYPE_MANAGED)
    for account in managed_accounts:
        add(
            str(account.get("account_id") or ""),
            str(account.get("account_name") or ""),
            normalize.ACCOUNT_TYPE_MANAGED,
        )
    for user in users:
        add(
            str(user.get("account_id") or ""),
            str(user.get("account_name") or ""),
            normalize.ACCOUNT_TYPE_IAM_USER,
            login_name=str(user.get("login_name") or ""),
            real_name=str(user.get("real_name") or ""),
            email=str(user.get("email") or ""),
            main_account_id=str(user.get("main_account_id") or ""),
        )
    for key in api_keys:
        add(
            str(key.get("account_id") or ""),
            str(key.get("account_name") or ""),
            str(key.get("account_type") or ""),
        )
    for account_id in overrides.values():
        add(str(account_id), "", normalize.ACCOUNT_TYPE_CUSTOM)

    return directory


def resolve_key_accounts(
    api_keys: Sequence[Mapping[str, Any]],
    directory: Mapping[str, Mapping[str, str]],
    overrides: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Map ``KeyId -> account`` so token usage can be indexed by RAM account."""
    mapping: dict[str, dict[str, str]] = {}
    for key in api_keys:
        key_id = str(key.get("key_id") or "")
        if not key_id:
            continue
        account_id = str(overrides.get(key_id) or key.get("account_id") or "")
        entry = dict(directory.get(account_id, {})) if account_id else {}
        mapping[key_id] = {
            "account_id": account_id,
            "account_name": str(
                entry.get("account_name") or key.get("account_name") or ""
            ),
            "account_type": str(
                entry.get("account_type") or key.get("account_type") or ""
            ),
        }
    return mapping


def resolve_product_group_code(
    bill_client: KsyunClient, cfg: AppConfig
) -> str | None:
    if cfg.product_group_code:
        return cfg.product_group_code

    try:
        groups = endpoints.list_product_groups(bill_client)
    except KsyunAPIError as exc:
        log.warning("ListProductGroups failed (%s); set --product-group-code", exc)
        return None

    wanted = (cfg.product_group_name or "").strip().lower()
    for group in groups:
        name = str(group.get("Name") or "")
        en_name = str(group.get("EnName") or "")
        if wanted and wanted in name.lower():
            log.info("resolved product group %r -> %s", name, group.get("Code"))
            return str(group.get("Code") or "")
    for group in groups:
        haystack = f"{group.get('Name', '')} {group.get('EnName', '')}".upper()
        if "星流" in haystack or "KSP" in haystack or "AICP" in haystack:
            log.info("resolved product group %r -> %s", group.get("Name"), group.get("Code"))
            return str(group.get("Code") or "")

    log.warning(
        "could not find product group containing %r; set --product-group-code. "
        "Available groups: %s",
        cfg.product_group_name,
        ", ".join(f"{g.get('Code')}={g.get('Name')}" for g in groups[:20]),
    )
    return None


def fetch_api_keys(aicp_client: KsyunClient, cfg: AppConfig) -> list[dict[str, Any]]:
    raw_keys = endpoints.describe_apikeys(aicp_client, max_pages=cfg.max_pages)
    keys = [normalize.normalize_api_key_row(item) for item in raw_keys]
    keys = [key for key in keys if key["key_id"]]
    log.info("fetched %d API key(s)", len(keys))
    return keys


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_token_rows(
    aicp_client: KsyunClient,
    cfg: AppConfig,
    days: Sequence[date],
    api_keys: Sequence[Mapping[str, Any]],
    key_accounts: Mapping[str, Mapping[str, str]],
    *,
    group_by: str = "model",
) -> list[dict[str, Any]]:
    """Query usage per API key × reasoning type (as the workflow doc specifies).

    ``QueryTokenData`` has no account field, so the RAM account comes from
    the key's ``CreateUserId`` / ``CreateUserName`` (resolved by
    :func:`resolve_key_accounts`).
    """
    key_ids: list[str | None] = [str(key["key_id"]) for key in api_keys if key.get("key_id")]
    if not key_ids:
        key_ids = [None]

    key_projects = {
        str(key["key_id"]): (
            str(key.get("project_id") or ""),
            str(key.get("project_name") or ""),
        )
        for key in api_keys
        if key.get("key_id")
    }

    rows: list[dict[str, Any]] = []
    for day in days:
        start_ms, end_ms = day_bounds_ms(day)
        for key_id in key_ids:
            for reasoning_type in cfg.reasoning_types:
                try:
                    raw_rows = endpoints.query_token_data(
                        aicp_client,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        key_id=key_id,
                        group_by=group_by,
                        reasoning_type=reasoning_type,
                        page_size=10000,
                        max_pages=cfg.max_pages,
                    )
                except KsyunAPIError as exc:
                    log.warning(
                        "QueryTokenData failed for key=%s type=%s: %s",
                        key_id,
                        reasoning_type,
                        exc,
                    )
                    continue

                for raw in raw_rows:
                    normalised = normalize.normalize_token_row(
                        raw,
                        usage_date=day.isoformat(),
                        reasoning_type=reasoning_type,
                        key_id=key_id or "",
                        group_by=group_by,
                    )
                    if not normalised["key_id"]:
                        normalised["key_id"] = key_id or ""
                    project_id, project_name = key_projects.get(
                        normalised["key_id"], ("", "")
                    )
                    normalised["project_id"] = normalised["project_id"] or project_id
                    normalised["project_name"] = normalised["project_name"] or project_name
                    account = key_accounts.get(normalised["key_id"], {})
                    normalised["account_id"] = str(account.get("account_id") or "")
                    normalised["account_name"] = str(account.get("account_name") or "")
                    normalised["account_type"] = str(account.get("account_type") or "")
                    rows.append(normalised)

    log.info("fetched %d token-usage row(s)", len(rows))
    return rows


def load_prices(
    aicp_client: KsyunClient,
    cfg: AppConfig,
    model_names: Iterable[str],
) -> pricing.PriceBook:
    raw_models = endpoints.describe_models(aicp_client, max_pages=cfg.max_pages)
    models = [normalize.normalize_model_row(item) for item in raw_models]

    name_to_id: dict[str, str] = {}
    for model in models:
        if model["model_name"]:
            name_to_id[model["model_name"]] = model["model_id"]
        if model["model_id"]:
            name_to_id.setdefault(model["model_id"], model["model_id"])

    book = pricing.PriceBook()
    for model_name in sorted({name for name in model_names if name}):
        model_id = name_to_id.get(model_name, "")
        detail: dict[str, Any] = {}
        if model_id:
            try:
                detail = endpoints.get_model_detail(aicp_client, model_id)
            except KsyunAPIError as exc:
                log.warning("GetModelDetail(%s) failed: %s", model_id, exc)
        else:
            log.warning("model %r not present in DescribeModels", model_name)

        prices = pricing.parse_model_detail(detail, model_id=model_id, model_name=model_name)
        for field_name, value in cfg.price_overrides.get(model_name, {}).items():
            if hasattr(prices, field_name):
                setattr(prices, field_name, normalize.to_decimal(value))
            else:
                log.warning("unknown price override field %r for %r", field_name, model_name)
        book.add(prices)

    log.info("loaded prices for %d model(s)", len(book))
    return book


def fetch_bills(
    bill_client: KsyunClient,
    cfg: AppConfig,
    months: Sequence[str],
    sub_accounts: Sequence[Mapping[str, str]],
    *,
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    targets: list[Mapping[str, str] | None] = list(sub_accounts) or [None]
    bills: list[dict[str, Any]] = []

    for month in months:
        for target in targets:
            sub_account_id = target["account_id"] if target else None
            try:
                raw_bills = endpoints.query_item_bills(
                    bill_client,
                    bill_month=month,
                    product_group_code=cfg.product_group_code,
                    pay_type=cfg.pay_type,
                    size=page_size,
                    sub_account_id=sub_account_id,
                    max_pages=cfg.max_pages,
                )
            except KsyunAPIError as exc:
                log.warning(
                    "QueryItemBills failed for month=%s sub_account=%s: %s",
                    month,
                    sub_account_id,
                    exc,
                )
                continue

            for raw in raw_bills:
                normalised = normalize.normalize_bill_row(raw, bill_month=month)
                if target and not normalised["account_id"]:
                    normalised["account_id"] = target["account_id"]
                    normalised["account_name"] = target["account_name"]
                bills.append(normalised)

    log.info("fetched %d item bill row(s)", len(bills))
    return bills


def fetch_consume(
    bill_client: KsyunClient,
    cfg: AppConfig,
    days: Sequence[date],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    product_rows: list[dict[str, Any]] = []
    project_rows: list[dict[str, Any]] = []

    for day in days:
        day_text = day.isoformat()
        try:
            for item in endpoints.query_product_consume(
                bill_client, start_day=day_text, end_day=day_text, max_pages=cfg.max_pages
            ):
                product_rows.append(normalize.normalize_product_consume_row(item))
        except KsyunAPIError as exc:
            log.warning("QueryProductConsume failed for %s: %s", day_text, exc)

        try:
            for item in endpoints.query_project_consume(
                bill_client, start_day=day_text, end_day=day_text, max_pages=cfg.max_pages
            ):
                project_rows.append(normalize.normalize_project_consume_row(item))
        except KsyunAPIError as exc:
            log.warning("QueryProjectConsume failed for %s: %s", day_text, exc)

    log.info(
        "fetched %d product-consume and %d project-consume row(s)",
        len(product_rows),
        len(project_rows),
    )
    return product_rows, project_rows


def propagate_accounts(
    token_rows: Sequence[dict[str, Any]],
    bill_rows: Sequence[Mapping[str, Any]],
    directory: Mapping[str, Mapping[str, str]],
) -> None:
    """Fill missing account / project attribution on token rows from bills.

    Token rows get their RAM account from the API key owner; bills are
    queried per managed sub-account and therefore carry the authoritative
    ``account_id`` when one applies.  Project ids come from the API key or
    from the bill line.
    """
    account_by_key: dict[str, str] = {}
    account_by_model: dict[str, str] = {}
    project_by_key: dict[str, tuple[str, str]] = {}
    project_by_model: dict[str, tuple[str, str]] = {}

    for bill in bill_rows:
        key_id = str(bill.get("key_id") or "")
        model_name = str(bill.get("model_name") or "")

        account_id = str(bill.get("account_id") or "")
        if account_id:
            if key_id:
                account_by_key.setdefault(key_id, account_id)
            if model_name:
                account_by_model.setdefault(model_name, account_id)

        project_id = str(bill.get("project_id") or "")
        project_name = str(bill.get("project_name") or "")
        if project_id or project_name:
            if key_id:
                project_by_key.setdefault(key_id, (project_id, project_name))
            if model_name:
                project_by_model.setdefault(model_name, (project_id, project_name))

    for row in token_rows:
        key_id = str(row.get("key_id") or "")
        model_name = str(row.get("model_name") or "")

        if not row.get("account_id"):
            account_id = account_by_key.get(key_id) or account_by_model.get(model_name) or ""
            if account_id:
                entry = directory.get(account_id, {})
                row["account_id"] = account_id
                row["account_name"] = str(entry.get("account_name") or "")
                row["account_type"] = str(entry.get("account_type") or "")

        if not row.get("project_id") and not row.get("project_name"):
            project = project_by_key.get(key_id) or project_by_model.get(model_name)
            if project:
                row["project_id"], row["project_name"] = project


def official_totals_by_day(
    product_rows: Sequence[Mapping[str, Any]], cfg: AppConfig
) -> dict[str, Decimal]:
    """Sum the 星流 platform official amount per day."""
    totals: dict[str, Decimal] = {}
    wanted = (cfg.product_group_name or "").strip().lower()
    for row in product_rows:
        name = str(row.get("product_group_name") or "")
        if wanted and wanted not in name.lower():
            continue
        day = str(row.get("bill_day") or "")
        amount = row.get("bill_final_amount") or row.get("realtime_bills_amount") or Decimal("0")
        totals[day] = totals.get(day, Decimal("0")) + Decimal(str(amount))
    return totals


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def aggregate_daily(
    reconciled_rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    return reconcile.aggregate(
        reconciled_rows,
        group_fields=group_fields,
        value_fields=("estimated_cost", "official_cost", "allocated_cost"),
    )


def write_reports(
    output_dir: Path,
    *,
    token_rows: Sequence[Mapping[str, Any]],
    api_keys: Sequence[Mapping[str, Any]],
    accounts: Sequence[Mapping[str, Any]],
    bills: Sequence[Mapping[str, Any]],
    product_rows: Sequence[Mapping[str, Any]],
    project_rows: Sequence[Mapping[str, Any]],
    reconciled_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    written["token_usage"] = reports.write_csv(
        output_dir / "token_usage.csv", reports.TOKEN_USAGE_FIELDS, token_rows
    )
    written["accounts"] = reports.write_csv(
        output_dir / "accounts.csv", reports.USER_FIELDS, accounts
    )
    written["api_keys"] = reports.write_csv(
        output_dir / "api_keys.csv", reports.API_KEY_FIELDS, api_keys
    )
    written["item_bills"] = reports.write_csv(
        output_dir / "item_bills.csv", reports.ITEM_BILL_FIELDS, bills
    )
    written["product_consume"] = reports.write_csv(
        output_dir / "product_consume.csv", reports.PRODUCT_CONSUME_FIELDS, product_rows
    )
    written["project_consume"] = reports.write_csv(
        output_dir / "project_consume.csv", reports.PROJECT_CONSUME_FIELDS, project_rows
    )
    written["reconciled_daily"] = reports.write_csv(
        output_dir / "reconciled_daily.csv", reports.RECONCILED_FIELDS, reconciled_rows
    )
    written["ram_account_daily"] = reports.write_csv(
        output_dir / "ram_account_daily.csv",
        reports.RAM_ACCOUNT_FIELDS,
        aggregate_daily(
            reconciled_rows,
            group_fields=(
                "usage_date",
                "account_id",
                "account_name",
                "account_type",
                "model_name",
            ),
        ),
    )
    written["account_model_summary"] = reports.write_csv(
        output_dir / "account_model_summary.csv",
        reports.ACCOUNT_SUMMARY_FIELDS,
        aggregate_daily(
            reconciled_rows,
            group_fields=("account_id", "account_name", "account_type"),
        ),
    )
    written["api_key_daily"] = reports.write_csv(
        output_dir / "api_key_daily.csv",
        reports.DAILY_COST_FIELDS,
        aggregate_daily(
            reconciled_rows,
            group_fields=(
                "usage_date",
                "account_id",
                "account_name",
                "account_type",
                "project_id",
                "project_name",
                "key_id",
            ),
        ),
    )
    written["model_daily"] = reports.write_csv(
        output_dir / "model_daily.csv",
        reports.DAILY_COST_FIELDS,
        aggregate_daily(reconciled_rows, group_fields=("usage_date", "model_name")),
    )
    written["summary"] = reports.write_json(output_dir / "summary.json", summary)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ksyun-starflow",
        description=(
            "Pull 星流 (MaaS) token usage and official bills from the KSYUN OpenAPI "
            "and write CSV reports, including per-RAM-sub-account daily 星流 cost."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Credentials resolve in order: CLI --ak/--sk -> KSYUN_ACCESS_KEY / "
            "KSYUN_SECRET_KEY -> config file -> interactive prompt.\n\n"
            "Examples:\n"
            "  ksyun-starflow --date 2026-07-31 --output-dir output\n"
            "  ksyun-starflow --start 2026-07-01 --end 2026-07-31 \\\n"
            "      --sub-account-id 2031447841 --sub-account-id 2031447842\n"
            "  KSYUN_ACCESS_KEY=... KSYUN_SECRET_KEY=... ksyun-starflow --date 2026-07-31\n"
        ),
    )

    # Date selection
    parser.add_argument("--date", help="single day, YYYY-MM-DD")
    parser.add_argument("--start", help="range start, YYYY-MM-DD (with --end)")
    parser.add_argument("--end", help="range end, YYYY-MM-DD (inclusive)")
    parser.add_argument("--month", help="bill month override, YYYYMM (default: derived from dates)")

    # Credentials / endpoints
    parser.add_argument("--ak", help="AccessKey ID (prefer env or config file)")
    parser.add_argument("--sk", help="Secret Access Key (prefer env or config file)")
    parser.add_argument("--region", help="signing region (default: cn-beijing-6)")
    parser.add_argument(
        "--config", help="config.yaml / config.json / accesskey.txt (console credential file)"
    )
    parser.add_argument("--product-group-code", help="星流 product group code")
    parser.add_argument("--product-group-name", help="product group name to match (default: 星流)")

    # Sub-account attribution
    parser.add_argument(
        "--sub-account-id",
        action="append",
        help="managed sub-account (托管子账号) ID; repeatable",
    )
    parser.add_argument(
        "--no-account-discovery",
        action="store_true",
        help="do not call ListAccounts/ListUsers; attribute from API keys and overrides only",
    )
    parser.add_argument(
        "--key-subaccount-map",
        help="JSON file mapping KeyId -> SubAccountId (or key_id/sub_account_id objects)",
    )

    # Query tuning
    parser.add_argument(
        "--reasoning-type",
        action="append",
        help="reasoning type (normal/batch/web); repeatable (default: all three)",
    )
    parser.add_argument(
        "--group-by",
        default="model",
        choices=["model", "keyId", "endpointId"],
        help="QueryTokenData grouping (default: model)",
    )
    parser.add_argument("--price-overrides", help="JSON file of per-model price overrides")
    parser.add_argument(
        "--pay-type",
        type=int,
        help="QueryItemBills pay type: 0=prepaid, 1=realtime, 2=postpaid (default: 1)",
    )
    parser.add_argument("--timeout", type=float, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, help="HTTP retries")
    parser.add_argument("--max-pages", type=int, help="pagination safety cap")
    parser.add_argument("--bill-page-size", type=int, default=1000, help="QueryItemBills page size")

    # Output / behaviour
    parser.add_argument("--output-dir", help="output directory (default: ./output)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--no-token", action="store_true", help="skip QueryTokenData")
    parser.add_argument("--no-official", action="store_true", help="skip QueryItemBills")
    parser.add_argument("--no-consume", action="store_true", help="skip QueryProductConsume/QueryProjectConsume")
    parser.add_argument("--dry-run", action="store_true", help="print config and exit without API calls")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        # Allow a dry run without credentials.
        args.ak = args.ak or "DRYRUN"
        args.sk = args.sk or "DRYRUN"

    cfg = resolve(args)
    setup_logging(args.log_level, cfg.output_dir)
    log.info("config: %s", cfg.redacted())

    days = parse_dates(args)
    if args.month and not (len(args.month) == 6 and args.month.isdigit()):
        raise SystemExit("--month must be YYYYMM, e.g. 202607")
    months = [args.month] if args.month else sorted({month_of_day(day) for day in days})
    log.info("dates: %s .. %s (%d day(s)); bill months: %s", days[0], days[-1], len(days), months)

    if args.dry_run:
        print("dry run: no API calls made")
        print(f"  dates        : {days[0]} .. {days[-1]}")
        print(f"  bill months  : {', '.join(months)}")
        print(f"  output dir   : {cfg.output_dir}")
        print(f"  product group: {cfg.product_group_name} ({cfg.product_group_code or 'auto'})")
        print(f"  sub accounts : {args.sub_account_id or 'auto-discover'}")
        return 0

    aicp_client, bill_client, _iam_client, rm_client = build_clients(cfg)

    cfg.product_group_code = resolve_product_group_code(bill_client, cfg)
    if not cfg.product_group_code:
        log.warning(
            "ProductGroupCode is not set; QueryItemBills will return all product lines"
        )

    if args.sub_account_id:
        sub_accounts = [
            {
                "account_id": str(account_id),
                "account_name": "",
                "account_type": normalize.ACCOUNT_TYPE_MANAGED,
            }
            for account_id in args.sub_account_id
        ]
    elif args.no_account_discovery:
        sub_accounts = []
    else:
        sub_accounts = discover_accounts(rm_client, cfg)

    users: list[dict[str, str]] = []
    if not args.no_account_discovery:
        users = discover_users(_iam_client, cfg)

    api_keys: list[dict[str, Any]] = []
    if not args.no_token:
        api_keys = fetch_api_keys(aicp_client, cfg)

    # KeyId -> RAM account, from CreateUserId / CreateUserName plus overrides.
    directory = build_account_directory(
        api_keys,
        users,
        sub_accounts,
        cfg.key_subaccount_map,
        sub_account_ids=args.sub_account_id or (),
    )
    key_accounts = resolve_key_accounts(api_keys, directory, cfg.key_subaccount_map)

    token_rows: list[dict[str, Any]] = []
    if not args.no_token:
        token_rows = fetch_token_rows(
            aicp_client,
            cfg,
            days,
            api_keys,
            key_accounts,
            group_by=args.group_by,
        )

    bill_rows: list[dict[str, Any]] = []
    if not args.no_official:
        bill_rows = fetch_bills(
            bill_client, cfg, months, sub_accounts, page_size=args.bill_page_size
        )

    # Bills are queried per managed sub-account, so they are the authoritative
    # source for account / project attribution when the key has no owner.
    propagate_accounts(token_rows, bill_rows, directory)

    if not args.no_token:
        model_names = {row["model_name"] for row in token_rows if row["model_name"]}
        price_book = load_prices(aicp_client, cfg, model_names)
        for row in token_rows:
            prices = price_book.require(model_name=row["model_name"])
            row["estimated_cost"] = pricing.estimate_cost(row, prices)

    product_rows: list[dict[str, Any]] = []
    project_rows: list[dict[str, Any]] = []
    if not args.no_consume:
        product_rows, project_rows = fetch_consume(bill_client, cfg, days)

    official_totals = official_totals_by_day(product_rows, cfg)

    reconciled_rows: list[dict[str, Any]] = []
    for day in days:
        day_text = day.isoformat()
        estimated = [row for row in token_rows if row["usage_date"] == day_text]
        official = [row for row in bill_rows if row["bill_date"] == day_text]
        reconciled_rows.extend(
            row.as_dict()
            for row in reconcile.reconcile(
                estimated,
                official,
                bill_month=month_of_day(day),
                official_total=official_totals.get(day_text),
            )
        )

    total_estimated = sum((Decimal(str(row["estimated_cost"])) for row in reconciled_rows), Decimal("0"))
    total_official = sum((Decimal(str(row["official_cost"])) for row in reconciled_rows), Decimal("0"))
    total_allocated = sum((Decimal(str(row["allocated_cost"])) for row in reconciled_rows), Decimal("0"))

    account_summary = aggregate_daily(
        reconciled_rows, group_fields=("account_id", "account_name", "account_type")
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date_range": {"start": days[0].isoformat(), "end": days[-1].isoformat()},
        "bill_months": months,
        "product_group": {
            "name": cfg.product_group_name,
            "code": cfg.product_group_code,
        },
        "config": cfg.redacted(),
        "accounts": list(directory.values()),
        "key_accounts": key_accounts,
        "counts": {
            "api_keys": len(api_keys),
            "accounts": len(directory),
            "token_rows": len(token_rows),
            "item_bills": len(bill_rows),
            "product_consume": len(product_rows),
            "project_consume": len(project_rows),
            "reconciled_rows": len(reconciled_rows),
        },
        "totals": {
            "estimated_cost": f"{total_estimated:.6f}",
            "official_cost": f"{total_official:.6f}",
            "allocated_cost": f"{total_allocated:.6f}",
        },
        "by_account": [
            {
                "account_id": row["account_id"],
                "account_name": row["account_name"],
                "account_type": row["account_type"],
                "estimated_cost": f"{row['estimated_cost']:.6f}",
                "official_cost": f"{row['official_cost']:.6f}",
                "allocated_cost": f"{row['allocated_cost']:.6f}",
            }
            for row in account_summary
        ],
    }

    written = write_reports(
        cfg.output_dir,
        token_rows=token_rows,
        api_keys=api_keys,
        accounts=list(directory.values()),
        bills=bill_rows,
        product_rows=product_rows,
        project_rows=project_rows,
        reconciled_rows=reconciled_rows,
        summary=summary,
    )

    print()
    print("星流 daily cost report complete")
    print(f"  dates      : {days[0]} .. {days[-1]}")
    print(f"  estimated  : {total_estimated:.6f}")
    print(f"  official   : {total_official:.6f}")
    print(f"  allocated  : {total_allocated:.6f}")
    print()
    print("  by RAM account:")
    for row in account_summary:
        label = row["account_name"] or row["account_id"] or "UNATTRIBUTED"
        print(
            f"    - {label:<24} estimated={row['estimated_cost']:.6f}"
            f"  official={row['official_cost']:.6f}"
            f"  allocated={row['allocated_cost']:.6f}"
        )
    print()
    print(f"  output     : {cfg.output_dir.resolve()}")
    for name, path in written.items():
        print(f"    - {name}: {path.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KsyunAPIError as exc:
        log.error(
            "KSYUN API error: %s (status=%s, code=%s, request_id=%s)",
            exc,
            exc.status,
            exc.code,
            exc.request_id,
        )
        return 1
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
