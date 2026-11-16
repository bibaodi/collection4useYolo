"""Configuration and credential resolution.

Resolution order (first non-empty wins), matching the requested behaviour:

1. CLI flags (``--ak`` / ``--sk`` / ``--region`` ...)
2. environment variables (``KSYUN_ACCESS_KEY`` ...)
3. config file (``config.yaml`` or ``config.json``)
4. interactive prompt (AK/SK only)

Secrets are never written to disk by this module.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "region": "cn-beijing-6",
    "aicp_endpoint": "https://aicp.api.ksyun.com",
    "bill_endpoint": "https://bill-union.api.ksyun.com",
    "iam_endpoint": "https://iam.api.ksyun.com",
    "rm_endpoint": "https://resourcemanager.api.ksyun.com",
    "aicp_service": "aicp",
    "bill_service": "bill-union",
    "iam_service": "iam",
    "rm_service": "resourcemanager",
    "product_group_name": "星流",
    "product_group_code": None,
    "pay_type": 1,
    "reasoning_types": ["normal", "batch", "web"],
    "timeout": 60.0,
    "retries": 3,
    "max_pages": 1000,
    "output_dir": "output",
}

ENV_MAP = {
    "access_key": "KSYUN_ACCESS_KEY",
    "secret_key": "KSYUN_SECRET_KEY",
    "region": "KSYUN_REGION",
    "aicp_endpoint": "KSYUN_AICP_ENDPOINT",
    "bill_endpoint": "KSYUN_BILL_ENDPOINT",
    "iam_endpoint": "KSYUN_IAM_ENDPOINT",
    "rm_endpoint": "KSYUN_RM_ENDPOINT",
    "aicp_service": "KSYUN_AICP_SERVICE",
    "bill_service": "KSYUN_BILL_SERVICE",
    "iam_service": "KSYUN_IAM_SERVICE",
    "rm_service": "KSYUN_RM_SERVICE",
    "product_group_name": "KSYUN_PRODUCT_GROUP_NAME",
    "product_group_code": "KSYUN_PRODUCT_GROUP_CODE",
}


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


_ACCESS_KEY_LABELS = ("accesskeyid", "access_key_id", "accesskey", "ak")
_SECRET_KEY_LABELS = ("secretaccesskey", "secret_access_key", "secretkey", "私有访问密钥", "密钥", "sk")


def parse_credentials_text(text: str) -> dict[str, str]:
    """Parse ``AccessKeyId：…`` / ``SecretAccessKey=…`` style files.

    Accepts ASCII or full-width (Chinese) colons so the console's
    ``accesskey.txt`` download can be used directly via ``--config``.
    """
    credentials: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        normalised = line.replace("：", ":")
        if ":" in normalised:
            key, value = normalised.split(":", 1)
        elif "=" in normalised:
            key, value = normalised.split("=", 1)
        else:
            continue
        key = key.strip().lower()
        value = value.strip().strip("'\"")
        if not value:
            continue
        if "secret" in key or "密钥" in key or key == "sk":
            credentials.setdefault("secret_key", value)
        elif "accesskey" in key or "access_key" in key or key == "ak":
            credentials.setdefault("access_key", value)
    return credentials


def load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    data: Any = {}
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # imported lazily so JSON configs need no dependency

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    except (ValueError, ImportError):
        data = {}

    if not isinstance(data, dict) or not data:
        # Fall back to the console's credential-file format.
        credentials = parse_credentials_text(text)
        if credentials:
            return credentials
        if data not in ({}, None):
            raise ValueError(f"config file {path} must contain a mapping")
        return {}

    # Accept both a nested ``ksyun:`` block and flat keys.
    if isinstance(data.get("ksyun"), dict):
        merged = dict(data["ksyun"])
        for key, value in data.items():
            if key != "ksyun":
                merged.setdefault(key, value)
        return merged
    return dict(data)


def load_key_subaccount_map(path: Path | None) -> dict[str, str]:
    """Load a ``KeyId -> SubAccountId`` mapping.

    Accepts either ``{"API-KEY-xxx": "2031447841"}`` or a list of
    ``{"key_id": "...", "sub_account_id": "..."}`` objects.
    """
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = value.get("sub_account_id") or value.get("subAccountId") or value.get("account_id")
            mapping[str(key)] = str(value)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            key = item.get("key_id") or item.get("KeyId") or item.get("api_key")
            value = (
                item.get("sub_account_id")
                or item.get("SubAccountId")
                or item.get("account_id")
                or item.get("accountId")
            )
            if key and value:
                mapping[str(key)] = str(value)
    return mapping


def load_price_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("price overrides must be a JSON object keyed by model name")
    return {str(key): dict(value) for key, value in data.items() if isinstance(value, dict)}


@dataclass
class AppConfig:
    access_key: str
    secret_key: str
    region: str
    aicp_endpoint: str
    bill_endpoint: str
    iam_endpoint: str
    rm_endpoint: str
    aicp_service: str
    bill_service: str
    iam_service: str
    rm_service: str
    product_group_name: str
    product_group_code: str | None
    pay_type: int
    reasoning_types: list[str]
    timeout: float
    retries: int
    max_pages: int
    output_dir: Path
    key_subaccount_map: dict[str, str] = field(default_factory=dict)
    price_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    config_file: Path | None = None

    def redacted(self) -> dict[str, Any]:
        return {
            "access_key": (self.access_key[:4] + "***") if self.access_key else "",
            "secret_key": "***" if self.secret_key else "",
            "region": self.region,
            "aicp_endpoint": self.aicp_endpoint,
            "bill_endpoint": self.bill_endpoint,
            "iam_endpoint": self.iam_endpoint,
            "rm_endpoint": self.rm_endpoint,
            "product_group_name": self.product_group_name,
            "product_group_code": self.product_group_code,
            "pay_type": self.pay_type,
            "reasoning_types": self.reasoning_types,
            "output_dir": str(self.output_dir),
        }


def resolve(args: Any) -> AppConfig:
    """Merge CLI args, environment, config file and interactive input."""
    config_path = Path(args.config).expanduser() if getattr(args, "config", None) else None
    file_cfg = load_config_file(config_path)

    def file_value(key: str) -> Any:
        return file_cfg.get(key)

    def env_value(key: str) -> Any:
        env_name = ENV_MAP.get(key)
        return os.environ.get(env_name) if env_name else None

    def resolve_value(key: str) -> Any:
        return _first(
            getattr(args, key.replace("-", "_"), None),
            env_value(key),
            file_value(key),
            DEFAULTS.get(key),
        )

    access_key = _first(
        getattr(args, "ak", None),
        os.environ.get("KSYUN_ACCESS_KEY"),
        file_value("access_key"),
    )
    if not access_key:
        access_key = input("KSYUN AccessKey ID: ").strip()

    secret_key = _first(
        getattr(args, "sk", None),
        os.environ.get("KSYUN_SECRET_KEY"),
        file_value("secret_key"),
    )
    if not secret_key:
        secret_key = getpass.getpass("KSYUN Secret AccessKey: ").strip()

    if not access_key or not secret_key:
        raise SystemExit("KSYUN AK/SK are required (CLI, env, config file, or prompt).")

    reasoning_types = getattr(args, "reasoning_type", None) or file_value("reasoning_types")
    if not reasoning_types:
        reasoning_types = DEFAULTS["reasoning_types"]
    if isinstance(reasoning_types, str):
        reasoning_types = [reasoning_types]

    output_dir = Path(
        _first(getattr(args, "output_dir", None), file_value("output_dir"), DEFAULTS["output_dir"])
    ).expanduser()

    key_map_path = getattr(args, "key_subaccount_map", None)
    if key_map_path is None and file_value("key_subaccount_map"):
        key_map_path = file_value("key_subaccount_map")
    if isinstance(key_map_path, dict):
        key_subaccount_map = {str(k): str(v) for k, v in key_map_path.items()}
    else:
        key_subaccount_map = load_key_subaccount_map(
            Path(key_map_path).expanduser() if key_map_path else None
        )

    overrides_path = getattr(args, "price_overrides", None)
    if overrides_path is None and file_value("price_overrides"):
        overrides_path = file_value("price_overrides")
    if isinstance(overrides_path, dict):
        price_overrides = {
            str(k): dict(v) for k, v in overrides_path.items() if isinstance(v, dict)
        }
    else:
        price_overrides = load_price_overrides(
            Path(overrides_path).expanduser() if overrides_path else None
        )

    return AppConfig(
        access_key=str(access_key),
        secret_key=str(secret_key),
        region=str(resolve_value("region")),
        aicp_endpoint=str(resolve_value("aicp_endpoint")).rstrip("/"),
        bill_endpoint=str(resolve_value("bill_endpoint")).rstrip("/"),
        iam_endpoint=str(resolve_value("iam_endpoint")).rstrip("/"),
        rm_endpoint=str(resolve_value("rm_endpoint")).rstrip("/"),
        aicp_service=str(resolve_value("aicp_service")),
        bill_service=str(resolve_value("bill_service")),
        iam_service=str(resolve_value("iam_service")),
        rm_service=str(resolve_value("rm_service")),
        product_group_name=str(resolve_value("product_group_name")),
        product_group_code=(
            str(resolve_value("product_group_code"))
            if resolve_value("product_group_code")
            else None
        ),
        pay_type=int(resolve_value("pay_type")),
        reasoning_types=[str(item) for item in reasoning_types],
        timeout=float(resolve_value("timeout")),
        retries=int(resolve_value("retries")),
        max_pages=int(resolve_value("max_pages")),
        output_dir=output_dir,
        key_subaccount_map=key_subaccount_map,
        price_overrides=price_overrides,
        config_file=config_path,
    )
