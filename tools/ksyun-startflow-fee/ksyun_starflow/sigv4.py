"""AWS Signature Version 4 signer for Kingsoft Cloud (KSYUN) OpenAPI.

KSYUN's OpenAPI is signature-compatible with AWS SigV4:

    Authorization: AWS4-HMAC-SHA256
        Credential=<AK>/<YYYYMMDD>/<region>/<service>/aws4_request,
        SignedHeaders=content-type;host;x-amz-date,
        Signature=<hex>

Only the headers KSYUN documents (``content-type``, ``host``,
``x-amz-date``) are signed.  The implementation is dependency free and is
cross-checked against ``botocore`` in ``tests/test_sigv4.py``.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import quote, urlparse

ALGORITHM = "AWS4-HMAC-SHA256"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, msg: str | bytes) -> bytes:
    if isinstance(msg, str):
        msg = msg.encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).digest()


def uri_encode(value: Any) -> str:
    """RFC 3986 percent-encoding, as required by SigV4.

    Everything except the unreserved set ``- _ . ~`` is encoded, including
    ``/`` (callers that need to keep path separators use ``quote(..., safe="/~")``).
    """
    return quote(str(value), safe="-_.~")


def canonical_query_string(params: Mapping[str, Any] | None) -> str:
    """Return a sorted, RFC 3986-encoded canonical query string.

    List/tuple values become repeated ``key=value`` pairs.  ``None`` values
    are skipped.  Sorting is done on the *encoded* key and value, which is
    what SigV4 specifies.
    """
    if not params:
        return ""
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is not None:
                    pairs.append((str(key), str(item)))
        else:
            pairs.append((str(key), str(value)))
    pairs.sort(key=lambda kv: (uri_encode(kv[0]), uri_encode(kv[1])))
    return "&".join(f"{uri_encode(k)}={uri_encode(v)}" for k, v in pairs)


def _canonical_headers(headers: Mapping[str, str]) -> tuple[str, str]:
    """Return (canonical_headers, signed_headers)."""
    normalised = {str(k).lower(): " ".join(str(v).split()) for k, v in headers.items()}
    signed_headers = ";".join(sorted(normalised))
    canonical = "".join(f"{k}:{normalised[k]}\n" for k in sorted(normalised))
    return canonical, signed_headers


def build_canonical_request(
    method: str,
    path: str,
    query: str,
    headers: Mapping[str, str],
    body: bytes,
) -> tuple[str, str]:
    """Return ``(canonical_request, signed_headers)``."""
    canonical_uri = quote(path or "/", safe="/~") or "/"
    canonical_headers, signed_headers = _canonical_headers(headers)
    payload_hash = _sha256_hex(body)
    canonical_request = "\n".join(
        [
            method.upper(),
            canonical_uri,
            query,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    return canonical_request, signed_headers


def signing_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    """Derive the SigV4 signing key (the ``kSigning`` HMAC chain)."""
    k_date = _hmac_sha256(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    return _hmac_sha256(k_service, "aws4_request")


@dataclass(frozen=True)
class SignedRequest:
    url: str
    headers: dict[str, str]
    body: bytes
    canonical_request: str
    string_to_sign: str
    signature: str


def sign_request(
    method: str,
    endpoint: str,
    *,
    access_key: str,
    secret_key: str,
    region: str,
    service: str,
    query: Mapping[str, Any] | None = None,
    body: bytes | str = b"",
    content_type: str = "application/x-www-form-urlencoded",
    headers: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> SignedRequest:
    """Sign a request and return everything needed to send it.

    ``endpoint`` may include a path (e.g. ``https://host/api``); its existing
    query string, if any, is preserved ahead of ``query``.
    """
    method = method.upper()
    moment = now or datetime.now(timezone.utc)
    amz_date = moment.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = moment.strftime("%Y%m%d")

    if isinstance(body, str):
        body = body.encode("utf-8")

    parsed = urlparse(endpoint)
    host = parsed.netloc
    path = parsed.path or "/"

    extra_query = canonical_query_string(query)
    full_query = "&".join(part for part in (parsed.query, extra_query) if part)

    all_headers: dict[str, str] = {
        "content-type": content_type,
        "host": host,
        "x-amz-date": amz_date,
    }
    if headers:
        all_headers.update({str(k).lower(): str(v) for k, v in headers.items()})

    canonical_request, signed_headers = build_canonical_request(
        method, path, full_query, all_headers, body
    )
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            ALGORITHM,
            amz_date,
            credential_scope,
            _sha256_hex(canonical_request.encode("utf-8")),
        ]
    )
    signature = _hmac_sha256(
        signing_key(secret_key, date_stamp, region, service), string_to_sign
    ).hex()
    authorization = (
        f"{ALGORITHM} Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    url = f"{parsed.scheme}://{parsed.netloc}{path}"
    if full_query:
        url = f"{url}?{full_query}"

    out_headers = dict(all_headers)
    out_headers["authorization"] = authorization
    return SignedRequest(
        url=url,
        headers=out_headers,
        body=body,
        canonical_request=canonical_request,
        string_to_sign=string_to_sign,
        signature=signature,
    )
