"""HTTP client for KSYUN OpenAPI endpoints (signed with SigV4)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import requests

from .sigv4 import sign_request

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class KsyunAPIError(RuntimeError):
    """Raised when a KSYUN OpenAPI call fails."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
        code: str | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.request_id = request_id
        self.code = code
        self.payload = payload


def _extract_error(payload: Any) -> tuple[str | None, str | None, str | None]:
    """Return (code, message, request_id) from an API error payload."""
    if not isinstance(payload, dict):
        return None, None, None

    code = payload.get("Code") or payload.get("code")
    message = payload.get("Message") or payload.get("message")
    request_id = payload.get("RequestId") or payload.get("requestId")

    for key in ("Error", "error", "ResponseMetadata", "Response"):
        node = payload.get(key)
        if isinstance(node, dict):
            code = code or node.get("Code") or node.get("code")
            message = message or node.get("Message") or node.get("message")
            request_id = request_id or node.get("RequestId") or node.get("requestId")

    if payload.get("Success") is False and not message:
        message = payload.get("Msg") or payload.get("msg")
    return code, message, request_id


@dataclass
class KsyunClient:
    """A thin signed-request client for one KSYUN service endpoint."""

    access_key: str
    secret_key: str
    region: str
    service: str
    endpoint: str
    timeout: float = 60.0
    retries: int = 3
    backoff: float = 1.5
    session: requests.Session = field(default_factory=requests.Session)
    last_request_id: str | None = None

    def __post_init__(self) -> None:
        self.session.headers.update({"User-Agent": "ksyun-starflow/0.1 (+python-requests)"})

    # -- public helpers -------------------------------------------------
    def get(self, query: Mapping[str, Any], **kwargs: Any) -> Any:
        return self.request("GET", query=query, **kwargs)

    def post_form(
        self,
        query: Mapping[str, Any],
        form: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        from urllib.parse import urlencode

        body = urlencode({k: v for k, v in (form or {}).items() if v is not None})
        return self.request(
            "POST",
            query=query,
            body=body,
            content_type="application/x-www-form-urlencoded",
            **kwargs,
        )

    def post_json(
        self,
        query: Mapping[str, Any],
        payload: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return self.request(
            "POST",
            query=query,
            body=body,
            content_type="application/json",
            **kwargs,
        )

    # -- core ------------------------------------------------------------
    def request(
        self,
        method: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: bytes | str = b"",
        content_type: str = "application/x-www-form-urlencoded",
    ) -> Any:
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            signed = sign_request(
                method,
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                region=self.region,
                service=self.service,
                query=query,
                body=body,
                content_type=content_type,
            )
            log.debug("%s %s", method, signed.url)
            log.debug("canonical_request:\n%s", signed.canonical_request)

            try:
                response = self.session.request(
                    method,
                    signed.url,
                    headers=signed.headers,
                    data=signed.body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                log.warning(
                    "%s %s failed (%s), attempt %d/%d",
                    method,
                    signed.url,
                    exc,
                    attempt,
                    self.retries,
                )
                if attempt < self.retries:
                    time.sleep(self.backoff ** attempt)
                    continue
                raise KsyunAPIError(f"request failed: {exc}") from exc

            self.last_request_id = response.headers.get("X-Ksyun-Request-Id") or response.headers.get(
                "x-amz-request-id"
            )

            if response.status_code in RETRY_STATUS and attempt < self.retries:
                log.warning(
                    "%s %s returned %d, retrying %d/%d",
                    method,
                    signed.url,
                    response.status_code,
                    attempt,
                    self.retries,
                )
                time.sleep(self.backoff ** attempt)
                continue

            payload = self._decode(response)
            if response.status_code >= 400:
                code, message, request_id = _extract_error(payload)
                self.last_request_id = request_id or self.last_request_id
                raise KsyunAPIError(
                    message or f"HTTP {response.status_code}",
                    status=response.status_code,
                    request_id=self.last_request_id,
                    code=code,
                    payload=payload,
                )

            return payload

        raise KsyunAPIError(f"request failed after {self.retries} attempts: {last_error}")

    @staticmethod
    def _decode(response: requests.Response) -> Any:
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            text = response.text
            # Some KSYUN endpoints return XML; keep it as a string.
            return {"_raw": text}
