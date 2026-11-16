from __future__ import annotations

from datetime import datetime, timezone

from ksyun_starflow.sigv4 import (
    canonical_query_string,
    sign_request,
    signing_key,
    uri_encode,
)

AK = "AKIDEXAMPLE"
SK = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
REGION = "us-east-1"
SERVICE = "service"


def test_uri_encode_unreserved_and_reserved():
    assert uri_encode("a b/c~d") == "a%20b%2Fc~d"


def test_canonical_query_string_sorted_and_encoded():
    query = {"b": "2", "a": "1 2", "c": ["x", "y"], "skip": None}
    assert canonical_query_string(query) == "a=1%202&b=2&c=x&c=y"


def test_signing_key_is_deterministic():
    first = signing_key(SK, "20150830", REGION, SERVICE)
    second = signing_key(SK, "20150830", REGION, SERVICE)
    assert first == second
    assert len(first) == 32


def _botocore_signature(
    method: str, url: str, body: bytes, timestamp: str, region: str, service: str
) -> str:
    """Compute botocore's SigV4 signature without freezing the clock.

    ``add_auth`` overwrites ``context['timestamp']`` with the current time, so
    the low-level ``canonical_request`` / ``string_to_sign`` / ``signature``
    methods are used with an explicit ``X-Amz-Date`` header.
    """
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    auth = SigV4Auth(Credentials(AK, SK), service, region)
    request = AWSRequest(
        method=method,
        url=url,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Amz-Date": timestamp,
        },
        data=body,
    )
    request.context["timestamp"] = timestamp
    canonical = auth.canonical_request(request)
    string_to_sign = auth.string_to_sign(request, canonical)
    return auth.signature(string_to_sign, request)


def test_signature_matches_botocore_get():
    """Cross-validate against botocore's reference SigV4 implementation."""
    expected = _botocore_signature(
        "GET",
        "https://example.amazonaws.com/?Param1=value1&Param2=value2",
        b"",
        "20150830T123600Z",
        REGION,
        SERVICE,
    )

    signed = sign_request(
        "GET",
        "https://example.amazonaws.com/",
        access_key=AK,
        secret_key=SK,
        region=REGION,
        service=SERVICE,
        query={"Param1": "value1", "Param2": "value2"},
        body=b"",
        now=datetime(2015, 8, 30, 12, 36, 0, tzinfo=timezone.utc),
    )
    assert signed.signature == expected


def test_signature_matches_botocore_post_form():
    body = b"Marker=1&MaxResults=100"
    expected = _botocore_signature(
        "POST",
        "https://aicp.api.ksyun.com/?Action=DescribeModels&Version=2024-06-12",
        body,
        "20260731T010000Z",
        "cn-beijing-6",
        "aicp",
    )

    signed = sign_request(
        "POST",
        "https://aicp.api.ksyun.com/",
        access_key=AK,
        secret_key=SK,
        region="cn-beijing-6",
        service="aicp",
        query={"Action": "DescribeModels", "Version": "2024-06-12"},
        body=body,
        now=datetime(2026, 7, 31, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert signed.signature == expected


def test_authorization_header_shape():
    signed = sign_request(
        "GET",
        "https://bill-union.api.ksyun.com/",
        access_key=AK,
        secret_key=SK,
        region="cn-beijing-6",
        service="bill-union",
        query={"Action": "ListProductGroups", "Version": "2020-01-01"},
        now=datetime(2026, 7, 31, 1, 0, 0, tzinfo=timezone.utc),
    )
    auth = signed.headers["authorization"]
    assert auth.startswith(
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260731/cn-beijing-6/bill-union/aws4_request,"
    )
    assert "SignedHeaders=content-type;host;x-amz-date" in auth
    assert signed.headers["x-amz-date"] == "20260731T010000Z"
