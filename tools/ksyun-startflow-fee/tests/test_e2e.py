"""End-to-end test: real CLI + CSV output against an in-process mock API."""

from __future__ import annotations

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from ksyun_starflow import cli
from ksyun_starflow.normalize import to_decimal

TOKEN_ROW = {
    "GroupByValue": "deepseek-r1-0528",
    "InputToken": 100,
    "OutputToken": 50,
    "TotalToken": 150,
    "CacheToken": 10,
    "CacheMissToken": 90,
    "CacheWrite5mToken": 5,
    "CacheWrite1hToken": 3,
    "WebSearch": 1,
    "WebParser": 2,
    "ImageCount": 0,
    "VoiceWordCount": 0,
    "VideoDuration": 0,
}

BILL_ROW = {
    "BillStartTime": "2026-07-31 00:00:00",
    "BillEndTime": "2026-07-31 23:59:59",
    "BillRealAmount": "1.5",
    "ProjectId": "0",
    "ProjectName": "ksyun-trial",
    "ProductGroupCode": "XINGLIU",
    "ProductGroupName": "星流平台",
    "ProductTypeName": "MaaS",
    "BillItemCode": "item-1",
    "ItemName": "模型调用",
    "InstanceId": "inst-1",
    "InstanceName": "deepseek-r1-0528",
    "ConfigInfo": json.dumps({"ModelName": "deepseek-r1-0528", "KeyId": "API-KEY-1"}),
    "ExtraInfo": "",
}


LIST_USERS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ListUsersResponse>
  <ListUserResult>
    <Users>
      <member>
        <Item>
          <UserId><![CDATA[USER-1]]></UserId>
          <UserName><![CDATA[eton]]></UserName>
          <RealName><![CDATA[eton]]></RealName>
          <Email><![CDATA[eton@example.com]]></Email>
          <Krn><![CDATA[krn:ksc:iam::2000192131:user/eton]]></Krn>
        </Item>
      </member>
    </Users>
    <IsTruncated>false</IsTruncated>
    <Total>1</Total>
  </ListUserResult>
  <ResponseMetadata><RequestId>r-users</RequestId></ResponseMetadata>
</ListUsersResponse>
"""


def respond(action: str) -> dict | str:
    if action == "ListUsers":
        return LIST_USERS_XML
    if action == "ListProductGroups":
        return {
            "RequestId": "r1",
            "Success": True,
            "Data": [{"Id": 1, "Code": "XINGLIU", "Name": "星流平台", "EnName": "Xingliu"}],
        }
    if action == "ListAccounts":
        return {
            "RequestId": "r2",
            "Success": True,
            "Data": [{"AccountId": 2031447841, "AccountName": "child-a", "Type": "ResourceAccount"}],
        }
    if action == "DescribeModels":
        return {
            "RequestId": "r3",
            "ModelSet": [{"ModelId": "m1", "ModelName": "deepseek-r1-0528"}],
            "TotalCount": 1,
        }
    if action == "GetModelDetail":
        return {
            "RequestId": "r4",
            "ModelId": "m1",
            "ModelName": "deepseek-r1-0528",
            "OnlineCalculateInput": "0.004",
            "OnlineCalculateRight": "0.0008",
            "OnlineCalculateOutput": "0.016",
            "BatchCalculateInput": "0.002",
            "BatchCalculateCacheRight": "0.0004",
            "BatchCalculateOutput": "0.008",
            "PricingRules": [
                {"PricingRule": "CacheWrite5m", "Price": "0.005"},
                {"PricingRule": "CacheWrite1h", "Price": "0.01"},
            ],
            "ImageModelModeMap": {},
            "VideoModelModeSet": [],
            "VoicePrice": "0",
        }
    if action == "DescribeApikeys":
        return {
            "RequestId": "r5",
            "KeySet": [
                {
                    "KeyId": "API-KEY-1",
                    "Name": "key1",
                    "ProjectId": "0",
                    "ProjectName": "ksyun-trial",
                    "Status": "Active",
                    "CreateUserId": "USER-1",
                    "CreateUserName": "eton",
                }
            ],
            "TotalCount": 1,
        }
    if action == "QueryTokenData":
        return {"RequestId": "r6", "Data": [TOKEN_ROW], "LastKey": ""}
    if action == "QueryItemBills":
        return {"RequestId": "r7", "Data": {"Results": [BILL_ROW], "LastSortValue": "", "PageSize": 1}}
    if action == "QueryProductConsume":
        return {
            "RequestId": "r8",
            "Data": [
                {
                    "billDay": "2026-07-31",
                    "productGroupId": "1",
                    "productGroupName": "星流平台",
                    "realtimeBillsAmount": "2.0",
                    "billFinalAmount": "2.0",
                    "cash": "2.0",
                    "reward": "0",
                    "vouchers": "0",
                    "cloudTicketDenomination": "0",
                }
            ],
        }
    if action == "QueryProjectConsume":
        return {
            "RequestId": "r9",
            "Data": [
                {
                    "billDay": "2026-07-31",
                    "projectId": "0",
                    "projectName": "ksyun-trial",
                    "realtimeBillsAmount": "2.0",
                    "postPayBillsAmount": "2.0",
                    "billFinalAmount": "2.0",
                    "cash": "2.0",
                    "reward": "0",
                    "vouchers": "0",
                }
            ],
        }
    return {"RequestId": "unknown", "Code": "InvalidAction", "Message": f"unknown action {action}"}


class Handler(BaseHTTPRequestHandler):
    def _handle(self) -> None:
        parsed = urlparse(self.path)
        action = (parse_qs(parsed.query).get("Action") or ["?"])[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        result = respond(action)
        if isinstance(result, str):
            body = result.encode("utf-8")
            content_type = "application/xml"
        else:
            body = json.dumps(result).encode("utf-8")
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _handle
    do_POST = _handle

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401 - silence
        pass


@pytest.fixture()
def api_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_end_to_end(tmp_path, monkeypatch, api_endpoint):
    monkeypatch.setenv("KSYUN_ACCESS_KEY", "AKIDEXAMPLE")
    monkeypatch.setenv("KSYUN_SECRET_KEY", "SK")
    monkeypatch.setenv("KSYUN_AICP_ENDPOINT", api_endpoint)
    monkeypatch.setenv("KSYUN_BILL_ENDPOINT", api_endpoint)
    monkeypatch.setenv("KSYUN_IAM_ENDPOINT", api_endpoint)
    monkeypatch.setenv("KSYUN_RM_ENDPOINT", api_endpoint)

    exit_code = cli.main(
        ["--date", "2026-07-31", "--output-dir", str(tmp_path), "--log-level", "ERROR"]
    )
    assert exit_code == 0

    ram_rows = _read_csv(tmp_path / "ram_account_daily.csv")
    assert len(ram_rows) == 1
    assert ram_rows[0]["account_id"] == "USER-1"
    assert ram_rows[0]["account_name"] == "eton"
    assert ram_rows[0]["account_type"] == "iam_user"
    assert ram_rows[0]["model_name"] == "deepseek-r1-0528"
    assert to_decimal(ram_rows[0]["estimated_cost"]) == to_decimal("0.120003")
    assert to_decimal(ram_rows[0]["official_cost"]) == to_decimal("1.5")
    assert to_decimal(ram_rows[0]["allocated_cost"]) == to_decimal("2.0")

    token_rows = _read_csv(tmp_path / "token_usage.csv")
    assert {row["reasoning_type"] for row in token_rows} == {"normal", "batch", "web"}
    assert all(row["account_id"] == "USER-1" for row in token_rows)
    assert all(row["account_name"] == "eton" for row in token_rows)
    assert all(row["key_id"] == "API-KEY-1" for row in token_rows)

    key_rows = _read_csv(tmp_path / "api_key_daily.csv")
    assert len(key_rows) == 1
    assert key_rows[0]["key_id"] == "API-KEY-1"
    assert key_rows[0]["account_name"] == "eton"
    assert key_rows[0]["project_name"] == "ksyun-trial"

    keys = _read_csv(tmp_path / "api_keys.csv")
    assert keys[0]["account_id"] == "USER-1"
    assert keys[0]["account_name"] == "eton"

    account_summary = _read_csv(tmp_path / "account_model_summary.csv")
    assert len(account_summary) == 1
    assert account_summary[0]["account_name"] == "eton"

    model_rows = _read_csv(tmp_path / "model_daily.csv")
    assert len(model_rows) == 1
    assert model_rows[0]["model_name"] == "deepseek-r1-0528"

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["product_group"]["code"] == "XINGLIU"
    assert summary["counts"]["token_rows"] == 3
    assert summary["counts"]["item_bills"] == 1
    assert summary["key_accounts"]["API-KEY-1"]["account_name"] == "eton"
