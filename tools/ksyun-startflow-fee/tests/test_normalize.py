from __future__ import annotations

from decimal import Decimal

from ksyun_starflow import normalize


def test_pick_is_case_insensitive_and_skips_empty():
    row = {"KeyId": "", "key_id": "API-KEY-1", "Other": "x"}
    assert normalize.pick(row, "KeyId", "key_id") == "API-KEY-1"
    assert normalize.pick(row, "missing", default="fallback") == "fallback"


def test_normalise_date_formats():
    assert normalize.normalise_date("2026-07-31 00:00:00") == "2026-07-31"
    assert normalize.normalise_date("2026/07/31") == "2026-07-31"
    assert normalize.normalise_date("20260731") == "2026-07-31"
    assert normalize.normalise_date(1785456000000) == "2026-07-31"
    assert normalize.normalise_date("", fallback="2026-01-01") == "2026-01-01"


def test_normalise_token_row_camel_case():
    raw = {
        "GroupByValue": "deepseek-r1-0528",
        "InputToken": "100",
        "OutputToken": 20,
        "TotalToken": "120",
        "CacheToken": "10",
        "CacheMissToken": "90",
        "CacheWrite5mToken": "5",
        "CacheWrite1hToken": "7",
        "WebSearch": "2",
        "WebParser": "1",
        "ImageCount": "3",
        "VoiceWordCount": "4",
        "VideoDuration": "5",
    }
    row = normalize.normalize_token_row(
        raw, usage_date="2026-07-31", reasoning_type="normal", key_id="API-KEY-1"
    )
    assert row["usage_date"] == "2026-07-31"
    assert row["model_name"] == "deepseek-r1-0528"
    assert row["key_id"] == "API-KEY-1"
    assert row["input_token"] == 100
    assert row["output_token"] == 20
    assert row["cache_token"] == 10
    assert row["cache_miss_token"] == 90
    assert row["cache_write_5m_token"] == 5
    assert row["cache_write_1h_token"] == 7
    assert row["web_search"] == 2
    assert row["web_parser"] == 1
    assert row["image_count"] == 3
    assert row["voice_word_count"] == 4
    assert row["video_duration"] == 5


def test_normalise_token_row_group_by_key_id():
    raw = {"GroupByValue": "API-KEY-9", "InputToken": 1}
    row = normalize.normalize_token_row(
        raw, usage_date="2026-07-31", reasoning_type="normal", group_by="keyId"
    )
    assert row["key_id"] == "API-KEY-9"


def test_normalise_bill_row_extracts_ids_from_config_info():
    raw = {
        "BillStartTime": "2026-07-31 00:00:00",
        "BillRealAmount": "1.234567",
        "ProjectId": "0",
        "ProjectName": "ksyun-trial",
        "ProductGroupName": "星流平台",
        "BillItemCode": "item-1",
        "ItemName": "模型调用",
        "ConfigInfo": '{"ModelName": "deepseek-r1-0528", "KeyId": "API-KEY-1"}',
        "ExtraInfo": "",
        "SubAccountId": "2031447841",
    }
    row = normalize.normalize_bill_row(raw, bill_month="202607")
    assert row["bill_date"] == "2026-07-31"
    assert row["bill_month"] == "202607"
    assert row["bill_real_amount"] == Decimal("1.234567")
    assert row["model_name"] == "deepseek-r1-0528"
    assert row["key_id"] == "API-KEY-1"
    assert row["project_id"] == "0"
    assert row["project_name"] == "ksyun-trial"
    assert row["account_id"] == "2031447841"


def test_normalise_bill_row_falls_back_to_item_name_and_price_label():
    raw = {
        "BillStartTime": "2026-11-16 16:00:00",
        "BillRealAmount": 0.02,
        "ItemName": "deepseek-v4.1-flash-1017",
        "ConfigInfo": ["deepseek-v4.1-flash-1017(千Token):24.8340"],
        "UserId": 2000192131,
        "LoginName": "xin1zp",
    }
    row = normalize.normalize_bill_row(raw, bill_month="202611")
    assert row["model_name"] == "deepseek-v4.1-flash-1017"
    assert row["billing_account_id"] == "2000192131"
    assert row["billing_account_name"] == "xin1zp"


def test_normalise_api_key_row_maps_owner_to_account():
    raw = {
        "KeyId": "API-KEY-1319677108893802496",
        "Name": "etonKey01",
        "CreateUserId": "ARkgUxbRQWc3oWfiL0n8A",
        "CreateUserName": "eton",
        "ProjectId": "127860",
        "ProjectName": "ksyun-trial",
    }
    row = normalize.normalize_api_key_row(raw)
    assert row["key_id"] == "API-KEY-1319677108893802496"
    assert row["account_id"] == "ARkgUxbRQWc3oWfiL0n8A"
    assert row["account_name"] == "eton"
    assert row["account_type"] == normalize.ACCOUNT_TYPE_IAM_USER


def test_normalise_user_row_extracts_main_account_from_krn():
    row = normalize.normalize_user_row(
        {
            "UserId": "ARkgUxbRQWc3oWfiL0n8A",
            "UserName": "eton",
            "RealName": "eton",
            "Email": "bibd@medicalxy.com",
            "Krn": "krn:ksc:iam::2000192131:user/eton",
        }
    )
    assert row["account_id"] == "ARkgUxbRQWc3oWfiL0n8A"
    assert row["account_name"] == "eton"
    assert row["main_account_id"] == "2000192131"


def test_normalise_bill_row_tolerates_opaque_config_info():
    raw = {
        "BillStartTime": "2026-07-31",
        "BillRealAmount": 0.5,
        "ConfigInfo": "not-json",
        "ExtraInfo": None,
    }
    row = normalize.normalize_bill_row(raw)
    assert row["model_name"] == ""
    assert row["key_id"] == ""
    assert row["bill_real_amount"] == Decimal("0.5")


def test_normalise_account_row():
    row = normalize.normalize_account_row(
        {"AccountId": 2031447841, "AccountName": "child", "Type": "ResourceAccount"}
    )
    assert row["account_id"] == "2031447841"
    assert row["account_name"] == "child"
    assert row["account_type"] == "ResourceAccount"
