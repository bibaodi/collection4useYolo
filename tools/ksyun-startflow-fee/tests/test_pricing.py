from __future__ import annotations

from decimal import Decimal

from ksyun_starflow.pricing import PriceBook, estimate_cost, parse_model_detail

DETAIL = {
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
    "ImageModelModeMap": {"text2img": {"Price": "0.2"}},
    "VideoModelModeSet": [{"Price": "1.5"}],
    "VoicePrice": "0.0001",
}

USAGE = {
    "reasoning_type": "normal",
    "input_token": 0,
    "output_token": 200,
    "cache_token": 500,
    "cache_miss_token": 1000,
    "cache_write_5m_token": 100,
    "cache_write_1h_token": 50,
    "web_search": 2,
    "web_parser": 1,
    "image_count": 3,
    "voice_word_count": 5,
    "video_duration": 4,
}


def test_parse_model_detail_extracts_all_prices():
    prices = parse_model_detail(DETAIL)
    assert prices.online_input == Decimal("0.004")
    assert prices.online_cache_right == Decimal("0.0008")
    assert prices.online_output == Decimal("0.016")
    assert prices.batch_input == Decimal("0.002")
    assert prices.batch_cache_right == Decimal("0.0004")
    assert prices.batch_output == Decimal("0.008")
    assert prices.cache_write_5m == Decimal("0.005")
    assert prices.cache_write_1h == Decimal("0.01")
    assert prices.image_price == Decimal("0.2")
    assert prices.video_price == Decimal("1.5")
    assert prices.voice_price == Decimal("0.0001")


def test_estimate_online_cost_matches_doc_formula():
    prices = parse_model_detail(DETAIL)
    expected = (
        (Decimal(1000) * Decimal("0.004") / Decimal(1_000_000))
        + (Decimal(500) * Decimal("0.0008") / Decimal(1_000_000))
        + (Decimal(200) * Decimal("0.016") / Decimal(1_000_000))
        + (Decimal(100) * Decimal("0.005") / Decimal(1_000_000))
        + (Decimal(50) * Decimal("0.01") / Decimal(1_000_000))
        + (Decimal(2) * Decimal("0.03"))
        + (Decimal(1) * Decimal("0.015"))
        + (Decimal(3) * Decimal("0.2"))
        + (Decimal(4) * Decimal("1.5"))
        + (Decimal(5) * Decimal("0.0001"))
    )
    assert estimate_cost(USAGE, prices) == expected
    assert estimate_cost(USAGE, prices) == Decimal("6.6755086")


def test_estimate_batch_cost_uses_batch_prices_only():
    prices = parse_model_detail(DETAIL)
    usage = dict(USAGE, reasoning_type="batch")
    expected = (
        (Decimal(1000) * Decimal("0.002") / Decimal(1_000_000))
        + (Decimal(500) * Decimal("0.0004") / Decimal(1_000_000))
        + (Decimal(200) * Decimal("0.008") / Decimal(1_000_000))
    )
    assert estimate_cost(usage, prices) == expected


def test_price_book_lookup_by_name_and_id():
    book = PriceBook()
    book.add(parse_model_detail(DETAIL))
    assert book.get(model_name="deepseek-r1-0528") is not None
    assert book.get(model_id="m1") is not None
    assert book.get(model_name="missing") is None
    assert len(book) == 2  # one id + one name entry


def test_parse_model_detail_tolerates_missing_fields():
    prices = parse_model_detail({})
    assert prices.online_input == Decimal("0")
    assert not prices.has_any_price()


def test_zero_cache_write_price_is_honoured():
    prices = parse_model_detail(
        {"PricingRules": [{"PricingRule": "CacheWrite5m", "Price": "0"}]}
    )
    assert prices.cache_write_5m == Decimal("0")


def test_price_config_price_item_labels():
    """The live API returns prices as ``priceConfig.priceItem`` records."""
    detail = {
        "ModelId": "369840453853585408",
        "ModelName": "deepseek-v4.1-flash-1017",
        "ExtraFields": {
            "priceConfig": {
                "priceItem": [
                    {"label": "在线推理输入", "price": 2.0, "unit": "元/百万Token"},
                    {"label": "在线推理输出", "price": 8.0, "unit": "元/百万Token"},
                    {"label": "在线推理缓存命中", "price": 0.04, "unit": "元/百万Token"},
                    {"label": "5分钟缓存写入", "price": 0.5, "unit": "元/百万Token"},
                    {"label": "1小时缓存写入", "price": 1.5, "unit": "元/百万Token"},
                ],
                "pricingRules": [],
            }
        },
    }
    prices = parse_model_detail(detail)
    assert prices.online_input == Decimal("2.0")
    assert prices.online_output == Decimal("8.0")
    assert prices.online_cache_right == Decimal("0.04")
    assert prices.cache_write_5m == Decimal("0.5")
    assert prices.cache_write_1h == Decimal("1.5")


def test_price_item_units_are_normalised():
    detail = {
        "ExtraFields": {
            "priceConfig": {
                "priceItem": [
                    {"label": "在线推理输入", "price": 0.002, "unit": "元/千Token"},
                    {"label": "联网搜索", "price": 30, "unit": "元/千次"},
                ]
            }
        }
    }
    prices = parse_model_detail(detail)
    assert prices.online_input == Decimal("2.000")
    assert prices.web_search_unit == Decimal("0.030")
