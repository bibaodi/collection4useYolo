"""Model price extraction and estimated-cost calculation.

Implements the formulas in section 5 of
``星流 MaaS 模型与 API Key 日账单接口调用步骤.md``.

Prices are expressed per 1,000,000 tokens unless the docs say otherwise
(web search: 30 / 1000 calls, web parser: 15 / 1000 calls).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Mapping

from .normalize import MONEY, pick, to_decimal

log = logging.getLogger(__name__)

MILLION = Decimal("1000000")
WEB_SEARCH_UNIT = Decimal("30") / Decimal("1000")
WEB_PARSER_UNIT = Decimal("15") / Decimal("1000")


def _norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _iter_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_dicts(value)


def _numeric(value: Any) -> Decimal | None:
    if value in (None, "", True, False):
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    text = str(value).strip().replace(",", "")
    for symbol in ("¥", "￥", "$", " "):
        text = text.replace(symbol, "")
    try:
        return Decimal(text)
    except Exception:  # noqa: BLE001 - malformed prices are non-fatal
        return None


def _first_numeric(obj: Any) -> Decimal | None:
    if isinstance(obj, dict):
        for key in ("Price", "UnitPrice", "Amount", "Value", "Calculate", "PricePerUnit", "Cost"):
            if key in obj:
                found = _numeric(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = _first_numeric(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _first_numeric(value)
            if found is not None:
                return found
    else:
        return _numeric(obj)
    return None


def _find_price_by_label(obj: Any, *label_parts: str) -> Decimal | None:
    """Find a price whose sibling key or value label contains all parts.

    Handles both ``{"CacheWrite5mPrice": 0.1}`` style maps and
    ``[{"PricingRule": "CacheWrite5m", "Price": 0.1}]`` style rules.
    """
    wanted = [part.lower() for part in label_parts]
    for node in _iter_dicts(obj):
        for key, value in node.items():
            normalised_key = _norm_key(key)
            normalised_value = _norm_key(value) if isinstance(value, str) else ""
            for candidate in (normalised_key, normalised_value):
                if candidate and all(part in candidate for part in wanted):
                    price = _numeric(value)
                    if price is not None:
                        return price
                    price = _first_numeric(
                        {k: v for k, v in node.items() if _norm_key(k) != normalised_key}
                    )
                    if price is not None:
                        return price
    return None


def _find_price(obj: Any, *keyword_groups: tuple[str, ...]) -> Decimal | None:
    for group in keyword_groups:
        found = _find_price_by_label(obj, *group)
        if found is not None:
            return found
    return None


MONEY_FIELDS = (
    "online_input",
    "online_output",
    "online_cache_right",
    "batch_input",
    "batch_output",
    "batch_cache_right",
    "cache_write_5m",
    "cache_write_1h",
)
PER_CALL_FIELDS = ("web_search_unit", "web_parser_unit")


def _classify_price_label(label: str) -> str | None:
    """Map a ``priceConfig.priceItem`` label to a ``ModelPrices`` field."""
    text = str(label or "").replace(" ", "")
    lowered = text.lower()
    batch = "批处理" in text or "batch" in lowered

    if "缓存" in text or "cache" in lowered:
        if "命中" in text or "hit" in lowered:
            return "batch_cache_right" if batch else "online_cache_right"
        if "写入" in text or "创建" in text or "write" in lowered or "creation" in lowered:
            if "5" in text or "五" in text:
                return "cache_write_5m"
            if "1" in text or "一" in text:
                return "cache_write_1h"
            return "cache_write_5m"
        return None
    if "输入" in text or "input" in lowered:
        return "batch_input" if batch else "online_input"
    if "输出" in text or "output" in lowered:
        return "batch_output" if batch else "online_output"
    if "搜索" in text or "search" in lowered:
        return "web_search_unit"
    if "解析" in text or "parser" in lowered:
        return "web_parser_unit"
    if "图片" in text or "图像" in text or "image" in lowered:
        return "image_price"
    if "视频" in text or "video" in lowered:
        return "video_price"
    if "语音" in text or "voice" in lowered or "audio" in lowered:
        return "voice_price"
    return None


def _normalise_price_unit(field: str, price: Decimal, unit: Any) -> Decimal:
    """Convert ``元/千Token`` style units to the units the formulas expect."""
    text = str(unit or "").replace(" ", "")
    per_thousand = "千" in text and "百万" not in text
    if field in MONEY_FIELDS and per_thousand:
        return price * 1000
    if field in PER_CALL_FIELDS and per_thousand:
        return price / 1000
    return price


def price_items(detail: Any) -> dict[str, Decimal]:
    """Extract ``ExtraFields.priceConfig.priceItem`` prices by field name.

    The live API exposes prices as a list of
    ``{"label": "在线推理输入", "price": 2.0, "unit": "元/百万Token"}``
    records rather than the flat ``OnlineCalculateInput`` fields the
    workflow doc describes; support both.
    """
    items: list[Mapping[str, Any]] = []
    for node in _iter_dicts(detail):
        candidate = node.get("priceItem") or node.get("PriceItem")
        if isinstance(candidate, list):
            items = [item for item in candidate if isinstance(item, Mapping)]
            if items:
                break

    prices: dict[str, Decimal] = {}
    for item in items:
        label = item.get("label") or item.get("Label") or item.get("name") or ""
        field = _classify_price_label(str(label))
        if field is None or field in prices:
            continue
        raw_price = item.get("price", item.get("Price"))
        value = _numeric(raw_price)
        if value is None:
            continue
        prices[field] = _normalise_price_unit(field, value, item.get("unit"))
    return prices


def _coalesce(*values: Any) -> Decimal:
    for value in values:
        if value is not None:
            return to_decimal(value)
    return MONEY


@dataclass
class ModelPrices:
    model_id: str = ""
    model_name: str = ""
    online_input: Decimal = MONEY
    online_cache_right: Decimal = MONEY
    online_output: Decimal = MONEY
    batch_input: Decimal = MONEY
    batch_cache_right: Decimal = MONEY
    batch_output: Decimal = MONEY
    cache_write_5m: Decimal = MONEY
    cache_write_1h: Decimal = MONEY
    web_search_unit: Decimal = WEB_SEARCH_UNIT
    web_parser_unit: Decimal = WEB_PARSER_UNIT
    image_price: Decimal = MONEY
    video_price: Decimal = MONEY
    voice_price: Decimal = MONEY
    raw: dict[str, Any] = field(default_factory=dict)

    def has_any_price(self) -> bool:
        return any(
            value
            for value in (
                self.online_input,
                self.online_cache_right,
                self.online_output,
                self.batch_input,
                self.batch_cache_right,
                self.batch_output,
                self.cache_write_5m,
                self.cache_write_1h,
                self.image_price,
                self.video_price,
                self.voice_price,
            )
        )


def parse_model_detail(
    detail: Mapping[str, Any],
    *,
    model_id: str = "",
    model_name: str = "",
) -> ModelPrices:
    pricing_rules = pick(detail, "PricingRules", "pricingRules", default=None)
    by_label = price_items(detail)

    cache_write_5m = _find_price(
        pricing_rules,
        ("cachewrite5",),
        ("creationcache5",),
        ("cachecreation5",),
        ("cache", "write", "5m"),
        ("cache", "write", "5min"),
    )
    cache_write_1h = _find_price(
        pricing_rules,
        ("cachewrite1",),
        ("creationcache1",),
        ("cachecreation1",),
        ("cache", "write", "1h"),
    )

    if cache_write_5m is None or cache_write_1h is None:
        # Fall back to top-level sibling fields if PricingRules is opaque.
        cache_write_5m = cache_write_5m if cache_write_5m is not None else _find_price(
            detail, ("cachewrite5",), ("cache", "write", "5")
        )
        cache_write_1h = cache_write_1h if cache_write_1h is not None else _find_price(
            detail, ("cachewrite1",), ("cache", "write", "1")
        )

    return ModelPrices(
        model_id=str(pick(detail, "ModelId", "modelId", default=model_id)),
        model_name=str(pick(detail, "ModelName", "modelName", default=model_name)),
        online_input=_coalesce(
            pick(detail, "OnlineCalculateInput", "onlineCalculateInput"), by_label.get("online_input")
        ),
        online_cache_right=_coalesce(
            pick(detail, "OnlineCalculateRight", "onlineCalculateRight"),
            by_label.get("online_cache_right"),
        ),
        online_output=_coalesce(
            pick(detail, "OnlineCalculateOutput", "onlineCalculateOutput"),
            by_label.get("online_output"),
        ),
        batch_input=_coalesce(
            pick(detail, "BatchCalculateInput", "batchCalculateInput"), by_label.get("batch_input")
        ),
        batch_cache_right=_coalesce(
            pick(detail, "BatchCalculateCacheRight", "batchCalculateCacheRight"),
            by_label.get("batch_cache_right"),
        ),
        batch_output=_coalesce(
            pick(detail, "BatchCalculateOutput", "batchCalculateOutput"), by_label.get("batch_output")
        ),
        cache_write_5m=_coalesce(cache_write_5m, by_label.get("cache_write_5m")),
        cache_write_1h=_coalesce(cache_write_1h, by_label.get("cache_write_1h")),
        web_search_unit=_coalesce(
            _first_numeric(pick(detail, "WebSearchPrice", "webSearchPrice", default=None)),
            by_label.get("web_search_unit"),
            WEB_SEARCH_UNIT,
        ),
        web_parser_unit=_coalesce(
            _first_numeric(pick(detail, "WebParserPrice", "webParserPrice", default=None)),
            by_label.get("web_parser_unit"),
            WEB_PARSER_UNIT,
        ),
        image_price=_coalesce(
            _first_numeric(pick(detail, "ImageModelModeMap", "imageModelModeMap", default=None)),
            by_label.get("image_price"),
        ),
        video_price=_coalesce(
            _first_numeric(pick(detail, "VideoModelModeSet", "videoModelModeSet", default=None)),
            by_label.get("video_price"),
        ),
        voice_price=_coalesce(
            _first_numeric(pick(detail, "VoicePrice", "voicePrice", default=None)),
            by_label.get("voice_price"),
        ),
        raw=dict(detail),
    )


def estimate_cost(usage: Mapping[str, Any], prices: ModelPrices) -> Decimal:
    """Return the estimated cost for one normalised token-usage row."""
    cache_miss = Decimal(int(usage.get("cache_miss_token") or 0))
    cache_hit = Decimal(int(usage.get("cache_token") or 0))
    output = Decimal(int(usage.get("output_token") or 0))

    reasoning_type = str(usage.get("reasoning_type") or "normal").lower()

    if reasoning_type == "batch":
        return (
            cache_miss * prices.batch_input / MILLION
            + cache_hit * prices.batch_cache_right / MILLION
            + output * prices.batch_output / MILLION
        )

    cache_write_5m = Decimal(int(usage.get("cache_write_5m_token") or 0))
    cache_write_1h = Decimal(int(usage.get("cache_write_1h_token") or 0))
    web_search = Decimal(int(usage.get("web_search") or 0))
    web_parser = Decimal(int(usage.get("web_parser") or 0))
    image_count = Decimal(int(usage.get("image_count") or 0))
    video_duration = Decimal(int(usage.get("video_duration") or 0))
    voice_word_count = Decimal(int(usage.get("voice_word_count") or 0))

    return (
        cache_miss * prices.online_input / MILLION
        + cache_hit * prices.online_cache_right / MILLION
        + output * prices.online_output / MILLION
        + cache_write_5m * prices.cache_write_5m / MILLION
        + cache_write_1h * prices.cache_write_1h / MILLION
        + web_search * prices.web_search_unit
        + web_parser * prices.web_parser_unit
        + image_count * prices.image_price
        + video_duration * prices.video_price
        + voice_word_count * prices.voice_price
    )


class PriceBook:
    """Cache of model prices keyed by both model id and model name."""

    def __init__(self) -> None:
        self._by_id: dict[str, ModelPrices] = {}
        self._by_name: dict[str, ModelPrices] = {}

    def add(self, prices: ModelPrices) -> None:
        if prices.model_id:
            self._by_id[prices.model_id] = prices
        if prices.model_name:
            self._by_name[prices.model_name] = prices

    def get(self, *, model_id: str = "", model_name: str = "") -> ModelPrices | None:
        return self._by_id.get(model_id) or self._by_name.get(model_name)

    def require(self, *, model_id: str = "", model_name: str = "") -> ModelPrices:
        found = self.get(model_id=model_id, model_name=model_name)
        if found is not None:
            return found
        log.warning("No price found for model_id=%r model_name=%r", model_id, model_name)
        return ModelPrices(model_id=model_id, model_name=model_name)

    def __len__(self) -> int:
        return len(set(self._by_id) | set(self._by_name))
