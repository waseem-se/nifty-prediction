"""News scraping layer for the NIFTY prediction agent.

Pulls market-relevant headlines from a small set of RSS feeds, normalizes
them into :class:`NewsItem` objects, deduplicates, applies a look-back
window, and caches results on disk so repeated runs in the same hour do
not re-hit the network.

The scraper is designed to fail soft: any individual feed that errors out
is skipped, and if ``feedparser`` is not installed the caller receives an
empty list rather than a hard crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence


# Default RSS sources. Kept small and high-signal to control LLM cost.
DEFAULT_FEEDS: tuple[tuple[str, str], ...] = (
    ("Moneycontrol", "https://www.moneycontrol.com/rss/marketsnews.xml"),
    ("Economic Times Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Livemint Markets", "https://www.livemint.com/rss/markets"),
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
)

DEFAULT_CACHE_PATH = Path.home() / ".cache" / "nifty" / "news.json"
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_MAX_ITEMS = 40
DEFAULT_CACHE_TTL_SECONDS = 60 * 30  # 30 minutes


@dataclass(frozen=True)
class NewsItem:
    """A normalized news headline."""

    title: str
    summary: str
    source: str
    url: str
    published_at: datetime

    def fingerprint(self) -> str:
        """Stable hash used for deduplication."""
        key = (self.url or self.title).strip().lower()
        return hashlib.sha1(key.encode("utf-8")).hexdigest()


@dataclass
class ScrapeResult:
    items: list[NewsItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _parse_published(entry: object) -> datetime:
    """Best-effort extraction of a publish datetime from a feedparser entry."""
    for attr in ("published_parsed", "updated_parsed"):
        value = getattr(entry, attr, None)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return datetime.now(timezone.utc)


def _strip_html(text: str) -> str:
    """Strip simple HTML tags from RSS summaries without adding deps."""
    if not text:
        return ""
    out: list[str] = []
    in_tag = False
    for ch in text:
        if ch == "<":
            in_tag = True
            continue
        if ch == ">":
            in_tag = False
            continue
        if not in_tag:
            out.append(ch)
    return " ".join("".join(out).split())


def _parse_feed(source: str, url: str) -> list[NewsItem]:
    """Parse a single RSS feed using feedparser.

    Imported lazily so the package remains importable in test/offline
    environments where ``feedparser`` is not installed.
    """
    try:
        import feedparser  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError("feedparser is required for live scraping") from exc

    parsed = feedparser.parse(url)
    items: list[NewsItem] = []
    for entry in getattr(parsed, "entries", []) or []:
        title = _strip_html(getattr(entry, "title", "") or "")
        if not title:
            continue
        summary = _strip_html(getattr(entry, "summary", "") or "")
        link = getattr(entry, "link", "") or ""
        items.append(
            NewsItem(
                title=title,
                summary=summary,
                source=source,
                url=link,
                published_at=_parse_published(entry),
            )
        )
    return items


def _filter_and_dedupe(
    items: Iterable[NewsItem],
    lookback_hours: int,
    max_items: int,
) -> list[NewsItem]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    seen: set[str] = set()
    kept: list[NewsItem] = []
    for item in sorted(items, key=lambda i: i.published_at, reverse=True):
        if item.published_at < cutoff:
            continue
        fp = item.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(item)
        if len(kept) >= max_items:
            break
    return kept


def _serialize(items: Sequence[NewsItem]) -> list[dict]:
    out: list[dict] = []
    for item in items:
        data = asdict(item)
        data["published_at"] = item.published_at.isoformat()
        out.append(data)
    return out


def _deserialize(raw: Sequence[dict]) -> list[NewsItem]:
    items: list[NewsItem] = []
    for entry in raw:
        try:
            published = datetime.fromisoformat(entry["published_at"])
        except (KeyError, ValueError):
            published = datetime.now(timezone.utc)
        items.append(
            NewsItem(
                title=entry.get("title", ""),
                summary=entry.get("summary", ""),
                source=entry.get("source", ""),
                url=entry.get("url", ""),
                published_at=published,
            )
        )
    return items


def _load_cache(cache_path: Path, ttl_seconds: int) -> list[NewsItem] | None:
    if not cache_path.exists():
        return None
    try:
        if time.time() - cache_path.stat().st_mtime > ttl_seconds:
            return None
        with cache_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return _deserialize(payload.get("items", []))
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(cache_path: Path, items: Sequence[NewsItem]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump({"items": _serialize(items)}, fh)
    except OSError:
        # Caching is best-effort; ignore filesystem issues.
        pass


def scrape_news(
    feeds: Sequence[tuple[str, str]] = DEFAULT_FEEDS,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    max_items: int = DEFAULT_MAX_ITEMS,
    cache_path: Path | None = DEFAULT_CACHE_PATH,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    use_cache: bool = True,
) -> ScrapeResult:
    """Scrape configured RSS feeds and return a filtered, deduped result.

    Always returns a :class:`ScrapeResult` (never raises). Per-feed errors
    are collected in ``result.errors`` instead of propagating.
    """
    cache_file = Path(cache_path) if cache_path else None
    if use_cache and cache_file:
        cached = _load_cache(cache_file, cache_ttl_seconds)
        if cached is not None:
            return ScrapeResult(
                items=_filter_and_dedupe(cached, lookback_hours, max_items)
            )

    result = ScrapeResult()
    for source, url in feeds:
        try:
            result.items.extend(_parse_feed(source, url))
        except Exception as exc:  # pragma: no cover - network-dependent
            result.errors.append(f"{source}: {exc}")

    result.items = _filter_and_dedupe(result.items, lookback_hours, max_items)
    if use_cache and cache_file and result.items:
        _save_cache(cache_file, result.items)
    return result


def items_from_dicts(raw: Sequence[dict]) -> list[NewsItem]:
    """Public helper used by tests / external callers to build NewsItems."""
    return _deserialize(raw)
