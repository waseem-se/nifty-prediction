import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from news_scraper import (
    NewsItem,
    _filter_and_dedupe,
    _strip_html,
    items_from_dicts,
    scrape_news,
)


def _make_item(title: str, hours_ago: float = 1.0, url: str | None = None) -> NewsItem:
    return NewsItem(
        title=title,
        summary="",
        source="test",
        url=url if url is not None else f"https://example.com/{title}",
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


class TestNewsScraper(unittest.TestCase):
    def test_strip_html_removes_tags(self) -> None:
        self.assertEqual(_strip_html("<p>Hello <b>world</b></p>"), "Hello world")
        self.assertEqual(_strip_html(""), "")

    def test_filter_drops_old_items(self) -> None:
        items = [_make_item("recent", 1), _make_item("old", 100)]
        kept = _filter_and_dedupe(items, lookback_hours=24, max_items=10)
        self.assertEqual([i.title for i in kept], ["recent"])

    def test_dedupe_by_url(self) -> None:
        items = [
            _make_item("A", 1, url="https://x/1"),
            _make_item("A-dup", 2, url="https://x/1"),
            _make_item("B", 1, url="https://x/2"),
        ]
        kept = _filter_and_dedupe(items, lookback_hours=24, max_items=10)
        self.assertEqual(len(kept), 2)

    def test_max_items_caps_results(self) -> None:
        items = [_make_item(f"t{i}", 1, url=f"https://x/{i}") for i in range(10)]
        kept = _filter_and_dedupe(items, lookback_hours=24, max_items=3)
        self.assertEqual(len(kept), 3)

    def test_items_from_dicts_round_trip(self) -> None:
        raw = [
            {
                "title": "headline",
                "summary": "body",
                "source": "src",
                "url": "https://e/1",
                "published_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        items = items_from_dicts(raw)
        self.assertEqual(items[0].title, "headline")

    def test_scrape_news_uses_cache_when_fresh(self) -> None:
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "news.json"
            now = datetime.now(timezone.utc).isoformat()
            cache.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "title": "cached",
                                "summary": "",
                                "source": "test",
                                "url": "https://e/cached",
                                "published_at": now,
                            }
                        ]
                    }
                )
            )
            # No feeds passed; if cache is read, we still get the cached item.
            result = scrape_news(
                feeds=(), lookback_hours=24, cache_path=cache, cache_ttl_seconds=3600
            )
            self.assertEqual([i.title for i in result.items], ["cached"])

    def test_scrape_news_no_feeds_returns_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "missing.json"
            result = scrape_news(feeds=(), cache_path=cache)
            self.assertEqual(result.items, [])
            self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
