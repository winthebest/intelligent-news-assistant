"""Unified entry point for the VnExpress crawling pipeline."""

from pathlib import Path
from typing import Optional

from config import settings

from .url_crawler import crawl_urls
from .content_crawler import crawl_content


def run_pipeline(
    start_page: int = 1,
    end_page: int = 5,
    step: str = "all",
    urls_file: Optional[Path] = None,
    articles_file: Optional[Path] = None,
    days: Optional[int] = 7,
    max_old_streak: Optional[int] = 5,
) -> dict:
    """Run URL + content crawl. ``step`` selects a stage ("urls", "content",
    or "all"). ``days`` is the publication window (default 7); ``None`` or
    <= 0 disables it. ``max_old_streak`` breaks the crawl early after N
    consecutive out-of-window articles."""
    if urls_file is None:
        urls_file = settings.RAW_DATA_DIR / "vnexpress_links.csv"
    if articles_file is None:
        articles_file = settings.RAW_DATA_DIR / "vnexpress_articles.json"

    if days is not None and days <= 0:
        days = None
    if max_old_streak is not None and max_old_streak <= 0:
        max_old_streak = None

    results = {}

    if step in ["urls", "all"]:
        num_urls = crawl_urls(start_page, end_page, urls_file)
        results["urls_crawled"] = num_urls

    if step in ["content", "all"]:
        num_articles = crawl_content(
            urls_file,
            articles_file,
            days=days,
            max_old_streak=max_old_streak,
        )
        results["articles_crawled"] = num_articles

    return results
