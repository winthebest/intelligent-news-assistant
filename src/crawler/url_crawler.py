import csv
import logging
import time
from pathlib import Path
from typing import Set
from urllib.parse import urlparse, urlunparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from config import settings

logger = logging.getLogger(__name__)


def canonicalize_url(url: str) -> str:
    """Strip whitespace, query string, and fragment from ``url``."""
    if not url:
        return ""
    url = url.strip()
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def configure_driver() -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.set_capability('pageLoadStrategy', 'none')

    if settings.SELENIUM_HEADLESS:
        chrome_options.add_argument('--headless')

    chrome_options.add_argument("--disable-features=ScriptStreaming")
    chrome_options.add_argument("--disable-features=PreloadMediaEngagementData")

    chrome_options.add_experimental_option(
        "prefs",
        {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.videos": 2,
            "intl.accept_languages": "vi,en-US;q=0.8,en;q=0.5",
        },
    )

    driver = webdriver.Chrome(options=chrome_options)
    return driver


def crawl_urls(start_page: int, end_page: int, output_file: Path) -> int:

    print(f"\n{'='*60}")
    print(f"STEP 1: Crawl URLs from page {start_page} to {end_page}")
    print(f"{'='*60}\n")

    logger.info(f"Crawling URLs from page {start_page} to {end_page}")

    existing_canonical_urls: Set[str] = set()
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "canonical_url" in row and row["canonical_url"]:
                        existing_canonical_urls.add(row["canonical_url"])
                    elif "URL" in row and row["URL"]:
                        existing_canonical_urls.add(canonicalize_url(row["URL"]))
            logger.info(f"Found {len(existing_canonical_urls)} existing URLs in {output_file}")
        except (FileNotFoundError, csv.Error, KeyError):
            existing_canonical_urls = set()
            logger.warning(f"Could not read existing URLs from {output_file}, starting fresh")

    driver = configure_driver()
    new_canonical_urls: list[str] = []

    try:
        base_url = settings.VNEXPRESS_BASE_URL.rstrip("/")
        for i in range(start_page, end_page + 1):
            url = f"{base_url}-p{i}" if i > 1 else base_url
            print(f"Crawling page {i}...")
            logger.info(f"Crawling page {i}: {url}")
            driver.get(url)
            time.sleep(settings.CRAWL_DELAY)

            links = driver.find_elements(By.XPATH, "//a[@data-medium and @href]")
            page_hrefs = [
                link.get_attribute("href")
                for link in links
                if link.get_attribute("href") and link.get_attribute("href").endswith(".html")
            ]

            page_new_count = 0
            seen_on_this_run = set(new_canonical_urls)
            for href in page_hrefs:
                canonical = canonicalize_url(href)
                if (
                    canonical
                    and canonical not in existing_canonical_urls
                    and canonical not in seen_on_this_run
                ):
                    new_canonical_urls.append(canonical)
                    seen_on_this_run.add(canonical)
                    page_new_count += 1

            print(
                f"  Found {len(page_hrefs)} links, {page_new_count} new URLs "
                f"(total new: {len(new_canonical_urls)}, "
                f"total: {len(existing_canonical_urls) + len(new_canonical_urls)})..."
            )
            logger.info(
                f"Page {i}: Found {len(page_hrefs)} links, {page_new_count} new URLs"
            )
    finally:
        driver.quit()

    # New URLs go first so the CSV stays newest-first, which lets the
    # content crawler's auto-stop trigger early on old pages.
    all_canonical_urls: list[str] = list(new_canonical_urls) + [
        u for u in existing_canonical_urls if u not in new_canonical_urls
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_url"])
        for canonical_url in all_canonical_urls:
            writer.writerow([canonical_url])

    print(
        f"\n✓ Saved {len(all_canonical_urls)} total URLs "
        f"({len(new_canonical_urls)} new) to {output_file}"
    )
    logger.info(
        f"Saved {len(all_canonical_urls)} total URLs "
        f"({len(new_canonical_urls)} new) to {output_file}"
    )
    return len(all_canonical_urls)


if __name__ == "__main__":

    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Crawl URLs from VNExpress")
    parser.add_argument("--start-page", type=int, default=1, help="Starting page number")
    parser.add_argument("--end-page", type=int, default=2, help="Ending page number")
    parser.add_argument("--output-file", type=Path,
                       default=settings.RAW_DATA_DIR / "vnexpress_links.csv",
                       help="Output CSV file")

    args = parser.parse_args()
    total = crawl_urls(args.start_page, args.end_page, args.output_file)
    print(f"\n✓ Total URLs: {total}")
