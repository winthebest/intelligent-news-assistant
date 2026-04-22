import csv
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By

from config import settings, VIETNAMESE_ABBREVIATIONS

from .url_crawler import configure_driver, canonicalize_url

logger = logging.getLogger(__name__)

VN_TZ = timezone(timedelta(hours=7))


def now_vn_iso() -> str:
    return datetime.now(VN_TZ).isoformat()


def replace_abbreviations(text: str) -> str:
    if not text:
        return ""
    for abbr, full_form in VIETNAMESE_ABBREVIATIONS.items():
        text = re.sub(rf'\b{abbr}\b', full_form, text, flags=re.IGNORECASE)
    return text


def wait_for_page_load(driver: webdriver.Chrome, timeout: int = 30) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            state = driver.execute_script("return document.readyState")
        except Exception:
            time.sleep(1)
            continue
        if state == "complete":
            return True
        time.sleep(1)

    logger.warning("Page did not fully load after timeout!")
    return False


# VnExpress "related articles" / sidebar boxes stripped from the DOM before
# we call ``.text`` on ``.fck_detail`` — reduces downstream noise a lot.
_RELATED_SELECTORS = [
    ".fck_detail .list-news-subfolder",
    ".fck_detail .box-tinlienquanv2",
    ".fck_detail .box_tinlienquanv2",
    ".fck_detail .item_related_box",
    ".fck_detail .item-news",
    ".fck_detail .list_news_folder",
    ".fck_detail .block_tinlienquan",
    ".fck_detail .block-tinlienquan",
    ".fck_detail .width_common .list-news-subfolder",
    ".fck_detail .page-guicauhoi",
    ".fck_detail section.page-guicauhoi",
    ".fck_detail .box-wg-guicauhoi",
    ".fck_detail .box-wg-guicauhoi-v2",
]


def _strip_related_dom(driver: webdriver.Chrome) -> None:
    script = (
        "const sel = arguments[0].join(',');"
        "document.querySelectorAll(sel).forEach(el => el.remove());"
    )
    try:
        driver.execute_script(script, _RELATED_SELECTORS)
    except Exception as e:
        logger.debug(f"_strip_related_dom: non-fatal JS error: {e}")


def extract_author(driver: webdriver.Chrome) -> Optional[str]:
    candidates = [
        (By.CSS_SELECTOR, ".author_mail"),
        (By.CSS_SELECTOR, ".author"),
        (By.CSS_SELECTOR, ".name-author"),
        (By.CSS_SELECTOR, ".article-author"),
    ]
    for by, sel in candidates:
        try:
            t = driver.find_element(by, sel).text.strip()
            if t:
                return t
        except Exception:
            continue
    return None



def parse_vnexpress_date(raw: str) -> Optional[str]:
    """Parse 'Thứ bảy, 12/1/2025, 10:30 (GMT+7)' -> '2025-01-12'."""
    if not raw:
        return None
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if not m:
        return None
    d, mo, y = m.groups()
    return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"


def parse_vnexpress_datetime(raw: str) -> Optional[str]:
    """Parse 'Thứ bảy, 12/1/2025, 10:30 (GMT+7)' -> ISO-8601 with offset."""
    if not raw:
        return None

    date_m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if not date_m:
        return None
    d, mo, y = date_m.groups()

    time_m = re.search(r"(\d{1,2}):(\d{2})", raw)
    if time_m:
        hh, mm = time_m.groups()
    else:
        hh, mm = "00", "00"

    return (
        f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
        f"T{hh.zfill(2)}:{mm}:00+07:00"
    )


def crawl_content(
    input_file: Path,
    output_file: Path,
    days: Optional[int] = None,
    max_old_streak: Optional[int] = 5,
) -> int:
    """
    Crawl content from saved URLs.

    Args:
        input_file:  CSV file containing URLs.
        output_file: JSON file to save articles.
        days:        Optional publication window. When set (e.g. 7), articles
                     with a parseable ``date_published`` older than
                     ``now - days`` are skipped and not written to the output
                     JSON. Articles whose date cannot be parsed are always
                     kept (with a warning log) to avoid silent data loss.
        max_old_streak: Auto-stop the whole crawl after this many consecutive
                     "too old" articles. Requires ``days`` to be set and the
                     input CSV to be ordered newest-first (as produced by the
                     current ``url_crawler``). Pass ``None`` or 0 to disable.

    Returns:
        Number of articles currently stored in ``output_file`` after the run.
    """
    print(f"\n{'='*60}")
    print(f"STEP 2: Crawl content from URLs")
    print(f"{'='*60}\n")

    logger.info(f"Crawling content from {input_file}")

    cutoff_date = None
    if days is not None and days > 0:
        cutoff_date = (datetime.now(VN_TZ) - timedelta(days=days)).date()
        logger.info(f"Applying publication window: skip articles older than {cutoff_date}")
        print(f"Publication window: keep articles published on/after {cutoff_date}\n")

    # Stop crawl if no publication window is set (<= 0)
    if cutoff_date is None:
        max_old_streak = None
    elif max_old_streak is not None and max_old_streak <= 0:
        max_old_streak = None

    if max_old_streak:
        logger.info(f"Auto-stop enabled: break after {max_old_streak} consecutive old articles")
        print(f"Auto-stop: break after {max_old_streak} consecutive old articles\n")

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    url_list: List[str] = []
    with input_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "canonical_url" in row and row["canonical_url"]:
                url_list.append(row["canonical_url"])
            elif "URL" in row and row["URL"]:
                url_list.append(canonicalize_url(row["URL"]))

    if not url_list:
        logger.warning("No URLs to crawl!")
        print("No URLs to crawl!")
        return 0

    articles: List[Dict] = []
    crawled_canonical_urls = set()
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as f:
                articles = json.load(f)
                for article in articles:
                    canonical = article.get("canonical_url")
                    if canonical:
                        crawled_canonical_urls.add(canonical)
                    else:
                        # Fallback for legacy records that only have ``url``.
                        old_url = article.get("url", "")
                        if old_url:
                            crawled_canonical_urls.add(canonicalize_url(old_url))
        except (json.JSONDecodeError, KeyError):
            articles = []
            crawled_canonical_urls = set()

    new_urls = [u for u in url_list if u not in crawled_canonical_urls]

    if not new_urls:
        logger.info(f"All {len(url_list)} URLs have been crawled. Nothing new.")
        print(f"All {len(url_list)} URLs have been crawled. Nothing new.")
        return len(articles)

    logger.info(f"Already have {len(articles)} articles. Will crawl {len(new_urls)} new URLs.")
    print(f"Already have {len(articles)} articles. Will crawl {len(new_urls)} new URLs.\n")

    driver = configure_driver()
    old_streak = 0  # consecutive "too old" articles, for auto-stop

    try:
        for idx, canonical_url in enumerate(new_urls, 1):
            print(f"[{idx}/{len(new_urls)}] Crawling: {canonical_url[:60]}...")
            logger.info(f"[{idx}/{len(new_urls)}] Crawling: {canonical_url[:60]}...")

            try:
                driver.get(canonical_url)
                if not wait_for_page_load(driver):
                    print(f"Page did not fully load, skipping...")
                    logger.warning(f"Page did not fully load: {canonical_url}")
                    continue
            except Exception as e:
                print(f"Error loading page: {e}")
                logger.error(f"Error loading page {canonical_url}: {e}")
                continue

            # Read the date first so we can bail out on old articles without
            # paying the cost of extracting title/content.
            try:
                date = driver.find_element(By.CLASS_NAME, "date").text.strip()
            except Exception as e:
                date = ""
                logger.warning(f"No .date element on {canonical_url}: {e}")

            date_published = parse_vnexpress_date(date)
            published_at = parse_vnexpress_datetime(date)

            if cutoff_date is not None and date_published:
                try:
                    pub = datetime.strptime(date_published, "%Y-%m-%d").date()
                    if pub < cutoff_date:
                        old_streak += 1
                        print(
                            f"Skipping (published {pub}, older than window) "
                            f"[streak {old_streak}]"
                        )
                        logger.info(
                            f"Skipping old article {canonical_url}: "
                            f"{pub} < {cutoff_date} (streak={old_streak})"
                        )
                        if max_old_streak and old_streak >= max_old_streak:
                            print(
                                f"Auto-stop: {old_streak} consecutive old "
                                f"articles >= {max_old_streak}. Stopping crawl."
                            )
                            logger.info(
                                f"Auto-stop triggered: {old_streak} "
                                f"consecutive old articles >= {max_old_streak}"
                            )
                            break
                        continue
                except ValueError:
                    logger.warning(
                        f"Unparseable date_published for {canonical_url}: {date_published}"
                    )
            elif cutoff_date is not None and not date_published:
                logger.warning(f"Missing date_published for {canonical_url}; keeping article")

            old_streak = 0

            try:
                title = driver.find_element(By.CLASS_NAME, "title-detail").text.strip()
                description = driver.find_element(By.CLASS_NAME, "description").text.strip()

                _strip_related_dom(driver)
                content = driver.find_element(By.CLASS_NAME, "fck_detail").text.strip()

                title = replace_abbreviations(title)
                description = replace_abbreviations(description)
                content = replace_abbreviations(content)

                author = extract_author(driver)

                article = {
                    "canonical_url": canonical_url,
                    "title": title,
                    "description": description,
                    "content": content,
                    "date": date,
                    "date_published": date_published,
                    "published_at": published_at,
                    "source": "vnexpress",
                    "category": settings.VNEXPRESS_CATEGORY,
                    "author": author,
                    "crawl_time": now_vn_iso(),
                }

                articles.append(article)

                # Persist after every article so a crash doesn't lose the run.
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(articles, f, ensure_ascii=False, indent=4)

                print(f"Crawled and saved")
                logger.info(f"Crawled and saved: {canonical_url}")

            except Exception as e:
                print(f"Error extracting data: {e}")
                logger.error(f"Error extracting data from {canonical_url}: {e}")
                continue

    finally:
        driver.quit()
    
    print(f"\nComplete! Total {len(articles)} articles in {output_file}")
    logger.info(f"Complete! Total {len(articles)} articles in {output_file}")
    return len(articles)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Crawl content from URLs")
    parser.add_argument("--input-file", type=Path,
                       default=settings.RAW_DATA_DIR / "vnexpress_links.csv",
                       help="Input CSV file with URLs")
    parser.add_argument("--output-file", type=Path,
                       default=settings.RAW_DATA_DIR / "vnexpress_articles.json",
                       help="Output JSON file for articles")
    parser.add_argument("--days", type=int, default=7,
                       help="Keep only articles published within the last N days "
                            "(default: 7). Use 0 to disable the filter.")
    parser.add_argument("--max-old-streak", type=int, default=5,
                       help="Auto-stop the whole crawl after N consecutive "
                            "too-old articles. Requires --days > 0 and a "
                            "chronologically ordered CSV (newest first). "
                            "Use 0 to disable. Default: 5.")

    args = parser.parse_args()
    total = crawl_content(
        args.input_file,
        args.output_file,
        days=args.days if args.days > 0 else None,
        max_old_streak=args.max_old_streak if args.max_old_streak > 0 else None,
    )
    print(f"\nTotal articles: {total}")
