"""CLI entry point for Phase 4. Reads ``data/clean/articles.json`` +
``data/analysis/keywords_final.json`` and writes
``reports/weekly_report_<date>.{md,json}``. See README for usage."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import settings

from .formatter import render_report
from .highlighter import HighlightedArticle, rank_articles
from .summarizer import generate_executive_summary

logger = logging.getLogger(__name__)

VN_TZ = timezone(timedelta(hours=7))


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_period(articles: Sequence[Dict]) -> Tuple[str, Optional[str], Optional[str]]:
    """Return ``(human_string, oldest_iso, newest_iso)`` derived from
    ``date_published``, or a "7 ngày gần nhất" fallback."""
    dates: List[str] = [
        a.get("date_published") for a in articles if a.get("date_published")
    ]
    if not dates:
        return "7 ngày gần nhất", None, None
    oldest = min(dates)
    newest = max(dates)
    if oldest == newest:
        return f"Ngày {oldest}", oldest, newest
    return f"Từ {oldest} đến {newest}", oldest, newest


def _unique_sources(articles: Sequence[Dict]) -> List[str]:
    seen: List[str] = []
    for a in articles:
        s = a.get("source")
        if s and s not in seen:
            seen.append(s)
    return seen


def run_reporter(
    articles_file: Path,
    keywords_file: Path,
    output_dir: Path,
    topic: str,
    top_articles: int,
    top_keywords: int,
    recency_decay: float,
    max_overlap: int,
    use_llm: bool,
) -> Dict:
    logger.info("Loading articles from %s", articles_file)
    articles: List[Dict] = _load_json(articles_file)  # type: ignore[assignment]
    logger.info("Loading keywords from %s", keywords_file)
    keywords: List[Dict] = _load_json(keywords_file)  # type: ignore[assignment]
    logger.info("Corpus: %d articles, %d keywords", len(articles), len(keywords))

    period, oldest, newest = _compute_period(articles)
    sources = _unique_sources(articles)

    highlighted: List[HighlightedArticle] = rank_articles(
        articles,
        keywords,
        top_n=top_articles,
        recency_decay=recency_decay,
        max_overlap=max_overlap,
    )
    highlighted_dicts = [h.to_dict() for h in highlighted]

    summary_text = generate_executive_summary(
        keywords=keywords,
        highlighted=highlighted_dicts,
        topic=topic,
        period=period,
        article_count=len(articles),
        use_llm=use_llm,
    )

    generated_at = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M %z")
    md = render_report(
        topic=topic,
        period=period,
        article_count=len(articles),
        generated_at=generated_at,
        executive_summary=summary_text,
        keywords=keywords,
        highlighted=highlighted_dicts,
        keywords_top=top_keywords,
        llm_model=settings.LLM_MODEL,
        sources=sources,
        analysis_artifacts=[
            "data/analysis/keywords_tfidf.json",
            "data/analysis/keywords_ner.json",
            "data/analysis/keywords_final.json",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(VN_TZ).date().isoformat()
    md_path = output_dir / f"weekly_report_{today}.md"
    json_path = output_dir / f"weekly_report_{today}.json"

    md_path.write_text(md, encoding="utf-8")
    payload = {
        "topic": topic,
        "period": period,
        "oldest_date": oldest,
        "newest_date": newest,
        "article_count": len(articles),
        "sources": sources,
        "generated_at": generated_at,
        "llm_model": settings.LLM_MODEL,
        "top_keywords": keywords[:top_keywords],
        "highlighted": highlighted_dicts,
        "executive_summary": summary_text,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "markdown_file": str(md_path),
        "json_file": str(json_path),
        "article_count": len(articles),
        "highlighted_count": len(highlighted_dicts),
        "keyword_count": min(top_keywords, len(keywords)),
        "period": period,
    }


def _print_summary(summary: Dict) -> None:
    print()
    print("=" * 60)
    print("REPORTER SUMMARY")
    print("=" * 60)
    print(f"  Period           : {summary['period']}")
    print(f"  Articles         : {summary['article_count']}")
    print(f"  Keywords shown   : {summary['keyword_count']}")
    print(f"  Highlighted news : {summary['highlighted_count']}")
    print(f"  Markdown         : {summary['markdown_file']}")
    print(f"  JSON companion   : {summary['json_file']}")
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Render the weekly news report (Phase 4)."
    )
    parser.add_argument(
        "--articles",
        type=Path,
        default=settings.CLEAN_DATA_DIR / "articles.json",
        help="Cleaned articles JSON (filter output).",
    )
    parser.add_argument(
        "--keywords",
        type=Path,
        default=settings.ANALYSIS_DATA_DIR / "keywords_final.json",
        help="Merged keyword list (analyzer output).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.REPORTS_DIR,
        help="Where to write the Markdown and JSON report files.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Khoa học & Công nghệ",
        help="Human-readable topic label used in the report title.",
    )
    parser.add_argument(
        "--top-articles",
        type=int,
        default=settings.REPORT_TOP_ARTICLES,
        help="How many articles to highlight.",
    )
    parser.add_argument(
        "--top-keywords",
        type=int,
        default=settings.REPORT_TOP_KEYWORDS,
        help="How many keywords to include in the trending table.",
    )
    parser.add_argument(
        "--recency-decay",
        type=float,
        default=settings.REPORT_RECENCY_DECAY,
        help="Per-day geometric decay for recency (0 disables).",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=settings.REPORT_DIVERSITY_MAX_OVERLAP,
        help="Max shared keywords between highlighted picks.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM exec-summary call; use deterministic template instead.",
    )
    args = parser.parse_args()

    summary = run_reporter(
        articles_file=args.articles,
        keywords_file=args.keywords,
        output_dir=args.output_dir,
        topic=args.topic,
        top_articles=args.top_articles,
        top_keywords=args.top_keywords,
        recency_decay=args.recency_decay,
        max_overlap=args.max_overlap,
        use_llm=not args.no_llm,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
