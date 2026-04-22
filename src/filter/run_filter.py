from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from config import settings

from .cleaner import clean_articles
from .dedup import dedup_exact, dedup_fuzzy

logger = logging.getLogger(__name__)


def run_filter(
    input_file: Path,
    output_file: Path,
    report_file: Path,
    dedup_threshold: int = 85,
    min_content_length: int = 100,
) -> Dict:
    """Run clean + dedup pipeline on a raw articles JSON. Returns summary dict."""
    logger.info(f"Loading raw articles from {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        raw: List[Dict] = json.load(f)
    logger.info(f"Loaded {len(raw)} raw articles")

    cleaned, invalid = clean_articles(raw, min_content_length=min_content_length)
    after_exact, exact_drops = dedup_exact(cleaned)
    after_fuzzy, fuzzy_drops = dedup_fuzzy(after_exact, threshold=dedup_threshold)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(after_fuzzy, f, ensure_ascii=False, indent=2)

    summary = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "dedup_threshold": dedup_threshold,
        "min_content_length": min_content_length,
        "counts": {
            "raw": len(raw),
            "after_clean": len(cleaned),
            "after_exact_dedup": len(after_exact),
            "after_fuzzy_dedup": len(after_fuzzy),
            "invalid": len(invalid),
            "exact_duplicates": len(exact_drops),
            "fuzzy_duplicates": len(fuzzy_drops),
        },
        "invalid": invalid,
        "exact_duplicates": exact_drops,
        "fuzzy_duplicates": fuzzy_drops,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def _print_summary(summary: Dict) -> None:
    c = summary["counts"]
    print()
    print("=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    print(f"  Raw articles            : {c['raw']}")
    print(f"  After cleaning          : {c['after_clean']}   (dropped {c['invalid']})")
    print(f"  After exact URL dedup   : {c['after_exact_dedup']}   (dropped {c['exact_duplicates']})")
    print(f"  After fuzzy title dedup : {c['after_fuzzy_dedup']}   (dropped {c['fuzzy_duplicates']})")
    print(f"  Fuzzy threshold         : {summary['dedup_threshold']}")
    print()
    print(f"  Output : {summary['output_file']}")
    print(f"  Report : {summary.get('report_file', '(see --report)')}")
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Clean + dedup raw VnExpress articles")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.RAW_DATA_DIR / "vnexpress_articles.json",
        help="Raw articles JSON (crawler output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.CLEAN_DATA_DIR / "articles.json",
        help="Cleaned + deduped articles JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=settings.CLEAN_DATA_DIR / "dedup_report.json",
        help="Per-run dedup audit log",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=85,
        help="token_set_ratio cutoff 0..100 (default 85)",
    )
    parser.add_argument(
        "--min-content-length",
        type=int,
        default=100,
        help="Articles shorter than this are dropped (default 100 chars)",
    )
    args = parser.parse_args()

    summary = run_filter(
        input_file=args.input,
        output_file=args.output,
        report_file=args.report,
        dedup_threshold=args.dedup_threshold,
        min_content_length=args.min_content_length,
    )
    summary["report_file"] = str(args.report)
    _print_summary(summary)


if __name__ == "__main__":
    main()
