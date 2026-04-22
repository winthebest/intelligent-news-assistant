"""Phase 3 CLI: cleaned articles → TF-IDF baseline + LLM-NER + merged list.

Writes four artifacts under ``data/analysis/``: ``keywords_tfidf.json``,
``keywords_ner.json``, ``keywords_final.json`` (used by the reporter), and
``analyzer_report.json`` (run metadata). Pass ``--skip-llm`` to degrade
gracefully to TF-IDF only when the LLM is unreachable. See README for
invocation details."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from config import settings

from .keyword_merger import merge_keywords
from .keyword_ner import extract_ner_entities
from .keyword_tfidf import extract_tfidf_keywords
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def run_analyzer(
    input_file: Path,
    output_dir: Path,
    top_k: int,
    skip_llm: bool,
    llm_model: str | None = None,
) -> Dict:
    """End-to-end analyzer. Returns the run summary (also persisted to disk)."""
    logger.info("Loading cleaned articles from %s", input_file)
    with input_file.open("r", encoding="utf-8") as f:
        articles: List[Dict] = json.load(f)
    logger.info("Loaded %d articles", len(articles))

    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "top_k": top_k,
        "article_count": len(articles),
        "skip_llm": skip_llm,
        "llm_model": llm_model or settings.LLM_MODEL,
        "stages": {},
    }

    t0 = time.time()
    tfidf = extract_tfidf_keywords(articles, top_k=top_k)
    report["stages"]["tfidf"] = {
        "count": len(tfidf),
        "latency_ms": int((time.time() - t0) * 1000),
    }
    (output_dir / "keywords_tfidf.json").write_text(
        json.dumps([k.to_dict() for k in tfidf], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ner: list = []
    if skip_llm:
        logger.warning("Skipping LLM-NER (--skip-llm). Final list = TF-IDF only.")
        report["stages"]["ner"] = {"count": 0, "skipped": True}
    else:
        try:
            client = LLMClient(model=llm_model) if llm_model else LLMClient()
            t0 = time.time()
            ner = extract_ner_entities(articles, client=client, top_k=top_k)
            report["stages"]["ner"] = {
                "count": len(ner),
                "latency_ms": int((time.time() - t0) * 1000),
                "model": client.model,
            }
        except Exception as e:  # noqa: BLE001
            logger.error("LLM-NER failed: %s. Falling back to TF-IDF only.", e)
            report["stages"]["ner"] = {"count": 0, "error": str(e)}
            ner = []

    (output_dir / "keywords_ner.json").write_text(
        json.dumps([e.to_dict() for e in ner], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    final = merge_keywords(tfidf, ner, top_k=top_k)
    report["stages"]["merge"] = {
        "count": len(final),
        "latency_ms": int((time.time() - t0) * 1000),
    }
    (output_dir / "keywords_final.json").write_text(
        json.dumps([m.to_dict() for m in final], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (output_dir / "analyzer_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _print_summary(report: Dict, final_path: Path) -> None:
    stages = report["stages"]
    print()
    print("=" * 60)
    print("ANALYZER SUMMARY")
    print("=" * 60)
    print(f"  Articles analysed    : {report['article_count']}")
    print(f"  LLM model            : {report['llm_model']}")
    print(
        "  TF-IDF baseline      : "
        f"{stages['tfidf']['count']} terms ({stages['tfidf']['latency_ms']} ms)"
    )
    ner_stage = stages["ner"]
    if ner_stage.get("skipped"):
        print("  LLM-NER              : SKIPPED (--skip-llm)")
    elif ner_stage.get("error"):
        print(f"  LLM-NER              : ERROR ({ner_stage['error'][:60]}…)")
    else:
        print(
            "  LLM-NER              : "
            f"{ner_stage['count']} entities ({ner_stage.get('latency_ms', '?')} ms)"
        )
    print(
        "  Merged final list    : "
        f"{stages['merge']['count']} terms ({stages['merge']['latency_ms']} ms)"
    )
    print()

    try:
        top = json.loads(final_path.read_text(encoding="utf-8"))[:10]
    except (OSError, json.JSONDecodeError):
        top = []
    if top:
        print("  Top keywords (preview):")
        print(f"    {'Rank':<5} {'Term':<32} {'Score':<7} {'DocFreq':<8} Sources")
        print(f"    {'-' * 5} {'-' * 32} {'-' * 7} {'-' * 8} {'-' * 15}")
        for i, row in enumerate(top, 1):
            src = "+".join(row.get("sources") or [])
            df = row.get("doc_freq", "-")
            print(
                f"    {i:<5} {row['term'][:32]:<32} "
                f"{row['final_score']:<7.3f} {str(df):<8} {src}"
            )
    print()
    print(f"  Artifacts: {report['output_dir']}")
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Phase 3: TF-IDF baseline + LLM-NER on cleaned articles",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.CLEAN_DATA_DIR / "articles.json",
        help="Cleaned articles JSON (filter output)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.ANALYSIS_DATA_DIR,
        help="Where to write analyzer artifacts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.KEYWORD_TOP_K,
        help="How many final keywords to surface (default from KEYWORD_TOP_K)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override settings.LLM_MODEL for this run (e.g. ollama/llama3.2:3b)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-NER. Final list = TF-IDF only. Useful offline.",
    )
    args = parser.parse_args()

    settings.ensure_dirs()

    report = run_analyzer(
        input_file=args.input,
        output_dir=args.output_dir,
        top_k=args.top_k,
        skip_llm=args.skip_llm,
        llm_model=args.llm_model,
    )
    _print_summary(report, args.output_dir / "keywords_final.json")


if __name__ == "__main__":
    main()
