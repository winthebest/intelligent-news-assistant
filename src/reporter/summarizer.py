"""Generate the weekly Executive Summary via one LLM call, with a
deterministic template fallback when the LLM is disabled or errors."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from config import settings

from src.analyzer.llm_client import LLMClient

logger = logging.getLogger(__name__)


PROMPT_FILENAME = "executive_summary.txt"


def _load_prompt_template() -> str:
    path = settings.PROMPTS_DIR / PROMPT_FILENAME
    return path.read_text(encoding="utf-8")


def _format_keywords_block(keywords: Sequence[Dict], top: int = 10) -> str:
    lines: List[str] = []
    for k in keywords[:top]:
        term = k.get("term", "").strip()
        df = k.get("doc_freq")
        if not term:
            continue
        if df is not None:
            lines.append(f"- {term} (xuất hiện trong {df} bài)")
        else:
            lines.append(f"- {term}")
    return "\n".join(lines) if lines else "(không có keyword trending)"


def _format_titles_block(highlighted: Sequence[Dict], top: int = 8) -> str:
    lines: List[str] = []
    for i, h in enumerate(highlighted[:top], 1):
        title = h.get("title") or ""
        if not title and "article" in h:
            title = (h.get("article") or {}).get("title", "")
        if title:
            lines.append(f"{i}. {title.strip()}")
    return "\n".join(lines) if lines else "(không có tiêu đề)"


def _template_summary(
    keywords: Sequence[Dict],
    highlighted: Sequence[Dict],
    topic: str,
    article_count: int,
) -> str:
    top_terms = [k.get("term", "") for k in keywords[:3] if k.get("term")]
    if not top_terms:
        return (
            f"Trong tuần qua, {article_count} bài báo thuộc chủ đề "
            f"{topic} đã được tổng hợp, tuy nhiên không xác định được "
            f"xu hướng nổi bật."
        )

    top_titles = []
    for h in highlighted[:2]:
        t = h.get("title") or (h.get("article") or {}).get("title", "")
        if t:
            top_titles.append(f'"{t.strip()}"')
    titles_sentence = (
        f" Một số bài đáng chú ý: {', '.join(top_titles)}."
        if top_titles
        else ""
    )

    return (
        f"Trong tuần qua, {article_count} bài báo chủ đề {topic} cho thấy "
        f"các xu hướng nổi bật xoay quanh {', '.join(top_terms)}. "
        f"Các từ khóa này xuất hiện lặp lại trên nhiều nguồn, phản ánh mối "
        f"quan tâm lớn của báo chí trong giai đoạn phân tích.{titles_sentence}"
    )


def generate_executive_summary(
    keywords: Sequence[Dict],
    highlighted: Sequence[Dict],
    topic: str,
    period: str,
    article_count: int,
    use_llm: Optional[bool] = None,
    client: Optional[LLMClient] = None,
) -> str:
    """Return a single paragraph summarising the week. Falls back to a
    deterministic template when the LLM is disabled or errors."""
    if use_llm is None:
        use_llm = settings.REPORT_USE_LLM_SUMMARY

    if not use_llm:
        logger.info("Executive summary: LLM disabled -> template fallback.")
        return _template_summary(keywords, highlighted, topic, article_count)

    try:
        template = _load_prompt_template()
    except OSError as e:
        logger.warning(
            "Executive summary: prompt template unreadable (%s) -> template fallback.",
            e,
        )
        return _template_summary(keywords, highlighted, topic, article_count)

    prompt = template.format(
        topic=topic,
        period=period,
        article_count=article_count,
        keywords_block=_format_keywords_block(keywords),
        titles_block=_format_titles_block(highlighted),
    )

    try:
        llm = client or LLMClient()
        resp = llm.complete(prompt)
    except Exception as e:  # noqa: BLE001
        logger.warning("Executive summary LLM call failed (%s) -> template fallback.", e)
        return _template_summary(keywords, highlighted, topic, article_count)

    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    if not text:
        logger.warning("Executive summary LLM returned empty -> template fallback.")
        return _template_summary(keywords, highlighted, topic, article_count)

    logger.info(
        "Executive summary: %d chars, model=%s, cached=%s, latency=%dms",
        len(text),
        resp.model,
        resp.cached,
        resp.latency_ms,
    )
    return text
