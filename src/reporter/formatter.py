"""Render the weekly Markdown report: header, executive summary, trending
keywords table, highlighted news, footer. Plain string concatenation — the
structure is short and stable enough to not warrant a templating engine."""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


def _escape_md(text: str) -> str:
    # Only the chars likely to break our tables; full CommonMark escape would
    # be overkill for a curated VN news corpus.
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _short_date(ts: str) -> str:
    if not ts or len(ts) < 10:
        return ts or "?"
    return ts[:10]


def render_header(
    topic: str,
    period: str,
    article_count: int,
    generated_at: str,
) -> str:
    return (
        f"# Báo cáo tin tức tuần — {topic}\n\n"
        f"_{period} · {article_count} bài đã phân tích · "
        f"tạo lúc {generated_at}_\n\n"
    )


def render_executive_summary(text: str) -> str:
    return "## 1. Executive Summary\n\n" + text.strip() + "\n\n"


def render_trending_keywords(keywords: Sequence[Dict], top: int) -> str:
    lines = [
        "## 2. Trending Keywords\n",
        "| # | Từ khóa | Điểm | Số bài | Nguồn |",
        "|---|---|---:|---:|---|",
    ]
    if not keywords:
        lines.append("| _(không có keyword trending)_ |  |  |  |  |")
    for i, k in enumerate(keywords[:top], 1):
        term = _escape_md(k.get("term", ""))
        score = k.get("final_score", 0.0)
        df = k.get("doc_freq", "-")
        sources = "+".join(k.get("sources") or []) or "-"
        lines.append(f"| {i} | {term} | {score:.3f} | {df} | {sources} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_highlighted_news(highlighted: Sequence[Dict]) -> str:
    lines = ["## 3. Highlighted News\n"]
    if not highlighted:
        lines.append("_(không có bài nào đạt ngưỡng highlight)_\n")
        return "\n".join(lines)

    for i, h in enumerate(highlighted, 1):
        title = _escape_md(h.get("title") or "")
        url = h.get("canonical_url") or ""
        published = _short_date(h.get("published_at") or "")
        matched = h.get("matched_keywords") or []
        desc = _escape_md(h.get("description") or "")
        score = h.get("score", 0.0)

        link = f"[{title}]({url})" if url else title
        matched_str = ", ".join(f"`{m}`" for m in matched[:5])
        if len(matched) > 5:
            matched_str += f" _(+{len(matched) - 5})_"

        lines.append(f"### {i}. {link}")
        lines.append(
            f"_Xuất bản: {published} · điểm: {score:.3f} · "
            f"khớp từ khóa: {matched_str or '—'}_"
        )
        if desc:
            lines.append("")
            lines.append(f"> {desc}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_footer(
    llm_model: str,
    sources: Sequence[str],
    analysis_artifacts: Sequence[str],
) -> str:
    src = ", ".join(sources) if sources else "-"
    arts = ", ".join(f"`{a}`" for a in analysis_artifacts)
    return (
        "---\n\n"
        f"_Pipeline: VnExpress crawler → clean+dedup → TF-IDF + LLM-NER "
        f"({llm_model}) → merge → report._\n"
        f"_Nguồn: {src}. Artifacts: {arts}._\n"
    )


def render_report(
    *,
    topic: str,
    period: str,
    article_count: int,
    generated_at: str,
    executive_summary: str,
    keywords: Sequence[Dict],
    highlighted: Sequence[Dict],
    keywords_top: int,
    llm_model: str,
    sources: Sequence[str],
    analysis_artifacts: Sequence[str],
) -> str:
    parts: List[str] = [
        render_header(topic, period, article_count, generated_at),
        render_executive_summary(executive_summary),
        render_trending_keywords(keywords, keywords_top),
        render_highlighted_news(highlighted),
        render_footer(llm_model, sources, analysis_artifacts),
    ]
    return "".join(parts)
