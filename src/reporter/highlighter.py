"""Pick the top-N "highlighted news" articles for the weekly report.

Score: ``sum(final_score of matching keywords) * (1 - decay) ^ days_old``,
where "matching" is a word-boundary regex against ``title + description``
and ``days_old`` is measured from the freshest article in the corpus.
A greedy diversity pass then drops candidates that share ``max_overlap``
or more keywords with any already-picked article.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HighlightedArticle:
    article: Dict
    score: float
    matched_keywords: List[str] = field(default_factory=list)
    days_old: int = 0

    def to_dict(self) -> Dict:
        return {
            "canonical_url": self.article.get("canonical_url"),
            "title": self.article.get("title"),
            "description": self.article.get("description"),
            "published_at": self.article.get("published_at"),
            "score": round(self.score, 4),
            "matched_keywords": self.matched_keywords,
            "days_old": self.days_old,
        }


def _parse_published(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        logger.debug("Unparseable published_at=%r", ts)
        return None


@lru_cache(maxsize=512)
def _kw_pattern(term: str) -> re.Pattern:
    # Word-boundary regex instead of plain substring: short acronyms like
    # "AI" would otherwise match inside "Dubai" or inside Vietnamese
    # accented letters that decompose to "a + i" under NFD.
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def _matched_keywords(article: Dict, keyword_terms: Sequence[str]) -> List[str]:
    # Body excluded on purpose: too noisy for high-confidence picks.
    hay = " ".join([article.get("title") or "", article.get("description") or ""])
    return [kw for kw in keyword_terms if kw and _kw_pattern(kw).search(hay)]


def rank_articles(
    articles: Sequence[Dict],
    keywords: Sequence[Dict],
    top_n: int,
    recency_decay: float = 0.08,
    max_overlap: int = 3,
) -> List[HighlightedArticle]:
    if not articles or not keywords:
        return []

    kw_score: Dict[str, float] = {
        k["term"]: float(k.get("final_score", 0.0))
        for k in keywords
        if k.get("term")
    }
    terms = list(kw_score.keys())

    parsed_times: List[Tuple[int, datetime]] = []
    for i, art in enumerate(articles):
        dt = _parse_published(art.get("published_at"))
        if dt is not None:
            parsed_times.append((i, dt))
    newest = max((dt for _, dt in parsed_times), default=None)

    scored: List[HighlightedArticle] = []
    for art in articles:
        matched = _matched_keywords(art, terms)
        if not matched:
            continue
        base = sum(kw_score[t] for t in matched)

        dt = _parse_published(art.get("published_at"))
        if dt is not None and newest is not None:
            days_old = max(0, (newest - dt).days)
            factor = (1.0 - recency_decay) ** days_old
        else:
            days_old = 0
            factor = 1.0

        scored.append(
            HighlightedArticle(
                article=art,
                score=base * factor,
                matched_keywords=matched,
                days_old=days_old,
            )
        )

    scored.sort(key=lambda h: h.score, reverse=True)

    picked: List[HighlightedArticle] = []
    for cand in scored:
        if len(picked) >= top_n:
            break
        cand_set = set(cand.matched_keywords)
        if any(
            len(cand_set & set(p.matched_keywords)) >= max_overlap
            for p in picked
        ):
            continue
        picked.append(cand)

    logger.info(
        "Highlighted: %d candidates scored, %d picked (top_n=%d, decay=%.2f, "
        "overlap<%d)",
        len(scored),
        len(picked),
        top_n,
        recency_decay,
        max_overlap,
    )
    return picked
