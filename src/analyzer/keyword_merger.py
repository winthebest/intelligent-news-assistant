from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .keyword_ner import Entity
from .keyword_tfidf import KeywordScore

logger = logging.getLogger(__name__)


DEFAULT_W_NER = 0.6
DEFAULT_W_TFIDF = 0.4
AGREEMENT_BONUS = 0.15


@dataclass
class MergedKeyword:
    term: str
    final_score: float
    sources: List[str]
    doc_freq: int
    ner_count: Optional[int]
    tfidf_score: Optional[float]
    sample_titles: List[str]

    def to_dict(self) -> Dict:
        return {
            "term": self.term,
            "final_score": round(self.final_score, 4),
            "sources": self.sources,
            "doc_freq": self.doc_freq,
            "ner_count": self.ner_count,
            "tfidf_score": (
                round(self.tfidf_score, 4) if self.tfidf_score is not None else None
            ),
            "sample_titles": self.sample_titles,
        }


def _norm(text: str) -> str:
    """Lowercase + strip accents + collapse whitespace, for fuzzy matching."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text)
    return text


def _scale_0_1(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return [1.0] * len(values)
    return [(v - lo) / span for v in values]


def merge_keywords(
    tfidf: Sequence[KeywordScore],
    ner: Sequence[Entity],
    top_k: int = 15,
    w_ner: float = DEFAULT_W_NER,
    w_tfidf: float = DEFAULT_W_TFIDF,
    agreement_bonus: float = AGREEMENT_BONUS,
) -> List[MergedKeyword]:
    """Return the top-k merged keywords sorted by ``final_score``.

    Each source is normalised to [0, 1] independently, then the final score
    is ``w_ner * norm_ner + w_tfidf * norm_tfidf + bonus`` where ``bonus``
    fires when a term (accent-folded + lowercased) appears in both sources.
    """
    ner_scores = _scale_0_1([e.count for e in ner]) if ner else []
    tfidf_scores = _scale_0_1([k.score for k in tfidf]) if tfidf else []

    ner_by_key: Dict[str, tuple[Entity, float]] = {}
    for ent, ns in zip(ner, ner_scores):
        k = _norm(ent.entity)
        if k and k not in ner_by_key:
            ner_by_key[k] = (ent, ns)

    tfidf_by_key: Dict[str, tuple[KeywordScore, float]] = {}
    for kw, ts in zip(tfidf, tfidf_scores):
        k = _norm(kw.term)
        if k and k not in tfidf_by_key:
            tfidf_by_key[k] = (kw, ts)

    # Prefer the NER entity's surface form (proper case) over the
    # lowercased TF-IDF token when both sources carry the same key.
    merged: Dict[str, MergedKeyword] = {}

    for key, (ent, ns) in ner_by_key.items():
        tfidf_hit = tfidf_by_key.get(key)
        ts = tfidf_hit[1] if tfidf_hit else 0.0
        tfidf_raw = tfidf_hit[0].score if tfidf_hit else None
        doc_freq = tfidf_hit[0].doc_freq if tfidf_hit else ent.count
        sources = ["ner"] + (["tfidf"] if tfidf_hit else [])
        bonus = agreement_bonus if tfidf_hit else 0.0
        samples = ent.sample_titles or (tfidf_hit[0].sample_titles if tfidf_hit else [])
        merged[key] = MergedKeyword(
            term=ent.entity,
            final_score=w_ner * ns + w_tfidf * ts + bonus,
            sources=sources,
            doc_freq=doc_freq,
            ner_count=ent.count,
            tfidf_score=tfidf_raw,
            sample_titles=samples,
        )

    for key, (kw, ts) in tfidf_by_key.items():
        if key in merged:
            continue
        merged[key] = MergedKeyword(
            term=kw.term,
            final_score=w_tfidf * ts,
            sources=["tfidf"],
            doc_freq=kw.doc_freq,
            ner_count=None,
            tfidf_score=kw.score,
            sample_titles=kw.sample_titles,
        )

    ordered = sorted(merged.values(), key=lambda m: -m.final_score)[:top_k]
    logger.info(
        "Merged keywords: %d (of %d candidates), top term=%r score=%.3f",
        len(ordered),
        len(merged),
        ordered[0].term if ordered else None,
        ordered[0].final_score if ordered else 0.0,
    )
    return ordered
