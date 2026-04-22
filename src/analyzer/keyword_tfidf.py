"""Statistical keyword baseline using TF-IDF. Complements LLM-NER: catches
frequent n-grams (e.g. "điện thoại gập") that NER won't emit as named
entities, and doubles as a cheap ablation for the Technical Report."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from config import settings

logger = logging.getLogger(__name__)


# Tiny emergency fallback used only if ``settings.STOPWORDS_FILE`` is
# unreadable; production runs load the ~2k-entry VN community list. The
# "video/photo/..." group covers VnExpress filler the VN list tends to
# miss. Always unioned on top of the loaded list.
_INLINE_FALLBACK_STOPWORDS: frozenset[str] = frozenset(
    [
        "là", "của", "và", "với", "cho", "trong", "các", "những", "một",
        "này", "đó", "được", "có", "không",
        "video", "photo", "ảnh", "clip",
    ]
)


def _load_stopwords_file(path) -> set[str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            "Stopwords file %s not readable (%s); using inline fallback.",
            path,
            e,
        )
        return set()

    words: set[str] = set()
    for line in raw.splitlines():
        phrase = line.strip().lower()
        if not phrase or phrase.startswith("#"):
            continue
        words.add(phrase)
        # Explode multi-word phrases into unigrams so sklearn's default
        # token pattern can filter them out token-by-token.
        for token in phrase.split():
            if len(token) >= 2:
                words.add(token)
    return words


@dataclass
class KeywordScore:
    """One row in the TF-IDF leaderboard."""

    term: str
    score: float
    doc_freq: int
    sample_titles: List[str]

    def to_dict(self) -> Dict:
        return {
            "term": self.term,
            "score": round(self.score, 4),
            "doc_freq": self.doc_freq,
            "sample_titles": self.sample_titles,
        }


_WS_RE = re.compile(r"\s+")


def _doc_text(article: Dict) -> str:
    parts = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("content", ""),
    ]
    text = " ".join(p for p in parts if p)
    return _WS_RE.sub(" ", text).strip()


def _build_stopwords(extra: Optional[Sequence[str]] = None) -> List[str]:
    words = _load_stopwords_file(settings.STOPWORDS_FILE)
    if not words:
        logger.info(
            "Stopwords: using inline fallback (%d words).",
            len(_INLINE_FALLBACK_STOPWORDS),
        )
    else:
        logger.info(
            "Stopwords: loaded %d entries from %s.",
            len(words),
            settings.STOPWORDS_FILE.name,
        )
    words.update(_INLINE_FALLBACK_STOPWORDS)
    if extra:
        words.update(w.strip().lower() for w in extra if w)
    return sorted(words)


def extract_tfidf_keywords(
    articles: Sequence[Dict],
    top_k: Optional[int] = None,
    ngram_range: Optional[tuple[int, int]] = None,
    min_df: Optional[int] = None,
    max_df: Optional[float] = None,
    extra_stopwords: Optional[Sequence[str]] = None,
    sample_size: int = 3,
) -> List[KeywordScore]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not articles:
        return []

    top_k = top_k or settings.KEYWORD_TOP_K
    ngram_range = ngram_range or (
        settings.TFIDF_NGRAM_MIN,
        settings.TFIDF_NGRAM_MAX,
    )
    min_df = min_df if min_df is not None else settings.TFIDF_MIN_DF
    max_df = max_df if max_df is not None else settings.TFIDF_MAX_DF

    docs = [_doc_text(a) for a in articles]
    titles = [a.get("title", "") for a in articles]
    stopwords = _build_stopwords(extra_stopwords)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stopwords,
        lowercase=True,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    summed = matrix.sum(axis=0).A1
    doc_freq = (matrix > 0).sum(axis=0).A1
    top_idx = summed.argsort()[::-1][:top_k]

    results: List[KeywordScore] = []
    for idx in top_idx:
        term = vocab[idx]
        samples: List[str] = []
        for doc_idx, t in enumerate(titles):
            if len(samples) >= sample_size:
                break
            if term in t.lower():
                samples.append(t)
        results.append(
            KeywordScore(
                term=term,
                score=float(summed[idx]),
                doc_freq=int(doc_freq[idx]),
                sample_titles=samples,
            )
        )

    logger.info(
        "TF-IDF: vocab=%d, top_k=%d, min_df=%s, max_df=%s, ngrams=%s",
        len(vocab),
        top_k,
        min_df,
        max_df,
        ngram_range,
    )
    return results
