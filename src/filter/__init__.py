"""Filter layer: clean articles + dedup (Phase 2).

Public API:
    clean_articles(raw) -> (cleaned, dropped)
    dedup_exact(articles) -> (kept, dropped)
    dedup_fuzzy(articles, threshold) -> (kept, dropped)
"""

from .cleaner import clean_article, clean_articles, normalize_unicode, strip_related_articles, validate_article
from .dedup import dedup_exact, dedup_fuzzy

__all__ = [
    "clean_article",
    "clean_articles",
    "normalize_unicode",
    "strip_related_articles",
    "validate_article",
    "dedup_exact",
    "dedup_fuzzy",
]
