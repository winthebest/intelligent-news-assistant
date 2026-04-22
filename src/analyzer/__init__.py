"""Analyzer layer: LLM + statistical keyword extraction (Phase 3).

Public API:
    LLMClient(model=..., ...)            -> thin wrapper over litellm with cache
    extract_tfidf_keywords(articles, ...) -> list[KeywordScore]
    extract_ner_entities(articles, ...)   -> list[Entity]
    merge_keywords(tfidf, ner, ...)       -> list[dict]   (final ranked list)
"""

from .llm_client import LLMClient, LLMResponse
from .keyword_tfidf import extract_tfidf_keywords
from .keyword_ner import extract_ner_entities
from .keyword_merger import merge_keywords
from .resources import HostCapacity, probe_host

__all__ = [
    "LLMClient",
    "LLMResponse",
    "extract_tfidf_keywords",
    "extract_ner_entities",
    "merge_keywords",
    "HostCapacity",
    "probe_host",
]
