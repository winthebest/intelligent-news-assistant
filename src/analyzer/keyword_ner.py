from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from config import settings

from .entity_classifier import is_common_noun
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


PROMPT_PATH = settings.PROJECT_ROOT / "prompts" / "ner_titles.txt"


# Short system prompt: pins role so smaller models (Qwen 7b/14b) don't drift
# into summarisation, and leaves attention budget for the user prompt.
NER_SYSTEM_PROMPT = (
    "Bạn là hệ thống NER (Named Entity Recognition) cho tin công nghệ "
    "tiếng Việt. Output DUY NHẤT là JSON array thuần, mỗi phần tử có 2 "
    "trường ``entity`` và ``count``. Không prose, không markdown fence, "
    "không tạo trường mới."
)


def _coerce_to_list(raw: object) -> object:
    """Recover the two common schema-drift shapes from small models:
    a wrapper dict ``{"entities": [...]}`` or a flat ``{entity: count}``
    mapping. Anything else is returned unchanged for the validator."""
    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ("entities", "data", "result", "results", "items"):
            if isinstance(raw.get(key), list):
                return raw[key]

        # Only accept flat mappings where every value is numeric — avoids
        # eating a nested structure we don't recognise.
        if raw and all(isinstance(v, (int, float)) for v in raw.values()):
            logger.warning(
                "NER: recovered flat {entity: count} shape (%d items).",
                len(raw),
            )
            return [{"entity": k, "count": int(v)} for k, v in raw.items()]

    return raw


@dataclass
class Entity:
    entity: str
    count: int
    sample_titles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "entity": self.entity,
            "count": self.count,
            "sample_titles": self.sample_titles,
        }


def _format_articles_block(articles: Sequence[Dict]) -> str:
    lines = []
    for i, art in enumerate(articles, start=1):
        title = (art.get("title") or "").strip()
        desc = (art.get("description") or "").strip()
        if len(desc) > 180:
            desc = desc[:180].rstrip() + "…"
        if desc:
            lines.append(f"{i}. {title} || {desc}")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def _load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"NER prompt template not found at {PROMPT_PATH}. "
            "Did you check out prompts/ner_titles.txt?"
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


_ENTITY_FIELD_ALIASES = ("entity", "name", "term", "keyword")
_COUNT_FIELD_ALIASES = ("count", "frequency", "freq", "mentions", "occurrences")


def _first_field(raw: Dict, keys: tuple[str, ...]) -> object | None:
    for k in keys:
        if k in raw and raw[k] is not None:
            return raw[k]
    return None


def _validate_entity(raw: object) -> Optional[Dict]:
    """Lenient on field NAMES (accepts ``entity``/``name``/``term``/…,
    ``count``/``frequency``/…), strict on the output SHAPE — always emits
    the canonical ``{entity, count}`` schema or drops the row."""
    if not isinstance(raw, dict):
        return None

    entity_val = _first_field(raw, _ENTITY_FIELD_ALIASES)
    count_val = _first_field(raw, _COUNT_FIELD_ALIASES)
    if entity_val is None or count_val is None:
        logger.debug("NER: dropping entity with missing entity/count: %s", raw)
        return None

    entity = str(entity_val).strip()
    if not entity:
        return None
    try:
        count = int(count_val)
    except (TypeError, ValueError):
        return None

    return {"entity": entity, "count": max(count, 1)}


def _find_sample_titles(
    entity: str,
    articles: Sequence[Dict],
    limit: int = 3,
) -> List[str]:
    """Backfill up to ``limit`` real titles that mention ``entity``, so the
    sample titles in the output never come from the LLM."""
    needle = entity.lower()
    if not needle:
        return []
    hits: List[str] = []
    for art in articles:
        if len(hits) >= limit:
            break
        title = art.get("title") or ""
        if needle in title.lower():
            hits.append(title)
    return hits


def extract_ner_entities(
    articles: Sequence[Dict],
    client: Optional[LLMClient] = None,
    top_k: Optional[int] = None,
    min_count: Optional[int] = None,
) -> List[Entity]:
    """Single-call batch NER over the full article set."""
    if not articles:
        return []

    client = client or LLMClient()
    top_k = top_k or settings.KEYWORD_TOP_K
    min_count = min_count or settings.NER_MIN_ARTICLE_COUNT

    template = _load_prompt_template()
    articles_block = _format_articles_block(articles)
    prompt = template.format(
        n=len(articles),
        min_count=min_count,
        top_k=top_k,
        articles_block=articles_block,
    )

    logger.info(
        "NER prompt prepared: %d articles, ~%d chars",
        len(articles),
        len(prompt),
    )

    raw = client.complete_json(prompt, system_prompt=NER_SYSTEM_PROMPT)
    raw = _coerce_to_list(raw)
    if not isinstance(raw, list):
        raise ValueError(f"NER: expected JSON array, got {type(raw).__name__}")

    results: List[Entity] = []
    dropped_common_nouns = 0
    for item in raw:
        clean = _validate_entity(item)
        if clean is None:
            continue
        if clean["count"] < min_count:
            continue
        # Drop common-noun drift ("phi hành gia", "tấm chắn nhiệt") that
        # models sometimes emit instead of true named entities.
        if is_common_noun(clean["entity"]):
            dropped_common_nouns += 1
            continue
        samples = _find_sample_titles(clean["entity"], articles, limit=3)
        results.append(
            Entity(
                entity=clean["entity"],
                count=clean["count"],
                sample_titles=samples,
            )
        )
    if dropped_common_nouns:
        logger.info(
            "NER: dropped %d common-noun phrases (blocklist)",
            dropped_common_nouns,
        )

    results.sort(key=lambda e: (-e.count, len(e.entity)))
    results = results[:top_k]
    logger.info("NER: %d entities returned (after filter)", len(results))
    return results
