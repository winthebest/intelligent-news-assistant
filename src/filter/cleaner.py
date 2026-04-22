from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_WEIRD_SPACES = re.compile(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")


def normalize_unicode(text: str) -> str:
    """NFC-compose Vietnamese diacritics and tidy whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _WEIRD_SPACES.sub(" ", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


# Lines ending with these chars look like real sentences, not bylines.
_SENTENCE_END = (".", ",", ";", ":", "!", "?", '"', "'", ")", "]", "}")

# Byline detection: matches "Name (Theo Source)" or "Name + role suffix".
_BYLINE_WITH_SOURCE_RE = re.compile(
    r"^[^\s(][^()]{0,60}\([Tt]heo[^()]+\)\s*$"
)
_BYLINE_SUFFIX_RE = re.compile(
    r"^(?P<name>[^\s].{0,40}?)\s+"
    r"(tổng hợp|biên dịch|biên soạn|ghi nhận|lược dịch|tường thuật)\s*$",
    re.IGNORECASE,
)

def _is_byline(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 50:
        return False
    if s.endswith(_SENTENCE_END):
        return False
    words = s.split()
    if not (1 <= len(words) <= 4):
        return False
    for w in words:
        if not w or not w[0].isalpha() or not w[0].isupper():
            return False
        if len(w) > 1 and w.isupper():
            return False
    return True



def _is_byline_with_suffix(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 60:
        return False
    m = _BYLINE_SUFFIX_RE.match(s)
    if not m:
        return False
    name = m.group("name").strip()
    words = name.split()
    if not (1 <= len(words) <= 4):
        return False
    for w in words:
        if not w or not w[0].isalpha() or not w[0].isupper():
            return False
    return True


def _is_byline_with_source(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if not _BYLINE_WITH_SOURCE_RE.match(s):
        return False
    # The name before the parenthesis must be 1–5 Title-cased tokens.
    before = s.split("(", 1)[0].strip()
    words = before.split()
    if not (1 <= len(words) <= 5):
        return False
    for w in words:
        if not w or not w[0].isalpha() or not w[0].isupper():
            return False
    return True


def _ends_with_view_count(line: str) -> bool:
    """Detect junk trailing view/comment counts like "Robot X …  74", while
    protecting 4-digit years (1900–2100) from being flagged."""
    s = line.strip()
    if not s or len(s) > 250:
        return False
    parts = s.rsplit(None, 1)
    if len(parts) != 2:
        return False
    _, tail = parts
    if not tail.isdigit():
        return False
    if not (1 <= len(tail) <= 4):
        return False
    if len(tail) == 4 and 1900 <= int(tail) <= 2100:
        return False
    return True


def strip_related_articles(content: str) -> str:
    """Drop VnExpress's trailing "related articles" block by cutting at the
    earliest byline / byline-with-source / title-plus-view-count line found
    in the tail ~45%. Returns ``content`` unchanged if nothing matches."""
    if not content:
        return content

    lines = content.split("\n")
    n = len(lines)
    if n < 4:
        return content

    tail_start = max(1, int(n * 0.55))

    cut_at = -1
    for i in range(tail_start, n):
        line = lines[i]
        if _is_byline(line) and (n - i - 1) >= 2:
            cut_at = i
            break
        if _is_byline_with_source(line):
            cut_at = i
            break
        if _is_byline_with_suffix(line) and (n - i - 1) >= 1:
            cut_at = i
            break
        # View-count match only counts when neighbouring lines also look
        # short/title-like — keeps us from cutting a real paragraph that
        # happens to end with a number.
        if _ends_with_view_count(line):
            near_tail_short = sum(
                1 for j in range(max(0, i - 1), min(n, i + 3))
                if len(lines[j]) < 200
            )
            if near_tail_short >= 2:
                cut_at = i
                break

    if cut_at < 0:
        return content

    truncated = "\n".join(lines[:cut_at]).rstrip()
    logger.debug(
        f"strip_related_articles: cut {n - cut_at} trailing lines at "
        f"line {cut_at}: '{lines[cut_at].strip()[:60]}'"
    )
    return truncated


REQUIRED_FIELDS = ("title", "content", "date_published", "canonical_url")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_article(
    article: Dict,
    min_content_length: int = 100,
) -> Tuple[bool, Optional[str]]:
    for f in REQUIRED_FIELDS:
        v = article.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False, f"missing_field:{f}"

    date = article["date_published"]
    if not _DATE_RE.match(date):
        return False, f"bad_date_format:{date}"
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return False, f"unparseable_date:{date}"

    if len(article["content"]) < min_content_length:
        return False, f"content_too_short:{len(article['content'])}"

    return True, None


_DEPRECATED_KEYS = ("url_id", "url")


def clean_article(
    article: Dict,
    min_content_length: int = 100,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Clean one article. Returns ``(cleaned_dict, None)`` or ``(None, reason)``."""
    cleaned = dict(article)
    for field in ("title", "description", "content"):
        if cleaned.get(field):
            cleaned[field] = normalize_unicode(cleaned[field])

    if cleaned.get("content"):
        cleaned["content"] = strip_related_articles(cleaned["content"])

    for key in _DEPRECATED_KEYS:
        cleaned.pop(key, None)

    ok, reason = validate_article(cleaned, min_content_length=min_content_length)
    if not ok:
        return None, reason
    return cleaned, None


def clean_articles(
    articles: List[Dict],
    min_content_length: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """Clean a batch. Returns ``(kept, dropped)`` where each dropped entry is
    ``{"canonical_url", "reason", "title"}``."""
    kept: List[Dict] = []
    dropped: List[Dict] = []
    for art in articles:
        cleaned, reason = clean_article(art, min_content_length=min_content_length)
        if cleaned is None:
            dropped.append(
                {
                    "canonical_url": art.get("canonical_url"),
                    "title": art.get("title", "")[:120],
                    "reason": reason,
                }
            )
        else:
            kept.append(cleaned)
    logger.info(f"clean_articles: {len(kept)} kept, {len(dropped)} dropped")
    return kept, dropped
