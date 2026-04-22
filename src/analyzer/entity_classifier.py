"""Noise filter for LLM-NER output: drop common-noun phrases.

Small open-weight models (qwen2.5:7b/14b) occasionally emit phrases that
*look* like entities but are actually plain common nouns in Vietnamese
("phi hành gia" = astronaut, "tấm chắn nhiệt" = heat shield). A curated
blocklist drops these before they reach the merger.

Scope is intentionally small:
- We do NOT try to classify entities into PERSON/ORG/PRODUCT/TECH/LOC.
  That classification is either (a) metadata-only and unused downstream,
  or (b) done trivially in the final report as a flat ranked list.
- The blocklist is topic-specific to ``khoa-hoc-cong-nghe``; extend when
  switching VnExpress sections (e.g. add "bệnh nhân", "triệu chứng" for
  ``suc-khoe``).
"""

from __future__ import annotations

import re
import unicodedata


# Phrases that NER models sometimes emit but are NOT named entities.
# Keep tight - false drops cost more than false accepts (the merger will
# down-rank noise naturally via TF-IDF cross-check anyway).
_COMMON_NOUN_BLOCKLIST = frozenset(
    {
        # Space / science generic terms
        "phi hành gia", "phi hành đoàn", "tấm chắn nhiệt", "tàu vũ trụ",
        "nhà khoa học", "nghiên cứu sinh", "nghiên cứu sinh xuất sắc",
        "tàu cao tốc",
        # Government / business generic roles
        "doanh nghiệp", "doanh nghiệp nhỏ", "chính phủ", "nhà mạng",
        "công ty công nghệ", "công nghệ thông tin", "công nghệ số",
        # Event / product-line generic
        "ngày hội trí tuệ nhân tạo",
        "phiên bản", "thiết kế", "sản phẩm", "ứng dụng",
    }
)


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _norm(s: str) -> str:
    """Lowercase + strip accents + collapse whitespace - canonical key."""
    return re.sub(r"\s+", " ", _strip_accents(s.lower())).strip()


_NORM_BLOCKLIST = {_norm(x) for x in _COMMON_NOUN_BLOCKLIST}


def is_common_noun(entity: str) -> bool:
    """Return True if ``entity`` matches the common-noun blocklist.

    Used by the NER post-processor to drop drifted model outputs before
    they make it into the final keyword list.
    """
    return _norm(entity) in _NORM_BLOCKLIST
