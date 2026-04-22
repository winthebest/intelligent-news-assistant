"""Project-wide configuration package.

Re-exports the common symbols so callers can do `from config import settings`.
"""
from .config import settings
from .constants import VIETNAMESE_ABBREVIATIONS

__all__ = ["settings", "VIETNAMESE_ABBREVIATIONS"]
