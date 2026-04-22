"""Phase 4 — weekly report generator. CLI lives in ``run_reporter`` and is
not re-exported here (avoids the ``python -m pkg.submod`` RuntimeWarning
that Python emits when the submodule is also re-exported from the package).
"""

from .formatter import render_report
from .highlighter import HighlightedArticle, rank_articles
from .summarizer import generate_executive_summary

__all__ = [
    "HighlightedArticle",
    "rank_articles",
    "generate_executive_summary",
    "render_report",
]
