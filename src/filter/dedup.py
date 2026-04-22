from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def dedup_exact(articles: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Drop repeat ``canonical_url`` entries, keeping the first occurrence."""
    seen = set()
    kept: List[Dict] = []
    dropped: List[Dict] = []
    for art in articles:
        canon = art.get("canonical_url")
        if not canon:
            dropped.append(
                {
                    "canonical_url": None,
                    "title": art.get("title", "")[:120],
                    "reason": "missing_canonical_url",
                }
            )
            continue
        if canon in seen:
            dropped.append(
                {
                    "canonical_url": canon,
                    "title": art.get("title", "")[:120],
                    "reason": "exact_duplicate_url",
                }
            )
        else:
            seen.add(canon)
            kept.append(art)
    logger.info(f"dedup_exact: {len(kept)} kept, {len(dropped)} dropped")
    return kept, dropped


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def dedup_fuzzy(
    articles: List[Dict],
    threshold: int = 85,
) -> Tuple[List[Dict], List[Dict]]:
    """Cluster near-duplicate titles via rapidfuzz ``token_set_ratio`` +
    union-find, then keep one representative per cluster (the longest
    ``content``). Returns ``(kept, dropped)`` with dedup audit info."""
    n = len(articles)
    if n == 0:
        return [], []

    uf = _UnionFind(n)
    titles = [a.get("title", "") for a in articles]

    # O(n²) over titles — acceptable for a weekly corpus (~100 articles).
    for i in range(n):
        for j in range(i + 1, n):
            if fuzz.token_set_ratio(titles[i], titles[j]) >= threshold:
                uf.union(i, j)

    # Collect clusters in the order of their first member (stable output).
    clusters: Dict[int, List[int]] = {}
    first_seen_order: List[int] = []
    for i in range(n):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
            first_seen_order.append(root)
        clusters[root].append(i)

    kept: List[Dict] = []
    dropped: List[Dict] = []

    for root in first_seen_order:
        members = clusters[root]
        rep_idx = max(members, key=lambda k: len(articles[k].get("content", "")))
        rep = articles[rep_idx]
        rep_canonical = rep.get("canonical_url")

        kept.append(dict(rep))

        for k in members:
            if k == rep_idx:
                continue
            member = articles[k]
            sim = fuzz.token_set_ratio(member.get("title", ""), rep.get("title", ""))
            dropped.append(
                {
                    "canonical_url": member.get("canonical_url"),
                    "title": member.get("title", "")[:120],
                    "reason": "fuzzy_duplicate",
                    "similarity": int(sim),
                    "cluster_representative_canonical_url": rep_canonical,
                    "cluster_representative_title": rep.get("title", "")[:120],
                }
            )

    logger.info(
        f"dedup_fuzzy (threshold={threshold}): {len(kept)} clusters kept, "
        f"{len(dropped)} near-duplicates dropped"
    )
    return kept, dropped
