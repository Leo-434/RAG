"""Reciprocal Rank Fusion (RRF) for merging ranked lists."""

from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[Any]],
    key_fn,
    k: int = 60,
) -> list[tuple[Any, float]]:
    """
    Merge multiple ranked lists using RRF.

    Args:
        ranked_lists: List of ranked result lists (each already ordered best-first)
        key_fn: Function to extract a unique key from each item
        k: RRF constant (default 60)

    Returns:
        List of (item, rrf_score) tuples sorted by score descending.
        Only the first occurrence of each key is retained as the item.
    """
    scores: dict[str, float] = {}
    items: dict[str, Any] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = key_fn(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in items:
                items[key] = item

    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    return [(items[k], scores[k]) for k in sorted_keys]
