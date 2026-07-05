"""Byte-size pagination for large tool payloads. See docs/DESIGN.md "Pagination and size limits":
tool responses are capped and never silently drop data — a payload too big for one response comes
back as a bounded, labeled page (`truncated: true`) plus a way to get the rest (continuation
cursor, or a resource_uri for the full content).
"""

from __future__ import annotations

import json
from typing import Any


def canonical_size(obj: Any) -> int:
    # Compact separators so this matches the +1-per-comma accounting in paginate_items below —
    # json.dumps' default separators include a space after each comma, which otherwise makes
    # paginate_items silently underestimate page size and exceed max_bytes.
    return len(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode())


def paginate_items(items: list[Any], start: int, max_bytes: int) -> tuple[list[Any], int | None]:
    """Return as many whole items starting at `start` as fit within max_bytes, plus the index to
    resume from (None if every remaining item fit). Always includes at least one item so a single
    oversized item still makes progress rather than looping forever."""
    page: list[Any] = []
    used = 2  # enclosing []
    for i in range(start, len(items)):
        item_size = canonical_size(items[i]) + 1  # +1 for the separating comma
        if page and used + item_size > max_bytes:
            return page, i
        page.append(items[i])
        used += item_size
    return page, None
