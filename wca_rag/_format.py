"""
Shared formatting helpers for CLI surfaces.

Prefix-underscored to signal "internal to the CLIs" — not part of the
library's public surface. Lives here so query.py (retrieval-only debug)
and ask.py (--show-hits) render retrieved chunks identically. Before
this, each CLI had its own hand-rolled formatter and they drifted in
small ways (color codes, snippet length, which chunk fields were
shown).

The renderer reads `regulation_id` and `article` directly off the hit
(they are explicit attributes — see retriever.py) and `text` /
`full_path_id` / `article_title` off the chunk dict. The latter two are
optional: cached/reconstructed hits (e.g. eval.py) only carry `text`,
and the formatter degrades gracefully.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wca_rag.retriever import RetrievalHit


# ANSI escape codes. Terminals that don't support them just print the
# raw codes; pipe through `cat` if that bothers you. With color=False
# these are never emitted.
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def format_hit(hit: "RetrievalHit", *, snippet_chars: int = 200, color: bool = False) -> str:
    """One retrieved hit, human-readable.

    Args:
        hit: the RetrievalHit to format.
        snippet_chars: how many characters of the chunk body to show.
            The body is truncated with an ellipsis if longer.
        color: if True, emit ANSI bold/dim codes on the header. Off by
            default — safe for piping and for log files.
    """
    chunk = hit.chunk
    article_title = chunk.get("article_title", "")
    full_path = chunk.get("full_path_id", hit.regulation_id)

    # Snippet from the displayable text (not text_for_embedding — the prepended
    # header would clutter the output).
    body = chunk.get("text", "").strip()
    snippet = body[:snippet_chars]
    if len(body) > snippet_chars:
        snippet += "…"

    bold = _BOLD if color else ""
    dim = _DIM if color else ""
    reset = _RESET if color else ""

    # Suffix is the contextual breadcrumb: full hierarchical id and the
    # article title. Hidden when neither is available (cached hits).
    suffix_parts = []
    if full_path and full_path != hit.regulation_id:
        suffix_parts.append(full_path)
    if article_title:
        suffix_parts.append(f"Article {hit.article}: {article_title}")
    suffix = f"  {dim}({' — '.join(suffix_parts)}){reset}" if suffix_parts else ""

    header = f"{bold}#{hit.rank}  [{hit.regulation_id}]{reset}  score={hit.score:+.4f}{suffix}"
    return f"{header}\n{snippet}\n"
