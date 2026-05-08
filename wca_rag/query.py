"""Query entry point: ask one question, see top-k chunks.

Run: `python -m wca_rag.query "what happens when a competitor's puzzle pops?"`

Prints the top-k retrieved chunks with their similarity scores. Useful for
sanity-checking retrieval quality before the generator stage exists, and
later as a debugging tool when answers look wrong.

This script intentionally does not call an LLM. It only does retrieval.
The generator stage gets its own entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wca_rag.retriever import DEFAULT_DATA_DIR, DEFAULT_K, Retriever


# ANSI for terminal output. Terminals that don't support it just print the codes;
# pipe through `cat` if that bothers you.
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_hit(hit, snippet_chars: int = 200) -> str:
    """One hit, human-readable."""
    chunk = hit.chunk
    article = chunk.get("article", "?")
    article_title = chunk.get("article_title", "")
    full_path = chunk.get("full_path_id", hit.regulation_id)

    # Snippet from the displayable text (not text_for_embedding — the prepended
    # header would clutter the output).
    body = chunk.get("text", "").strip()
    snippet = body[:snippet_chars]
    if len(body) > snippet_chars:
        snippet += "…"

    header = f"{BOLD}#{hit.rank}  [{hit.regulation_id}]{RESET}  score={hit.score:+.4f}  {DIM}({full_path} — Article {article}: {article_title}){RESET}"
    return f"{header}\n{snippet}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", type=str, help="Question to retrieve chunks for")
    parser.add_argument(
        "-k", "--k", type=int, default=DEFAULT_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Index directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--snippet-chars", type=int, default=200,
        help="Characters of body text to show per hit (default: 200)",
    )
    args = parser.parse_args(argv)

    retriever = Retriever.from_disk(data_dir=args.data_dir)
    hits = retriever.retrieve(args.query, k=args.k)

    print(f"\n{DIM}Query:{RESET} {args.query}")
    print(f"{DIM}Top {len(hits)} of {len(retriever.chunk_ids)} chunks:{RESET}\n")
    for hit in hits:
        print(format_hit(hit, snippet_chars=args.snippet_chars))

    return 0


if __name__ == "__main__":
    sys.exit(main())
