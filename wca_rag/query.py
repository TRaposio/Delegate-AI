"""Query entry point: ask one question, see top-k chunks.

Run: `python -m wca_rag.query "what happens when a competitor's puzzle pops?"`

Prints the top-k retrieved chunks with their similarity scores. Useful for
sanity-checking retrieval quality before the generator stage exists, and
later as a debugging tool when answers look wrong.

This script intentionally does not call an LLM. It only does retrieval.
The generator stage gets its own entry point (ask.py).

Hit rendering is delegated to wca_rag._format so this CLI and ask.py
--show-hits stay in sync.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wca_rag._format import format_hit
from wca_rag.retriever import DEFAULT_DATA_DIR, RETRIEVAL_DEFAULT_K, Retriever


# ANSI dim/reset only for the small "Query:" header; the per-hit
# rendering owns its own color via _format.format_hit.
_DIM = "\033[2m"
_RESET = "\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", type=str, help="Question to retrieve chunks for")
    parser.add_argument(
        "-k", "--k", type=int, default=RETRIEVAL_DEFAULT_K,
        help=f"Number of chunks to retrieve (default: {RETRIEVAL_DEFAULT_K})",
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

    print(f"\n{_DIM}Query:{_RESET} {args.query}")
    print(f"{_DIM}Top {len(hits)} of {len(retriever.chunk_ids)} chunks:{_RESET}\n")
    for hit in hits:
        print(format_hit(hit, snippet_chars=args.snippet_chars, color=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
