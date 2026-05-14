"""
CLI entry point for the full RAG pipeline.

Run with:
    python -m wca_rag.ask "your question"
    python -m wca_rag.ask "your question" -k 5
    python -m wca_rag.ask "your question" --show-hits

`query.py` remains as the retrieval-only debugging entry point (no API
key needed). This module wires retriever + generator into the full
Phase-2 pipeline and is the entry point a delegate would actually use.

API key: set GEMINI_API_KEY in .env (see .env.example) or export it.
"""

from __future__ import annotations

import argparse
import sys

from wca_rag._format import format_hit
from wca_rag.embedder import SentenceTransformerEmbedder
from wca_rag.generator import GeminiGenerator, GroqGenerator, Generator
from wca_rag.pipeline import PIPELINE_DEFAULT_K, Pipeline
from wca_rag.retriever import Retriever


GENERATORS: dict[str, type[Generator]] = {
    "gemini": GeminiGenerator,
    "groq": GroqGenerator,
}
DEFAULT_GENERATOR = "gemini"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="wca_rag.ask",
        description="Ask a WCA-regulations question. Returns a cited answer.",
    )
    parser.add_argument("question", help="The delegate's question, in quotes.")
    parser.add_argument(
        "-k",
        type=int,
        default=PIPELINE_DEFAULT_K,
        help=f"Number of chunks to retrieve (default: {PIPELINE_DEFAULT_K}).",
    )
    parser.add_argument(
        "--generator",
        choices=sorted(GENERATORS),
        default=DEFAULT_GENERATOR,
        help=(
            f"LLM provider (default {DEFAULT_GENERATOR}). "
            "Tier 3 baseline runs should stay on gemini; "
            "use groq for exploratory work (ROADMAP §A2)."
        ),
    )
    parser.add_argument(
        "--show-hits",
        action="store_true",
        help="Print the retrieved chunks above the answer (for debugging).",
    )
    args = parser.parse_args()

    # Try to load .env if python-dotenv is available. Soft dependency:
    # if not installed, environment variables still work the normal way.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    embedder = SentenceTransformerEmbedder()
    retriever = Retriever.from_disk(embedder=embedder)
    generator = GENERATORS[args.generator]()
    pipeline = Pipeline(retriever=retriever, generator=generator)

    try:
        result = pipeline.ask(args.question, k=args.k)
    except RuntimeError as e:
        # GeminiGenerator raises RuntimeError if GEMINI_API_KEY is missing.
        # Catch it for a clean CLI message.
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.show_hits:
        print("=" * 70)
        print(f"RETRIEVED CHUNKS (k={args.k})")
        print("=" * 70)
        for hit in result.hits:
            # Plain (no ANSI) — ask.py output is more often piped/saved.
            print(format_hit(hit, snippet_chars=200, color=False))

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    print()

    # Token usage line, if the provider reported it. Useful for keeping
    # an eye on free-tier consumption.
    g = result.generation
    if g.input_tokens is not None and g.output_tokens is not None:
        print(
            f"[{g.model}  in: {g.input_tokens} tok  "
            f"out: {g.output_tokens} tok]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
