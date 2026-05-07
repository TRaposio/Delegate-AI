"""
Inspect data/chunks.jsonl after running the parser.

Prints summary stats, flags oversized chunks, lists articles covered, samples
a few full chunks for visual inspection, and reports cross-reference stats.

Run: python -m scripts.inspect_chunks
   or: python scripts/inspect_chunks.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"
WARN_CHARS = 5000
SAMPLE_SEED = 42  # deterministic so re-runs show the same sample
SAMPLE_K = 5


def _load() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"not found: {CHUNKS_PATH}. Run the parser first.")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    # Nearest-rank percentile; fine for sanity checks.
    idx = max(0, min(len(sorted_values) - 1, int(round(pct * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _print_summary(chunks: list[dict]) -> None:
    sizes = sorted(c["char_count"] for c in chunks)
    total = sum(sizes)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"chunks: {len(chunks)}")
    print(f"total chars: {total}")
    if sizes:
        print(
            f"char_count -- min={sizes[0]} "
            f"p50={_percentile(sizes, 0.50)} "
            f"p90={_percentile(sizes, 0.90)} "
            f"p95={_percentile(sizes, 0.95)} "
            f"max={sizes[-1]}"
        )
    print(f"source_version: {chunks[0]['source_version'] if chunks else 'n/a'}")


def _print_articles(chunks: list[dict]) -> None:
    counts = Counter(c["article"] for c in chunks)
    print()
    print("=" * 70)
    print("ARTICLES")
    print("=" * 70)
    # Sort numeric articles numerically, letter articles alphabetically, numeric first.
    numeric = sorted([a for a in counts if a.isdigit()], key=int)
    letters = sorted([a for a in counts if not a.isdigit()])
    for a in numeric + letters:
        print(f"  Article {a:>3}: {counts[a]:>3} chunk(s)")


def _print_oversized(chunks: list[dict]) -> None:
    big = [c for c in chunks if c["char_count"] > WARN_CHARS]
    print()
    print("=" * 70)
    print(f"OVERSIZED CHUNKS (> {WARN_CHARS} chars)")
    print("=" * 70)
    if not big:
        print("  none")
        return
    for c in big:
        print()
        print(f"--- {c['regulation_id']}  ({c['char_count']} chars) ---")
        print(c["text"])


def _print_sample(chunks: list[dict]) -> None:
    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(chunks, min(SAMPLE_K, len(chunks)))
    print()
    print("=" * 70)
    print(f"RANDOM SAMPLE (seed={SAMPLE_SEED}, k={SAMPLE_K})")
    print("=" * 70)
    for c in sample:
        print()
        print(f"--- {c['regulation_id']}  ({c['char_count']} chars) ---")
        print(c["text"])


def _print_xrefs(chunks: list[dict]) -> None:
    chunks_with_xrefs = [c for c in chunks if c["cross_references"]]
    target_counter: Counter[str] = Counter()
    for c in chunks:
        for x in c["cross_references"]:
            target_counter[f"{x['type']}:{x['id']}"] += 1

    print()
    print("=" * 70)
    print("CROSS-REFERENCES")
    print("=" * 70)
    print(f"chunks with at least one xref: {len(chunks_with_xrefs)} / {len(chunks)}")
    print(f"unique xref targets: {len(target_counter)}")
    print(f"total xref edges: {sum(target_counter.values())}")
    print()
    print("top 10 most-referenced targets:")
    for target, count in target_counter.most_common(10):
        print(f"  {count:>3}  {target}")


def main() -> None:
    chunks = _load()
    _print_summary(chunks)
    _print_articles(chunks)
    _print_oversized(chunks)
    _print_sample(chunks)
    _print_xrefs(chunks)


if __name__ == "__main__":
    main()
