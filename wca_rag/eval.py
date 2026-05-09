"""
Eval harness: run the golden set through the pipeline, score the results.

Read this file's overview before reading the code. The design here is
deliberately three-phase + cache-on-disk so that scoring can be re-run
without re-paying for retrieval or generation. That decoupling is the
whole point.

============================================================================
OVERVIEW
============================================================================

WHY THREE PHASES?
-----------------
Retrieval is fast, deterministic, and free (local embedder).
Generation is slow, rate-limited, and quota-eating (Gemini free tier:
~10 RPM, ~250 RPD, ~250k TPM as of 2026-04, post-Dec-2025 reduction).
Scoring is the part that gets iterated on while developing the harness.

If we did all three in one pass, every scorer change would require
re-running the (rate-limited, quota-eating) generation step. By
caching phase 1 and phase 2 outputs to disk, phase 3 becomes a pure
function over those artifacts and re-runs in milliseconds.

PHASES
------
Phase 1 — RETRIEVE (local, fast, deterministic):
    For each question in the golden set, run the retriever with k=DEFAULT_K.
    Persist the retrieved chunk ids + scores per question.
    No LLM calls. No quota consumed.
    Output: hits.json

Phase 2 — GENERATE (rate-limited, quota-eating, non-deterministic):
    For each question, load its cached hits, assemble the user prompt,
    call the generator. Sleep between calls to respect RPM. Fail loudly
    on 429 — do not silently retry.
    Output: answers.json

Phase 3 — SCORE (local, fast, pure):
    For each question, compute:
      - retrieval recall@k (sub-rule expected_ids → parent chunk membership)
      - citation set extracted from answer text (regex over [xxx])
      - citation accuracy (set intersection with expected_ids, claim-level)
      - confabulation count (cited ids whose text is NOT in any retrieved chunk)
      - quote validity fraction (verbatim quotes substring-matched against retrieved)
      - mode comparison (declared vs expected)
    Aggregate: means + mode confusion matrix.
    Output: metrics.json

CITATION GRANULARITY: OPTION A
------------------------------
The model is prompted to cite at the most specific id supported by
verbatim text in retrieved chunks (sub-regulation ids permitted, e.g.
[A6c] inside chunk A6's body). Consequences for this harness:

  - Citation accuracy: direct claim-level comparison. No sub-rule
    collapsing. Expected_ids in the golden set are written at the
    granularity the question demands.
  - Recall@k: still chunk-level (the retriever returns chunks, not
    sub-rules). Sub-rule expected_ids must be mapped back to their
    parent chunk before checking membership in retrieved hits. This
    is the ONE place sub-rule collapsing still happens.
  - Confabulation check: scan retrieved chunk bodies (text field) for
    each cited id. If id not present anywhere, the model invented it —
    this is a hard failure regardless of mode or accuracy.
  - Quote validity: extract "..." spans from the answer, normalize
    whitespace, substring-match against retrieved chunks. The verbatim-
    quote requirement is the load-bearing constraint under Option A;
    measuring it programmatically replaces eyeball verification.

RUN ARTIFACT LAYOUT
-------------------
One directory per run. Three (small) JSON files plus a config blob.

    evals/results/run-<TIMESTAMP>/
        config.json    # what was run: golden_set hash, k, model, prompt hash
        hits.json      # phase 1 output
        answers.json   # phase 2 output
        metrics.json   # phase 3 output (rerun --score-only updates this)

CONFIG HASHES
-------------
config.json records:
  - golden_set_hash: sha256 of the loaded YAML (sorted by id)
  - prompt_hash: sha256 of SYSTEM_PROMPT
  - model_name, k, rpm, timestamp

Two runs with matching golden_set_hash + prompt_hash + model are
comparable. Mismatches mean you're comparing tuning experiments, not
runs of the same system.

CLI
---
    python -m wca_rag.eval                       # all three phases, new run dir
    python -m wca_rag.eval --questions q01,q02   # subset
    python -m wca_rag.eval --rpm 10              # rate limit override (default 10)
    python -m wca_rag.eval --summary             # print aggregate to stdout
    python -m wca_rag.eval --score-only RUN_ID   # re-score existing run

GOLDEN SET FORMAT
-----------------
evals/golden_set.yaml — list of entries:

    - id: q01
      question: "..."
      expected_mode: ANSWER | PARTIAL | REFUSE
      expected_ids: [A6c]   # claim-level granularity (Option A)
      notes: "..."          # human-readable rationale

============================================================================
NOT IN V1 (deferred, on purpose)
============================================================================

  - LLM-as-judge scoring (faithfulness, answer relevance). Defer until
    baseline metrics are stable. ARCHITECTURE.md §5.
  - Token-bucket rate limiting with retry. Sleep-between-calls is fine
    for 30-question runs. Revisit at 100+ questions.
  - Per-question retry on transient errors. Fail loudly, re-run. Silent
    retries hide systematic problems.
  - Comparison/diff tooling between two run directories. Useful later;
    eyeball + jq is fine for now.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from wca_rag.generator import GeminiGenerator
from wca_rag.pipeline import DEFAULT_K
from wca_rag.prompts import SYSTEM_PROMPT, assemble_user_prompt
from wca_rag.retriever import RetrievalHit, Retriever


# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------

GOLDEN_SET_PATH = Path("evals/golden_set.yaml")
RESULTS_DIR = Path("evals/results")

# Default rate limit. Free-tier Gemini 2.5 Flash is ~10 RPM after the
# Dec 2025 reduction. Override with --rpm if Google's limits change.
DEFAULT_RPM = 10

# Citation regex. Matches WCA id shapes: digit-then-letters (11e, 11e++,
# 11e2a) and uppercase-letter-then-rest (A6c, E2c, H2). Excludes mode
# labels [ANSWER], [PARTIAL], [REFUSE] which contain only uppercase
# letters and would otherwise match.
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9]+\+*[a-z0-9+]*)\]")
MODE_LABELS = {"ANSWER", "PARTIAL", "REFUSE"}

# Quote regex. Matches "..." spans. Greedy-by-design — long quotes are
# fine, the substring check normalizes whitespace before matching.
QUOTE_PATTERN = re.compile(r'"([^"]+)"')

VALID_MODES = {"ANSWER", "PARTIAL", "REFUSE"}


# ----------------------------------------------------------------------------
# Data shapes
# ----------------------------------------------------------------------------


@dataclass
class GoldenQuestion:
    """One entry from golden_set.yaml."""

    id: str
    question: str
    expected_mode: str
    expected_ids: list[str]
    notes: str = ""

    def __post_init__(self) -> None:
        if self.expected_mode not in VALID_MODES:
            raise ValueError(
                f"{self.id}: expected_mode={self.expected_mode!r} not in {VALID_MODES}"
            )


@dataclass
class HitRecord:
    """One retrieved chunk for one question. Mirrors RetrievalHit but
    serializable and stripped to the fields the scorer needs."""

    rank: int
    score: float
    regulation_id: str
    text: str  # full chunk body — the scorer needs it for quote validation


@dataclass
class AnswerRecord:
    """One generation result for one question."""

    answer_text: str
    declared_mode: str | None  # None if model didn't emit a recognizable [MODE] label
    model: str
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class QuestionScore:
    """Per-question metrics. All fields populated by phase 3."""

    question_id: str
    declared_mode: str | None
    expected_mode: str
    mode_match: bool

    cited_ids: list[str]
    expected_ids: list[str]
    citation_accuracy: float            # |cited ∩ expected| / |expected|, 0 if expected empty
    citation_precision: float           # |cited ∩ expected| / |cited|, 1.0 if cited empty
    confabulated_ids: list[str]         # cited but NOT in any retrieved chunk's body

    retrieved_ids: list[str]            # chunk-level ids, parallel to phase-1 hits
    recall_at_k: float                  # of expected_ids (parent-chunk-collapsed), how many retrieved

    quotes_total: int
    quotes_valid: int
    quote_validity: float               # quotes_valid / quotes_total, 1.0 if no quotes


# ----------------------------------------------------------------------------
# Phase 0: load
# ----------------------------------------------------------------------------


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    """Load and validate golden set. Sorted by id for deterministic order."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it before running eval."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top level must be a list, got {type(raw).__name__}")

    questions = [GoldenQuestion(**entry) for entry in raw]

    # Duplicate-id check — easy to introduce, hard to debug.
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"duplicate question id: {q.id}")
        seen.add(q.id)

    questions.sort(key=lambda q: q.id)
    return questions


def hash_text(text: str) -> str:
    """sha256 hex digest, truncated to 16 chars for readability."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_golden_set(questions: list[GoldenQuestion]) -> str:
    """Hash of golden set content. Stable under sort-by-id."""
    payload = json.dumps(
        [asdict(q) for q in questions], sort_keys=True, ensure_ascii=False
    )
    return hash_text(payload)


# ----------------------------------------------------------------------------
# Phase 1: retrieve
# ----------------------------------------------------------------------------


def phase_retrieve(
    questions: list[GoldenQuestion],
    retriever: Retriever,
    k: int,
) -> dict[str, list[HitRecord]]:
    """Run retrieval for every question. Local, fast, deterministic.

    Returns a dict keyed by question id so phase 2 can look up by id
    without relying on list ordering.
    """
    hits_by_question: dict[str, list[HitRecord]] = {}
    for q in questions:
        raw_hits = retriever.retrieve(q.question, k=k)
        hits_by_question[q.id] = [
            HitRecord(
                rank=h.rank,
                score=h.score,
                regulation_id=h.regulation_id,
                text=h.chunk["text"],
            )
            for h in raw_hits
        ]
    return hits_by_question


# ----------------------------------------------------------------------------
# Phase 2: generate
# ----------------------------------------------------------------------------


def phase_generate(
    questions: list[GoldenQuestion],
    hits_by_question: dict[str, list[HitRecord]],
    generator: GeminiGenerator,
    rpm: int,
) -> dict[str, AnswerRecord]:
    """Run generation for every question. Rate-limited.

    Pacing strategy: simple sleep-between-calls. At rpm=10, that's 6s
    between requests. Dumb, predictable, fine for 30 questions.

    Failure strategy: any exception propagates. Re-running with
    --score-only would re-score whatever phase 2 wrote before the
    failure (currently we write at the end, so partial failures lose
    the run — see TODO).
    """
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    delay = 60.0 / rpm

    answers: dict[str, AnswerRecord] = {}
    last_call_at: float | None = None

    for q in questions:
        if last_call_at is not None:
            elapsed = time.monotonic() - last_call_at
            if elapsed < delay:
                time.sleep(delay - elapsed)

        # Rebuild the user prompt from cached hits. We need the original
        # RetrievalHit shape for assemble_user_prompt; fake the chunk
        # dict with just the fields prompts.py reads.
        cached_hits = hits_by_question[q.id]
        synthetic_hits = [
            RetrievalHit(
                rank=h.rank,
                score=h.score,
                regulation_id=h.regulation_id,
                chunk={
                    "regulation_id": h.regulation_id,
                    "article": _article_from_id(h.regulation_id),
                    "text": h.text,
                },
            )
            for h in cached_hits
        ]
        user_prompt = assemble_user_prompt(q.question, synthetic_hits)

        last_call_at = time.monotonic()
        result = generator.generate(SYSTEM_PROMPT, user_prompt)

        answers[q.id] = AnswerRecord(
            answer_text=result.text,
            declared_mode=extract_mode(result.text),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    return answers


def _article_from_id(regulation_id: str) -> str:
    """Best-effort article extraction from a regulation id.

    Used only to populate the `article` attribute on the synthetic
    chunk dict for prompt assembly. The real chunks have this from the
    parser; we approximate here to avoid round-tripping through
    chunks.jsonl during eval.
    """
    # Top-level digit articles (11e → 11), top-level appendix letters (A6c → A).
    m = re.match(r"([A-Z]|\d+)", regulation_id)
    return m.group(1) if m else ""


# ----------------------------------------------------------------------------
# Phase 3: score
# ----------------------------------------------------------------------------


def extract_mode(answer_text: str) -> str | None:
    """Pull the leading [MODE] label out of an answer. None if absent.

    The system prompt requires the answer to start with [ANSWER],
    [PARTIAL], or [REFUSE]. We match strictly on that contract — if the
    model drifted (e.g. emitted "ANSWER:" or put the label mid-text)
    the harness flags it as an unrecognized mode rather than guessing.
    """
    m = re.match(r"\s*\[(ANSWER|PARTIAL|REFUSE)\]", answer_text)
    return m.group(1) if m else None


def extract_citations(answer_text: str) -> list[str]:
    """Extract regulation ids cited in [xxx] form.

    Filters out mode labels. Preserves order and duplicates — the
    scorer dedupes when it needs to.
    """
    found = CITATION_PATTERN.findall(answer_text)
    return [cid for cid in found if cid not in MODE_LABELS]


def extract_quotes(answer_text: str) -> list[str]:
    """Extract verbatim quote spans (text inside double quotes)."""
    return QUOTE_PATTERN.findall(answer_text)


def normalize_for_quote_match(text: str) -> str:
    """Whitespace normalization for substring-matching quotes against chunks.

    Collapses any run of whitespace (including newlines) to a single
    space. The verbatim-quote contract is about words, not formatting;
    a quote that differs from the source only in line breaks should
    still count as valid.
    """
    return re.sub(r"\s+", " ", text).strip()


def parent_chunk_id(sub_rule_id: str, retrieved_ids: list[str]) -> str | None:
    """Map a (possibly sub-rule) id to its parent chunk id, if retrieved.

    Strategy: the retrieved chunk ids are at top level (e.g. `11e`,
    `A6e`). A sub-rule id like `11e2a` or `A6c` has its parent chunk
    id as a prefix. We pick the longest retrieved id that is a prefix
    of the sub-rule id.

    Returns None if no retrieved chunk could be the parent — caller
    treats that as a recall miss.

    Example:
      sub_rule_id = "A6c", retrieved_ids = ["A6", "A6e", "9f"]
      → "A6" (prefix match, longest among matching candidates)
    """
    candidates = [rid for rid in retrieved_ids if sub_rule_id.startswith(rid)]
    if not candidates:
        return None
    return max(candidates, key=len)


def score_question(
    q: GoldenQuestion,
    hits: list[HitRecord],
    answer: AnswerRecord,
) -> QuestionScore:
    """Score one question. Pure function over the cached artifacts."""

    retrieved_ids = [h.regulation_id for h in hits]
    chunks_concatenated = " ".join(normalize_for_quote_match(h.text) for h in hits)

    # --- citations ---
    cited_ids_raw = extract_citations(answer.answer_text)
    cited_ids = list(dict.fromkeys(cited_ids_raw))  # dedupe, preserve order

    expected_set = set(q.expected_ids)
    cited_set = set(cited_ids)
    if expected_set:
        citation_accuracy = len(cited_set & expected_set) / len(expected_set)
    else:
        # REFUSE cases have no expected_ids. Define accuracy as 1.0 if
        # the model also cited nothing, 0.0 if it cited anyway. This is
        # a defensible convention; document it.
        citation_accuracy = 1.0 if not cited_set else 0.0

    # Citation precision: of the ids the model cited, how many were
    # expected? Distinct from accuracy/recall — a model that cites 7
    # correct ids when 3 were expected scores 1.0 on accuracy
    # (over-citation isn't penalized) but lower on precision. For
    # PARTIAL/ANSWER cases where the expected set is "representative"
    # rather than exhaustive, precision will read low and that's fine
    # — it's diagnostic, not a quality signal on its own.
    if cited_set:
        citation_precision = len(cited_set & expected_set) / len(cited_set)
    else:
        # No citations at all. For REFUSE this is correct (1.0); for
        # ANSWER/PARTIAL it's a separate failure caught by mode/recall.
        citation_precision = 1.0 if not expected_set else 0.0

    # --- confabulation: cited id whose text is not in ANY retrieved chunk ---
    # We check whether the cited id appears as a discrete token in any
    # retrieved chunk body. The chunks format ids as "3l+) [ANNOTATION]..."
    # — id followed by `)`. We anchor on that rather than \b word
    # boundaries because \b breaks on `+`-suffixed ids: \b is a word/
    # non-word transition, and `+` is non-word, so \b after `+` doesn't
    # match where you'd intuitively expect.
    #
    # The id is bordered by:
    #   - leading: start-of-string OR non-alphanumeric char
    #     (whitespace, `(`, `[`, etc.)
    #   - trailing: `)` (the canonical chunk format), OR whitespace,
    #     OR end-of-string
    confabulated_ids = []
    for cid in cited_ids:
        # (?:^|[^A-Za-z0-9+])  : start, or a char that can't extend the id
        # {escaped id}
        # (?:\)|\s|$)          : chunk-format `)`, whitespace, or end
        pattern = re.compile(
            rf"(?:^|[^A-Za-z0-9+]){re.escape(cid)}(?:\)|\s|$)"
        )
        found = any(pattern.search(h.text) for h in hits)
        if not found:
            confabulated_ids.append(cid)

    # --- recall@k: parent-chunk-collapse expected ids, check retrieval ---
    recall_hits = 0
    for expected_id in q.expected_ids:
        if expected_id in retrieved_ids:
            recall_hits += 1
            continue
        parent = parent_chunk_id(expected_id, retrieved_ids)
        if parent is not None:
            recall_hits += 1
    recall_at_k = recall_hits / len(q.expected_ids) if q.expected_ids else 1.0

    # --- quote validity ---
    quotes = extract_quotes(answer.answer_text)
    quotes_valid = 0
    for quote in quotes:
        normalized_quote = normalize_for_quote_match(quote)
        if normalized_quote and normalized_quote in chunks_concatenated:
            quotes_valid += 1
    quote_validity = quotes_valid / len(quotes) if quotes else 1.0

    return QuestionScore(
        question_id=q.id,
        declared_mode=answer.declared_mode,
        expected_mode=q.expected_mode,
        mode_match=(answer.declared_mode == q.expected_mode),
        cited_ids=cited_ids,
        expected_ids=q.expected_ids,
        citation_accuracy=citation_accuracy,
        citation_precision=citation_precision,
        confabulated_ids=confabulated_ids,
        retrieved_ids=retrieved_ids,
        recall_at_k=recall_at_k,
        quotes_total=len(quotes),
        quotes_valid=quotes_valid,
        quote_validity=quote_validity,
    )


def aggregate(scores: list[QuestionScore]) -> dict[str, Any]:
    """Aggregate per-question scores into the run-level metrics block."""
    n = len(scores)
    if n == 0:
        return {"n": 0}

    confusion: dict[str, dict[str, int]] = {}
    for s in scores:
        row = confusion.setdefault(s.expected_mode, {})
        col = s.declared_mode if s.declared_mode is not None else "MISSING"
        row[col] = row.get(col, 0) + 1

    return {
        "n": n,
        "recall_at_k_mean": sum(s.recall_at_k for s in scores) / n,
        "citation_accuracy_mean": sum(s.citation_accuracy for s in scores) / n,
        "citation_precision_mean": sum(s.citation_precision for s in scores) / n,
        "quote_validity_mean": sum(s.quote_validity for s in scores) / n,
        "mode_accuracy": sum(1 for s in scores if s.mode_match) / n,
        "mode_confusion": confusion,
        "total_confabulated_ids": sum(len(s.confabulated_ids) for s in scores),
        "questions_with_confabulation": sum(
            1 for s in scores if s.confabulated_ids
        ),
    }


def phase_score(
    questions: list[GoldenQuestion],
    hits_by_question: dict[str, list[HitRecord]],
    answers: dict[str, AnswerRecord],
) -> dict[str, Any]:
    """Run phase 3. Pure function over cached artifacts."""
    per_question = [
        score_question(q, hits_by_question[q.id], answers[q.id]) for q in questions
    ]
    return {
        "aggregate": aggregate(per_question),
        "per_question": [asdict(s) for s in per_question],
    }


# ----------------------------------------------------------------------------
# Run-directory I/O
# ----------------------------------------------------------------------------


def make_run_dir(timestamp: datetime | None = None) -> Path:
    """Create evals/results/run-<TS>/ and return its path."""
    ts = timestamp or datetime.now(timezone.utc)
    run_id = ts.strftime("run-%Y-%m-%dT%H%M%S")
    path = RESULTS_DIR / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    # Dataclasses round-trip through asdict elsewhere; this is a backstop.
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def load_run(run_dir: Path) -> tuple[
    dict[str, list[HitRecord]],
    dict[str, AnswerRecord],
]:
    """Reload phase 1 + phase 2 artifacts from a run directory."""
    hits_raw = json.loads((run_dir / "hits.json").read_text(encoding="utf-8"))
    answers_raw = json.loads((run_dir / "answers.json").read_text(encoding="utf-8"))

    hits = {
        qid: [HitRecord(**h) for h in hit_list]
        for qid, hit_list in hits_raw.items()
    }
    answers = {qid: AnswerRecord(**a) for qid, a in answers_raw.items()}
    return hits, answers


# ----------------------------------------------------------------------------
# Summary printing
# ----------------------------------------------------------------------------


def print_summary(metrics: dict[str, Any]) -> None:
    agg = metrics["aggregate"]
    print()
    print("=" * 70)
    print(f"AGGREGATE  (n={agg['n']})")
    print("=" * 70)
    print(f"  recall@k mean         {agg['recall_at_k_mean']:.3f}")
    print(f"  citation accuracy     {agg['citation_accuracy_mean']:.3f}")
    print(f"  citation precision    {agg['citation_precision_mean']:.3f}")
    print(f"  quote validity        {agg['quote_validity_mean']:.3f}")
    print(f"  mode accuracy         {agg['mode_accuracy']:.3f}")
    print(f"  confabulated ids      {agg['total_confabulated_ids']} "
          f"(in {agg['questions_with_confabulation']} questions)")
    print()
    print("Mode confusion (rows = expected, cols = declared):")
    for expected, row in sorted(agg["mode_confusion"].items()):
        for declared, count in sorted(row.items()):
            print(f"  {expected:7s} → {declared:7s}  {count}")
    print()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WCA RAG eval harness")
    parser.add_argument(
        "--questions",
        help="comma-separated question ids (default: all)",
        default=None,
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_RPM,
        help=f"max requests per minute (default {DEFAULT_RPM})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"top-k for retrieval (default {DEFAULT_K})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print aggregate metrics to stdout after run",
    )
    parser.add_argument(
        "--score-only",
        metavar="RUN_ID",
        help="re-score an existing run directory (skip phases 1+2)",
        default=None,
    )
    args = parser.parse_args(argv)

    # Load .env before anything that reads env vars: GEMINI_API_KEY for
    # the generator, HF_TOKEN for the embedder. Soft dependency — if
    # python-dotenv isn't installed, regular env vars still work.
    # Same pattern as wca_rag/ask.py.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    questions = load_golden_set()
    if args.questions:
        wanted = set(args.questions.split(","))
        questions = [q for q in questions if q.id in wanted]
        if not questions:
            print(f"no questions matched: {args.questions}", file=sys.stderr)
            return 1

    if args.score_only:
        run_dir = RESULTS_DIR / args.score_only
        if not run_dir.exists():
            print(f"run dir not found: {run_dir}", file=sys.stderr)
            return 1
        hits, answers = load_run(run_dir)
        # Subset to the questions actually present in the run if user
        # didn't override — avoids KeyError if golden set grew since the run.
        questions = [q for q in questions if q.id in hits and q.id in answers]
        metrics = phase_score(questions, hits, answers)
        write_json(run_dir / "metrics.json", metrics)
        print(f"rescored: {run_dir}")
        if args.summary:
            print_summary(metrics)
        return 0

    # Full run.
    run_dir = make_run_dir()
    print(f"run dir: {run_dir}")

    print(f"phase 1: retrieve  ({len(questions)} questions, k={args.k})")
    retriever = Retriever.from_disk()
    hits = phase_retrieve(questions, retriever, k=args.k)
    write_json(run_dir / "hits.json", {qid: [asdict(h) for h in hl] for qid, hl in hits.items()})

    print(f"phase 2: generate  (rpm={args.rpm})")
    generator = GeminiGenerator()
    answers = phase_generate(questions, hits, generator, rpm=args.rpm)
    write_json(run_dir / "answers.json", {qid: asdict(a) for qid, a in answers.items()})

    print("phase 3: score")
    metrics = phase_score(questions, hits, answers)
    write_json(run_dir / "metrics.json", metrics)

    write_json(run_dir / "config.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_set_hash": hash_golden_set(questions),
        "prompt_hash": hash_text(SYSTEM_PROMPT),
        "model_name": generator.model_name,
        "k": args.k,
        "rpm": args.rpm,
        "question_ids": [q.id for q in questions],
    })

    print(f"done: {run_dir}")
    if args.summary:
        print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
