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
Phase 1 - RETRIEVE (local, fast, deterministic):
    For each question in the golden set, run the retriever with k=PIPELINE_DEFAULT_K.
    Persist the retrieved chunk ids + scores per question.
    No LLM calls. No quota consumed.
    Output: hits.json

Phase 2 - GENERATE (rate-limited, quota-eating, non-deterministic):
    For each question, load its cached hits, assemble the user prompt,
    call the generator. Sleep between calls to respect RPM. Fail loudly
    on 429 - do not silently retry.
    Output: answers.json

Phase 3 - SCORE (local, fast, pure):
    For each question, compute:
      - retrieval recall@k (sub-rule expected_ids -> parent chunk membership)
      - retrieval recall@1, @3, @5 (rank-aware slicing of the same hits)
      - MRR (1/rank of first expected hit, 0 if missed)
      - retrieval score signals: top-1 score, top1-top2 margin, top-k mean
      - citation set extracted from answer text (regex over [xxx])
      - citation accuracy (set intersection with expected_ids, claim-level)
      - citation_quote_alignment ("faithfulness proxy"): each cited id is
        backed by a verbatim quote that substring-matches THAT id's chunk
        body specifically - stricter than quote_validity which checks the
        concatenation of all retrieved chunks.
      - confabulation count (cited ids whose text is NOT in any retrieved chunk)
      - quote validity fraction (verbatim quotes substring-matched against retrieved)
      - mode comparison (declared vs expected)
      - self_confidence parsed from the trailing `Confidence: 0.XX` line
    Aggregate: means + mode confusion matrix + confidence threshold summary
    + retrieval score stats.
    Output: metrics.json + plots/ directory (6 PNGs).

EVAL_K vs K
-----------
Phase 1 retrieves `eval_k` chunks (default 10, override with --eval-k) so the
recall@k *curve* can plot beyond the generator's k. Only the top `k` (default
PIPELINE_DEFAULT_K=8) hits are fed to the generator - generation cost is
unchanged. The extra retrieval cost is negligible (NumPy matmul over 108 rows).

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
    each cited id. If id not present anywhere, the model invented it -
    this is a hard failure regardless of mode or accuracy.
  - Quote validity: extract "..." spans from the answer, normalize
    whitespace, substring-match against retrieved chunks. The verbatim-
    quote requirement is the load-bearing constraint under Option A;
    measuring it programmatically replaces eyeball verification.

RUN ARTIFACT LAYOUT
-------------------
One directory per run. Three (small) JSON files plus a config blob,
plus human-readable summaries written every scoring pass.

    evals/results/run-<TIMESTAMP>/
        config.json    # what was run: golden_set hash, k, model, prompt hash
        hits.json      # phase 1 output
        answers.json   # phase 2 output
        metrics.json   # phase 3 output (rerun --score-only updates this)
        summary.txt    # aggregate metrics, human-readable
        review.md      # flat per-question Q/A for eyeballing
        plots/         # six PNGs (rerun --score-only regenerates these)

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
    python -m wca_rag.eval                       # all three phases, new run dir; writes summary.txt + plots/
    python -m wca_rag.eval --questions q01,q02   # subset
    python -m wca_rag.eval --rpm 10              # rate limit override (default 10)
    python -m wca_rag.eval --k 8                 # generator top-k (default PIPELINE_DEFAULT_K)
    python -m wca_rag.eval --eval-k 10           # retrieval depth for recall curve (default 10)
    python -m wca_rag.eval --summary             # also print aggregate to stdout
    python -m wca_rag.eval --score-only RUN_ID   # re-score existing run; rewrites metrics.json + summary.txt + review.md + plots/

GOLDEN SET FORMAT
-----------------
evals/golden_set.yaml - list of entries:

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
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # no display; we only write PNGs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from wca_rag.generator import GeminiGenerator
from wca_rag.pipeline import PIPELINE_DEFAULT_K
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

# Default retrieval depth for the recall curve. Independent from the
# generator's k (which stays at PIPELINE_DEFAULT_K). Phase 1 retrieves
# this many chunks per question; the generator sees only top-k. The
# extra retrieval cost is negligible (NumPy matmul).
DEFAULT_EVAL_K = 10

# Recall@k curve break points. Computed against the top-N slice of the
# already-retrieved hits, so adding entries is free as long as they're
# <= eval_k.
RECALL_AT_K_BREAKS = (1, 3, 5)

# Confidence-threshold reporting break points for the summary stat.
# "If confidence >= T, what's the accuracy?" - drives the review-skip
# threshold decision.
CONFIDENCE_THRESHOLDS = (0.9, 0.8, 0.7)

# Citation regex. Matches WCA id shapes: digit-then-letters (11e, 11e++,
# 11e2a) and uppercase-letter-then-rest (A6c, E2c, H2). Excludes mode
# labels [ANSWER], [PARTIAL], [REFUSE] which contain only uppercase
# letters and would otherwise match.
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9]+\+*[a-z0-9+]*)\]")

# Quote regex. Matches "..." spans. Greedy-by-design - long quotes are
# fine, the substring check normalizes whitespace before matching.
QUOTE_PATTERN = re.compile(r'"([^"]+)"')

# Self-confidence line. Anchored to end-of-text-ish position - the
# prompt requires it on its own final line. Tolerant of trailing
# whitespace but strict on the numeric format. None if unparseable.
CONFIDENCE_PATTERN = re.compile(
    r"Confidence:\s*([01](?:\.\d+)?)\s*$", re.MULTILINE
)

# Single source of truth for response modes. Used both to filter
# citation regex hits (mode labels would otherwise match) and to
# validate golden-set entries.
MODES = {"ANSWER", "PARTIAL", "REFUSE"}


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
        if self.expected_mode not in MODES:
            raise ValueError(
                f"{self.id}: expected_mode={self.expected_mode!r} not in {MODES}"
            )


@dataclass
class HitRecord:
    """One retrieved chunk for one question. Mirrors RetrievalHit but
    serializable and stripped to the fields the scorer needs."""

    rank: int
    score: float
    regulation_id: str
    article: str
    text: str  # full chunk body - the scorer needs it for quote validation


@dataclass
class AnswerRecord:
    """One generation result for one question."""

    answer_text: str
    declared_mode: str | None  # None if model didn't emit a recognizable [MODE] label
    self_confidence: float | None  # None if model didn't emit a parseable Confidence line
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

    retrieved_ids: list[str]            # chunk-level ids, parallel to phase-1 hits (rank-ordered)
    recall_at_k: float                  # of expected_ids (parent-chunk-collapsed), how many retrieved
    recall_at_n: dict[str, float]       # rank-aware: {"1": ..., "3": ..., "5": ...}; keys are str
                                        # for JSON friendliness
    rank_of_first_expected: int | None  # 1-indexed; None if no expected id (or parent) retrieved
    mrr: float                          # 1 / rank_of_first_expected; 0.0 if miss; 1.0 for REFUSE (no expected)

    top1_score: float                   # similarity of rank-1 hit
    top1_top2_margin: float             # top1 - top2; 0 if only one hit
    topk_mean_score: float              # mean over all retrieved hits

    quotes_total: int
    quotes_valid: int
    quote_validity: float               # quotes_valid / quotes_total, 1.0 if no quotes

    # Faithfulness proxy: each cited id should be backed by a quote that
    # substring-matches THAT id's chunk body specifically. Stricter than
    # quote_validity. Honest name kept in code: not real NLI faithfulness.
    citation_quote_alignment: float     # see score_question for definition

    self_confidence: float | None       # mirrored from AnswerRecord for plotting/threshold convenience

    correct: bool                       # mode_match and citation_accuracy==1.0 and no confab; drives threshold summary


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

    # Duplicate-id check - easy to introduce, hard to debug.
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
    eval_k: int,
) -> dict[str, list[HitRecord]]:
    """Run retrieval for every question. Local, fast, deterministic.

    Retrieves `eval_k` chunks per question (default 10), wider than the
    generator's k. The extra chunks are used by phase 3 to plot the
    recall@N curve beyond the generator's slice. Phase 2 will subset
    to the first `k` hits before calling the generator.

    Returns a dict keyed by question id so phase 2 can look up by id
    without relying on list ordering.
    """
    hits_by_question: dict[str, list[HitRecord]] = {}
    for q in questions:
        raw_hits = retriever.retrieve(q.question, k=eval_k)
        hits_by_question[q.id] = [
            HitRecord(
                rank=h.rank,
                score=h.score,
                regulation_id=h.regulation_id,
                article=h.article,
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
    k: int,
) -> dict[str, AnswerRecord]:
    """Run generation for every question. Rate-limited.

    Pacing strategy: simple sleep-between-calls. At rpm=10, that's 6s
    between requests. Dumb, predictable, fine for 30 questions.

    Failure strategy: any exception propagates. Re-running with
    --score-only would re-score whatever phase 2 wrote before the
    failure (currently we write at the end, so partial failures lose
    the run - see TODO).

    Note `k` here is the GENERATOR's slice depth, distinct from the
    retrieval depth `eval_k`. Phase 1 retrieves `eval_k` hits; we feed
    only the top `k` of those to the model.
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

        # Slice to the generator's top-k. The full eval_k hits stay
        # cached in hits.json for the recall curve in phase 3.
        cached_hits = hits_by_question[q.id][:k]

        # Rebuild the user prompt from cached hits. format_chunk reads
        # `regulation_id` and `article` as attributes off the hit and
        # only `text` from the chunk dict, so the dict here is minimal
        # by design - that narrow contract is what makes this
        # reconstruction safe and stable across parser changes.
        synthetic_hits = [
            RetrievalHit(
                rank=h.rank,
                score=h.score,
                regulation_id=h.regulation_id,
                article=h.article,
                chunk={"text": h.text},
            )
            for h in cached_hits
        ]
        user_prompt = assemble_user_prompt(q.question, synthetic_hits)

        last_call_at = time.monotonic()
        result = generator.generate(SYSTEM_PROMPT, user_prompt)

        answers[q.id] = AnswerRecord(
            answer_text=result.text,
            declared_mode=extract_mode(result.text),
            self_confidence=extract_confidence(result.text),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    return answers


# ----------------------------------------------------------------------------
# Phase 3: score
# ----------------------------------------------------------------------------


def extract_mode(answer_text: str) -> str | None:
    """Pull the leading [MODE] label out of an answer. None if absent.

    The system prompt requires the answer to start with [ANSWER],
    [PARTIAL], or [REFUSE]. We match strictly on that contract - if the
    model drifted (e.g. emitted "ANSWER:" or put the label mid-text)
    the harness flags it as an unrecognized mode rather than guessing.
    """
    m = re.match(r"\s*\[(ANSWER|PARTIAL|REFUSE)\]", answer_text)
    return m.group(1) if m else None


def extract_confidence(answer_text: str) -> float | None:
    """Pull the trailing `Confidence: 0.XX` line. None if absent/unparseable.

    Contract: prompt instructs the model to emit a final line of the
    form `Confidence: 0.XX` with a value in [0, 1]. We're tolerant of
    where in the text it appears (use `MULTILINE` `$`) but strict on
    the numeric format. Out-of-range values are clamped to [0, 1] -
    LLMs occasionally emit 1.0 as `1` or values like `0.95.` with
    stray punctuation.
    """
    m = CONFIDENCE_PATTERN.search(answer_text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


def extract_citations(answer_text: str) -> list[str]:
    """Extract regulation ids cited in [xxx] form.

    Filters out mode labels. Preserves order and duplicates - the
    scorer dedupes when it needs to.
    """
    found = CITATION_PATTERN.findall(answer_text)
    return [cid for cid in found if cid not in MODES]


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

    Returns None if no retrieved chunk could be the parent - caller
    treats that as a recall miss.

    Example:
      sub_rule_id = "A6c", retrieved_ids = ["A6", "A6e", "9f"]
      -> "A6" (prefix match, longest among matching candidates)
    """
    candidates = [rid for rid in retrieved_ids if sub_rule_id.startswith(rid)]
    if not candidates:
        return None
    return max(candidates, key=len)


def _confab_pattern_for(cited_ids: list[str]) -> re.Pattern[str]:
    """Build the confabulation-check regex for a set of cited ids.

    Anchored on the chunk format `3l+) [ANNOTATION]...` - id followed
    by `)` - because `\\b` boundaries break on `+`-suffixed ids (`+` is
    non-word, so `\\b` after `+` doesn't behave intuitively).

    Border:
      - leading: start-of-string OR non-alphanumeric (whitespace, `(`, `[`)
      - trailing: `)` OR whitespace OR end-of-string

    Capture group lets the caller record which id matched.
    """
    id_alternation = "|".join(re.escape(cid) for cid in cited_ids)
    return re.compile(rf"(?:^|[^A-Za-z0-9+])({id_alternation})(?:\)|\s|$)")


def score_question(
    q: GoldenQuestion,
    hits: list[HitRecord],
    answer: AnswerRecord,
    k: int,
) -> QuestionScore:
    """Score one question. Pure function over the cached artifacts.

    `k` is the GENERATOR's slice depth. `hits` contains up to `eval_k`
    entries; recall@k uses the top-`k` slice, the recall curve in
    `recall_at_n` uses configurable slices of the same list.
    """

    # All hits up to eval_k. Rank-ordered (rank 1 = best). Used for the
    # recall@N curve. Generation-time chunks are the top-k slice.
    retrieved_ids_full = [h.regulation_id for h in hits]
    retrieved_ids = retrieved_ids_full[:k]   # what the generator actually saw
    chunks_concatenated = " ".join(
        normalize_for_quote_match(h.text) for h in hits[:k]
    )

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
    # expected? Distinct from accuracy/recall - a model that cites 7
    # correct ids when 3 were expected scores 1.0 on accuracy
    # (over-citation isn't penalized) but lower on precision. For
    # PARTIAL/ANSWER cases where the expected set is "representative"
    # rather than exhaustive, precision will read low and that's fine
    # - it's diagnostic, not a quality signal on its own.
    if cited_set:
        citation_precision = len(cited_set & expected_set) / len(cited_set)
    else:
        # No citations at all. For REFUSE this is correct (1.0); for
        # ANSWER/PARTIAL it's a separate failure caught by mode/recall.
        citation_precision = 1.0 if not expected_set else 0.0

    # --- confabulation: cited id whose text is not in ANY retrieved chunk ---
    if cited_ids:
        confab_pattern = _confab_pattern_for(cited_ids)
        seen_ids: set[str] = set()
        for h in hits[:k]:
            seen_ids.update(confab_pattern.findall(h.text))
        confabulated_ids = [cid for cid in cited_ids if cid not in seen_ids]
    else:
        confabulated_ids = []

    # --- recall@k (chunk-collapsed parent matching for sub-rule ids) ---
    def _recall_at(n: int) -> tuple[float, int | None]:
        """Recall against the top-n slice; also return rank-of-first-hit."""
        slice_ids = retrieved_ids_full[:n]
        if not q.expected_ids:
            # REFUSE cases: no expected ids -> recall is 1.0 by convention,
            # rank-of-first-hit undefined (None).
            return 1.0, None
        hits_count = 0
        first_rank: int | None = None
        for expected_id in q.expected_ids:
            # Either the expected id IS a retrieved chunk id, or its
            # parent (longest-prefix retrieved chunk) was retrieved.
            matched_rank: int | None = None
            if expected_id in slice_ids:
                matched_rank = slice_ids.index(expected_id) + 1
            else:
                parent = parent_chunk_id(expected_id, slice_ids)
                if parent is not None:
                    matched_rank = slice_ids.index(parent) + 1
            if matched_rank is not None:
                hits_count += 1
                if first_rank is None or matched_rank < first_rank:
                    first_rank = matched_rank
        return hits_count / len(q.expected_ids), first_rank

    recall_at_k, rank_of_first_expected = _recall_at(k)
    recall_at_n = {str(n): _recall_at(n)[0] for n in RECALL_AT_K_BREAKS}

    # MRR: 1/rank of first expected hit. Convention: REFUSE -> 1.0
    # (no expected ids, vacuously perfect); miss -> 0.0.
    if not q.expected_ids:
        mrr = 1.0
    elif rank_of_first_expected is None:
        mrr = 0.0
    else:
        mrr = 1.0 / rank_of_first_expected

    # --- retrieval score signals (over the top-k slice the generator saw) ---
    topk_hits = hits[:k]
    if topk_hits:
        top1_score = topk_hits[0].score
        top1_top2_margin = (
            top1_score - topk_hits[1].score if len(topk_hits) > 1 else 0.0
        )
        topk_mean_score = sum(h.score for h in topk_hits) / len(topk_hits)
    else:
        top1_score = 0.0
        top1_top2_margin = 0.0
        topk_mean_score = 0.0

    # --- quote validity (existing): match against any retrieved chunk ---
    quotes = extract_quotes(answer.answer_text)
    quotes_valid = 0
    for quote in quotes:
        normalized_quote = normalize_for_quote_match(quote)
        if normalized_quote and normalized_quote in chunks_concatenated:
            quotes_valid += 1
    quote_validity = quotes_valid / len(quotes) if quotes else 1.0

    # --- citation_quote_alignment (faithfulness proxy) ---
    # For each cited id that has a verbatim quote in the same paragraph,
    # does the quote substring-match THAT id's chunk body specifically?
    # Stricter than quote_validity (which only checks "did this quoted
    # text exist anywhere in the retrieved set?"). Catches the failure
    # mode: model quotes text from chunk A but cites chunk B.
    #
    # Heuristic for "associated quote": look in the same paragraph as
    # the `[id]` citation. Paragraphs are split on blank lines.
    #
    # Important design choice: only QUOTE-BACKED citations are scored.
    # Citations without a nearby quote are excluded from the denominator
    # (not counted as misaligned). Rationale: under the verbatim-quote
    # contract every primary claim needs a quote, but PARTIAL answers
    # legitimately list "also relevant" pointer-citations without quoting
    # each one. Penalizing those as misaligned would conflate two
    # different failure modes. Confabulation + quote_validity already
    # catch the cases this would otherwise flag.
    #
    # REFUSE cases: 1.0 by convention (no claims to back).
    if not cited_ids:
        citation_quote_alignment = 1.0 if not expected_set else 0.0
    else:
        # Build a lookup: id -> normalized chunk body. Sub-rule ids
        # (e.g. A6c cited from inside chunk A6) inherit their parent's
        # body via parent_chunk_id lookup.
        id_to_body: dict[str, str] = {
            h.regulation_id: normalize_for_quote_match(h.text) for h in hits[:k]
        }
        paragraphs = re.split(r"\n\s*\n", answer.answer_text)
        aligned = 0
        considered = 0
        for cid in cited_ids:
            # Find the body to check this id against.
            body = id_to_body.get(cid)
            if body is None:
                parent = parent_chunk_id(cid, retrieved_ids)
                if parent is not None:
                    body = id_to_body.get(parent)
            if body is None:
                # Cited but not retrieved -> already caught by confabulation.
                # Don't double-count here.
                continue
            # Look for a paragraph that contains BOTH this citation and
            # at least one quote.
            for para in paragraphs:
                if f"[{cid}]" not in para:
                    continue
                quotes_in_para = extract_quotes(para)
                if not quotes_in_para:
                    continue
                # This is a quote-backed citation; score it.
                considered += 1
                for quote in quotes_in_para:
                    nq = normalize_for_quote_match(quote)
                    if nq and nq in body:
                        aligned += 1
                        break
                break  # only score the first paragraph that has both
        # If no citations had associated quotes, the proxy is vacuously
        # 1.0 (nothing to disprove). quote_validity stays as the signal
        # for "did the model quote at all".
        citation_quote_alignment = aligned / considered if considered else 1.0

    # --- derived "correct" flag for the threshold summary ---
    # Strict: mode is right AND every expected id was cited AND nothing
    # was confabulated. Tunable later by editing this single line.
    correct = (
        (answer.declared_mode == q.expected_mode)
        and (citation_accuracy == 1.0)
        and (not confabulated_ids)
    )

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
        recall_at_n=recall_at_n,
        rank_of_first_expected=rank_of_first_expected,
        mrr=mrr,
        top1_score=top1_score,
        top1_top2_margin=top1_top2_margin,
        topk_mean_score=topk_mean_score,
        quotes_total=len(quotes),
        quotes_valid=quotes_valid,
        quote_validity=quote_validity,
        citation_quote_alignment=citation_quote_alignment,
        self_confidence=answer.self_confidence,
        correct=correct,
    )


def _confidence_threshold_summary(
    scores: list[QuestionScore],
    thresholds: tuple[float, ...] = CONFIDENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Build the threshold-style stat: at confidence >= T, what's accuracy?

    Returns a dict with one entry per threshold and counts of unparseable
    confidences. Questions without a parseable Confidence line are
    excluded from above/below buckets - they're reported separately.
    """
    parseable = [s for s in scores if s.self_confidence is not None]
    missing = len(scores) - len(parseable)

    buckets = []
    for t in thresholds:
        above = [s for s in parseable if s.self_confidence >= t]
        below = [s for s in parseable if s.self_confidence < t]
        n_above = len(above)
        n_below = len(below)
        buckets.append({
            "threshold": t,
            "n_above": n_above,
            "n_below": n_below,
            "frac_above": n_above / len(parseable) if parseable else 0.0,
            # Accuracy is the fraction of "correct" (strict) questions in each bucket.
            # None when bucket is empty - caller prints "n/a".
            "accuracy_above": (
                sum(1 for s in above if s.correct) / n_above if n_above else None
            ),
            "accuracy_below": (
                sum(1 for s in below if s.correct) / n_below if n_below else None
            ),
        })

    return {
        "n_with_confidence": len(parseable),
        "n_missing_confidence": missing,
        "by_threshold": buckets,
    }


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

    # Recall curve aggregates: mean recall at each break point.
    recall_curve = {
        f"recall_at_{n_break}_mean": (
            sum(s.recall_at_n.get(str(n_break), 0.0) for s in scores) / n
        )
        for n_break in RECALL_AT_K_BREAKS
    }

    # Retrieval score stats - over the top-k slice the generator saw.
    top1_scores = [s.top1_score for s in scores]
    margins = [s.top1_top2_margin for s in scores]
    topk_means = [s.topk_mean_score for s in scores]
    retrieval_score_stats = {
        "top1_mean": sum(top1_scores) / n,
        "top1_min": min(top1_scores),
        "top1_max": max(top1_scores),
        "margin_mean": sum(margins) / n,
        "topk_mean_overall": sum(topk_means) / n,
    }

    # Rank-of-first-expected aggregates (excludes REFUSE / None).
    valid_ranks = [s.rank_of_first_expected for s in scores if s.rank_of_first_expected is not None]
    misses = sum(
        1 for s in scores
        if s.expected_ids and s.rank_of_first_expected is None
    )
    rank_stats: dict[str, Any] = {
        "mean": sum(valid_ranks) / len(valid_ranks) if valid_ranks else None,
        "misses": misses,
    }

    return {
        "n": n,
        "recall_at_k_mean": sum(s.recall_at_k for s in scores) / n,
        **recall_curve,
        "mrr_mean": sum(s.mrr for s in scores) / n,
        "rank_of_first_expected": rank_stats,
        "citation_accuracy_mean": sum(s.citation_accuracy for s in scores) / n,
        "citation_precision_mean": sum(s.citation_precision for s in scores) / n,
        "quote_validity_mean": sum(s.quote_validity for s in scores) / n,
        "citation_quote_alignment_mean": (
            sum(s.citation_quote_alignment for s in scores) / n
        ),
        "mode_accuracy": sum(1 for s in scores if s.mode_match) / n,
        "mode_confusion": confusion,
        "total_confabulated_ids": sum(len(s.confabulated_ids) for s in scores),
        "questions_with_confabulation": sum(
            1 for s in scores if s.confabulated_ids
        ),
        "retrieval_score_stats": retrieval_score_stats,
        "confidence_threshold_summary": _confidence_threshold_summary(scores),
        "strict_correct_count": sum(1 for s in scores if s.correct),
        "strict_correct_fraction": sum(1 for s in scores if s.correct) / n,
    }


def phase_score(
    questions: list[GoldenQuestion],
    hits_by_question: dict[str, list[HitRecord]],
    answers: dict[str, AnswerRecord],
    k: int,
) -> dict[str, Any]:
    """Run phase 3. Pure function over cached artifacts.

    `k` is the generator's slice depth - needed because score_question
    distinguishes "all retrieved" (for the recall curve) from "what the
    generator saw" (for everything else).
    """
    per_question = [
        score_question(q, hits_by_question[q.id], answers[q.id], k=k) for q in questions
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


def format_summary(metrics: dict[str, Any]) -> str:
    """Render aggregate metrics as a human-readable string.

    Single source of truth for the summary text - both summary.txt and
    --summary stdout output go through here, so they can't drift.
    """
    agg = metrics["aggregate"]
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"AGGREGATE  (n={agg['n']})")
    lines.append("=" * 70)
    if agg["n"] == 0:
        lines.append("  (no questions scored)")
        lines.append("")
        return "\n".join(lines)

    lines.append("Retrieval")
    for n_break in RECALL_AT_K_BREAKS:
        key = f"recall_at_{n_break}_mean"
        if key in agg:
            lines.append(f"  recall@{n_break}              {agg[key]:.3f}")
    lines.append(f"  recall@k mean         {agg['recall_at_k_mean']:.3f}")
    lines.append(f"  MRR                   {agg['mrr_mean']:.3f}")
    rank_stats = agg.get("rank_of_first_expected", {})
    mean_rank = rank_stats.get("mean")
    misses = rank_stats.get("misses", 0)
    mean_rank_str = f"{mean_rank:.2f}" if mean_rank is not None else "n/a"
    lines.append(
        f"  rank of 1st expected  mean={mean_rank_str}  misses={misses}"
    )
    lines.append("")

    lines.append("Citations & answer quality")
    lines.append(f"  citation accuracy     {agg['citation_accuracy_mean']:.3f}")
    lines.append(f"  citation precision    {agg['citation_precision_mean']:.3f}")
    lines.append(f"  quote validity        {agg['quote_validity_mean']:.3f}")
    lines.append(
        f"  faithfulness (quote-alignment proxy)  "
        f"{agg['citation_quote_alignment_mean']:.3f}"
    )
    lines.append(f"  mode accuracy         {agg['mode_accuracy']:.3f}")
    lines.append(
        f"  confabulated ids      {agg['total_confabulated_ids']} "
        f"(in {agg['questions_with_confabulation']} questions)"
    )
    lines.append(
        f"  strict correct        {agg['strict_correct_count']}/{agg['n']} "
        f"({agg['strict_correct_fraction']:.3f})"
    )
    lines.append("")

    lines.append("Mode confusion (rows = expected, cols = declared):")
    for expected, row in sorted(agg["mode_confusion"].items()):
        for declared, count in sorted(row.items()):
            lines.append(f"  {expected:7s} -> {declared:7s}  {count}")
    lines.append("")

    # Retrieval signal block.
    rstats = agg.get("retrieval_score_stats", {})
    if rstats:
        lines.append("Retrieval score signals (top-k slice seen by generator):")
        lines.append(
            f"  top-1 score   mean={rstats['top1_mean']:.3f}  "
            f"min={rstats['top1_min']:.3f}  max={rstats['top1_max']:.3f}"
        )
        lines.append(f"  top1-top2 gap mean={rstats['margin_mean']:.3f}")
        lines.append(f"  top-k mean    overall={rstats['topk_mean_overall']:.3f}")
        lines.append("")

    # Confidence threshold summary.
    cts = agg.get("confidence_threshold_summary")
    if cts:
        lines.append(
            "Confidence threshold summary "
            "(correct = mode_match and cit_acc=1.0 and no confab):"
        )
        n_conf = cts["n_with_confidence"]
        n_miss = cts["n_missing_confidence"]
        if n_conf == 0:
            lines.append(
                f"  (no questions had a parseable Confidence line; "
                f"missing={n_miss})"
            )
        else:
            for bucket in cts["by_threshold"]:
                t = bucket["threshold"]
                na = bucket["n_above"]
                nb = bucket["n_below"]
                aa = bucket["accuracy_above"]
                ab = bucket["accuracy_below"]
                aa_str = f"{aa:.3f}" if aa is not None else "n/a"
                ab_str = f"{ab:.3f}" if ab is not None else "n/a"
                lines.append(
                    f"  conf >= {t:.2f}: above={na} (acc={aa_str})  "
                    f"below={nb} (acc={ab_str})"
                )
            if n_miss:
                lines.append(f"  questions missing Confidence line: {n_miss}")
        lines.append("")

    return "\n".join(lines)


def write_summary(run_dir: Path, metrics: dict[str, Any]) -> None:
    (run_dir / "summary.txt").write_text(format_summary(metrics), encoding="utf-8")


# ----------------------------------------------------------------------------
# Per-run review report (human eyeball pass)
# ----------------------------------------------------------------------------
#
# review.md is a flat Q/A render of one run, sorted by question id. It
# joins the golden set (question + notes), the cached answers, and the
# strict-correct flag from phase 3 into one scrollable file. Purpose:
# eyeballing individual answers without jq. Aggregate stats stay in
# summary.txt / metrics.json; this is the per-question view.
#
# Written on every full run and every --score-only rerun, same as
# summary.txt and plots/.


def format_review(
    questions: list[GoldenQuestion],
    answers: dict[str, AnswerRecord],
    metrics: dict[str, Any],
) -> str:
    """Render a flat Q/A review of one run, sorted by question id.

    Per-question dicts come from metrics["per_question"] which is
    already asdict()-ed by phase_score; no rescoring needed.

    `correct` mirrors QuestionScore.correct exactly: mode_match AND
    citation_accuracy==1.0 AND no confabulated_ids. A check mark does
    NOT mean the answer is well-written or substantively right - only
    that the harness's strict checks passed. Read the answer text.
    """
    per_q_by_id = {pq["question_id"]: pq for pq in metrics["per_question"]}

    lines: list[str] = []
    lines.append("# Run review")
    lines.append("")
    lines.append(
        "One entry per question, in golden-set order. "
        "`correct` = strict-correct flag from QuestionScore "
        "(mode match + citation_accuracy==1.0 + no confabulation). "
        "Always read the answer text before trusting the flag."
    )
    lines.append("")

    for q in questions:
        score = per_q_by_id.get(q.id)
        answer = answers.get(q.id)

        if score is None or answer is None:
            # Question is in the golden set but not in this run
            # (e.g. --questions subset, or partial-failure run).
            lines.append(f"## {q.id}  *(not in this run)*")
            lines.append("")
            continue

        mark = "✓" if score["correct"] else "✗"
        declared = score["declared_mode"] or "MISSING"
        expected = score["expected_mode"]
        mode_field = (
            expected if declared == expected else f"{declared}  (expected {expected})"
        )

        lines.append(f"## {q.id}  {mark}  [{mode_field}]")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(q.question)
        lines.append("")
        lines.append("**Answer**")
        lines.append("")
        # Render as a blockquote so multi-paragraph answers stay
        # visually distinct from the question and the notes.
        answer_lines = answer.answer_text.splitlines() or [""]
        for line in answer_lines:
            lines.append(f"> {line}" if line else ">")
        lines.append("")
        lines.append("**Golden-set note**")
        lines.append("")
        lines.append(q.notes.strip() if q.notes else "_(none)_")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def write_review(
    run_dir: Path,
    questions: list[GoldenQuestion],
    answers: dict[str, AnswerRecord],
    metrics: dict[str, Any],
) -> None:
    """Write review.md. Idempotent; overwrites on every call."""
    (run_dir / "review.md").write_text(
        format_review(questions, answers, metrics), encoding="utf-8"
    )


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
#
# All plots are pure functions over `metrics` (the same dict written to
# metrics.json). They write PNGs to `<run_dir>/plots/`. Run on every
# full run and every --score-only invocation. Failing to produce a plot
# is logged but never aborts scoring - plot bugs shouldn't lose metric
# data.

PLOT_DPI = 120

# Seaborn theme applied via rc_context in write_plots(). "whitegrid"
# gives the subtle horizontal grid lines that helped readability in
# the original plots without the heavy default ggplot look.
SEABORN_CONTEXT = "notebook"
SEABORN_STYLE = "whitegrid"


def _color_for_mode(mode: str) -> str:
    return {"ANSWER": "#2a7", "PARTIAL": "#f80", "REFUSE": "#a33"}.get(mode, "#888")


def _ensure_plots_dir(run_dir: Path) -> Path:
    p = run_dir / "plots"
    p.mkdir(exist_ok=True)
    return p


def plot_reliability_diagram(metrics: dict[str, Any], path: Path) -> None:
    """Bin questions by self_confidence, plot mean accuracy per bin.

    Answers the motivating question: "if confidence > 0.9, is accuracy
    also high enough that I can skip review?" Diagonal overlay = perfect
    calibration. Bars above the diagonal = underconfident; below =
    overconfident.

    Skip silently if no questions have parseable confidence.
    """
    rows = [r for r in metrics["per_question"] if r.get("self_confidence") is not None]
    if not rows:
        return

    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    bin_labels = ["[0, 0.5)", "[0.5, 0.7)", "[0.7, 0.85)", "[0.85, 0.95)", "[0.95, 1.0]"]
    bin_centers = [(lo + min(hi, 1.0)) / 2 for lo, hi in bins]
    accuracies: list[float | None] = []
    counts: list[int] = []
    for lo, hi in bins:
        b = [r for r in rows if lo <= r["self_confidence"] < hi]
        counts.append(len(b))
        accuracies.append(
            (sum(1 for r in b if r["correct"]) / len(b)) if b else None
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = list(range(len(bins)))
    bar_heights = [a if a is not None else 0 for a in accuracies]
    bars = ax.bar(xs, bar_heights, color="#4a8", alpha=0.8, edgecolor="#266")
    # Annotate bars with n.
    for i, (bar, c, a) in enumerate(zip(bars, counts, accuracies)):
        label = f"n={c}"
        if a is None:
            ax.text(i, 0.02, label, ha="center", va="bottom", fontsize=9, color="#888")
        else:
            ax.text(i, bar.get_height() + 0.02, label, ha="center", va="bottom", fontsize=9)
    # Diagonal "perfect calibration" reference.
    ax.plot(xs, bin_centers, "k--", alpha=0.4, label="perfect calibration")
    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels, rotation=15)
    ax.set_xlabel("Self-reported confidence bin")
    ax.set_ylabel("Empirical accuracy (strict)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Reliability diagram: confidence vs. accuracy")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_mode_confusion(metrics: dict[str, Any], path: Path) -> None:
    """Heatmap of declared vs expected modes. Annotated with counts.

    Asymmetric error costs (REFUSE->ANSWER vs ANSWER->REFUSE) are visible
    by eye on the heatmap - no explicit weighting in the metric.
    """
    confusion = metrics["aggregate"].get("mode_confusion", {})
    if not confusion:
        return

    mode_order = ["ANSWER", "PARTIAL", "REFUSE"]
    declared_modes = set()
    for row in confusion.values():
        declared_modes.update(row.keys())
    cols = [m for m in mode_order if m in declared_modes]
    if "MISSING" in declared_modes:
        cols.append("MISSING")

    rows_present = [m for m in mode_order if m in confusion]
    grid = np.array(
        [[confusion.get(r, {}).get(c, 0) for c in cols] for r in rows_present],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        grid,
        annot=True, fmt="d",
        cmap="Blues",
        xticklabels=cols, yticklabels=rows_present,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Declared mode")
    ax.set_ylabel("Expected mode")
    ax.set_title("Mode confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_recall_curve(metrics: dict[str, Any], path: Path) -> None:
    """Recall@N curve for N in RECALL_AT_K_BREAKS + the generator's k.

    One line for the overall mean. Separate lines per expected_mode for
    ANSWER and PARTIAL. REFUSE excluded (recall is 1.0 by convention,
    not informative).

    Reads per-question `recall_at_n` for breaks < k, then uses
    `recall_at_k` for the final point.
    """
    per_q = metrics["per_question"]
    if not per_q:
        return

    # Determine k from any per-question record (recall_at_k key exists)
    # and use it as the rightmost x.
    breakpoints = list(RECALL_AT_K_BREAKS)
    # Infer k from the retrieval slice size - len(retrieved_ids) is the
    # generator's k.
    k_inferred = len(per_q[0].get("retrieved_ids", []))
    if k_inferred and k_inferred not in breakpoints:
        breakpoints.append(k_inferred)
    breakpoints = sorted(set(breakpoints))

    def _mean_recall_at(rows: list[dict], n_break: int) -> float | None:
        if not rows:
            return None
        if n_break == k_inferred:
            vals = [r["recall_at_k"] for r in rows]
        else:
            vals = [r["recall_at_n"][str(n_break)] for r in rows]
        return sum(vals) / len(vals)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    overall = [_mean_recall_at(per_q, n) for n in breakpoints]
    ax.plot(breakpoints, overall, "o-", color="#333", linewidth=2, label="all")

    for mode in ("ANSWER", "PARTIAL"):
        subset = [r for r in per_q if r["expected_mode"] == mode]
        if not subset:
            continue
        ys = [_mean_recall_at(subset, n) for n in breakpoints]
        ax.plot(breakpoints, ys, "o--", color=_color_for_mode(mode), label=f"{mode} (n={len(subset)})")

    ax.set_xlabel("Top-N retrieved chunks")
    ax.set_ylabel("Mean recall")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(breakpoints)
    ax.set_title("Recall@N curve")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_per_question_scorecard(metrics: dict[str, Any], path: Path) -> None:
    """Heatmap: rows = questions, cols = key metrics, cells colored 0->1.

    Spots problem questions instantly. mode_match is 0/1; no_confab is
    1 if zero confabulations, 0 otherwise (more interpretable than raw
    count in a heatmap).
    """
    per_q = metrics["per_question"]
    if not per_q:
        return

    cols = [
        ("recall@k", "recall_at_k"),
        ("cit_acc", "citation_accuracy"),
        ("cit_prec", "citation_precision"),
        ("quote_val", "quote_validity"),
        ("faithful", "citation_quote_alignment"),
        ("mode_ok", None),     # special
        ("no_confab", None),   # special
        ("correct", "correct"),
    ]

    def _cell(row: dict, label: str, key: str | None) -> float:
        if label == "mode_ok":
            return 1.0 if row["mode_match"] else 0.0
        if label == "no_confab":
            return 1.0 if not row["confabulated_ids"] else 0.0
        if label == "correct":
            return 1.0 if row["correct"] else 0.0
        return float(row[key])

    grid = np.array(
        [[_cell(r, label, key) for label, key in cols] for r in per_q]
    )
    qids = [r["question_id"] for r in per_q]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(qids) + 1)))
    sns.heatmap(
        grid,
        annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        xticklabels=[c[0] for c in cols],
        yticklabels=qids,
        annot_kws={"fontsize": 7},
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Per-question scorecard")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_retrieval_score_distribution(metrics: dict[str, Any], path: Path) -> None:
    """Histogram of top-1 retrieval scores, split by expected_mode.

    Hypothesis: REFUSE cases should have low top-1 (out-of-domain -> no
    on-topic chunk). High top-1 on a REFUSE case = suspicious (retriever
    found something semi-related, model might over-claim).
    """
    per_q = metrics["per_question"]
    if not per_q:
        return

    data = {
        "top1_score": [r["top1_score"] for r in per_q],
        "expected_mode": [r["expected_mode"] for r in per_q],
    }
    palette = {m: _color_for_mode(m) for m in set(data["expected_mode"])}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(
        data=data,
        x="top1_score",
        hue="expected_mode",
        palette=palette,
        bins=15,
        multiple="layer",
        alpha=0.55,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_xlabel("Top-1 cosine similarity")
    ax.set_ylabel("Question count")
    ax.set_title("Top-1 retrieval score by expected mode")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_confidence_vs_correctness(metrics: dict[str, Any], path: Path) -> None:
    """Scatter: self_confidence (x) vs correctness (y), sized by top-1 score.

    Tiny jitter on y so 0/1 points don't fully overlap. Lets you eyeball
    calibration when n is small (<=30) and the binned reliability diagram
    can be misleading.
    """
    rows = [r for r in metrics["per_question"] if r.get("self_confidence") is not None]
    if not rows:
        return

    rng = np.random.default_rng(0)
    data = {
        "self_confidence": [r["self_confidence"] for r in rows],
        "correct_jittered": [
            (1.0 if r["correct"] else 0.0) + rng.uniform(-0.04, 0.04)
            for r in rows
        ],
        "top1_score": [r["top1_score"] for r in rows],
        "expected_mode": [r["expected_mode"] for r in rows],
        "question_id": [r["question_id"] for r in rows],
    }
    palette = {m: _color_for_mode(m) for m in set(data["expected_mode"])}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=data,
        x="self_confidence",
        y="correct_jittered",
        size="top1_score",
        sizes=(40, 290),
        hue="expected_mode",
        palette=palette,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    for qid, x, y in zip(data["question_id"], data["self_confidence"], data["correct_jittered"]):
        ax.annotate(
            qid, (x, y),
            fontsize=7, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["wrong", "correct"])
    ax.set_xlabel("Self-reported confidence")
    ax.set_ylabel("")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.3, 1.3)
    ax.set_title("Confidence vs correctness (point size = top-1 retrieval score)")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


PLOTTERS = [
    ("reliability_diagram.png", plot_reliability_diagram),
    ("mode_confusion_matrix.png", plot_mode_confusion),
    ("recall_at_k_curve.png", plot_recall_curve),
    ("per_question_scorecard.png", plot_per_question_scorecard),
    ("retrieval_score_distribution.png", plot_retrieval_score_distribution),
    ("confidence_vs_correctness.png", plot_confidence_vs_correctness),
]


def write_plots(run_dir: Path, metrics: dict[str, Any]) -> None:
    """Generate all six plot PNGs. Failures per-plot are logged, not raised.

    Plot bugs shouldn't lose the metrics - phase 3 already wrote them.
    """
    if metrics["aggregate"].get("n", 0) == 0:
        return
    plots_dir = _ensure_plots_dir(run_dir)
    # Seaborn theme is set globally for the duration of the plotting
    # block. The reset at the end keeps matplotlib's globals untouched
    # for any caller that imports this module and plots elsewhere.
    sns.set_theme(context=SEABORN_CONTEXT, style=SEABORN_STYLE)
    try:
        for fname, fn in PLOTTERS:
            try:
                fn(metrics, plots_dir / fname)
            except Exception as exc:
                print(f"  plot {fname} failed: {exc}", file=sys.stderr)
    finally:
        sns.reset_defaults()


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
        default=PIPELINE_DEFAULT_K,
        help=f"top-k for generator (default {PIPELINE_DEFAULT_K})",
    )
    parser.add_argument(
        "--eval-k",
        type=int,
        default=DEFAULT_EVAL_K,
        help=(
            f"retrieval depth for the recall@N curve (default {DEFAULT_EVAL_K}). "
            "Must be >= --k. Phase 2 still feeds only top --k chunks to the generator."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="also print aggregate metrics to stdout (summary.txt is always written)",
    )
    parser.add_argument(
        "--score-only",
        metavar="RUN_ID",
        help="re-score an existing run directory (skip phases 1+2)",
        default=None,
    )
    args = parser.parse_args(argv)

    if args.eval_k < args.k:
        print(
            f"--eval-k ({args.eval_k}) must be >= --k ({args.k}); "
            f"bumping eval_k to {args.k}",
            file=sys.stderr,
        )
        args.eval_k = args.k

    # Load .env before anything that reads env vars: GEMINI_API_KEY for
    # the generator, HF_TOKEN for the embedder. Soft dependency - if
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
        # didn't override - avoids KeyError if golden set grew since the run.
        questions = [q for q in questions if q.id in hits and q.id in answers]
        # Pull k from the run's config if available; fall back to CLI default.
        config_path = run_dir / "config.json"
        run_k = args.k
        if config_path.exists():
            try:
                run_k = json.loads(config_path.read_text())["k"]
            except (KeyError, json.JSONDecodeError):
                pass
        metrics = phase_score(questions, hits, answers, k=run_k)
        write_json(run_dir / "metrics.json", metrics)
        write_summary(run_dir, metrics)
        write_review(run_dir, questions, answers, metrics)
        write_plots(run_dir, metrics)
        print(f"rescored: {run_dir}")
        if args.summary:
            print()
            print(format_summary(metrics))
        return 0

    # Full run.
    run_dir = make_run_dir()
    print(f"run dir: {run_dir}")

    print(f"phase 1: retrieve  ({len(questions)} questions, eval_k={args.eval_k}, k={args.k})")
    retriever = Retriever.from_disk()
    hits = phase_retrieve(questions, retriever, eval_k=args.eval_k)
    write_json(run_dir / "hits.json", {qid: [asdict(h) for h in hl] for qid, hl in hits.items()})

    print(f"phase 2: generate  (rpm={args.rpm})")
    generator = GeminiGenerator()
    answers = phase_generate(questions, hits, generator, rpm=args.rpm, k=args.k)
    write_json(run_dir / "answers.json", {qid: asdict(a) for qid, a in answers.items()})

    print("phase 3: score")
    metrics = phase_score(questions, hits, answers, k=args.k)
    write_json(run_dir / "metrics.json", metrics)

    write_json(run_dir / "config.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_set_hash": hash_golden_set(questions),
        "prompt_hash": hash_text(SYSTEM_PROMPT),
        "model_name": generator.model_name,
        "k": args.k,
        "eval_k": args.eval_k,
        "rpm": args.rpm,
        "question_ids": [q.id for q in questions],
    })

    write_summary(run_dir, metrics)
    write_review(run_dir, questions, answers, metrics)
    write_plots(run_dir, metrics)

    print(f"done: {run_dir}")
    if args.summary:
        print()
        print(format_summary(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
