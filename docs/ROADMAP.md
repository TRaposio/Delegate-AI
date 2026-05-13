# Roadmap

Forward-looking plan for the WCA RAG project. Source of truth for
"what's next and why." Decisions move out of here into
`ARCHITECTURE.md` once they're implemented.

Status legend: 🟢 done · 🟡 in progress · ⬜ not started · 🔴 deferred / out of scope for now

---

## Operating principle: tiered evaluation

The Gemini free tier (~20 RPD as of 2026-04) makes naive full eval runs
the bottleneck for everything downstream. The harness already separates
retrieval (free, local, deterministic) from generation (rate-limited);
the roadmap leans on that separation hard.

| Tier | What it runs | Cost | When |
|---|---|---|---|
| **Tier 1** | Phase 1 retrieve + Phase 3 score against cached `answers.json` (or retrieval-only metrics if no cached answers apply) | Free | Every change — embedding swap, retrieval strategy, parser tweak |
| **Tier 2** | Phase 2 generation on a 5-question smoke set + full Phase 3 | ~5 LLM calls | After prompt edits, before declaring a Tier 1 winner |
| **Tier 3** | Phase 2 generation on the full golden set + full Phase 3 | ~20 LLM calls | Once per Phase milestone (A baseline, end of B, end of C) |

The discipline: never spend LLM calls on questions the retrieval layer
can answer. Recall@k, MRR, top-1 score, retrieval signal stats — all
free. Citation accuracy, mode accuracy, quote validity, faithfulness —
all need the LLM, all Tier 2/3.

**Tier 3 cadence is deliberately rigid (once per phase milestone).**
"Whenever a Tier 1 experiment looks promising" sounds disciplined but
in practice every experiment looks promising at Tier 1, and the budget
evaporates. Anchoring Tier 3 to phase boundaries gives clean
retrospective comparisons and forces Tier 1 to do most of the work.

---

## Phase A — make the eval set trustworthy

Prerequisite to literally everything else. Without a trustworthy eval
loop, every downstream comparison is on vibes.

### A1 ⬜ Uncomment the golden set (~20 questions)

Already written, gated behind RPD. Uncomment, sanity-check ids and
expected_modes, commit.

No LLM cost.

### A2 ⬜ Add `GroqGenerator`

Frees the rest of the project from the 20 RPD bottleneck. Groq's free
tier offers Llama 3.3 70B at ~30 RPM with no daily cap (verify before
relying on it; their free-tier terms shift).

Implementation: one new file following the `GeminiGenerator` pattern.
The `Generator` ABC and `Pipeline` are already designed for this swap.
`ask.py` and `eval.py` should grow a `--generator` flag (default still
Gemini for reproducibility; opt into Groq for bulk experimentation).

Quality caveat: Llama 3.3 70B is not Gemini 2.5 Flash. Citation
adherence and mode discipline may differ. **For Tier 3 runs that
establish baseline metrics, keep Gemini** — switch to Groq for
exploratory work where the question is "did retrieval get better?",
not "what's the absolute quality score?". Cross-checking the same
prompt across two generators is also a free sanity signal on prompt
robustness.

### A3 ⬜ Variance baseline (N=3 runs, same config)

Run the harness N=3 times on the golden set without changing anything.
Measure metric stdev. This is the noise floor — any future Tier 3 delta
smaller than ~2× the stdev is not signal.

Cost: ~60 LLM calls (3 × 20 questions). Spread across 3 days if on
Gemini, or 10 minutes if on Groq.

Output: a row in this doc recording the stdev of recall@k,
citation_accuracy, mode_accuracy, and strict_correct_fraction. Update
when prompt or model changes (variance is config-specific).

### A4 ⬜ Anchor Tier 3 baseline run

One clean Tier 3 run after A1–A3, on the default config (Gemini,
bge-small, k=8, current prompt). This is the "before" snapshot every
Phase B experiment will compare against. Tag the run dir with
`-baseline-v1`.

---

## Phase B — retrieval quality

Mostly Tier 1. The cost problem largely dissolves here.

### B1 ⬜ Embedding model bake-off

Compare `bge-small-en-v1.5` (current) against ≥2 alternatives:
- `bge-base-en-v1.5` — same family, ~3× larger, ~768 dim. Direct upgrade test.
- `intfloat/e5-small-v2` — different family, similar size. Tests whether bge's prefix asymmetry was the right pick for this corpus.
- Optional: `BAAI/bge-large-en-v1.5` — feasibility test on the 2015 MacBook (may be painfully slow at index time but query-time is one matrix multiply).

Tier 1 metrics only: recall@1/3/5/k, MRR, top-1 score distribution,
top1-top2 margin. Pick a winner on retrieval metrics alone.

Implementation: each model is one `SentenceTransformerEmbedder(model_name=...)`
swap. The index pipeline already detects stale embeddings via the
sidecar fingerprint and forces a rebuild — no manual cache busting.

Closing move: **one Tier 3 run with the winner** to confirm the
retrieval improvement actually propagates to citation accuracy. If it
doesn't, that's a finding worth understanding (retrieval bottleneck has
moved elsewhere — chunking? prompt? both?).

### B2 ⬜ Hybrid retrieval (BM25 + vector)

The single biggest expected quality win for this corpus. WCA
regulations are dense with explicit ids (`[A6c]`, `[Article 11]`) that
users sometimes paste verbatim. BM25 nails exact-id queries; embeddings
nail paraphrased ones. The standard hybrid recipe: BM25 and vector each
return top-N, scores are combined (Reciprocal Rank Fusion is the
simple, parameterless default).

Implementation:
- Add `rank-bm25` (pure Python, no native deps).
- New `HybridRetriever` class that composes a `BM25Retriever` (new) and
  the existing vector `Retriever` and merges results via RRF.
- Tier 1 metrics: same as B1.

Care points: the chunk's `text_for_embedding` is not the right input for
BM25 — the prepended `Article N: Title` header would bias scoring. BM25
should index `text` (or a tokenized version of it). Document the
distinction in CONCEPTS.md.

Closing move: **one Tier 3 run with the winner** of {vector-only,
hybrid}.

### B3 ⬜ Layered retrieval via `cross_references`

Already half-built — the parser extracts cross-references into chunk
metadata; the retriever currently ignores them. The intervention: after
top-k retrieval, expand the set with chunks referenced by the top-k.
Bounded by a hop budget (1 hop, capped at N additional chunks) so the
context window doesn't explode.

Two design choices to make before implementing:
1. Expand only top-1 or top-k? Top-1 is conservative and cheap; top-k
   inflates context fast.
2. Re-rank after expansion, or append at the bottom? Re-ranking is more
   principled but adds a cross-encoder dependency. Appending is the
   minimum viable version.

Tier 1 metrics: recall@k (does the right chunk show up more often?).
But the real win is downstream — better citation accuracy when the
answer requires reading rule X *and* rule Y together. So **Tier 3 is
load-bearing here** in a way it isn't for B1/B2.

CONCEPTS.md "Cross-references and graph-augmented retrieval" already
covers the motivation; this is the implementation.

### B4 🔴 Cross-encoder reranking

Standard production pattern: bi-encoder retrieves top-50, cross-encoder
reranks to top-5. Skip in v1 unless B1–B3 plateau. At 108 chunks the
"top-50 from a wide net" recipe is approximately "the whole corpus,"
which removes most of the speed-vs-quality tradeoff that justifies
rerankers in the first place. Revisit when the corpus grows.

---

## Phase C — infrastructure

Do nothing here unless Phase B work demands it.

### C1 🔴 ChromaDB migration

ARCHITECTURE.md §3.2 marks Chroma as 🟡; in practice the NumPy retriever
is faster than any vector DB at 108 chunks and the math is exposed for
learning. The two triggers for actually migrating:

- Phase B work (especially B3 with metadata filters like "only Article 11
  chunks") gets ugly to hand-roll in NumPy.
- Corpus grows past ~10k chunks (Delegate Handbook, past rulings).

Neither holds today. Defer.

### C2 ⬜ Chat memory / multi-turn

Two distinct features confusingly called the same thing:
- **Conversation memory**: the model remembers earlier turns in the same
  session ("what about the next rule?"). Trivial — pass prior
  user/assistant turns in `contents`.
- **Cross-session memory**: the system remembers things across runs
  ("last week you told me X"). Needs persistent storage and a retrieval
  mechanism on top of it. Whole separate project.

v1 of "memory" is conversation memory only. Cross-session is a Phase D
question.

Defer unless a real use case appears. Single-turn Q&A is currently the
right shape for an incident-handling tool — delegates ask one question
and need one answer, not a dialogue.

---

## Phase D — product

Different shape of work. Stops being a RAG project, starts being a web
project.

### D1 ⬜ Deployment

Three tiers, in increasing cost:

1. **Personal CLI** — what we have. Done.
2. **Streamlit Cloud / Hugging Face Spaces** — free tier, deploy in an
   afternoon. Imports the existing `wca_rag` package directly. Good for
   "you + a few delegate friends, occasional use." No auth, no rate
   limiting beyond what the platform gives.
3. **Real web app** — FastAPI backend + a frontend (React or plain
   HTML), hosted on Railway / Fly.io / Render. Auth, rate limits,
   monitoring, paid LLM API. Weeks of work, mostly outside RAG.

(2) is a reasonable Phase D step. (3) becomes a separate project.

### D2 ⬜ Library vs application separation

Stay in one repo. The "fork to a clean version" instinct should be
resisted:
- The educational comments cost nothing at runtime.
- Forks drift; bug fixes have to land in two places.
- A "cleaned up" version is just the same code with worse documentation.

The right separation already exists: `wca_rag/` is a library, the
scripts/CLIs are applications on top of it. A future deployment layer
(FastAPI app, Streamlit script) is one more application — same repo,
new directory.

### D3 🔴 Cross-session memory

See C2. Out of scope until D1 establishes there are actual users.

### D4 🔴 Multilingual support

Translate question → English → answer → translate back. Material work,
not RAG learning. Out of scope.

### D5 🔴 Additional corpora

Delegate Handbook, past WRC rulings, regulation interpretations forum.
Each is a separate scoping decision (corpus quality? versioning?
retrieval contamination?). The CLAUDE.md warning still applies — do not
blindly merge into the same chunk index. Out of scope until D1.

---

## Closed questions (decided, moved to ARCHITECTURE.md)

For reference. These were once open; they aren't anymore.

- Classic RAG vs long-context stuffing → RAG. ARCHITECTURE.md §2.
- Embedding model → bge-small-en-v1.5. ARCHITECTURE.md §3.1. (May
  change after B1.)
- Vector store → NumPy in-memory. ARCHITECTURE.md §3.2. (May change in
  C1.)
- Generation LLM → Gemini 2.5 Flash. ARCHITECTURE.md §3.3. (Augmented
  by Groq in A2; Gemini remains the Tier 3 reference.)
- Chunking strategy → custom parser, one chunk per top-level
  regulation. ARCHITECTURE.md §3.4.
- Metadata schema → see ARCHITECTURE.md §3.5.
- Citation granularity → claim-level (Option A). ARCHITECTURE.md §3.9,
  CONCEPTS.md.
- Evaluation strategy → three-phase harness, claim-level scoring, six
  plots. ARCHITECTURE.md §5.
