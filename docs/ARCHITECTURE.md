# Architecture

Status legend: 🟢 decided · 🟡 proposed (awaiting feedback)

This doc is the source of truth for architectural decisions that are
already made or in progress. Forward-looking plans live in
`ROADMAP.md`.

---

## 1. Goal

Build a RAG (Retrieval-Augmented Generation) system that answers WCA-regulation
questions for delegates handling incidents at competitions. The answer must
cite the specific regulations applied.

Primary goals (equal weight):
1. Learn how RAG works end-to-end.
2. Produce a tool actually usable at competitions.

Non-goals for v1:
- Production deployment, hosting, multi-user.
- Multi-language support (English only).
- Fine-tuning models.
- Sources beyond the WCA Regulations (handbook, rulings).

---

## 2. Why RAG (and not alternatives)

The WCA Regulations corpus is ~22k words / ~30k tokens. This is small enough
that several architectures are viable:

| Approach | Pro | Con | Verdict |
|---|---|---|---|
| **Long-context stuffing** (whole corpus in every prompt) | No retrieval errors. Model sees everything. | "Lost in the middle" attention degradation. Higher per-query cost. | Possible fallback / baseline. |
| **Classic RAG** (chunk → embed → top-k → LLM) | Cheap. Fast. Scales to more corpora later. Standard. | Retrieval miss = wrong answer. | 🟢 chosen for v1. |

**Decision (🟢):** classic RAG for v1.

References for further reading:
- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — original RAG paper.
- Liu et al. 2023, *Lost in the Middle* — context-window attention degradation.
- Pinecone's RAG learning section (`pinecone.io/learn/`).

---

## 3. Component choices

### 3.1 Embedding model 🟢

**Proposed:** `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local).

| Option | Size | Quality (MTEB avg) | Cost | Local feasibility |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 90MB | ~58 | free | great |
| **`bge-small-en-v1.5`** | 130MB | ~62 | free | great |
| `bge-large-en-v1.5` | 1.3GB | ~64 | free | painful on 2015 Mac |
| OpenAI `text-embedding-3-small` | API | ~62 | paid | n/a |
| Voyage `voyage-3-lite` | API | strong | free tier | n/a |
| Gemini `text-embedding-004` | API | strong | free tier | n/a |

**Rationale:** for a 22k-word English corpus, `bge-small` is the quality sweet
spot at the local-and-free price point. The marginal MTEB gap to `bge-large`
will not matter at this corpus size. Behind an `Embedder` interface for easy
swap.

**Implemented.** `wca_rag/embedder.py` defines the `Embedder` ABC with
`encode_documents` / `encode_query` (separate methods, not a flag, to
make the bge query-document asymmetry a contract). Default
implementation: `SentenceTransformerEmbedder` wrapping
`BAAI/bge-small-en-v1.5`. Vectors are L2-normalized at encode time so
the retriever can use dot product directly.

**Index outputs** (`wca_rag/index.py`, run via `python -m wca_rag.index`):
- `data/embeddings.npy` — `(108, 384)` float32, L2-normalized.
- `data/chunk_ids.json` — list of regulation_ids, parallel to
  embedding rows. Acts as the join key from the matrix back to
  `chunks.jsonl`.
- `data/embeddings.meta.json` — sidecar recording model name, embedding
  dim, normalization flag, chunk count, corpus fingerprint
  (sha1 of sorted `text_hash` values), and creation timestamp. Used to
  detect stale embeddings after re-parsing.

**Validation.** `scripts/inspect_embeddings.py` runs on the persisted
artifacts: checks shape/dtype/normalization, verifies `chunk_ids` are
row-aligned with `chunks.jsonl`, confirms the corpus fingerprint matches
the current parser output, asserts every chunk is its own nearest
neighbor (sanity check on normalization + the matrix multiply), and
prints similarity-distribution percentiles plus sample top-k neighbors
for eyeball validation.

### 3.2 Vector store 🟡

**Proposed:** ChromaDB (file-based, local).

Alternatives considered:
- **FAISS** — Meta's library, the de-facto standard for local vector search. Faster, more low-level. Educational value: you build the index yourself. Tradeoff: more friction, separate metadata storage.
- **NumPy array + cosine similarity** — fully from-scratch. Pedagogically the most informative for ~150 chunks (literally `embeddings @ query.T`). Tradeoff: write everything yourself.
- **Qdrant / Weaviate** — production-grade vector DBs. Overkill for local use.
- **pgvector** — Postgres extension. User knows Postgres so this would be familiar, but adds setup overhead.

**Rationale for Chroma:** Python-native, no server, persists to disk, has a
sane Python API. Current implementation is NumPy in-memory — at 108 chunks
the math is so simple that brute force is the fastest and most pedagogical
option. Chroma becomes worth it when metadata filtering or larger corpora
demand it.

### 3.3 Generation LLM 🟢

**Decided:** Google Gemini API (`gemini-2.5-flash`, free tier).

| Option | Cost | Quality for citation/refusal | Notes |
|---|---|---|---|
| **Gemini 2.5 Flash (free tier)** | free at hobby scale | strong | requires API key |
| Anthropic Claude Haiku 4.5 | $1/$5 per MTok | strongest | trial credit only, then paid |
| OpenAI GPT-4o-mini | paid | strong | paid only |
| Groq (Llama 3.3 70B) | free at hobby scale | reasonable | free-tier ToS uncertain |
| Local Llama 3.2 1B (Ollama) | free | weak for nuanced citation | educational but quality concern |

**Rationale:** user requirement is free. Gemini's free tier is the most
generous from a major lab. Behind a `Generator` interface to swap.

**Implemented.** `wca_rag/generator.py` defines the `Generator` ABC
with one method (`generate(system_prompt, user_prompt) -> GenerationResult`)
and one property (`model_name`). Default implementation:
`GeminiGenerator` wraps `google-genai` (the current SDK;
`google-generativeai` is deprecated and must not be used).
Lazy-imports the SDK so the rest of the package doesn't pull it in
for indexing-only commands. Temperature defaults to 0.1 —
citation-heavy QA is not creative writing.

**API key.** `GEMINI_API_KEY` read from environment, with `.env`
support via `python-dotenv` (soft dependency — falls back to plain env
vars if not installed). `.env.example` checked in. `.env` gitignored.

### 3.4 Chunking strategy 🟢

**Decided:** custom parser. One chunk per top-level regulation, including its
`+`/`++` annotations and nested children.

The corpus has explicit hierarchical structure:
- Articles (`Article 1`, `Article 11`, `Article A`, ...)
- Top-level regulations (`1a)`, `11e)`, `A2)`, ...)
- Annotations (`1c+)`, `11e++)` — same logical unit, just clarifications/examples)
- Nested regulations (`11e1)`, `11e2)`, `11e2a)` — children of `11e`)

**Chunking unit:** for each top-level regulation (e.g. `11e`), bundle into
one chunk:
- the regulation itself
- all its `+`-suffixed annotations (`11e+`, `11e++`, `11e++++++`)
- all its numbered children (`11e1`, `11e1+`, `11e2`, `11e2a`, ...)

**Why:** these are logical units. Splitting `11e` from `11e1` would break
retrieval — the children are conditions/exceptions to the parent. Keeping
them together preserves meaning and makes citations natural.

**Rejected alternatives:**
- *Generic markdown splitter (e.g. `RecursiveCharacterTextSplitter`).* Splits by
  character count, ignores semantic structure. Standard tutorial choice; bad
  for legal/regulatory text.
- *One chunk per leaf regulation.* Smaller, more precise chunks but loses
  context (a child rule is meaningless without the parent).
- *One chunk per article.* Too large; a single article like Article 11 is
  ~5k words — would dominate retrieval and dilute precision.
- *Embedding-based semantic chunking* (split on similarity drops between
  consecutive sentences). Loses regulation IDs as chunk anchors, which our
  citation strategy depends on. May revisit narrowly as a sub-splitter for
  oversized chunks (`9f`, `A1`, `A7`, `E2`) if eval shows they hurt
  retrieval. See `CONCEPTS.md` for details.

**Outcome (post-implementation):**
- **108 chunks** across all 16 articles (initial estimate was ~150).
- Distribution skewed by article: Article 9 has 15 chunks (densest), Article C has 1 (shortest).
- 4 oversized chunks (`9f`, `A1`, `A7`, `E2`) kept intact — coherent units, splitting would harm retrieval.
- 2 orphan annotations (`5c+`, `9q+`) — annotations whose parent regulation doesn't exist in the source, likely an artifact of regulations being deleted without renumbering. Handled with audit warnings during parsing; treated as standalone chunks.

### 3.5 Metadata schema 🟢

**Implemented per-chunk fields** (matches `wca_rag/parser.py`):

```python
{
    "regulation_id": "11e",                 # primary citation key
    "article": "11",
    "article_title": "Incidents",
    "full_path_id": "11 > 11e",             # human-readable path for display/debugging
    "label": "CLARIFICATION" | None,        # top-level regulation's label, if any
    "is_annotation": False,                 # True for +/++/+++ rules
    "depth": 1,
    "parent_id": None,
    "cross_references": [                   # extracted from regulation text
        {"type": "regulation", "id": "11i2"},
        {"type": "article", "id": "I"},
    ],
    "char_count": 412,
    "text_hash": "a3f1...",                 # sha1 of body, for change detection
    "source_version": "April 1, 2026",
    "text": "...",                          # raw chunk body
    "text_for_embedding": "...",            # body + prepended article/regulation header
}
```

Used for:
- **Citation** in answers (`regulation_id`).
- **Filtering** (`where article == "11"` for incident-only queries).
- **Debugging** retrieval (human-readable chunk inspection via `full_path_id`).
- **Cross-reference-aware retrieval** via `cross_references`. The
  `type` distinguishes regulation references (one chunk) from article
  references (many chunks); these expand differently.
- **Change detection on re-parse** via `text_hash` — diff two `chunks.jsonl`
  files across corpus versions.

**`text` vs `text_for_embedding` — important distinction.** The parser emits
two versions of the chunk body:

- `text` is the raw body of the regulation (just the rule text and its
  annotations/children).
- `text_for_embedding` prepends a stable header: `Article {N}: {title}\nRegulation {id}\n\n{text}`.

The embedder reads `text_for_embedding`, not `text`. The prompt assembler
that builds the LLM context reads `text` plus uses `regulation_id` for the
citation marker. This split exists because a chunk like `e2)` is meaningless
to an embedding model in isolation — the prepended header gives the
embedding model the article context it needs to place the chunk in semantic
space. See `CONCEPTS.md` → "Metadata-prepend" for the underlying pattern.

**Cross-reference graph data** (useful when revisiting link-aware retrieval):
- 55 / 108 chunks contain at least one cross-reference (~51%).
- 159 total edges, 124 unique targets — sparse graph with a few hubs.
- Top referenced targets: `2k` (disqualification), `9u` (end of competition),
  `3l` (logos), `10f` (misalignment limits), `Article I`, `Article A`, `11i1`
  (incorrect scrambles).

### 3.6 Interface 🟢

`.py` scripts only. No notebooks. Intermediate artifacts dumped to `data/`
for inspection.

- `python -m wca_rag.index` — build chunks + embeddings, persist to Chroma.
- `python -m wca_rag.query "your question"` — retrieve + generate.
- `python -m wca_rag.eval` — run golden set and report metrics.

### 3.7 Retriever 🟢

**Implemented.** `wca_rag/retriever.py` exposes a `Retriever` class with
a `from_disk()` classmethod and a `retrieve(query, k=5)` method. CLI
entry point: `wca_rag/query.py`, run via
`python -m wca_rag.query "your question"`.

**Algorithm.** Brute-force cosine similarity over the full matrix:

```python
query_vec = embedder.encode_query(query)         # (384,)
scores = embeddings @ query_vec                  # (108,) cosine sim
top_k_idx = np.argsort(scores)[-k:][::-1]
```

Three lines, no index structure. At 108 chunks this is several orders
of magnitude below where an ANN index would help; brute force is both
the fastest and the simplest option. See `CONCEPTS.md → Brute-force
retrieval` for the scaling argument.

**Why NumPy and not ChromaDB.** ChromaDB will replace this when (a) we
need metadata filtering (e.g. "only Article 11 chunks") or (b) the
corpus grows past ~10k chunks. Until then, NumPy is dependency-free
and the math is exposed for learning. The `Retriever` API is small
enough that swapping its internals later won't ripple into callers.

**Defensive checks at construction.** The `Retriever` constructor
validates that:
- embeddings row count == len(chunk_ids),
- every chunk_id is present in chunks.jsonl,
- embeddings dim == embedder.embedding_dim (catches a model swap with a
  stale index).

These fail loud at construction rather than silently producing wrong
retrievals at query time.

**Output: `RetrievalHit` dataclass.** Each hit carries `rank`, `score`
(cosine in [-1, 1]), `regulation_id`, and the full `chunk` dict. The
generator stage reads `chunk["text"]` for the LLM prompt and
`regulation_id` for the citation marker — note: `text`, not
`text_for_embedding` (the prepended header is for the embedder only,
not for human display or LLM context). See §3.5.

### 3.8 Generator pipeline 🟢

**Implemented.** Three modules + one CLI entry point:

- `wca_rag/prompts.py` — `SYSTEM_PROMPT` constant +
  `assemble_user_prompt(question, hits)` helper.
- `wca_rag/generator.py` — `Generator` ABC +
  `GeminiGenerator` impl. Returns `GenerationResult` (text + model name
  + optional token counts).
- `wca_rag/pipeline.py` — `Pipeline` class composing a Retriever and a
  Generator into one `ask(question, k=8)` call.
- `wca_rag/ask.py` — CLI entry point: `python -m wca_rag.ask "your
  question"`. Prints answer; `--show-hits` flag also prints retrieved
  chunks for debugging.

**Default k = 8 at query time.** Retriever default remains k=5 (used by
`query.py` for retrieval-only debugging). Generator pipeline defaults
to k=8 — multi-regulation questions are common in real delegate
incidents, and at this corpus size + chunk length the lost-in-the-middle
risk is negligible (~6k tokens of context). `Pipeline.ask()` accepts a
`k` override; the CLI exposes it as `-k`.

**System prompt design.** Three load-bearing decisions:

1. **Three-way refusal taxonomy.** ANSWER (chunks fully cover question)
   / PARTIAL (some coverage, some gaps — answer covered part, state
   gaps explicitly) / REFUSE (no coverage). Model labels its response
   with the mode. Binary answer/refuse was rejected because real
   delegate questions often touch multiple regulations and forcing a
   binary choice biases toward over-refusing or over-answering.

2. **Mandatory verbatim quoting + justification.** For every claim, the
   model must (a) state the conclusion, (b) quote the supporting
   regulation text in quotation marks, (c) explain in one sentence how
   the quote supports the conclusion. Step (c) is the real
   anti-confabulation safeguard — quoting alone is not enough because
   models will quote adjacent-but-irrelevant text. Producing a coherent
   justification is hard when the quote does not actually fit, which
   forces the model toward PARTIAL or REFUSE.

3. **Inline citations with bracketed regulation_id.** `[11e]`,
   `[11e++]`, `[A2b]` immediately after the supported claim. Granular
   enough to audit per-claim. Examples in the prompt deliberately
   include `+` suffixes so the model produces them correctly.

**Prompt assembly format.** Each retrieved chunk wrapped in:

```
<regulation id="11e" article="11">
{chunk["text"]}
</regulation>
```
XML-ish wrapper chosen over plain `[Regulation 11e]` headers for two
reasons: clearer chunk boundaries (LLMs respect XML-shaped tags well),
and the citation instruction becomes literal — "cite using the `id`
attribute". Uses `chunk["text"]`, NOT `chunk["text_for_embedding"]`.

**Generator interface: batch only in v1.** Streaming would complicate
citation parsing (you can't validate citations until the answer is
fully generated) and the CLI does not benefit. The ABC can grow a
`stream()` method later without breaking `generate()`.

**Citation granularity: Option A (claim-level) 🟢.** The model cites
the most specific id whose verbatim text appears in a retrieved
chunk, including sub-regulations nested inside chunk bodies (e.g.
`[A6c]` cited from inside chunk `A6`'s body, not just `[A6]`).
Sub-regulation ids appear in chunk text but not as `id` attributes
on `<regulation>` tags, so the constraint is now phrased as "cite
ids whose text is present in a retrieved chunk" — the verbatim quote
is the load-bearing audit, not the tag attribute.

Implications:
- **`prompts.py`** patched with the explicit "prefer narrowest id"
  preference and a worked example using a sub-rule citation.
- **Eval harness** scores citation accuracy at claim-level (no
  sub-rule collapsing). Recall@k is the only place sub-rule ids are
  collapsed to parent chunks, because the retriever returns chunks.
- **Confabulation check** scans retrieved chunk bodies for each cited
  id; if not present, the model invented it — hard failure
  regardless of mode or accuracy.
- **Quote validity** is measured programmatically (substring match
  against retrieved chunks, whitespace-normalized) and replaces the
  eyeball check the verbatim-quote rule otherwise relies on.

See `CONCEPTS.md` ("Chunk-level vs claim-level citation granularity")
for the full reasoning.

---

## 4. End-to-end pipeline

### Indexing (offline)

```
data/raw/wca-regulations.md
        │
        ▼
[parser]   ──► chunks + metadata (data/chunks.jsonl)         ✅
        │
        ▼
[embedder] ──► (data/embeddings.npy,                         ✅
                data/chunk_ids.json,
                data/embeddings.meta.json)
```

### Query (online)

```
user question
        │
        ▼
[embedder.encode_query]  ──► query vector                    ✅
        │
        ▼
[retriever]              ──► top-k hits + metadata           ✅
        │
        ▼
[prompt assembler]       ──► system prompt + chunks + question  ✅
        │
        ▼
[generator]              ──► answer with regulation IDs cited   ✅
```

---

## 5. Evaluation strategy 🟢

**Built.** `wca_rag/eval.py` is a three-phase, cache-on-disk harness.
Each phase writes to `evals/results/run-<TIMESTAMP>/` so the
expensive phases don't have to re-run when the scorer changes.

### Why three phases

Retrieval is fast, deterministic, and free (local embedder).
Generation is rate-limited and quota-eating (Gemini free tier:
~10 RPM, ~250 RPD as of 2026-04). Scoring is the part that gets
iterated on. If all three ran in one pass, every scorer change
would cost generator quota. By caching phase 1 and phase 2 outputs,
phase 3 becomes a pure function over those artifacts and re-runs
in milliseconds.

- **Phase 1 — Retrieve.** Run the retriever for each question with
  configurable k. No LLM calls. Output: `hits.json`.
- **Phase 2 — Generate.** Load cached hits, assemble user prompt,
  call generator. Sleep between calls to respect RPM. Fail loudly
  on 429 — silent retries hide systematic problems.
  Output: `answers.json`.
- **Phase 3 — Score.** Pure function over the cached artifacts.
  Output: `metrics.json` + `summary.txt`.

### Metrics (v2)

Phase 3 computes per-question metrics across three families:
retrieval quality, answer quality, and confidence calibration.
Aggregated to run-level means + a strict "correct" rollup.

**Retrieval.**

- **Recall@k** — fraction of expected ids whose parent chunk is in
  the retrieved set. Sub-rule expected ids are collapsed to parent
  chunks for this check (the retriever returns chunks, not sub-rules).
- **Recall@N curve (N = 1, 3, 5, k)** — same metric, sliced at
  multiple ranks. Re-uses the already-retrieved hits; computing the
  curve costs nothing. Phase 1 retrieves `eval_k` chunks per question
  (default 10) so the curve can extend beyond the generator's `k`
  without changing generation cost. Plotted as one line for "all"
  and per-mode lines for ANSWER and PARTIAL. REFUSE excluded
  (recall is 1.0 by convention).
- **MRR** — mean reciprocal rank of the first expected hit. Captures
  ordering quality within top-k that plain recall ignores.
- **Retrieval score signals** — top-1 cosine, top1-top2 margin, top-k
  mean. Cheap retrieval-derived confidence indicators; also useful
  diagnostics for "is the question out-of-domain?" (low top-1 score
  + high self-confidence is the dangerous pattern).

**Answer quality.**

- **Citation accuracy** — fraction of expected ids cited in the
  answer. Claim-level under Option A; no sub-rule collapsing.
- **Citation precision** — fraction of cited ids that were expected.
  Often <1.0 by design when the golden set lists representative ids
  rather than exhaustive ones.
- **Quote validity** — fraction of `"..."` spans in the answer that
  substring-match a retrieved chunk after whitespace normalization.
  This is the programmatic replacement for eyeball-checking the
  verbatim-quote requirement.
- **Citation-quote alignment (faithfulness proxy)** — stricter than
  quote validity: for each cited id that has a quote in the same
  paragraph, does that quote substring-match THAT id's chunk body
  specifically (not just any retrieved chunk)? Catches the failure
  "quote from chunk A, cite chunk B" that quote_validity misses.
  Honest naming: this is a programmatic proxy, not real NLI
  faithfulness. Only quote-backed citations enter the denominator
  to avoid penalizing PARTIAL pointer-citations that legitimately
  lack per-id quotes.
- **Mode accuracy + confusion matrix** — declared mode vs expected
  mode (ANSWER / PARTIAL / REFUSE). Confusion matrix shows where
  collapses happen (e.g. ANSWER -> PARTIAL). Asymmetric error costs
  (REFUSE -> ANSWER far worse than ANSWER -> REFUSE) are visible by
  eye on the heatmap; no explicit weighting in the metric.
- **Confabulation count** — cited ids whose text is not present in
  any retrieved chunk. Hard failure regardless of other metrics.

**Confidence & calibration.**

- **Self-reported confidence** — model emits a final
  `Confidence: 0.XX` line per response. Parsed by the harness; `None`
  if the line is missing or malformed (drift detection — visible in
  the summary as `n_missing_confidence`).
- **Confidence threshold summary** — at thresholds {0.9, 0.8, 0.7},
  how many questions are above/below, and what's the empirical
  accuracy in each bucket? Drives the "skip review above T"
  decision. "Correct" defined strictly: `mode_match AND
  citation_accuracy == 1.0 AND no_confabulation`.
- **Strict correct count** — same definition as above, run-level
  aggregate.

### Plots

Six PNGs generated to `<run_dir>/plots/` on every full run AND every
`--score-only`. Pure functions over `metrics.json`. Plot failures are
logged but never raise — phase 3 already wrote metrics, plot bugs
should not lose them.

- **reliability_diagram.png** — confidence-bin -> empirical accuracy,
  with the diagonal "perfect calibration" reference overlaid. Bars
  above the diagonal = underconfident; below = overconfident. The
  plot that directly answers the motivating question ("can I use
  confidence as a review threshold?").
- **mode_confusion_matrix.png** — annotated heatmap.
- **recall_at_k_curve.png** — recall@1/3/5/k, overall and per-mode.
- **per_question_scorecard.png** — heatmap of questions x metrics,
  red->green by score. Spots which questions drag aggregates down.
- **retrieval_score_distribution.png** — top-1 cosine histogram split
  by expected mode. Validates the hypothesis that REFUSE cases
  should have low top-1.
- **confidence_vs_correctness.png** — scatter, point size = top-1
  retrieval score. Useful when n is small (<= 30) and binned
  reliability diagrams are noisy.

Stack: seaborn (`whitegrid` theme) on matplotlib. No pandas
dependency — data passed as `dict[str, list]`.

### Run artifacts

```
evals/results/run-<TIMESTAMP>/
    config.json     # golden_set hash, prompt hash, model, k, eval_k, rpm, timestamp
    hits.json       # phase 1 output (eval_k chunks per question; generator sees top-k)
    answers.json    # phase 2 output
    metrics.json    # phase 3 output (per-question scores + aggregate)
    summary.txt     # human-readable aggregate, written every run
    plots/          # six PNGs, regenerated by --score-only
```

`config.json` records `golden_set_hash` (sha256 of sorted YAML),
`prompt_hash` (sha256 of `SYSTEM_PROMPT`), and model name. Two runs
with matching hashes + model are comparable; mismatches mean you're
comparing tuning experiments, not the same system.

### CLI

```
python -m wca_rag.eval                       # full run; writes summary.txt + plots/
python -m wca_rag.eval --questions q01,q02   # subset
python -m wca_rag.eval --rpm 10              # rate limit override
python -m wca_rag.eval --k 8                 # top-k passed to generator
python -m wca_rag.eval --eval-k 10           # retrieval depth for recall curve
python -m wca_rag.eval --summary             # also print aggregate to stdout
python -m wca_rag.eval --score-only RUN_ID   # rescore cached run; rewrites
                                             # metrics.json + summary.txt + plots/
```

`summary.txt` and `plots/` are always (re)written, full run and
`--score-only`. `--summary` only controls whether the aggregate is
also printed to stdout. `--score-only` overwrites in place — fine
while iterating on the scorer; if scorer-versioned output becomes
useful, add a `--tag` flag to write `metrics.<tag>.json` instead.

`--eval-k` decouples retrieval depth from the generator's `k`. Phase
1 retrieves `eval_k` chunks per question; phase 2 still feeds only
the top `k` to the generator (cost unchanged). The extra chunks
power the recall@N curve without re-running retrieval.

### Prompt-contract implication

The harness reads two model-emitted contracts: the `[MODE]` label at
the start of the response and the `Confidence: 0.XX` line at the end.
Both are parsed strictly:

- Missing/malformed `[MODE]` -> declared mode is `None`, shows up as
  `MISSING` in the confusion matrix.
- Missing/malformed `Confidence:` line -> `self_confidence` is `None`;
  question excluded from threshold buckets and reported as
  `n_missing_confidence` in the summary.

Both being visible in the summary is intentional — silent fallbacks
would hide prompt drift, exactly the failure mode the strict parser
exists to detect.

### Golden set

`evals/golden_set.yaml`. Schema:

```yaml
- id: q01                       # string, sortable
  question: "..."
  expected_mode: ANSWER         # ANSWER | PARTIAL | REFUSE
  expected_ids: [A6c]           # claim-level granularity (Option A)
  notes: "..."                  # human-readable rationale
```

---

## 6. Repo layout 🟢

```
wca-rag/
├── CLAUDE.md               # at repo root
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md          # forward-looking plan; source of truth for "what's next"
│   └── CONCEPTS.md         # learning notes — RAG concepts as we hit them
├── README.md
├── pyproject.toml          # or environment.yml for conda
├── .gitignore
│
├── wca_rag/                # source package
│   ├── __init__.py
│   ├── parser.py           # markdown → chunks + metadata
│   ├── embedder.py         # Embedder interface + impls
│   ├── store.py            # Chroma wrapper
│   ├── retriever.py        # query → top-k chunks
│   ├── generator.py        # Generator interface + impls (Gemini, Anthropic, ...)
│   ├── pipeline.py         # orchestration
│   ├── prompts.py          # prompt templates
│   ├── index.py            # entry point: build the index
│   ├── query.py            # entry point: retrieval-only debug
│   ├── ask.py              # entry point: one question end-to-end
│   └── eval.py             # entry point: three-phase eval harness
│
├── data/
│   ├── raw/
│   │   └── wca-regulations.md
│   ├── chunks.jsonl        # parsed chunks (gitignored)
│   ├── embeddings/         # gitignored
│   └── chroma/             # gitignored
│
├── evals/
│   ├── golden_set.yaml
│   └── results/            # run outputs (gitignored)
│
└── scripts/
    └── inspect_chunks.py   # debugging helpers
```

Light footprint. No `tests/` directory yet — add when the codebase
justifies it.
