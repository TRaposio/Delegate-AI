# Architecture

Status legend: 🟢 decided · 🟡 proposed (awaiting feedback) · 🔴 deferred (future)

This doc is the source of truth for architectural decisions. Update when
decisions move between statuses.

Last update: after first round of external feedback. Free-tier constraint is
hard (no trials, no paid tiers).

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
- Sources beyond the WCA Regulations (handbook, rulings — deferred).
- Paid APIs, including free trials.

---

## 2. Why RAG (and not alternatives)

The WCA Regulations corpus is ~22k words / ~30k tokens. Modern LLMs handle
200k+ context windows, so this fits.

| Approach | Pro | Con | Verdict |
|---|---|---|---|
| **Long-context stuffing** | No retrieval errors. | "Lost in the middle" attention degradation. Per-query cost when scaling. Defeats learning goal. | Possible baseline for sanity check. |
| **Classic RAG** | Cheap. Fast. Standard. Teaches the fundamentals. Scales when adding sources. | Retrieval miss = wrong answer. | 🟢 chosen for v1. |
| **Hybrid (BM25 + vector + rerank)** | Higher retrieval quality, especially for keyword-heavy queries. | More moving parts. | 🔴 v2 (priority 1). |
| **Agentic / iterative retrieval** | Handles multi-hop. | Slow, multiple LLM calls. | 🔴 v2/v3. |

Decision rationale: at this corpus size RAG is not strictly necessary, but the
project's primary purpose is learning, and we want extensibility for future
sources (handbook, rulings).

References:
- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Liu et al. 2023, *Lost in the Middle*
- Pinecone Learn (`pinecone.io/learn/`)

---

## 3. Component choices

### 3.1 Embedding model 🟢

**Decided:** `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local).

| Option | Size | Quality (MTEB avg) | Cost | Local feasibility |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 90MB | ~58 | free | great |
| **`bge-small-en-v1.5`** | 130MB | ~62 | free | great |
| `bge-large-en-v1.5` | 1.3GB | ~64 | free | painful on 2015 Mac |
| OpenAI `text-embedding-3-small` | API | ~62 | paid (excluded) |  |
| Voyage `voyage-3-lite` | API | strong | free trial only (excluded) |  |
| Gemini `text-embedding-004` | API | strong | free tier | possible alternative |

Rationale: free constraint, runs on a 2015 Mac, MTEB top 30. Quality gap to
larger models is unlikely to be the bottleneck given strong structural
features in the corpus (article numbers, IDs, headers). Reranking is the
better v2 lever than upgrading embeddings.

Behind an `Embedder` interface so swap to `text-embedding-004` for A/B is one
file change.

### 3.2 Vector store 🟢

**Decided:** numpy from-scratch.

For ~150 chunks, store embeddings as a single `(n_chunks, embedding_dim)`
numpy array on disk. Retrieval is `scores = E @ q` where `q` is the
unit-normalized query embedding. Top-k via `np.argpartition` or `np.argsort`.

Why this over Chroma/FAISS:
- **Educational value.** Cosine similarity becomes one line of code. You
  see the math, not a black box.
- **Zero dependencies.** numpy is already installed for everything else.
- **Plenty fast.** ~150 chunks × 384-dim float32 = ~230KB. Dot product is
  microseconds.
- **No persistence layer needed.** `np.save` / `np.load`.

Chroma/FAISS are valid but hide the part we want to understand. Revisit if
chunk count grows beyond 10k or filtering needs become complex.

### 3.3 Generation LLM 🟢

**Decided:** Google Gemini API, `gemini-2.5-flash`, free tier.

Free constraint excludes Anthropic, OpenAI, paid APIs, free trials. Among
free options:

| Option | Cost | Citation discipline | Notes |
|---|---|---|---|
| **Gemini 2.5 Flash (free tier)** | free at hobby scale | good | most generous free tier from a major lab |
| Groq (Llama 3.3 70B) | free at hobby scale | reasonable | free-tier ToS uncertain long-term |
| Local Llama 3.2 1B/3B (Ollama) | free | weak for nuanced citation | educational fallback |
| Local Llama 3.1 8B (Ollama) | free | reasonable | marginal on 2015 Mac, slow |

Risk acknowledged: external feedback flagged that "model obedience" matters
more than embedding quality for citation-heavy QA. Mitigations:
1. Strong prompt design with explicit refusal wording (see §5).
2. Golden-set evaluation will surface citation failures quantitatively.
3. If Gemini Flash quality is insufficient, fall back to Groq Llama 3.3 70B
   (still free) before considering paid options.

Behind a `Generator` interface to swap.

### 3.4 Chunking strategy 🟢

**Decided:** custom parser. One chunk per top-level regulation, including its
`+`/`++` annotations and nested children. Parent-path context prepended to
chunk text *before embedding*.

The corpus has explicit hierarchical structure:
- Articles (`Article 1`, `Article 11`, `Article A`, ...)
- Top-level regulations (`1a)`, `11e)`, `A2)`, ...)
- Annotations (`1c+)`, `11e++)` — same logical unit, just clarifications/examples)
- Nested regulations (`11e1)`, `11e2)`, `11e2a)` — children of `11e`)

**Chunking unit:** for each top-level regulation (e.g. `11e`), bundle
into one chunk:
- the regulation itself
- all its `+`-suffixed annotations (`11e+`, `11e++`, ...)
- all its numbered children (`11e1`, `11e1+`, `11e2`, `11e2a`, ...)

**Path context:** each chunk's *embedded text* is prepended with its
hierarchical path:

```
Article 11 — Incidents
Regulation 11e
[chunk body...]
```

The plain `text` field stored in metadata stays unprepended for citation.
Only the version sent to the embedder includes the path. This is a known
trick (a.k.a. contextual chunk headers) that materially improves retrieval
on hierarchical legal/regulatory text.

**Estimate:** ~150 chunks for the current corpus. Variable size from ~50
to ~2000 chars — fine because boundaries follow logic, not characters.

**Rejected alternatives:**
- *Generic markdown splitter* (e.g. `RecursiveCharacterTextSplitter`). Splits
  by character count, ignores structure. Bad for this corpus.
- *One chunk per leaf.* Too small, loses parent context.
- *One chunk per article.* Too large, dilutes retrieval precision.

### 3.5 Metadata schema 🟢

```python
{
    "regulation_id": "11e",              # primary citation key
    "article": "11",
    "article_title": "Incidents",
    "full_path_id": "11 > 11e",          # hierarchical breadcrumb
    "label": None,                       # CLARIFICATION | EXAMPLE | RECOMMENDATION | ADDITION | REMINDER | EXPLANATION
    "is_annotation": False,              # True for +/++/+++ rules
    "depth": 1,
    "parent_id": None,
    "cross_references": [],              # extracted from regulation text — phase 2
    "char_count": 412,
    "text_hash": "sha1:abc123...",       # for versioning / mismatch detection
    "source_version": "2026-04-01",
}
```

Used for:
- **Citation** in answers (`regulation_id`).
- **Filtering** (`where article == "11"`).
- **Debugging** retrieval (human-readable inspection).
- **Versioning** via `text_hash` and `source_version`.
- **Future link-aware retrieval** via `cross_references`.

`cross_references` extraction is best-effort phase 2. The field is included
now (as `[]`) so the schema is forward-compatible.

### 3.6 Interface 🟢

`.py` scripts only. No notebooks. Intermediate artifacts dumped to `data/`
for inspection.

- `python -m wca_rag.index` — build chunks + embeddings, persist to disk.
- `python -m wca_rag.query "your question"` — retrieve + generate.
- `python -m wca_rag.eval` — run golden set, report metrics.

Streamlit UI deferred. CLI first.

### 3.7 Cross-reference handling 🔴

Deferred. v1 ignores cross-references between regulations. Measure impact
via golden set. If meaningful, implement 1-hop link expansion in v2.

### 3.8 Reranking 🔴

Deferred to v2 priority 1. External feedback flagged this as the most likely
v2 quality win. Plan: add a cross-encoder reranker (e.g.
`BAAI/bge-reranker-base`, free, runs locally) over the top-N retrieved
chunks before passing top-k to the generator. Keep the retriever interface
stable so this is a drop-in addition.

---

## 4. End-to-end pipeline

### Indexing (offline)

```
data/raw/wca-regulations.md
        │
        ▼
[parser]    ──► structured chunks + metadata (data/chunks.jsonl)
        │
        ▼
[embedder]  ──► (n, d) float32 array (data/embeddings.npy)
                + parallel chunks file (data/chunks.jsonl)
```

### Query (online)

```
user question
        │
        ▼
[embedder]            ──► query vector (d,)
        │
        ▼
[numpy retriever]     ──► top-k chunks + metadata
                          (k tbd, start at 5)
        │
        ▼
[prompt assembler]    ──► system prompt + chunks + question
        │
        ▼
[Gemini generator]    ──► answer with regulation IDs cited
```

---

## 5. Prompt design 🟢

The single biggest lever for citation quality. Key patterns:

### Refusal pattern

The prompt explicitly instructs the model:

> If the answer is not explicitly supported by the retrieved regulations,
> respond: "No applicable regulation found in the provided context."

This wording is more effective than the common "if unsure, say I don't know"
because it gives a concrete output the model can produce, and ties the
condition to the retrieved context (not the model's general knowledge).

### Citation format

**Inline per-claim citations**, not footnotes or end-lists:

> "The competitor must stop the attempt [11e]."

Why inline:
- Easier to verify during competition use (read claim, read ID, look up rule).
- Reduces "citation drift" where the citation list is divorced from the
  claims it supports.

### Quote vs paraphrase

Default to short paraphrase. Quote the regulation verbatim only when:
- The decision hinges on exact wording (e.g. "must" vs "should").
- Ambiguity exists between paraphrase and original.

Never quote full regulations — bloats the answer and reduces usability under
competition time pressure.

### System prompt skeleton (v1)

```
You are an assistant for WCA Delegates handling competition incidents.
Your job: given a delegate's incident description and the relevant WCA
regulations retrieved for it, produce a concise ruling that cites the
specific regulations applied.

Rules:
- Use ONLY the regulations provided below as the basis for your answer.
- Cite regulation IDs inline in square brackets (e.g. [11e], [A3c1]).
- If the provided regulations do not cover the incident, respond exactly:
  "No applicable regulation found in the provided context."
- Prefer concise paraphrase. Quote regulation text verbatim only when
  exact wording matters for the decision.
- Do not speculate beyond what the provided regulations say.

Retrieved regulations:
{chunks}

Incident:
{question}
```

To be iterated on against the golden set.

---

## 6. Evaluation strategy 🟢

### Golden set

`evals/golden_set.yaml`, ≥20 hand-written incident questions to start,
expanding over time. Three case types:

1. **Standard cases** — clear regulation applies. `expected_rules` is set.
2. **Negative cases** — no regulation applies. `expected_answer` is the
   refusal string. Tests anti-hallucination.
3. **Ambiguous cases** — multiple regulations could apply or interpretations
   conflict. `expected_rules` lists all that should be surfaced.

Coverage diversity > raw count. Aim for spread across articles and
difficulty levels.

### Metrics for v1

**Retrieval:**
- **Recall@k**: of expected regulation IDs, how many are in the top-k retrieved chunks?
- **MRR (Mean Reciprocal Rank)**: average of `1 / rank_of_first_correct_hit`.
  Tells us *how high* the right rule is ranked, not just whether it's in top-k.

**Generation:**
- **Citation accuracy**: do the answer's cited IDs match `expected_rules`?
- **Refusal accuracy**: on negative cases, does the model produce the refusal string?

**Manual:**
- 1–5 quality grade by the user on a sample, with rubric.

### Versioning discipline

Every eval run records:
- `source_version` of the regulations used.
- Git commit hash of the code.
- Timestamp.

Eval results are not compared across `source_version` without an explicit
flag. Prevents "improvements" that are actually corpus changes.

### Metrics deferred to v2

- Faithfulness (does the answer follow from retrieved context?)
- Answer relevance
- LLM-as-judge automated grading
- RAGAS / TruLens / DeepEval framework integration

---

## 7. Repo layout 🟢

```
wca-rag/
├── CLAUDE.md
├── ARCHITECTURE.md
├── OPEN_QUESTIONS.md
├── README.md
├── environment.yml
├── .gitignore
│
├── wca_rag/                # source package
│   ├── __init__.py
│   ├── parser.py           # markdown → chunks + metadata
│   ├── embedder.py         # Embedder interface + impls
│   ├── store.py            # numpy-backed vector store
│   ├── retriever.py        # query → top-k chunks
│   ├── generator.py        # Generator interface + impls (Gemini, ...)
│   ├── pipeline.py         # orchestration
│   ├── prompts.py          # prompt templates
│   ├── index.py            # entry point: build the index
│   ├── query.py            # entry point: ask one question
│   └── eval.py             # entry point: run golden set
│
├── data/
│   ├── raw/
│   │   └── wca-regulations.md
│   ├── chunks.jsonl        # parsed chunks (gitignored)
│   ├── embeddings.npy      # gitignored
│   └── chunk_index.json    # id → row index map (gitignored)
│
├── evals/
│   ├── golden_set.yaml
│   └── results/            # run outputs (gitignored)
│
└── scripts/
    └── inspect_chunks.py   # debugging helpers
```

No `tests/` directory yet — add when the codebase justifies it.

---

## 8. Future enhancements (🔴 deferred)

In rough priority order:

1. **Reranking** (cross-encoder over top-N).
2. **Hybrid retrieval** (BM25 + vector).
3. **Cross-reference / link-aware retrieval.**
4. **Multilingual interface** (translate question → English → answer → translate back).
5. **Additional sources** (Delegate Handbook, past WRC rulings) as separate sub-indexes.
6. **Streamlit UI.**
7. **LLM-as-judge automated evaluation** (RAGAS or similar).
8. **A/B test embedding models** (Gemini API embeddings vs `bge-small`).
