# Architecture

Status legend: 🟢 decided · 🟡 proposed (awaiting feedback) · 🔴 deferred (future)

This doc is the source of truth for architectural decisions. Update when
decisions move between statuses.

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

---

## 2. Why RAG (and not alternatives)

The WCA Regulations corpus is ~22k words / ~30k tokens. This is small enough
that several architectures are viable:

| Approach | Pro | Con | Verdict |
|---|---|---|---|
| **Long-context stuffing** (whole corpus in every prompt) | No retrieval errors. Model sees everything. | "Lost in the middle" attention degradation. Higher per-query cost. | Possible fallback / baseline. |
| **Classic RAG** (chunk → embed → top-k → LLM) | Cheap. Fast. Scales to more corpora later. Standard. | Retrieval miss = wrong answer. | 🟢 chosen for v1. |
| **Hybrid (BM25 + vector + rerank)** | Higher retrieval quality, especially for keyword-heavy queries (e.g. "Article 11i"). | More moving parts. | 🔴 v2. |
| **Agentic / iterative retrieval** | Handles multi-hop ("rule X cites rule Y...") | Slow, multiple LLM calls. | 🔴 v2/v3. |

**Decision (🟢):** classic RAG for v1, designed so hybrid is a natural extension.

References for further reading:
- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — original RAG paper.
- Liu et al. 2023, *Lost in the Middle* — context-window attention degradation.
- Pinecone's RAG learning section (`pinecone.io/learn/`).

---

## 3. Component choices

### 3.1 Embedding model 🟡

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

### 3.2 Vector store 🟡

**Proposed:** ChromaDB (file-based, local).

Alternatives considered:
- **FAISS** — Meta's library, the de-facto standard for local vector search. Faster, more low-level. Educational value: you build the index yourself. Tradeoff: more friction, separate metadata storage.
- **NumPy array + cosine similarity** — fully from-scratch. Pedagogically the most informative for ~150 chunks (literally `embeddings @ query.T`). Tradeoff: write everything yourself.
- **Qdrant / Weaviate** — production-grade vector DBs. Overkill for local use.
- **pgvector** — Postgres extension. User knows Postgres so this would be familiar, but adds setup overhead.

**Rationale for Chroma:** Python-native, no server, persists to disk, has a
sane Python API. For learning purposes, the vector-similarity math is so
simple at this corpus size that we can also write a from-scratch numpy version
as a learning exercise alongside Chroma — to be discussed.

### 3.3 Generation LLM 🟡

**Proposed:** Google Gemini API (`gemini-2.5-flash`, free tier).

| Option | Cost | Quality for citation/refusal | Notes |
|---|---|---|---|
| **Gemini 2.5 Flash (free tier)** | free at hobby scale | strong | requires API key |
| Anthropic Claude Haiku 4.5 | $1/$5 per MTok | strongest | trial credit only, then paid |
| OpenAI GPT-4o-mini | paid | strong | paid only |
| Groq (Llama 3.3 70B) | free at hobby scale | reasonable | free-tier ToS uncertain |
| Local Llama 3.2 1B (Ollama) | free | weak for nuanced citation | educational but quality concern |

**Rationale:** user requirement is free. Gemini's free tier is the most
generous from a major lab. Behind a `Generator` interface to swap.

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

**Outcome (post-implementation):**
- **108 chunks** across all 16 articles (initial estimate was ~150).
- Distribution skewed by article: Article 9 has 15 chunks (densest), Article C has 1 (shortest).
- 4 oversized chunks (`9f`, `A1`, `A7`, `E2`) kept intact — coherent units, splitting would harm retrieval.
- 2 orphan annotations (`5c+`, `9q+`) — annotations whose parent regulation doesn't exist in the source, likely an artifact of regulations being deleted without renumbering. Handled with audit warnings during parsing; treated as standalone chunks.

### 3.5 Metadata schema 🟢

**Implemented per-chunk metadata:**

```python
{
    "regulation_id": "11e",              # primary citation key
    "article": "11",
    "article_title": "Incidents",
    "is_annotation": False,              # True for +/++/+++ rules
    "depth": 1,
    "parent_id": None,
    "cross_references": ["11i2", "9l"],  # extracted from regulation text
    "char_count": 412,
    "source_version": "2026-04-01",
}
```

Used for:
- **Citation** in answers (`regulation_id`).
- **Filtering** (`where article == "11"` for incident-only queries).
- **Debugging** retrieval (human-readable chunk inspection).
- **Future link-aware retrieval** via `cross_references`.

**Diff from initial proposal:** the `label` field (`CLARIFICATION`, `EXAMPLE`, etc.)
was dropped from chunk-level metadata since labels apply to individual annotations
within a chunk, not to the chunk as a whole. The label text is preserved in the
chunk content itself.

### 3.6 Interface 🟢

`.py` scripts only. No notebooks. Intermediate artifacts dumped to `data/`
for inspection.

- `python -m wca_rag.index` — build chunks + embeddings, persist to Chroma.
- `python -m wca_rag.query "your question"` — retrieve + generate.
- `python -m wca_rag.eval` — run golden set and report metrics.

Streamlit UI deferred. CLI first.

### 3.7 Cross-reference handling 🔴

Deferred. Many regulations reference others. v1 ignores this. Measure impact
via golden set, then decide.

**Data from parser run** (useful starting point when this is revisited):
- 55 / 108 chunks contain at least one cross-reference (~51%).
- 159 total edges, 124 unique targets — sparse graph with a few hubs.
- Top referenced targets: `2k` (disqualification), `9u` (end of competition),
  `3l` (logos), `10f` (misalignment limits), `Article I`, `Article A`, `11i1`
  (incorrect scrambles).
- These hubs are good candidates for "always retrieve alongside" link-aware
  retrieval, when we get to v2.

---

## 4. End-to-end pipeline

### Indexing (offline)

```
data/raw/wca-regulations.md
        │
        ▼
[parser]  ──► structured chunks + metadata (data/chunks.jsonl)
        │
        ▼
[embedder] ──► vectors (data/embeddings/)
        │
        ▼
[chroma]   ──► persisted vector DB (data/chroma/)
```

### Query (online)

```
user question
        │
        ▼
[embedder] ──► query vector
        │
        ▼
[chroma top-k] ──► k chunks + metadata (k tbd, start at 5)
        │
        ▼
[prompt assembler] ──► system prompt + chunks + question
        │
        ▼
[generator] ──► answer with regulation IDs cited
```

---

## 5. Evaluation strategy 🟡

**Golden set:** `evals/golden_set.yaml`, 15–30 hand-written incident
questions with expected regulation IDs.

**Metrics for v1:**
- **Retrieval recall@k**: of expected regulation IDs, how many are in top-k retrieved chunks?
- **Citation accuracy**: does the final answer reference the expected IDs?
- **Manual quality grade**: 1–5 from the user, on a sample.

**Scoring nuance:** golden set lists `expected_rules` at sub-chunk granularity
(e.g. `11e++++`, `11j3`), but chunks are at top-level only (`11e`, `11j`).
When computing recall@k, sub-rule IDs must be collapsed to their parent chunk
ID before scoring, otherwise valid retrievals look like misses. Handled in
`wca_rag/eval.py`.

**Metrics deferred to v2:**
- Faithfulness (does the answer follow from retrieved context?)
- Answer relevance
- LLM-as-judge automated grading
- RAGAS / TruLens / DeepEval framework integration

---

## 6. Repo layout 🟢

```
wca-rag/
├── CLAUDE.md               # at repo root
├── docs/
│   ├── ARCHITECTURE.md
│   ├── OPEN_QUESTIONS.md
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
│   ├── query.py            # entry point: ask one question
│   └── eval.py             # entry point: run golden set
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

---

## 7. Future enhancements (🔴 deferred)

In rough priority order:

1. **Hybrid retrieval (BM25 + vector + reranker).** Likely the single biggest quality win.
2. **Cross-reference / link-aware retrieval.** Follow regulation references in retrieved chunks.
3. **Multilingual interface.** Translate question → English → answer → translate back.
4. **Additional sources.** Delegate Handbook, past WRC rulings.
5. **Streamlit UI.**
6. **LLM-as-judge automated evaluation** (RAGAS or similar).
7. **Prompt caching** for the system prompt + retrieved chunks (cost optimization once on a paid API).
8. **Re-indexing automation** when WCA publishes a new regulations version.
