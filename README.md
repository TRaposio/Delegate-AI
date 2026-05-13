# WCA RAG

A retrieval-augmented Q&A tool over the [WCA Regulations](https://www.worldcubeassociation.org/regulations/),
aimed at helping WCA Delegates handle incidents at competitions. Ask
"this incident happened, how should I handle it?" and get an answer
that cites the specific regulations applied, with verbatim quotes from
the source.

This is also — equally — a **learning project**. I built it from
primitives (no LangChain, no LlamaIndex) to actually understand how
RAG works end-to-end: chunking, embeddings, vector similarity, prompt
engineering for grounded generation, refusal taxonomies, and eval
harness design. Architectural decisions and tradeoffs are documented
in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and the underlying
concepts are explained in [`docs/CONCEPTS.md`](docs/CONCEPTS.md).

The corpus is small (~22k words, 108 chunks), local execution only, no
deployment.

---

## Setup

Requires Python 3.11+ and conda (or any equivalent env manager).

```bash
# 1. Clone and enter the repo
git clone https://github.com/TRaposio/Delegate-AI.git
cd Delegate-AI

# 2. Create the environment
conda env create -f environment.yml
conda activate wca-rag

# 3. Configure API keys
cp .env.example .env
# Edit .env and fill in:
#   GEMINI_API_KEY  — free key from https://aistudio.google.com/apikey
#   HF_TOKEN        — token from https://huggingface.co/ (for the embedder)

# 4. Build the index (parses the corpus and computes embeddings)
python -m wca_rag.parser
python -m wca_rag.index

# 5. Ask a question
python -m wca_rag.ask "Can a competitor stop the timer while still touching the puzzle?"

# Optional: also print the retrieved chunks for debugging
python -m wca_rag.ask "..." --show-hits

# Run the eval harness against the golden set
python -m wca_rag.eval --summary
```

All scripts run from the repo root as `python -m wca_rag.<script>`.
Intermediate artifacts (chunks, embeddings, run results) land under
`data/` and `evals/results/` and are gitignored.

---

## Architecture

Classic RAG: chunk → embed → top-k vector retrieval → LLM generation
with mandatory citations. Each component sits behind an interface
(`Embedder`, `Retriever`, `Generator`) so swapping providers is a
one-file change.

**Indexing (offline):**

```
data/raw/wca-regulations.md
        │
        ▼
[parser]    ──► chunks + metadata (data/chunks.jsonl)
        │
        ▼
[embedder]  ──► (data/embeddings.npy, data/chunk_ids.json, ...meta.json)
```

**Query (online):**

```
user question
        │
        ▼
[embedder.encode_query]   ──►  query vector
        │
        ▼
[retriever]               ──►  top-k hits + metadata
        │
        ▼
[prompt assembler]        ──►  system prompt + chunks + question
        │
        ▼
[generator]               ──►  answer with regulation IDs cited
```

**Key choices:**

| Component | Choice | Why |
|---|---|---|
| Chunking | Custom parser, one chunk per top-level regulation (e.g. `11e` + its `+`/`++` annotations + numbered children `11e1`, `11e2a`, ...) | Generic character splitters break the semantic structure of regulatory text. Children are conditions/exceptions to the parent — keep them together. |
| Embedding model | `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local) | Quality sweet spot at the local-and-free price point for a 22k-word English corpus. |
| Vector store | NumPy matrix on disk (Chroma planned) | At 108 chunks, full pairwise cosine is trivially fast. Built it from scratch first to actually understand the math. |
| Generation LLM | Gemini 2.5 Flash (free tier) | Strong free tier from a major lab. Behind a `Generator` ABC for easy swap. |
| Prompt design | Three-way refusal taxonomy (`ANSWER` / `PARTIAL` / `REFUSE`) + mandatory verbatim quoting + inline `[regulation_id]` citations | Refusal as a first-class output prevents confident wrong answers in a regulatory context. Verbatim quotes + a one-sentence justification are the load-bearing anti-confabulation mechanism. |

Full rationale, rejected alternatives, and references in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repo layout

```
Delegate-AI/
├── CLAUDE.md                 # instructions for Claude when working in this repo
├── README.md                 # this file
├── environment.yml           # conda environment
├── .env.example              # API key template (copy to .env)
│
├── docs/
│   ├── ARCHITECTURE.md       # source of truth for design decisions
│   ├── CONCEPTS.md           # learning notes on RAG concepts
│   └── OPEN_QUESTIONS.md     # design questions still open
│
├── wca_rag/                  # source package
│   ├── parser.py             # markdown → chunks + metadata
│   ├── embedder.py           # Embedder ABC + sentence-transformers impl
│   ├── retriever.py          # query → top-k chunks (NumPy similarity)
│   ├── generator.py          # Generator ABC + Gemini impl
│   ├── prompts.py            # system prompt + user-prompt assembler
│   ├── pipeline.py           # composes Retriever + Generator into ask()
│   ├── index.py              # entry point: build the index
│   ├── query.py              # entry point: retrieval-only debugging
│   ├── ask.py                # entry point: full RAG pipeline (CLI)
│   └── eval.py               # entry point: run golden set + score
│
├── data/
│   ├── raw/wca-regulations.md   # source corpus (tracked)
│   ├── chunks.jsonl             # parsed chunks (gitignored)
│   ├── embeddings.npy           # gitignored
│   └── chroma/                  # gitignored (planned)
│
├── evals/
│   ├── golden_set.yaml          # hand-written incident questions (tracked)
│   └── results/                 # one dir per run (gitignored)
│
└── scripts/
    ├── inspect_chunks.py        # debugging helpers
    └── inspect_embeddings.py    # alignment + self-similarity sanity checks
```

---

## Example: question, answer, and scoring

**Question** (from `evals/golden_set.yaml`, id `q01`):

> Can a competitor stop the timer while still touching the puzzle?

**Answer** (mode label, verbatim quote, justification, inline citation):

```
[ANSWER]

No. The competitor must release the puzzle before stopping the timer;
stopping while still touching the puzzle incurs a +2 second penalty.
"If the competitor stops the timer without first releasing the puzzle,
the result is a +2 penalty" [A6c] — this directly addresses the
scenario in the question and prescribes the +2 penalty.
```

**Scoring.** The eval harness produces three artifacts per run under
`evals/results/run-<timestamp>/`: `hits.json` (retrieval), `answers.json`
(generation), `metrics.json` (scoring). For `q01` against
`expected_mode: ANSWER`, `expected_ids: [A6c]`:

| Metric | Value | What it checks |
|---|---|---|
| `mode_match` | `true` | Declared `[ANSWER]` matches expected `ANSWER`. |
| `recall_at_k` | `1.0` | `A6c`'s parent chunk `A6` was in the top-k retrieved. |
| `citation_accuracy` | `1.0` | All expected ids (`A6c`) were cited. |
| `citation_precision` | `1.0` | All cited ids (`A6c`) were expected. |
| `confabulated_ids` | `[]` | Every cited id appears verbatim in some retrieved chunk's body. |
| `quote_validity` | `1.0` | The `"..."` span string-matches a retrieved chunk after whitespace normalization. |

Run-level aggregates are means of the per-question scores plus a
mode-confusion matrix (rows = expected, cols = declared) so you can
see e.g. how often `ANSWER` collapsed to `PARTIAL`. See
[`wca_rag/eval.py`](wca_rag/eval.py) for the full scoring contract.
