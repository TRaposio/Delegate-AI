# CLAUDE.md

Instructions for Claude when working in this repository.

## Project

A retrieval-augmented Q&A tool over the WCA (World Cube Association) Regulations.
The user (a WCA Delegate) asks: "this incident happened, how should I handle it?"
The tool retrieves relevant regulations and produces an answer that cites the
specific rules applied.

## User context

- Mid-level developer, primarily Python and SQL (pandas, pyspark basic, MySQL, PostgreSQL).
- No prior experience with LLMs, RAG, embeddings, or vector databases.
- Wants to learn the underlying concepts, not just ship code.
- Direct, concise communication. No filler. No flattery. Treat as peer.
- Will explicitly ask for depth when wanted.

## How to behave in this project

### Teaching mode is on

This is a learning project. When introducing any new concept (embeddings,
chunking strategies, vector similarity, prompt engineering patterns, evaluation
metrics, etc.):

1. Explain *what* it is.
2. Explain *why* this approach over alternatives.
3. Flag tradeoffs explicitly.
4. Link authoritative resources (papers, official docs) when they add value.
5. Get technical and mathematical when warranted. Do not dumb things down.

When making any architectural recommendation, present alternatives even if not
asked. The user wants to understand the decision space.

### Code review behavior

- Always point out mistakes explicitly. The user wants to learn from them.
- If something is unclear, ask before answering.
- Suggest better approaches even when not asked. Flag clearly so the user can
  choose whether to dig in.

### Don't

- Don't implement code unless explicitly asked. Default mode is planning and
  discussion.
- Don't write a roadmap until decisions are locked in `ARCHITECTURE.md`.
- Don't optimize for deployment. Local execution only for now.
- Don't introduce framework abstractions (LangChain, LlamaIndex) without
  explicit justification. The corpus is small; primitives teach more.

## Stack (current proposal — see ARCHITECTURE.md for status)

- Python 3.11+, conda environment.
- Embedding model: `bge-small-en-v1.5` via `sentence-transformers` (local).
- Vector store: ChromaDB (local, file-based).
- Generation LLM: Google Gemini API (free tier) — pluggable interface so swap
  to Anthropic / local Llama is one file.
- Custom markdown parser for the WCA regulations corpus.
- No notebooks. `.py` files with inspectable intermediate artifacts (JSON, parquet) on disk.

## Repo conventions

- All RAG components are behind interfaces (`Embedder`, `Retriever`, `Generator`)
  to keep swappability for experiments.
- Intermediate artifacts (parsed chunks, embeddings, retrieval results) are
  written to `data/` and gitignored.
- Source-of-truth corpus is in `data/raw/` and tracked.
- Eval golden set is in `evals/golden_set.yaml` and tracked.
- All scripts are runnable from repo root: `python -m wca_rag.<script>`.

## When the user uploads new context (delegate handbook, past rulings)

Treat them as separate sources. Do not blindly merge into the same chunk index.
Adding heterogeneous corpora can degrade retrieval quality. Discuss with the
user how to scope each source and whether to use metadata filters.

## Documentation upkeep

When decisions are made, update `ARCHITECTURE.md`. When new questions arise,
add them to `OPEN_QUESTIONS.md`. Keep these two files as the source of truth
for project state.
