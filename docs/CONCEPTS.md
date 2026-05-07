# Concepts

Reference notes for concepts encountered while building this project.
Organized by concept, not chronologically. Reread cold in 3 months and find
what you need.

Each entry follows the same shape:
- **What it is** — the definition.
- **Why it matters** — what problem it solves, why it's not trivial.
- **Tradeoffs** — what you give up to get the benefit.
- **Further reading** — papers / docs / blog posts when they add value.

---

## Table of contents

- [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
- [Why retrieve at all? (RAG vs long-context stuffing)](#why-retrieve-at-all-rag-vs-long-context-stuffing)
- [Chunking](#chunking)
- [Chunk metadata](#chunk-metadata)
- [Cross-references and graph-augmented retrieval](#cross-references-and-graph-augmented-retrieval)
- [Inspectable intermediate artifacts](#inspectable-intermediate-artifacts)
- [Golden set evaluation](#golden-set-evaluation)
- [Recall@k](#recallk)

---

## RAG (Retrieval-Augmented Generation)

**What it is.** A pattern where, instead of relying on the LLM's parametric
knowledge alone, you (1) retrieve relevant text from a corpus, (2) stuff it
into the prompt as context, and (3) ask the LLM to answer using that context.
The "retrieval" step is usually dense vector search: embed the query, find
the chunks whose embeddings are closest in vector space, return the top k.

**Why it matters.**
- **Freshness:** the LLM's training data has a cutoff. RAG lets you ground answers in your own up-to-date corpus.
- **Citation:** because you control what goes into the prompt, you can ask the model to cite which retrieved chunk it used. Crucial for legal/regulatory use cases.
- **Hallucination reduction:** the model is less likely to invent facts when the relevant facts are sitting in the prompt.
- **Cost:** much cheaper than fine-tuning, and the corpus updates without retraining.

**Tradeoffs.**
- **Retrieval miss = wrong answer.** If the right chunk isn't in the top k, the model can't cite it. Garbage in, garbage out.
- **Latency.** Two model calls (embedder + generator) instead of one.
- **System complexity.** Chunking, embedding, vector store, retrieval, prompt assembly — many moving parts to debug.
- **Context-window pollution.** Retrieved chunks compete for tokens against the system prompt and the generated answer.

**Further reading.**
- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — the original paper. The architecture they propose is more complex than what "RAG" means in practice today; the modern usage is closer to "stuff retrieved text into the prompt."
- Pinecone's RAG learning hub (`pinecone.io/learn/`) — accessible intro material.

---

## Why retrieve at all? (RAG vs long-context stuffing)

**What it is.** Modern LLMs handle 200k+ token context windows. If your corpus
fits, why not stuff it all into every prompt and skip retrieval?

**Why it matters.** This is the right baseline question. For a corpus this
small (~30k tokens), long-context stuffing is technically viable. The reasons
to prefer RAG anyway:

1. **"Lost in the middle" effect.** Liu et al. 2023 showed that LLMs pay less attention to information in the middle of long contexts. Putting the entire corpus in the prompt makes the relevant rule statistically more likely to be ignored.
2. **Cost.** Stuffing 30k tokens per query is wasteful when a query about Article 11 only needs ~1k tokens of context.
3. **Scalability.** When you add the Delegate Handbook and past rulings, the corpus quickly outgrows context. A RAG architecture handles that without rework.
4. **Learning value.** RAG forces you to think about chunking, embedding, retrieval quality — concepts you need anyway for any production-grade system.

**Tradeoffs.** Long-context stuffing wins on simplicity and on "no retrieval
errors." It's a legitimate baseline to compare against.

**Further reading.**
- Liu et al. 2023, *Lost in the Middle: How Language Models Use Long Contexts*.

---

## Chunking

**What it is.** Splitting your source documents into smaller units ("chunks")
that get embedded and retrieved independently. The chunk is the atomic unit
of retrieval.

**Why it matters.** Chunk design dominates retrieval quality. Get this wrong
and no embedding model or reranker will save you. Three forces in tension:

- **Smaller chunks** → more precise retrieval (the right chunk is more "purely" about the topic) but more fragmented context (a child rule without its parent is meaningless).
- **Larger chunks** → richer context but diluted embeddings (a chunk about 10 different things has an embedding that's "average-y" and doesn't match any specific query well).
- **Semantic boundaries** beat character-count boundaries. Cutting a regulation in half because you hit 500 chars is worse than keeping it intact at 800 chars.

**Common chunking strategies.**

| Strategy | When to use | Why not for WCA |
|---|---|---|
| Fixed character/token windows with overlap | Plain prose with no structure (blog posts, transcripts) | Ignores the explicit hierarchy in WCA regs; would split `11e` from `11e1` |
| Recursive character splitting (LangChain default) | Generic markdown / mixed content | Same problem — character-driven, not structure-driven |
| Sentence-level | When you need very precise retrieval and the source is well-formed prose | Each WCA sentence is meaningless without its rule context |
| Custom semantic chunks | When the document has explicit structural cues (headings, IDs, sections) | ✅ what we use — one chunk per top-level regulation + its annotations + children |

**For WCA specifically.** The regulations have explicit hierarchical IDs
(`11e`, `11e+`, `11e1`, `11e2a`). This structure encodes "these things belong
together." We chunk on top-level regulations and bundle their annotations
and children. See `ARCHITECTURE.md §3.4` for the decision and
`wca_rag/parser.py` for the implementation.

**Tradeoffs we accepted.**
- **Variable chunk size.** Some chunks are ~50 chars (`Article 9` only), others are ~5000 chars (`A1`, `E2`). Embeddings of long chunks are diluted. We accept this because splitting them at character boundaries would do more damage than the dilution.
- **No overlap.** Sliding-window overlap is common RAG advice for prose. Unnecessary here — the natural chunk boundaries are exact.

**Further reading.**
- Pinecone, "Chunking Strategies for LLM Applications" — surveys the common approaches.
- Greg Kamradt's "5 Levels of Text Splitting" tutorial — accessible walkthrough from naive to semantic.

---

## Chunk metadata

**What it is.** Per-chunk structured data attached alongside the text.
Things like `regulation_id`, `article`, `cross_references`, `char_count`.
Stored in the vector store next to the embedding.

**Why it matters.**
- **Citation.** When the LLM answers, it cites by `regulation_id`. The metadata is the bridge between the retrieved vector and the human-readable identifier.
- **Filtering.** "Only retrieve chunks where `article == '11'`" turns a 108-chunk search into a ~12-chunk search. Cheaper and more precise when the user query is scoped.
- **Debugging.** When retrieval is wrong, you need to know what was retrieved. Metadata makes chunks human-readable in a debugger or in `inspect_chunks.py`.
- **Future features.** `cross_references` enables link-aware retrieval later without re-parsing.

**Tradeoffs.** Storage cost (negligible at this scale). Schema rigidity —
once you commit to a schema, changing it requires re-indexing. Worth pinning
the schema in `ARCHITECTURE.md §3.5` so it doesn't drift.

**Connection to SQL world.** Metadata is the "columns" of your vector store.
A vector DB is morally a table with one weird column (the embedding) that
supports `ORDER BY similarity(embedding, query) LIMIT k` instead of normal
predicates. Metadata is everything else — and you filter on it the same way
you'd filter a SQL query.

---

## Cross-references and graph-augmented retrieval

**What it is.** Many corpora have internal links: regulation `11e` references
`11i2` and `9l`. If retrieval surfaces `11e` but the answer also depends on
`11i2`, you've under-retrieved. Graph-augmented retrieval extracts these
links and follows them: after vector retrieval, pull the chunks that the
top-k chunks reference.

**Why it matters.** For dense reference networks (legal text, technical
specs, scientific papers), single-hop vector retrieval routinely misses
context. The reference is *there*, but the retriever can't see it because
embeddings don't encode link structure.

**For WCA specifically.** From parser output: 55/108 chunks have at least one
cross-reference. 159 edges total, 124 unique targets. Top hubs are `2k`,
`9u`, `3l`, `10f`. These hubs are referenced by many other rules and are
strong candidates for "always include adjacent" expansion.

**Tradeoffs.**
- **Context bloat.** Following every reference quickly fills the context window. Need a budget.
- **Quality of extraction.** Garbage extraction (regex catches false positives) gives you garbage neighbors.
- **Diminishing returns.** 1-hop expansion is usually high-value. 2-hop ("references of references") often pulls in noise.

**Why deferred to v2.** v1 ignores cross-refs to keep the architecture
simple. The metric to watch on the golden set: when retrieval recall@k looks
fine but answer quality is poor, that's the signal that cross-refs are
hurting.

**Further reading.**
- Microsoft's GraphRAG — production-grade system that combines knowledge graphs with vector retrieval. Heavyweight but conceptually instructive.

---

## Inspectable intermediate artifacts

**What it is.** Every stage of the pipeline writes its output to disk in a
human-readable (or near-human-readable) format. `data/chunks.jsonl` after
parsing, `data/embeddings.npy` after embedding, retrieval results dumped to
JSON during eval. No black-box "the index" — every step has a file you can
`cat` or load in Python and poke at.

**Why it matters.**
- **Debugging.** When the system gives a bad answer, you can walk backwards: was the right chunk retrieved? Was it embedded? Was it parsed correctly? Each artifact lets you check one stage in isolation.
- **Iteration speed.** Re-running the parser doesn't require re-running embeddings. Re-running embeddings doesn't require re-running retrieval. Cache the slow stages.
- **Learning.** You see what each stage actually produces. Black-box frameworks (LangChain, LlamaIndex) hide these artifacts behind abstractions.

**Tradeoffs.** Disk space (negligible here). Discipline — you have to
*actually look* at the artifacts. The `inspect_chunks.py` script exists for
this reason.

**Connection to your day job.** Same pattern as a multi-stage SQL pipeline
where each stage materializes to a table you can `SELECT *` from. Don't
build pipelines whose intermediate state you can't see.

---

## Golden set evaluation

**What it is.** A small, hand-written set of (query, expected output) pairs
used to measure retrieval and generation quality. The "test set" of a RAG
system. For us: ~20 incident scenarios with the regulations a correct answer
must cite, in `evals/golden_set.yaml`.

**Why it matters.**
- **Without it, you can't tell if changes are improvements.** Switched embedding models — better or worse? Changed chunking — better or worse? Tweaked the prompt — better or worse? Without a golden set, you're going on vibes.
- **Forces you to define "correct."** Writing the expected outputs makes you confront edge cases (which rule is the right citation?) before the system does.
- **Anchors discussions with stakeholders.** "The system gets 14/20 on the golden set" is a concrete claim. "It seems pretty good" isn't.

**Tradeoffs.**
- **Effort to build.** Each entry takes 5–15 minutes if you want it to be high-quality. 20 entries = a few hours.
- **Coverage gaps.** A 20-entry set doesn't cover every failure mode. Bias toward edge cases the system is most likely to fail on — those are the highest-information samples.
- **Drift.** The golden set has to be maintained as the regulations change. Pin the source version.

**Sizing rule of thumb.** For early-stage prototyping, 15–30 entries is
enough to detect large quality changes (e.g. swapping embedding models). For
production, you want 100+ and ideally categorized by difficulty/type. Below
~10 you can't detect anything reliably.

**Further reading.**
- Hamel Husain, "Your AI Product Needs Evals" — practical, opinionated guide.

---

## Recall@k

**What it is.** A retrieval metric. Of the documents that *should* be
retrieved for a query (the "relevant set"), what fraction appears in the
top-k retrieved? Range: 0 (none retrieved) to 1 (all retrieved).

```
recall@k = |relevant ∩ retrieved_top_k| / |relevant|
```

**Why it matters.**
- **Decouples retrieval from generation.** A bad answer can be the retriever's fault (didn't surface the right chunk) or the generator's fault (had the right chunk, ignored it). Recall@k isolates the retrieval question.
- **Sweep over k.** Plotting recall@1, recall@3, recall@5, recall@10 tells you whether your retriever is "right at the top" or "right but buried." Different fixes for each.

**Tradeoffs.**
- **Doesn't capture ordering within top-k.** Recall@5 is the same whether the right chunk is at rank 1 or rank 5. Use MRR (Mean Reciprocal Rank) or NDCG when ordering matters.
- **Doesn't capture answer quality.** A retriever can have perfect recall and the generator still produce garbage. Need a complementary "answer correctness" or "citation accuracy" metric.
- **Sensitive to how you define "relevant."** For us: the regulation IDs in `expected_rules`. Sub-rule IDs (`11e++++`) need to be collapsed to their parent chunk ID (`11e`) before comparison, since chunks live at top level.

**Connection to your day job.** Same shape as `precision` and `recall` for
classification, just applied to a ranked list.

**Further reading.**
- Manning, Raghavan, Schütze, *Introduction to Information Retrieval*, ch. 8 — canonical IR metrics reference.
