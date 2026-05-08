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
- [Semantic Chunking](#semantic-chunking)
- [Chunk metadata](#chunk-metadata)
- [Metadata-prepend (what you embed ≠ what you display)](#metadata-prepend-what-you-embed--what-you-display)
- [Cross-references and graph-augmented retrieval](#cross-references-and-graph-augmented-retrieval)
- [Inspectable intermediate artifacts](#inspectable-intermediate-artifacts)
- [Golden set evaluation](#golden-set-evaluation)
- [Recall@k](#recallk)
- [Embeddings](#embeddings)
- [Cosine similarity, dot product, L2 normalization](#cosine-similarity-dot-product-l2-normalization)
- [Bi-encoder vs cross-encoder](#bi-encoder-vs-cross-encoder)
- [Query-document asymmetry](#query-document-asymmetry)
- [Brute-force retrieval (and when to stop)](#brute-force-retrieval-and-when-to-stop)

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

## Semantic chunking

**What it is.** A family of chunking strategies that split text by *meaning*
rather than by character or token count. The shared idea: chunk boundaries
should fall where one coherent unit ends and another begins, not where a
counter happens to hit 500. The three common approaches differ in how they
detect those boundaries.

**The three flavors.**

| Approach | How it finds boundaries | When to use it |
|---|---|---|
| **Structure-aware** | Uses explicit cues already in the document — headings, IDs, sections, list items, code fences. Boundaries are deterministic and free. | The document has reliable structure: legal/regulatory text, technical specs, structured markdown, API docs. ✅ what we use for WCA. |
| **Embedding-based** | Embeds adjacent sentences (or windows), measures similarity drift, splits where similarity drops below a threshold. | Prose with no structural markers but real topic shifts: long-form articles, transcripts, books. |
| **LLM-based** | Asks an LLM to segment the document directly ("split this into self-contained chunks"). | Complex prose where neither structure nor sentence similarity is reliable, and you can afford the inference cost. |

**Why it matters.**
- **Retrieval quality is bottlenecked by chunk coherence.** A chunk that mixes two unrelated topics gets an "average-y" embedding that matches neither well. A chunk cut mid-thought leaves the answer unreachable.
- **It's the cheapest, highest-leverage tuning lever.** Better chunking usually beats a better embedding model at this corpus size. You don't need to retrain or pay more — you just need to cut in the right places.
- **Most "RAG isn't working" stories trace back to chunking.** Before reaching for rerankers, hybrid search, or fine-tuning, fix the chunks.

**Tradeoffs.**
- **Structure-aware is free but brittle.** Requires the document to actually have reliable structure. Garbage markdown → garbage chunks. Pays off massively when the structure is clean (WCA's `11e`/`11e+`/`11e1` IDs are gold).
- **Embedding-based is general but tunable.** You pick the similarity threshold, which is dataset-dependent. Too tight → too many tiny chunks; too loose → falls back to character-count behavior. Usually requires eyeballing a sample to pick a threshold.
- **LLM-based is highest quality but slowest and most expensive.** One LLM call per document at indexing time, non-deterministic outputs, harder to debug. Can be worth it for small high-value corpora.
- **All three break under the same failure mode:** if the document genuinely has no semantic structure (a chaotic dump, mixed-topic emails, scraped web noise), no chunking strategy saves you. Clean the input first.

**For WCA specifically.** Structure-aware is the obvious choice — the
hierarchical IDs (`11e`, `11e+`, `11e1`, `11e2a`) are explicit, machine-readable
markers of what belongs together. Embedding-based would be a regression: it
would rediscover, imperfectly, the structure already encoded in the IDs. See
the `Chunking` entry above for the WCA-specific decision and trade record.

**Connection to your day job.** Structure-aware chunking is the same instinct
as a well-designed SQL schema: respect the natural keys and relationships
already in the data, don't impose an arbitrary one. Embedding-based chunking
is closer to a clustering job — group rows by similarity and accept that the
cuts won't always land where a human would put them.

**Further reading.**
- Greg Kamradt, "5 Levels of Text Splitting" — walks all three flavors plus naive baselines, with code.
- LlamaIndex docs on `SemanticSplitterNodeParser` — reference implementation of embedding-based splitting.
- LangChain's `MarkdownHeaderTextSplitter` — canonical structure-aware splitter for markdown.

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

## Metadata-prepend (what you embed ≠ what you display)

**What it is.** A pattern where the text fed to the embedder is *not* the
same string that gets returned to the user or sent to the generator. The
embedder sees a version with extra context prepended — typically the
parent section title, document name, or hierarchical path. The display
text stays clean.

For WCA, every chunk carries two text fields:

- `text` — the raw regulation body. What the LLM sees as context. What gets cited.
- `text_for_embedding` — the same body with `Article {N}: {Title}\nRegulation {id}` prepended. What goes through the embedder.

So `11e` gets embedded as something like:

> Article 11: Incidents <br>
> Regulation 11e <br>
> [regulation body...] 

…but if it's retrieved, the LLM only sees the body and the metadata fields
separately. See `wca_rag/parser.py` (the `flush()` function) for the
construction.

**Why it matters.**
- **Embeddings only know what's in the input string.** A chunk that just says "The competitor must signal the WCA Delegate within 10 minutes" has no idea it belongs to Article 11 ("Incidents"). A query like "incident reporting deadline" might miss it because the literal word "incident" never appears in the chunk text.
- **It's a form of cheap, lossless context expansion.** You pay no inference cost, no extra storage, no extra retrieval complexity — you just front-load the parent context into the string before embedding.
- **It separates two jobs that should be separate.** Embedding wants *all* the disambiguating context. Display wants the *minimum* clean text the user needs. Conflating them either pollutes the LLM's context window with redundant headers or starves the embedder of context.

**The broader pattern.** This is one instance of "what you embed isn't always
what you display." Other instances of the same pattern:

- **Hypothetical questions for embedding.** Generate likely questions a chunk could answer, embed those, store the chunk for display. (HyDE-adjacent at index time.)
- **Summaries for embedding, full text for display.** Embed a short LLM-generated summary; return the long original on retrieval. Useful for very long chunks whose embeddings would otherwise be diluted.
- **Translations for embedding.** Embed an English translation; display the source language. Useful when the embedding model is much stronger in English than in the source language.

The unifying principle: the embedding string is a *retrieval key*, not the
content itself. Optimize the key for retrieval, optimize the content for the
LLM and the user.

**Tradeoffs.**
- **You're paying for embedding tokens you don't display.** Negligible at this scale (a few extra tokens per chunk, embedded once), but matters if you're on a paid embedding API and your prepend is heavy.
- **The prepended context dominates the embedding for very short chunks.** A 20-word regulation prepended with a 15-word header becomes "mostly header." For WCA this is mild because most chunks are long enough that the prepend is a small fraction. Worth checking on the shortest chunks (`Article 9`, ~50 chars) once retrieval is running.
- **Asymmetric query side.** For this trick to actually help, the query embedding has to land near where the prepended chunks now live. With `bge-small`, the query-side instruction prefix (`"Represent this sentence for searching relevant passages: ..."`) handles part of this; explicit query rewriting handles the rest.
- **Drift risk.** Two text fields means two things to keep in sync. If you ever re-derive `text_for_embedding` from `text` + metadata, do it in one place (the parser) so the format doesn't fork.

**For WCA specifically.** The prepend is `Article {N}: {Title}\nRegulation
{id}`. This buys two things: (1) chunks that don't mention their article by
name still match queries phrased in article-level terms ("rules about
incidents"), (2) the `regulation_id` is in the embedded text, so queries that
reference an ID directly ("what does 11e say about...") have a direct lexical
hook for the embedding to grab onto.

**Connection to your day job.** Same shape as a generated/derived column in
SQL: the canonical data lives in one column, but you persist a derived form
optimized for a specific lookup pattern (an indexed lowercase copy for
case-insensitive search, a tsvector for full-text). `text` is the canonical
column, `text_for_embedding` is the derived index-optimized form. You wouldn't
display the tsvector to users; same logic here.

**Further reading.**
- Anthropic's "Contextual Retrieval" blog post — production-scale version of this pattern, where the prepended context is LLM-generated per chunk rather than templated. Same principle, much heavier mechanism.
- Pinecone, "Chunking Strategies" — touches on context prepending under "metadata enrichment."

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

---
 
## Embeddings
 
**What it is.** A function that maps a piece of text to a fixed-length
vector of floats — for `bge-small`, 384 numbers between roughly -1 and 1.
The vector is the model's compressed representation of the text's
*meaning*. Two texts with similar meaning produce vectors that are close
to each other in 384-dimensional space; texts with unrelated meaning
produce vectors that are far apart.
 
That's the whole pitch. "Closeness in vector space" is the proxy for
"similarity in meaning," and it's good enough that you can do dense
retrieval (find the closest documents to a query) by comparing vectors.
 
**Why it matters.**
- **Semantic search beats keyword search for natural-language queries.** A
  user asking "what happens when a competitor's puzzle pops?" never says
  "DNF" or "Article 11," but the embedding of their question lands near
  the embedding of the relevant regulation regardless. Keyword search
  would miss this; embeddings won't.
- **Embeddings are precomputed.** You embed your 108 chunks once, persist
  the matrix, and at query time you only embed the 1 question. The
  expensive work happens offline.
- **They generalize across phrasings.** "Puzzle came apart," "cube broke,"
  "pieces fell off" all land in roughly the same neighborhood. You don't
  have to enumerate synonyms.
**How it actually works (one paragraph).** A transformer reads the input
text, processes it through 12 attention layers, and produces a token-level
representation for each token. A pooling layer collapses those into a
single vector — `bge-small` mean-pools the token vectors and then projects
to 384 dims. The model was trained on millions of (similar text, similar
text) and (similar text, dissimilar text) pairs with a contrastive loss:
push similar pairs together in vector space, push dissimilar pairs apart.
After training, the geometry of the space *is* the meaning.
 
**Tradeoffs.**
- **It's a black box.** You can't read off "what dimension 47 means" — the
  space is learned, not designed. Debugging is by comparison
  (`scripts/inspect_embeddings.py`), not introspection.
- **Quality is bounded by training data.** A model trained on web English
  will be bad at code, bad at non-English, bad at niche jargon. Domain
  shift is real. (Cubing terminology happens to be common enough on the
  open web that bge handles it acceptably.)
- **Fixed dimensionality is a compression.** 384 floats can't perfectly
  represent every nuance of every paragraph. Information is lost. The
  loss is small enough to be useful, large enough that retrieval will
  occasionally surface a wrong neighbor.
- **Embeddings are model-specific.** Vectors from bge-small and vectors
  from OpenAI `text-embedding-3-small` are not interchangeable, even
  though both are 384-dim. They live in different spaces. Swap the model
  → re-embed everything.
**Connection to your day job.** Embeddings are dimensional reduction with
a learned objective. PCA reduces to N dims by preserving variance;
embeddings reduce to N dims by preserving "semantic similarity" (a thing
defined by the training data). If you've ever computed `cosine_similarity`
between TF-IDF vectors of documents, embeddings are the same idea — but
with a learned representation that captures meaning instead of raw word
co-occurrence.
 
**Further reading.**
- Jay Alammar, "The Illustrated Transformer" — visual intro to what
  transformers do under the hood.
- The `bge` paper (Xiao et al. 2023, *C-Pack*) — describes how the model
  we use was trained.
- HuggingFace MTEB leaderboard (`huggingface.co/spaces/mteb/leaderboard`)
  — practical comparison of embedding models. Useful when picking one.
---
 
## Cosine similarity, dot product, L2 normalization
 
**What it is.** Three closely-related concepts you can't avoid in dense
retrieval. Best understood together because the relationship between them
is what makes RAG fast.
 
**Cosine similarity** measures the angle between two vectors:
 
```
cos(a, b) = (a · b) / (|a| × |b|)
```
 
Range: -1 (opposite direction) to 0 (perpendicular) to 1 (same direction).
Magnitude doesn't matter, only direction. Two vectors pointing the same
way score 1 regardless of length.
 
**Dot product** is the numerator alone:
 
```
a · b = Σ a_i × b_i
```
 
Range: unbounded. Captures both direction and magnitude.
 
**L2 normalization** scales a vector to unit length:
 
```
a_normalized = a / |a|
```
 
After normalization, `|a| = 1`. Information about magnitude is discarded;
only direction remains.
 
**The trick.** If both `a` and `b` are L2-normalized, then `|a| × |b| = 1`,
so:
 
```
cos(a, b) = a · b
```
 
Cosine similarity reduces to dot product. No division needed.
 
**Why this matters for RAG.** With normalized embeddings, retrieval is
literally one matrix multiply: `embeddings @ query_vec` produces all 108
similarity scores in one CPU instruction sweep. No square roots, no
divisions, no per-pair cosine computations. The retriever code is three
lines:
 
```python
scores = embeddings @ query_vec      # (108,) cosine similarities
top_k_idx = np.argsort(scores)[-k:][::-1]
hits = [chunk_ids[i] for i in top_k_idx]
```
 
That third line is the entire retrieval algorithm at this corpus size.
 
**Why cosine and not Euclidean distance?** For text embeddings, direction
encodes meaning and magnitude is mostly noise (longer documents tend to
have larger embedding norms even when topically identical to short ones).
Cosine factors out magnitude; Euclidean conflates the two. The standard
result: cosine outperforms Euclidean on text retrieval benchmarks.
Normalize once, use dot product forever.
 
**Tradeoffs.**
- **You commit to "direction is meaning."** True for text. Less true for
  some other modalities — image embeddings sometimes use Euclidean.
- **Normalization is destructive.** If magnitude does carry information,
  you lose it. For text it doesn't, so this is fine.
- **The normalization invariant is silent if violated.** If you forget
  `normalize_embeddings=True` in one place, dot product still runs and
  still produces numbers — they're just not cosine similarities anymore,
  and your retrieval quality silently degrades. This is why
  `inspect_embeddings.py` asserts `np.linalg.norm(embeddings, axis=1) ≈ 1`.
**Connection to your day job.** This is the same family as Pearson
correlation, which is also "cosine similarity after centering." If you've
computed correlation between two columns in pandas, you've done this
math.
 
**Further reading.**
- Manning, Raghavan, Schütze, *Introduction to Information Retrieval*,
  ch. 6 — derives cosine similarity from first principles in the
  vector-space-model context.
- Pinecone, "Cosine Similarity" — short visual explainer.
---
 
## Bi-encoder vs cross-encoder
 
**What it is.** Two architectures for scoring "how relevant is document D
to query Q." They make opposite tradeoffs between accuracy and speed.
 
**Bi-encoder.** Encode `Q` and `D` independently into vectors. Score with
a similarity function (dot product or cosine). The two encodings never
interact during the model forward pass.
 
```
Q → encoder → q_vec ───┐
                       ├── score = q_vec · d_vec
D → encoder → d_vec ───┘
```
 
**Cross-encoder.** Concatenate `Q` and `D` into a single input, feed
through the model, output a scalar relevance score. The model attends
across both texts simultaneously.
 
```
[Q, D] → encoder → score
```
 
**Why it matters.** The choice of architecture determines what you can
precompute, which determines how fast your retrieval is.
 
| | Bi-encoder | Cross-encoder |
|---|---|---|
| Document encoding | Precomputed once | Recomputed every query |
| Query-time work for n docs | 1 query encode + n dot products | n full model forward passes |
| Accuracy | Lower (no Q×D interaction) | Higher (model sees both) |
| Use case | First-stage retrieval | Reranking a small candidate set |
 
**Concrete numbers.** For 108 chunks at query time:
- Bi-encoder: 1 model call + a 108×384 matrix multiply ≈ ~10ms.
- Cross-encoder: 108 model calls ≈ ~1.5 seconds, even on a small model.
For 100k chunks the bi-encoder is still ~10ms; the cross-encoder is
20+ minutes. Cross-encoders are not retrievers, they're rerankers.
 
**The standard production pattern.**
1. Bi-encoder retrieves top-50 from the full corpus (fast, approximate).
2. Cross-encoder reranks those 50 into a tighter top-5 (slow, accurate).
You get bi-encoder speed and cross-encoder quality, paying the
cross-encoder cost only on a small candidate set. This is on
`ARCHITECTURE.md §7` as the v2 hybrid retrieval upgrade.
 
**For WCA specifically.** v1 is bi-encoder only — `bge-small-en-v1.5`
embedding + dot product. Cross-encoder reranking comes after the golden
set is built and we can measure whether it actually moves the needle.
 
**Tradeoffs.**
- **Bi-encoder loses query-document interaction information.** A
  cross-encoder can notice "the query mentions X but the document
  contradicts X" — bi-encoders can't, because Q and D never see each
  other. This is the accuracy gap.
- **Cross-encoders can't be precomputed.** Every (Q, D) pair is a fresh
  forward pass. Fundamentally not a retrieval architecture for n > a few
  hundred.
- **Bi-encoder accuracy is not bad — just lower.** On standard benchmarks
  bi-encoders typically reach 80–90% of cross-encoder accuracy at 1/100
  the latency. The gap matters more for marginal cases than typical ones.
**Connection to your day job.** Bi-encoder is a hash-based join — both
sides hash independently, you join on equality of hashes. Cross-encoder
is a nested-loop join — for every pair, compute the relationship from
scratch. Same speed/accuracy tradeoff, same query-planner instinct: use
the cheap operation to filter, use the expensive one to refine.
 
**Further reading.**
- Reimers & Gurevych 2019, *Sentence-BERT* — introduced bi-encoders for
  sentence-level retrieval. The reason `sentence-transformers` exists.
- Nogueira & Cho 2019, *Passage Re-ranking with BERT* — cross-encoder
  reranking, the canonical reference.
---
 
## Query-document asymmetry
 
**What it is.** With many embedding models, queries and documents need
*different treatment* before being fed to the same encoder, even though
the encoder itself is one model. The asymmetry is in what wraps the input.
 
For `bge-small-en-v1.5`:
- Documents: embedded as-is.
- Queries: prefixed with `"Represent this sentence for searching relevant passages: "`.
Other models do it differently:
 
| Model | Query side | Document side |
|---|---|---|
| `bge` family | prefix string | no prefix |
| E5 family | `"query: "` prefix | `"passage: "` prefix |
| OpenAI `text-embedding-3` | symmetric (no prefix) | symmetric |
| Voyage / Cohere | `input_type="query"` API param | `input_type="document"` API param |
 
**Why it matters.** The model was *trained* on this asymmetry. During
contrastive training, the query side and document side were presented in
their respective formats. The model learned that text starting with the
query prefix is a *question looking for an answer*, and text without it
is *content that might be the answer*. The prefix shifts the query's
embedding toward the region of vector space where matching documents
live.
 
Skip the prefix and the query lands in a slightly wrong neighborhood.
Retrieval still works — the embeddings aren't completely decoupled — but
quality drops measurably. The bge model card reports a few-point hit on
benchmarks when the prefix is omitted.
 
**For WCA specifically.** `wca_rag/embedder.py` has two separate methods:
 
```python
def encode_documents(self, texts: list[str]) -> np.ndarray:
    # No prefix.
    return self._model.encode(texts, ...)
 
def encode_query(self, text: str) -> np.ndarray:
    prefixed = f"{self._query_prefix}{text}"
    return self._model.encode([prefixed], ...)[0]
```
 
Two methods (not one method with a flag) so the asymmetry is enforced by
the type system, not by caller discipline. The indexer can only call
`encode_documents`; the retriever can only call `encode_query`. Misuse
requires actively defeating the API.
 
**Tradeoffs.**
- **You add prefix-handling complexity to your code.** Worth it; the
  prefix tokens are a small embedding cost vs. the retrieval-quality
  benefit.
- **It's model-specific and easy to forget on swap.** Move from bge to
  E5 → prefix changes from `"Represent this sentence..."` to `"query: "`.
  Move to OpenAI → no prefix at all. Worth keeping the prefix in a named
  constant so the dependency is obvious in code review.
- **The asymmetry is invisible in the embedding output.** A query
  embedded *without* the prefix produces a vector that looks normal —
  same shape, same dtype, similar magnitude. The bug only shows as
  worse retrieval. This is one of those places where automated tests
  can't help; only golden-set evaluation will catch it.
**Connection to your day job.** This is the same shape as parameterized
queries vs. raw strings in SQL. The database treats `?`-bound parameters
differently from inline literals — same statement, different processing
paths. You learn the convention of your driver and stick to it.
 
**Further reading.**
- The bge-small model card on HuggingFace (`huggingface.co/BAAI/bge-small-en-v1.5`)
  — documents the prefix and the rationale.
- The E5 paper (Wang et al. 2022, *Text Embeddings by Weakly-Supervised
  Contrastive Pre-training*) — introduces the `query: ` / `passage: `
  convention for that model family.
---
 
## Brute-force retrieval (and when to stop)
 
**What it is.** Compute the similarity between the query vector and
*every* document vector, then sort. No index, no approximation, no tree
structure. For 108 chunks, this is `embeddings @ query_vec` followed by
`argsort` — three lines of NumPy.
 
**Why it matters.** Brute force is the fastest possible approach at small
scale, and it's the simplest possible approach at any scale. Vector
databases like ChromaDB exist to do something *cleverer* than brute force
when you have millions of vectors and brute force gets slow. Until you
hit that point, the database is overhead, not optimization.
 
**The numbers.**
 
| Corpus size | Brute-force latency (CPU) | Need an index? |
|---|---|---|
| 100 | ~0.05 ms | Absolutely not |
| 10,000 | ~5 ms | No |
| 100,000 | ~50 ms | Probably not |
| 1,000,000 | ~500 ms | Yes, switch to HNSW |
| 10,000,000+ | seconds | Yes, definitely |
 
A single `embeddings @ query_vec` is a BLAS-optimized matrix-vector
multiply. Modern CPUs can do this on a 100k×384 matrix in milliseconds.
The crossover where brute force becomes painful is around 1M vectors —
many orders of magnitude above where most projects actually live.
 
**For WCA specifically.** 108 chunks. Brute force is so far below the
crossover that anything else would be performance theater. The retriever
in `wca_rag/retriever.py` is intentionally just NumPy. ChromaDB enters
the picture later for two reasons that have nothing to do with speed:
 
1. **Metadata filtering.** "Retrieve only chunks where `article == '11'`"
   is a one-liner in Chroma; in NumPy it's an extra mask layer.
2. **API ergonomics.** Chroma's `add` / `query` API is a stable contract
   that survives swapping vector backends. NumPy is a contract you
   maintain yourself.
**The clever-index approaches (for context).**
- **HNSW (Hierarchical Navigable Small World)** — a graph structure
  where nearest-neighbor search becomes graph traversal. Approximate but
  very fast, sub-linear in corpus size. What ChromaDB / FAISS / Qdrant /
  Pinecone use under the hood.
- **IVF (Inverted File Index)** — cluster the corpus, search only the
  nearest clusters at query time. Used by FAISS for very large indexes.
- **Product Quantization** — compress vectors to 8-bit codes, do
  approximate similarity in the compressed space. Memory savings at the
  cost of accuracy. Used in conjunction with the above for billion-scale
  indexes.
All of these are *approximate* nearest neighbor (ANN) methods. They
trade a tiny amount of accuracy (typically <1% recall loss) for huge
speedups. At our scale we don't need any of them, and we don't pay the
accuracy cost.
 
**Tradeoffs.**
- **Brute force scales linearly in n.** Doubling the corpus doubles the
  retrieval time. ANN indexes scale sub-linearly (typically log n or
  better). The crossover is corpus-size-dependent and hardware-dependent
  — measure before optimizing.
- **Brute force is exact.** No approximation, no recall loss, no
  parameters to tune. The simplicity is a feature when you're debugging
  retrieval quality, because you've eliminated one source of error.
- **Brute force keeps the full matrix in memory.** 100k chunks × 384
  dims × 4 bytes = 150 MB. Fine. 10M chunks × 1024 dims × 4 bytes = 40
  GB. Not fine. Memory is the practical scaling limit, not CPU.
**Connection to your day job.** This is the SQL equivalent of `SELECT *
FROM chunks ORDER BY similarity(embedding, ?) DESC LIMIT 5` with no
index on the similarity column. The DB does a full table scan. For 108
rows that's a non-issue and probably faster than maintaining an index.
For 100M rows you build the index. The reasoning is the same: indexes
are an optimization, not a default.
 
**Further reading.**
- Erik Bernhardsson's ANN benchmarks (`ann-benchmarks.com`) — empirical
  comparison of ANN libraries on standard datasets. Useful if/when we
  need to pick one.
- Malkov & Yashunin 2016, *Efficient and robust approximate nearest
  neighbor search using HNSW graphs* — the HNSW paper. Worth reading
  once when ChromaDB stops feeling magical.