# Open Questions for Colleague Review

Context for reviewers: I'm a Python/SQL developer with no prior RAG/LLM
experience. I'm building a regulation Q&A tool over the WCA Regulations
(~22k words, ~30k tokens, hierarchical markdown with explicit IDs like `11e`,
`11e+`, `11e1`, `A2b`). Use case: WCA delegates asking incident questions
during competitions; answers must cite the specific regulations applied.

Constraints:
- Free / no paid APIs.
- Local execution on a 2015 MacBook Pro (CPU-only, 8GB RAM).
- English only for v1.
- Goal balance: learn RAG end-to-end **and** produce a usable tool.

The proposed stack and rationale are in `ARCHITECTURE.md`. Questions below
are the specific points where I want pushback before locking decisions.

---

## 1. Is classic RAG the right architecture given the corpus is small enough to fit in context?

The full regulations are ~30k tokens. Modern LLMs (Gemini, Claude, GPT) handle
200k+ context windows.

- Should v1 just stuff the whole corpus into every prompt and skip retrieval?
- If yes, what's lost from a learning standpoint?
- If no, at what corpus size does RAG actually start mattering vs long-context?

Current proposal: classic RAG, both for learning value and to support adding
more sources later (handbook, rulings).

## 2. Embedding model: `bge-small-en-v1.5` local vs free API tier?

Tradeoff:
- **Local** (`bge-small`): no API key, no rate limits, reproducible long-term, slight quality gap.
- **API free tier** (Gemini `text-embedding-004`, Voyage `voyage-3-lite`): better quality, possible rate limits, depends on free tier persisting.

For a 22k-word English regulatory corpus where retrieval precision matters
(legal-ish text), is the quality gap meaningful? Anyone have direct
experience comparing on similar corpora?

## 3. Vector store: ChromaDB vs FAISS vs numpy from scratch?

For ~150 chunks:
- ChromaDB: easy, persistent, slight magic.
- FAISS: educational, more control, but separate metadata management.
- numpy: at this scale, `embeddings @ query.T` works in <1ms and is the most
  pedagogically valuable.

Is there a reason to use anything heavier than numpy for a corpus this small?

## 4. Generation LLM: Gemini free tier vs Anthropic trial credit vs local Ollama?

Free constraint pushes toward Gemini. But:
- For citation-heavy outputs ("which regulation applies?"), instruction-following
  matters a lot. Claude Haiku 4.5 reportedly outperforms Gemini Flash here.
- Anthropic gives trial credits that would last well past v1.
- Local models (Llama 3.2 1B/3B) are too weak for this; Llama 3.1 8B is the
  smallest that does well, marginal on a 2015 Mac.

Recommendation on the Gemini Flash vs Claude Haiku tradeoff for this specific
use case (citation accuracy on legal-ish text)?

## 5. Chunking strategy

**Proposal:** custom parser. One chunk per top-level regulation (e.g. `11e`),
bundling all `+`-annotations and nested children (`11e1`, `11e2`, `11e2a`, ...).
Estimate: ~150 chunks, varying size from ~50 chars to ~2000 chars.

Concerns I'd like challenged:
- **Variable chunk size**: some regulations have many annotations and children
  (`11e` is large), others are one sentence. Does this hurt retrieval?
- **Should I instead split at a fixed level (always one chunk per leaf)?**
  Seems worse to me but maybe I'm wrong.
- **Should I include the parent article header in each chunk** (e.g. prepend
  "Article 11: Incidents — " to every chunk)? This is a known trick for
  improving embedding quality but I'm not sure how much it matters.
- **Sliding-window overlap?** Common RAG advice. Seems unnecessary for this
  corpus given the natural chunk boundaries. Confirm or push back?

## 6. Metadata schema

```python
{
    "regulation_id": "11e",
    "article": "11",
    "article_title": "Incidents",
    "label": None,                       # CLARIFICATION | EXAMPLE | RECOMMENDATION | ADDITION | REMINDER | EXPLANATION
    "is_annotation": False,
    "depth": 1,
    "parent_id": None,
    "cross_references": ["11i2", "9l"],  # phase 2
    "char_count": 412,
    "source_version": "2026-04-01",
}
```

Anything missing that you'd want for retrieval, filtering, or debugging?

## 7. Cross-references

Many regulations link to others (`see [Regulation 11i2]`). Examples in
Article 11 are dense with these.

- v1 plan: ignore them, measure how often this hurts via golden set.
- v2 plan: link-aware retrieval (after vector retrieval, follow refs and
  pull those chunks too).

Anyone done graph-augmented RAG on cross-referenced documents (legal,
regulatory, technical specs)? Worth doing in v1 or correctly deferred?

## 8. Evaluation

**v1 plan:**
- Hand-written `evals/golden_set.yaml`: ~20 incident questions with expected
  regulation IDs.
- Retrieval recall@k.
- Citation accuracy in final answer.
- Manual 1–5 quality grade on a sample.

**Deferred:** RAGAS / TruLens / DeepEval, LLM-as-judge, faithfulness, answer relevance.

Is 20 examples enough to detect meaningful quality changes? What's the
minimum useful golden set size in your experience? Any gotchas with
recall@k as the primary retrieval metric?

## 9. Prompt design

Not yet drafted. Concerns I'm thinking about:
- How to make the model say "I don't know" rather than confabulate when the
  retrieved chunks don't cover the question (hallucination is genuinely
  dangerous in this use case — wrong rulings at competitions).
- Citation format: footnote-style `[11e]` after each claim? End-of-answer
  list? Inline?
- Should the prompt instruct the model to *quote* the regulation text or
  paraphrase? Quoting is safer but more verbose.

Anyone have battle-tested patterns for citation-heavy QA prompts?

## 10. Anything obvious I'm missing?

Catch-all. Especially interested in:
- Common beginner mistakes I'm walking into.
- Things that look fine on paper but break in practice.
- Whether any of the v1/v2 boundaries above are clearly wrong (deferring
  something I should do now, or doing something I should defer).
