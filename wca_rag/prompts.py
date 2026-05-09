"""
Prompt templates and prompt assembly for the generator stage.

Two responsibilities:

1. SYSTEM_PROMPT — the contract with the LLM. Defines refusal taxonomy
   (answer / partial / refuse), citation format (inline [11e] markers),
   and the mandatory verbatim-quote-then-explain structure that is the
   primary anti-confabulation mechanism. See ARCHITECTURE.md §3.9.

2. assemble_user_prompt() — deterministic. Takes retrieved hits + the
   user question, formats each chunk inside <regulation> tags with the
   regulation_id as an attribute, then appends the question.

Kept separate from generator.py so the prompt can be edited and version-
controlled independently of the LLM call. Iterating on prompts is the
expected debug loop for this stage; isolating them makes diffs clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wca_rag.retriever import RetrievalHit


# ----------------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------------
#
# Design notes (full rationale in ARCHITECTURE.md §3.9):
#
# - Refusal taxonomy is three-way: ANSWER | PARTIAL | REFUSE. The LLM
#   decides which mode applies based on whether the retrieved chunks
#   cover the question. PARTIAL exists because real questions often
#   touch multiple regulations; covering some-but-not-all should not
#   collapse to a full refusal.
#
# - Mandatory verbatim quoting is the single most important
#   anti-confabulation mechanism. The model must (a) state the
#   conclusion, (b) quote the regulation text verbatim, (c) explain in
#   one sentence how the quote supports the conclusion. Step (c) is the
#   real safeguard — it is hard to write a coherent justification when
#   the quote does not actually fit. Without (c), models will quote
#   adjacent-but-irrelevant text.
#
# - Citation format: inline [regulation_id] immediately after the claim
#   it supports. Granular enough to audit; readable enough for a CLI.
#
# - Citation granularity: claim-level, not chunk-level (Option A). The
#   model may cite sub-regulation ids (e.g. [A6c], [11e++]) when their
#   verbatim text appears inside a retrieved chunk's body, even though
#   chunks themselves are at top-level granularity. The verbatim-quote
#   requirement is what makes this auditable: every cited id must be
#   backed by a quote that string-matches the retrieved set. See
#   ARCHITECTURE.md §3.9 and CONCEPTS.md "Chunk-level vs claim-level
#   citation granularity".
#
# - Examples in the prompt deliberately include WCA's unusual ID
#   suffixes (11e++, A2b, 9f1) so the model produces them correctly.
#   Without examples, models tend to drop the + suffixes.
#
# - Audience framing: the LLM is told it is advising a WCA delegate at a
#   competition. This raises the stakes in-prompt and biases toward
#   refusal over guessing.

SYSTEM_PROMPT = """\
You are a regulation assistant for World Cube Association (WCA) delegates \
handling incidents at competitions. Delegates rely on your answers to \
make rulings that affect competitor results. A confident wrong answer is \
worse than no answer at all.

You will be given a set of WCA Regulations chunks retrieved as relevant \
to a delegate's question, each wrapped in <regulation> tags with an \
`id` attribute. You must answer the question using ONLY these chunks. \
Do not use prior knowledge of WCA Regulations. If the retrieved chunks \
do not contain the answer, say so — do not guess.

# Response modes

Choose exactly one mode for your response:

- **ANSWER** — the retrieved regulations directly and completely cover \
the question.
- **PARTIAL** — the retrieved regulations cover part of the question \
but not all of it. Answer what is covered, then explicitly state what is \
not covered.
- **REFUSE** — the retrieved regulations do not contain the information \
needed to answer the question. Say so plainly. Suggest the delegate \
consult the regulations directly or escalate to a more senior delegate.

Begin your response with the mode label in brackets, e.g. `[ANSWER]`, \
`[PARTIAL]`, or `[REFUSE]`.

# Mandatory structure for every claim

For every substantive claim you make, you must:

1. State the conclusion.
2. Quote the supporting regulation text verbatim, in quotation marks.
3. Explain in one sentence how the quoted text supports the conclusion.
4. Cite the regulation inline with its id in square brackets, e.g. [11e].

If you cannot find verbatim text that supports a conclusion, you do not \
have grounds to state that conclusion. Switch to PARTIAL or REFUSE.

# Citation format

- Cite the most specific regulation id whose verbatim text in the \
retrieved chunks supports your claim. The id may appear either as the \
`id` attribute of a <regulation> tag OR inside the body text of a \
retrieved chunk (sub-regulations like `11e1`, annotations like `11e++`, \
and nested children like `A2b` are all valid citation targets when \
their text is present in a retrieved chunk).
- Place the citation in square brackets, immediately after the \
supported claim. Examples: [11e], [11e++], [A2b], [9f1].
- Preserve `+` suffixes exactly as they appear. `11e` and `11e++` are \
different regulations.
- Prefer the narrowest id that fully supports the claim. If only the \
text labeled `A6c` supports a claim, cite [A6c], not [A6].
- Every citation must be backed by a verbatim quote from the retrieved \
chunks (per the mandatory structure above). If you cannot find verbatim \
text for a candidate id, you may not cite that id.

# Style

- Be direct. Delegates are working under time pressure during a \
competition.
- No preamble like "Based on the regulations..." — go straight to the \
answer.
- If a regulation contains conditions or exceptions, state them \
explicitly. Burying a condition is the same as omitting it.

# Example response (ANSWER mode)

[ANSWER]
A competitor may stop the timer while still touching the puzzle, but \
this incurs a +2 second time penalty. Regulation A6c states: "The \
competitor must fully release the puzzle before stopping the timer. \
Penalty: time penalty (+2 seconds)." The regulation directly addresses \
the act of stopping the timer without first releasing the puzzle and \
specifies the +2 penalty as the consequence. [A6c]

# Example response (REFUSE mode)

[REFUSE]
The retrieved regulations do not address this question. They cover \
[brief description of what they do cover], but say nothing about \
[the specific topic asked]. Please consult the WCA Regulations directly \
or escalate to a more senior delegate.\
"""


# ----------------------------------------------------------------------------
# User prompt assembly
# ----------------------------------------------------------------------------


def format_chunk(hit: "RetrievalHit") -> str:
    """Format one retrieved chunk for inclusion in the prompt.

    XML-ish wrapper with regulation_id and article as attributes. The id
    attribute is the citation key the LLM is instructed to use; the
    article attribute is contextual (cheap to include, occasionally
    useful for the LLM to disambiguate similar ids across articles).

    Uses chunk["text"] (raw body), NOT chunk["text_for_embedding"]. The
    embedding-side header was added to help the embedder retrieve short
    chunks; including it in the LLM context would just be noise.
    """
    chunk = hit.chunk
    return (
        f'<regulation id="{chunk["regulation_id"]}" '
        f'article="{chunk["article"]}">\n'
        f'{chunk["text"]}\n'
        f"</regulation>"
    )


def assemble_user_prompt(question: str, hits: list["RetrievalHit"]) -> str:
    """Assemble the user-side prompt: retrieved chunks + the question.

    Order: chunks first, question last. "Lost in the middle" research
    suggests the model attends most strongly to the start and end of
    the context — putting the question at the end keeps it salient. The
    chunks themselves are ordered by retrieval rank (most similar
    first), which is also the order the model will see them.
    """
    if not hits:
        # Defensive: should never happen — retriever always returns k hits
        # — but if it does, refusing on empty context is the safe behavior.
        return (
            "No regulations were retrieved for this question.\n\n"
            f"Question: {question}"
        )

    chunks_block = "\n\n".join(format_chunk(h) for h in hits)
    return (
        "Retrieved WCA Regulations:\n\n"
        f"{chunks_block}\n\n"
        f"Delegate's question: {question}"
    )