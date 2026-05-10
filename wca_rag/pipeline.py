"""
Pipeline: orchestrates retrieval + prompt assembly + generation.

The full Phase-2 (query) pipeline lives here. Composes the existing
Retriever and a Generator into one `ask()` call.

This is the integration seam. Keep it thin — the pipeline should not
contain logic that belongs in retriever.py, prompts.py, or generator.py.
If you find yourself adding business logic here, it probably belongs in
one of those modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from wca_rag.generator import Generator, GenerationResult
from wca_rag.prompts import SYSTEM_PROMPT, assemble_user_prompt
from wca_rag.retriever import RetrievalHit, Retriever


# k=8 default: see ARCHITECTURE.md §3.9. Higher than retriever's k=5
# default (RETRIEVAL_DEFAULT_K) because the generator benefits from
# headroom (multi-regulation questions are common) and lost-in-the-
# middle is not a concern at this corpus size and chunk length.
#
# Naming: deliberately distinct from RETRIEVAL_DEFAULT_K so a stray
# `from wca_rag.retriever import DEFAULT_K` vs `from wca_rag.pipeline
# import DEFAULT_K` cannot silently produce different k's depending on
# which module a caller imported from.
PIPELINE_DEFAULT_K = 8


@dataclass
class PipelineResult:
    """Everything one `ask()` call produced. Returned as a single object
    so the CLI / future UI / eval harness can show retrieval and
    generation side by side."""

    question: str
    answer: str
    hits: list[RetrievalHit]
    generation: GenerationResult


class Pipeline:
    """Composes a Retriever and a Generator into a one-call ask() flow.

    Constructed once, queried many times. Both dependencies are passed
    in (no internal construction) so the pipeline is trivially testable
    and so the CLI can wire concrete impls itself.
    """

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self._retriever = retriever
        self._generator = generator

    def ask(self, question: str, k: int = PIPELINE_DEFAULT_K) -> PipelineResult:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        hits = self._retriever.retrieve(question, k=k)
        user_prompt = assemble_user_prompt(question, hits)
        generation = self._generator.generate(SYSTEM_PROMPT, user_prompt)

        return PipelineResult(
            question=question,
            answer=generation.text,
            hits=hits,
            generation=generation,
        )
