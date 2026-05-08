"""
Generator: takes a system prompt + user prompt, returns the LLM's answer.

Mirrors the embedder pattern: an ABC defines the contract, a concrete
implementation wraps the chosen provider. Default implementation uses
Google Gemini (gemini-2.5-flash, free tier — ARCHITECTURE.md §3.3).

v1 is batch-only (single call, full response). Streaming would
complicate citation validation downstream and the CLI does not benefit;
the ABC can grow a `stream()` method later without breaking `generate()`.

API key handling: GEMINI_API_KEY is read from the environment.
python-dotenv loads it from .env if present. See README → Setup.

SDK note: uses `google-genai` (the unified SDK as of 2025), NOT the
deprecated `google-generativeai`. The two packages have similar names
and very different APIs; do not mix them.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Output schema
# ----------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """The generator's output. Kept structured so the pipeline / CLI can
    surface metadata (model name, token usage) alongside the answer
    without parsing free text."""

    text: str
    model: str
    # Token counts are optional — different providers report them differently
    # and some don't report them on the free tier at all.
    input_tokens: int | None = None
    output_tokens: int | None = None


# ----------------------------------------------------------------------------
# ABC
# ----------------------------------------------------------------------------


class Generator(ABC):
    """Abstract generator. Implementations wrap a specific LLM provider."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        """Run one generation. Batch only — no streaming in v1."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model. Used for logging / eval
        result tagging."""
        ...


# ----------------------------------------------------------------------------
# Gemini implementation
# ----------------------------------------------------------------------------


class GeminiGenerator(Generator):
    """Default implementation. Wraps the google-genai SDK.

    Lazy-imports the SDK so the rest of the package (parser, embedder,
    retriever) does not pull in google-genai unless the generator is
    actually used. Useful when running indexing-only commands.

    The google-genai SDK construction model is different from the older
    google-generativeai SDK: there is one Client per process, and the
    model + system_instruction + temperature are passed per-call via
    the `config` argument. We construct the Client once in __init__
    and build the per-call config from constructor args.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        # Low temperature: this is a citation-heavy QA task, not a
        # creative one. We want the model to follow the system-prompt
        # contract closely. 0.1 leaves a tiny bit of variation for
        # phrasing without inviting confabulation.
        self._model = model
        self._temperature = temperature

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to .env (see .env.example) "
                "or export it in your shell."
            )

        # Lazy import — see class docstring.
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai. "
                "Note: this is NOT the same package as the deprecated "
                "google-generativeai."
            ) from e

        self._client = genai.Client(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        # `types` is imported at call time rather than at module load to
        # keep the lazy-import promise (see class docstring). Cheap —
        # Python caches module imports.
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=config,
        )

        # google-genai returns usage_metadata on the response object.
        # Field names match the older SDK. Guard with getattr because
        # the free tier sometimes omits usage info.
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = (
            getattr(usage, "candidates_token_count", None) if usage else None
        )

        return GenerationResult(
            text=response.text,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
