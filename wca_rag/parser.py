"""
Parser for the WCA Regulations markdown file.

Strategy: walk the file once, line-by-line. Each line is classified by the
shape of its regulation ID (e.g. "11e", "11e+", "11e1", "11e1+"). We
accumulate lines into a "current chunk" buffer, and flush it whenever a new
top-level regulation starts or the article changes.

A chunk is one top-level regulation (e.g. 11e) plus:
  - all its plus-suffix annotations (11e+, 11e++, ...)
  - all its nested children (11e1, 11e2a, ...)
  - all annotations of nested children (11e1+, 11e2++, ...)

Output: data/chunks.jsonl, one JSON object per line.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


@dataclass
class CrossRef:
    """A reference from one regulation to another regulation or article."""

    type: str  # "regulation" | "article"
    id: str    # e.g. "11i2" or "11"


@dataclass
class Chunk:
    """One unit of retrievable content. Schema documented in ARCHITECTURE.md §3.5."""

    regulation_id: str
    article: str
    article_title: str
    full_path_id: str
    label: str | None
    is_annotation: bool
    depth: int
    parent_id: str | None
    cross_references: list[CrossRef]
    char_count: int
    text_hash: str
    source_version: str
    text: str
    text_for_embedding: str

    def to_dict(self) -> dict:
        d = asdict(self)
        # CrossRef objects come out as nested dicts via asdict; that's what we want.
        return d


# ----------------------------------------------------------------------------
# Regexes — defined once, compiled at module load.
# ----------------------------------------------------------------------------

# Article heading: e.g. "## <article-11><incidents><incidents> Article 11: Incidents"
# Also matches lettered articles: "## <article-A><speedsolving><speedsolving> Article A: Speed Solving"
_ARTICLE_RE = re.compile(
    r"^##\s+<article-([0-9A-Z]+)>[^>]*>[^>]*>\s+Article\s+\1:\s+(.+?)\s*$"
)

# Source version line: "<version>Version: April 1, 2026"
_VERSION_RE = re.compile(r"^<version>Version:\s+(.+?)\s*$")

# Regulation line: "    - 11e) [CLARIFICATION] The WCA Delegate may grant..."
# Captures: leading whitespace, ID, optional [LABEL], remaining text.
#
# The ID alternation matches both shapes:
#   numeric article: digits + lowercase letter, optional digit-letter nested, optional pluses
#                    e.g. 11e, 11e+, 11e2, 11e2a, 11e2++, 1c+, 1h1a+
#   letter article:  uppercase letter + digits, optional letter-digit nested, optional pluses
#                    e.g. A1, A1a, A1a2, A1a2+, A2c1+, A7c6++++, B4c2, E2c4+
#
# Detailed validation happens in _classify_id; this regex's job is just to spot
# which lines are regulation lines at all.
_REGULATION_LINE_RE = re.compile(
    r"^(?P<indent>\s*)-\s+"
    r"(?P<id>"
    r"[0-9]+[a-z](?:[0-9]+[a-z]?)?\+*"
    r"|"
    r"[A-Z][0-9]+(?:[a-z][0-9]*[a-z]?)?\+*"
    r")\)\s*"
    r"(?:\[(?P<label>[A-Z]+)\]\s*)?"
    r"(?P<text>.*)$"
)

# Cross-reference link: "[anything](regulations:regulation:11i2)" or
# "[anything](regulations:article:11)". We capture type and id.
_XREF_RE = re.compile(r"\(regulations:(regulation|article):([^)]+)\)")


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

# Roles a regulation line can play within the chunking algorithm:
TOP_LEVEL = "top_level"            # 11e
ANNOTATION = "annotation"          # 11e+, 11e++
NESTED = "nested"                  # 11e1, 11e2a
NESTED_ANNOTATION = "nested_annot" # 11e1+, 11e2++


# A regulation ID has two shapes depending on the article kind:
#
# Numeric-article ID: e.g. "11e", "11e+", "11e2", "11e2a", "11e2+", "1c++"
#   article = digits        (the article number, "11")
#   key     = lowercase letter  (top-level discriminator within the article, "e")
#   nested  = optional digits + optional lowercase letter ("2", "2a")
#   plus    = optional run of '+' (annotation suffix)
#
# Letter-article ID: e.g. "A2", "A2b", "A1a2", "A1a2+", "A7c6++++"
#   article = single uppercase letter  (the article, "A")
#   key     = digits        (top-level discriminator within the article, "1", "7")
#   nested  = optional lowercase letter + optional digits + optional letter
#   plus    = optional run of '+'
#
# We use two regexes rather than one alternation so the named groups stay simple.
_ID_NUMERIC_RE = re.compile(
    r"^(?P<article>[0-9]+)"
    r"(?P<key>[a-z])"
    r"(?P<nested>[0-9]+[a-z]?)?"
    r"(?P<plus>\++)?$"
)
_ID_LETTER_RE = re.compile(
    r"^(?P<article>[A-Z])"
    r"(?P<key>[0-9]+)"
    r"(?P<nested>[a-z][0-9]*[a-z]?)?"
    r"(?P<plus>\++)?$"
)


def _classify_id(reg_id: str) -> tuple[str, str, str]:
    """Return (role, top_level_id, article_id) for a regulation ID.

    Handles both numeric articles (1..12) and letter articles (A, B, C, E, H, I).

    Examples:
        "11e"      -> ("top_level",     "11e",   "11")
        "11e++"    -> ("annotation",    "11e",   "11")
        "11e2"     -> ("nested",        "11e",   "11")
        "11e2a"    -> ("nested",        "11e",   "11")
        "11e2+"    -> ("nested_annot",  "11e",   "11")
        "A2"       -> ("top_level",     "A2",    "A")
        "A2c"      -> ("nested",        "A2",    "A")
        "A2c1+"    -> ("nested_annot",  "A2",    "A")
        "A1a2"     -> ("nested",        "A1",    "A")
    """
    m = _ID_NUMERIC_RE.match(reg_id)
    if m is None:
        m = _ID_LETTER_RE.match(reg_id)
    if m is None:
        raise ValueError(f"unrecognized regulation id: {reg_id!r}")

    article = m.group("article")
    key = m.group("key")
    nested = m.group("nested")
    plus = m.group("plus")
    top_level_id = f"{article}{key}"

    if nested is None and plus is None:
        return TOP_LEVEL, top_level_id, article
    if nested is None and plus is not None:
        return ANNOTATION, top_level_id, article
    if nested is not None and plus is None:
        return NESTED, top_level_id, article
    return NESTED_ANNOTATION, top_level_id, article


# ----------------------------------------------------------------------------
# Cross-reference extraction
# ----------------------------------------------------------------------------


def _extract_cross_references(text: str) -> list[CrossRef]:
    """Pull (type, id) pairs out of all regulations:* link targets in text.

    Deduplicates while preserving first-seen order.
    """
    seen: set[tuple[str, str]] = set()
    out: list[CrossRef] = []
    for kind, target in _XREF_RE.findall(text):
        key = (kind, target)
        if key in seen:
            continue
        seen.add(key)
        out.append(CrossRef(type=kind, id=target))
    return out


def _strip_cross_reference_markup(text: str) -> str:
    """Strip `[Display Text](regulations:...)` markdown links to just
    the display text.

    The WCA regulations encode cross-references as markdown links —
    `[Regulation A6c](regulations:regulation:A6c)`. Cross-reference
    metadata is captured separately by `_extract_cross_references`;
    here we collapse the link to its display text so the chunk body
    reads naturally for the LLM and any downstream display.

    Run this AFTER `_extract_cross_references` so the structured
    metadata is preserved.
    """
    # Match the full markdown link, capturing the display text only.
    # Pattern: `[anything](regulations:regulation:X)` or `[anything](regulations:article:X)`
    return re.sub(
        r"\[([^\]]+)\]\(regulations:(?:regulation|article):[^)]+\)",
        r"\1",
        text,
    )


# ----------------------------------------------------------------------------
# Main parse loop
# ----------------------------------------------------------------------------


@dataclass
class _ChunkBuilder:
    """Accumulator for the chunk currently being built."""

    regulation_id: str
    article: str
    article_title: str
    label: str | None
    lines: list[str] = field(default_factory=list)


def parse(markdown_path: Path) -> list[Chunk]:
    """Parse the WCA regulations markdown file into a list of Chunk objects."""
    text = markdown_path.read_text(encoding="utf-8")

    # Pull source version from the prefatory section.
    source_version = "unknown"
    for line in text.splitlines():
        m = _VERSION_RE.match(line)
        if m:
            source_version = m.group(1).strip()
            break

    chunks: list[Chunk] = []
    current_article: str | None = None
    current_article_title: str | None = None
    builder: _ChunkBuilder | None = None

    # Top-level IDs opened so far in the current article. Reset on article change.
    # Used to recognize "orphan annotations" — an annotation whose claimed parent
    # top-level was never opened (because that top-level was deleted in a past
    # revision; the WCA does not renumber on deletion). Such annotations are
    # absorbed into the most recently opened chunk in the same article.
    opened_in_article: set[str] = set()
    orphan_warnings: list[str] = []

    def flush():
        """Finalize the current builder into a Chunk and append it."""
        nonlocal builder
        if builder is None:
            return
        body = "\n".join(builder.lines)
        xrefs = _extract_cross_references(body)
        # Strip cross-reference markdown links from body for display/LLM use.
        # Must happen AFTER xref extraction (which needs the raw markup) and
        # BEFORE building text_for_embedding (which embeds the cleaned body).
        body = _strip_cross_reference_markup(body)
        text_for_embedding = (
            f"Article {builder.article}: {builder.article_title}\n"
            f"Regulation {builder.regulation_id}\n"
            f"\n"
            f"{body}"
        )
        chunk = Chunk(
            regulation_id=builder.regulation_id,
            article=builder.article,
            article_title=builder.article_title,
            full_path_id=f"{builder.article} > {builder.regulation_id}",
            label=builder.label,
            is_annotation=False,  # top-level chunks are never themselves annotations
            depth=1,
            parent_id=None,
            cross_references=xrefs,
            char_count=len(body),
            text_hash=hashlib.sha1(body.encode("utf-8")).hexdigest(),
            source_version=source_version,
            text=body,
            text_for_embedding=text_for_embedding,
        )
        chunks.append(chunk)
        builder = None

    seen_first_article = False

    for raw_line in text.splitlines():
        # Article heading?
        m = _ARTICLE_RE.match(raw_line)
        if m:
            flush()
            current_article = m.group(1)
            current_article_title = m.group(2).strip()
            opened_in_article = set()
            seen_first_article = True
            continue

        # Skip everything before the first article heading (table of contents,
        # notes, version metadata, etc.).
        if not seen_first_article:
            continue

        # Regulation line?
        m = _REGULATION_LINE_RE.match(raw_line)
        if not m:
            # Non-regulation lines (blank lines, prose paragraphs inside an
            # article — there shouldn't be any but be defensive) are appended
            # to the current chunk if one is open. Otherwise dropped.
            if builder is not None and raw_line.strip():
                builder.lines.append(raw_line)
            continue

        reg_id = m.group("id")
        label = m.group("label")
        # We don't actually need the captured text — we keep the whole raw line
        # in the chunk to preserve indentation, the leading "- ", and label.

        try:
            role, top_level_id, article_id = _classify_id(reg_id)
        except ValueError as exc:
            # Unknown ID shape — surface loudly during dev. If this fires on the
            # real corpus, the regex needs widening, not silent skipping.
            raise RuntimeError(f"line: {raw_line!r}") from exc

        # Sanity check: ID's article should match the article heading we're under.
        if current_article is not None and article_id != current_article:
            raise RuntimeError(
                f"regulation {reg_id!r} appears under Article "
                f"{current_article!r} but its ID claims article {article_id!r}"
            )

        if role == TOP_LEVEL:
            # New top-level regulation: flush previous chunk, start a new one.
            flush()
            builder = _ChunkBuilder(
                regulation_id=top_level_id,
                article=current_article or "",
                article_title=current_article_title or "",
                label=label,
                lines=[raw_line],
            )
            opened_in_article.add(top_level_id)
        elif role in (ANNOTATION, NESTED, NESTED_ANNOTATION):
            # Belongs to the currently-open top-level chunk.
            if builder is None:
                raise RuntimeError(
                    f"regulation {reg_id!r} (role={role}) appeared with no open "
                    f"top-level chunk. Likely a malformed file or a parser bug."
                )
            if top_level_id != builder.regulation_id:
                # Two distinct cases:
                #   (a) ANNOTATION whose claimed parent was never opened in this
                #       article -- this is the "deleted regulation" case (e.g.
                #       5c+ surviving after 5c was deleted). WCA convention: the
                #       annotation belongs with the most recent open chunk.
                #       Absorb and warn.
                #   (b) Anything else -- structurally suspicious. Raise.
                if role == ANNOTATION and top_level_id not in opened_in_article:
                    orphan_warnings.append(
                        f"orphan annotation {reg_id!r} (claims parent "
                        f"{top_level_id!r}, never opened in Article "
                        f"{current_article}); absorbed into "
                        f"{builder.regulation_id!r}"
                    )
                    builder.lines.append(raw_line)
                    continue
                raise RuntimeError(
                    f"regulation {reg_id!r} (role={role}) belongs to top-level "
                    f"{top_level_id!r}, but the open chunk is "
                    f"{builder.regulation_id!r}. The corpus may have a "
                    f"non-contiguous block, or the classifier is wrong."
                )
            builder.lines.append(raw_line)

    flush()

    if orphan_warnings:
        import sys
        print(
            f"WARNING: parser absorbed {len(orphan_warnings)} orphan "
            f"annotation(s):", file=sys.stderr,
        )
        for w in orphan_warnings:
            print(f"  {w}", file=sys.stderr)

    return chunks


# ----------------------------------------------------------------------------
# CLI: write chunks.jsonl and print a brief summary.
# ----------------------------------------------------------------------------


def _write_jsonl(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "data" / "raw" / "wca-regulations.md"
    dst = repo_root / "data" / "chunks.jsonl"

    if not src.exists():
        raise SystemExit(f"input not found: {src}")

    chunks = parse(src)
    _write_jsonl(chunks, dst)

    # Headline summary. Detailed inspection lives in scripts/inspect_chunks.py.
    sizes = sorted(c.char_count for c in chunks)
    n = len(sizes)
    print(f"parsed {n} chunks -> {dst.relative_to(repo_root)}")
    if n:
        print(
            f"  char_count: min={sizes[0]} "
            f"p50={sizes[n // 2]} "
            f"p90={sizes[int(n * 0.9)]} "
            f"max={sizes[-1]}"
        )
    big = [c for c in chunks if c.char_count > 5000]
    if big:
        print(f"  WARNING: {len(big)} chunk(s) > 5000 chars — inspect:")
        for c in big:
            print(f"    {c.regulation_id} ({c.char_count} chars)")


if __name__ == "__main__":
    main()
