"""Shared text normalization and overlap helpers."""

from __future__ import annotations

import json
import re
from typing import Any

STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "not",
        "but",
        "you",
        "your",
        "our",
        "their",
        "its",
        "into",
        "about",
        "what",
        "which",
        "when",
        "where",
        "who",
        "how",
        "why",
        "can",
        "should",
        "would",
        "could",
        "may",
        "than",
        "then",
        "also",
        "such",
        "using",
        "used",
        "use",
        "each",
        "any",
        "all",
        "per",
        "one",
        "two",
        "three",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Case-fold and collapse whitespace. Used by the deterministic validator."""
    return _SPACE_RE.sub(" ", value.casefold()).strip()


def content_words(value: str) -> set[str]:
    return {w for w in _WORD_RE.findall(value.casefold()) if w not in STOPWORDS}


def word_count(value: str) -> int:
    return len(re.findall(r"\S+", value))


def overlap_count(left: str, right: str) -> int:
    return len(content_words(left) & content_words(right))


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(stripped[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    return _SPACE_RE.sub(" ", text).strip()
