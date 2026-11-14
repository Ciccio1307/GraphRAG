"""Typed application settings, loaded and validated from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlparse

DEFAULT_MODEL = "models/gemini-2.5-flash-lite"

ALLOWED_NEO4J_SCHEMES = frozenset(
    {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """Raised when the environment does not contain a valid configuration."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration.

    The password is excluded from ``repr`` so settings can be logged safely.
    """

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str = field(repr=False)
    neo4j_database: str | None = None

    qa_model: str = DEFAULT_MODEL
    cypher_model: str = DEFAULT_MODEL
    agent_model: str = DEFAULT_MODEL

    top_k: int = 100
    query_timeout_s: float = 15.0
    max_question_chars: int = 2000
    max_history_messages: int = 20
    max_sessions: int = 1000
    enable_web_search: bool = True
    web_max_chars: int = 4000
    verbose: bool = False

    def __post_init__(self) -> None:
        scheme = urlparse(self.neo4j_uri).scheme.lower()
        if scheme not in ALLOWED_NEO4J_SCHEMES:
            raise ConfigError(
                f"NEO4J_URI has unsupported scheme {scheme!r}; "
                f"expected one of {sorted(ALLOWED_NEO4J_SCHEMES)}"
            )
        _require_positive("GRAPHRAG_TOP_K", self.top_k, upper=10_000)
        _require_positive("GRAPHRAG_QUERY_TIMEOUT_S", self.query_timeout_s, upper=600)
        _require_positive("GRAPHRAG_MAX_QUESTION_CHARS", self.max_question_chars, upper=100_000)
        _require_positive("GRAPHRAG_MAX_HISTORY_MESSAGES", self.max_history_messages, upper=1_000)
        _require_positive("GRAPHRAG_MAX_SESSIONS", self.max_sessions, upper=1_000_000)
        _require_positive("GRAPHRAG_WEB_MAX_CHARS", self.web_max_chars, upper=100_000)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from ``env`` (defaults to ``os.environ``)."""
        env = os.environ if env is None else env

        missing = [
            name
            for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
            if not _clean(env.get(name))
        ]
        if missing:
            raise ConfigError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        return cls(
            neo4j_uri=_clean(env["NEO4J_URI"]),
            neo4j_user=_clean(env["NEO4J_USER"]),
            neo4j_password=_clean(env["NEO4J_PASSWORD"]),
            neo4j_database=_clean(env.get("NEO4J_DATABASE")) or None,
            qa_model=_clean(env.get("GRAPHRAG_QA_MODEL")) or DEFAULT_MODEL,
            cypher_model=_clean(env.get("GRAPHRAG_CYPHER_MODEL")) or DEFAULT_MODEL,
            agent_model=_clean(env.get("GRAPHRAG_AGENT_MODEL")) or DEFAULT_MODEL,
            top_k=_int(env, "GRAPHRAG_TOP_K", 100),
            query_timeout_s=_float(env, "GRAPHRAG_QUERY_TIMEOUT_S", 15.0),
            max_question_chars=_int(env, "GRAPHRAG_MAX_QUESTION_CHARS", 2000),
            max_history_messages=_int(env, "GRAPHRAG_MAX_HISTORY_MESSAGES", 20),
            max_sessions=_int(env, "GRAPHRAG_MAX_SESSIONS", 1000),
            enable_web_search=_bool(env, "GRAPHRAG_ENABLE_WEB_SEARCH", True),
            web_max_chars=_int(env, "GRAPHRAG_WEB_MAX_CHARS", 4000),
            verbose=_bool(env, "GRAPHRAG_VERBOSE", False),
        )


def _clean(value: str | None) -> str:
    """Strip whitespace and one level of surrounding quotes (common in .env files)."""
    if value is None:
        return ""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _clean(env.get(name))
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _clean(env.get(name))
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        raise ConfigError(f"{name} must be a finite number, got {raw!r}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _clean(env.get(name)).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}")


def _require_positive(name: str, value: float, *, upper: float) -> None:
    if not 0 < value <= upper:
        raise ConfigError(f"{name} must be in (0, {upper}], got {value!r}")
