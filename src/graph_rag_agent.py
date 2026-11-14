"""Backwards-compatible entry point for the hybrid Graph-RAG chat agent.

``from graph_rag_agent import agent_with_chat_history`` keeps working; the agent
is built lazily on first access instead of at import time.
"""

from __future__ import annotations

import threading
from typing import Any

from dotenv import load_dotenv

from graphrag.config import Settings
from graphrag.factory import AgentBundle, build_agent

_lock = threading.Lock()
_bundle: AgentBundle | None = None


def get_agent_bundle() -> AgentBundle:
    global _bundle
    with _lock:
        if _bundle is None:
            load_dotenv()
            _bundle = build_agent(Settings.from_env())
        return _bundle


def __getattr__(name: str) -> Any:
    if name == "agent_with_chat_history":
        return get_agent_bundle().agent
    if name == "graph_rag_executor":
        return get_agent_bundle().service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
