"""Backwards-compatible entry point for the Graph QA chain.

``from graphRag_cypher_chain_cd import graph_cypher_chain`` keeps working, but the
chain is now built lazily on first access (no Neo4j connection at import time)
and is a :class:`graphrag.graph_qa.GraphQAPipeline`, which validates every
generated query as read-only instead of running it with
``allow_dangerous_requests=True``.
"""

from __future__ import annotations

import threading
from typing import Any

from dotenv import load_dotenv

from graphrag.config import Settings
from graphrag.factory import build_graph_qa_pipeline
from graphrag.prompts import ALLOWED_LABELS as labels  # noqa: F401  (legacy name)
from graphrag.prompts import CYPHER_PROMPT as cypher_prompt  # noqa: F401
from graphrag.prompts import QA_PROMPT as qa_prompt  # noqa: F401

_lock = threading.Lock()
_chain = None


def get_graph_cypher_chain():
    global _chain
    with _lock:
        if _chain is None:
            load_dotenv()
            _chain = build_graph_qa_pipeline(Settings.from_env())
        return _chain


def __getattr__(name: str) -> Any:
    if name == "graph_cypher_chain":
        return get_graph_cypher_chain()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
