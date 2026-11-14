"""Legacy OpenAI variant of the Graph QA chain.

Reuses the hardened :class:`graphrag.graph_qa.GraphQAPipeline` (read-only
Cypher validation + read-only bounded execution) with OpenAI chat models.
Requires ``pip install langchain-openai`` and ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from graphrag.config import Settings  # noqa: E402
from graphrag.factory import build_graph_qa_pipeline  # noqa: E402
from graphrag.prompts import ALLOWED_LABELS as labels  # noqa: E402,F401

_lock = threading.Lock()
_chain = None


def openai_llm_factory(model: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0)


def get_graph_cypher_chain():
    global _chain
    with _lock:
        if _chain is None:
            load_dotenv()
            _chain = build_graph_qa_pipeline(Settings.from_env(), llm_factory=openai_llm_factory)
        return _chain


def __getattr__(name: str) -> Any:
    if name == "graph_cypher_chain":
        return get_graph_cypher_chain()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
