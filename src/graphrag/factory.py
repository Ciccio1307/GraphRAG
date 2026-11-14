"""Wiring of real external services (Neo4j, Gemini, DuckDuckGo).

Everything that performs network I/O at construction time lives here, so the
rest of the package can be imported and tested without credentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from .config import Settings
from .executor import CypherExecutor, Neo4jReadOnlyExecutor
from .graph_qa import GraphQAPipeline
from .hybrid import HybridRAGService, WebSearch
from .memory import SessionHistoryStore
from .sanitize import truncate
from .prompts import ALLOWED_LABELS

logger = logging.getLogger(__name__)

LLMFactory = Callable[[str], BaseChatModel]


def gemini_llm_factory(model: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=0)


def build_schema(structured_schema: dict[str, Any], include_types: tuple[str, ...] = ALLOWED_LABELS) -> str:
    """Schema text shown to the Cypher LLM, restricted to ``include_types``."""
    from langchain_neo4j.chains.graph_qa.cypher import construct_schema

    return construct_schema(structured_schema, list(include_types), [], False)


def build_corrector(structured_schema: dict[str, Any]) -> Callable[[str], str]:
    """Relationship-direction corrector (what ``validate_cypher=True`` used to do)."""
    from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

    return CypherQueryCorrector(
        [Schema(r["start"], r["type"], r["end"]) for r in structured_schema.get("relationships", [])]
    )


def fetch_structured_schema(settings: Settings) -> dict[str, Any]:
    """Introspect the graph schema once at start-up (requires APOC)."""
    from langchain_neo4j import Neo4jGraph

    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        refresh_schema=True,
    )
    try:
        return graph.get_structured_schema
    finally:
        graph.close()


def build_executor(settings: Settings) -> Neo4jReadOnlyExecutor:
    import neo4j

    driver = neo4j.GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    driver.verify_connectivity()
    return Neo4jReadOnlyExecutor(
        driver,
        database=settings.neo4j_database,
        max_records=settings.top_k,
        timeout_s=settings.query_timeout_s,
    )


def build_web_search(settings: Settings) -> WebSearch | None:
    if not settings.enable_web_search:
        return None
    from langchain_community.tools import DuckDuckGoSearchRun

    tool = DuckDuckGoSearchRun()
    return tool.invoke


def build_graph_qa_pipeline(
    settings: Settings,
    *,
    llm_factory: LLMFactory = gemini_llm_factory,
    executor: CypherExecutor | None = None,
    structured_schema: dict[str, Any] | None = None,
) -> GraphQAPipeline:
    structured_schema = structured_schema if structured_schema is not None else fetch_structured_schema(settings)
    return GraphQAPipeline(
        cypher_llm=llm_factory(settings.cypher_model),
        qa_llm=llm_factory(settings.qa_model),
        executor=executor if executor is not None else build_executor(settings),
        schema=build_schema(structured_schema),
        corrector=build_corrector(structured_schema),
    )


class ChatAgent:
    """Per-session conversational wrapper around :class:`HybridRAGService`.

    Drop-in replacement for the (deprecated) ``RunnableWithMessageHistory``
    usage: ``agent.invoke({"input": q}, config={"configurable": {"session_id": sid}})``
    returns ``{"output": answer}`` and records the turn in that session only.
    """

    def __init__(self, service: HybridRAGService, store: SessionHistoryStore, *, max_stored_chars: int = 4000) -> None:
        self._service = service
        self._store = store
        self._max_stored_chars = max_stored_chars

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, str]:
        session_id = ((config or {}).get("configurable") or {}).get("session_id")
        if not session_id:
            raise ValueError("config['configurable']['session_id'] is required")
        history = self._store.get(str(session_id))
        question = inputs.get("input")
        output = self._service.run({"input": question, "chat_history": list(history.messages)})
        history.add_messages(
            [
                HumanMessage(content=truncate(str(question or ""), self._max_stored_chars)),
                AIMessage(content=truncate(output, self._max_stored_chars)),
            ]
        )
        return {"output": output}


@dataclass
class AgentBundle:
    agent: ChatAgent
    service: HybridRAGService
    history_store: SessionHistoryStore


def build_agent(
    settings: Settings,
    *,
    llm_factory: LLMFactory = gemini_llm_factory,
    graph_pipeline: GraphQAPipeline | None = None,
    web_search: WebSearch | None | bool = True,
    history_store: SessionHistoryStore | None = None,
) -> AgentBundle:
    """Assemble the chat agent.

    ``web_search=True`` means "build from settings"; pass a callable to inject
    one, or ``None``/``False`` to disable it.
    """
    pipeline = graph_pipeline or build_graph_qa_pipeline(settings, llm_factory=llm_factory)
    if web_search is True:
        search: WebSearch | None = build_web_search(settings)
    elif web_search is False:
        search = None
    else:
        search = web_search

    service = HybridRAGService(
        graph_answerer=pipeline.answer,
        synthesizer_llm=llm_factory(settings.agent_model),
        web_search=search,
        max_question_chars=settings.max_question_chars,
        web_max_chars=settings.web_max_chars,
        history_max_messages=settings.max_history_messages,
    )
    store = history_store or SessionHistoryStore(
        max_sessions=settings.max_sessions, max_messages=settings.max_history_messages
    )
    return AgentBundle(agent=ChatAgent(service, store), service=service, history_store=store)

