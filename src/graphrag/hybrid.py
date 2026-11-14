"""Hybrid Graph-RAG + web search answer synthesis."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Mapping, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .graph_qa import GraphQAResult, GraphQAStatus
from .prompts import SYNTHESIS_SYSTEM_PROMPT
from .sanitize import InvalidQuestionError, fence_untrusted, normalize_question, truncate

logger = logging.getLogger(__name__)

WebSearch = Callable[[str], str]
GraphAnswerer = Callable[[str], GraphQAResult]

MSG_EMPTY_QUESTION = "Please enter a question."
MSG_TOO_LONG = "La domanda è troppo lunga. Riformulala in modo più conciso."
MSG_SYNTHESIS_ERROR = "Si è verificato un errore durante la generazione della risposta. Riprova più tardi."

GRAPH_NO_INFO = "Nessuna informazione trovata nel database interno."
GRAPH_REJECTED = "La query generata per il database è stata bloccata per motivi di sicurezza."
GRAPH_ERROR = "Database interno temporaneamente non disponibile."
WEB_UNAVAILABLE = "Ricerca web non disponibile."
WEB_DISABLED = "Ricerca web disabilitata."

_GRAPH_STATUS_TEXT = {
    GraphQAStatus.EMPTY: GRAPH_NO_INFO,
    GraphQAStatus.REJECTED: GRAPH_REJECTED,
    GraphQAStatus.ERROR: GRAPH_ERROR,
}


class HybridRAGService:
    """Runs graph retrieval and web search **concurrently**, then synthesises.

    The original runner executed them sequentially, so latency was the *sum*
    of the two; now it is their maximum.
    """

    def __init__(
        self,
        *,
        graph_answerer: GraphAnswerer,
        synthesizer_llm: BaseLanguageModel,
        web_search: WebSearch | None = None,
        max_question_chars: int = 2000,
        web_max_chars: int = 4000,
        history_max_messages: int = 10,
        retrieval_timeout_s: float = 60.0,
        pool: ThreadPoolExecutor | None = None,
    ) -> None:
        self._graph_answerer = graph_answerer
        self._llm = synthesizer_llm
        self._web_search = web_search
        self._max_question_chars = max_question_chars
        self._web_max_chars = web_max_chars
        self._history_max_messages = history_max_messages
        self._timeout_s = retrieval_timeout_s
        self._pool = pool or ThreadPoolExecutor(max_workers=8, thread_name_prefix="graphrag")

    # Public entry point compatible with RunnableLambda / RunnableWithMessageHistory.
    def __call__(self, inputs: Mapping[str, Any]) -> dict[str, str]:
        return {"output": self.run(inputs)}

    def run(self, inputs: Mapping[str, Any]) -> str:
        try:
            question = normalize_question(inputs.get("input"), max_chars=self._max_question_chars)
        except InvalidQuestionError:
            return MSG_TOO_LONG
        if not question:
            return MSG_EMPTY_QUESTION

        graph_future = self._pool.submit(self._graph_answerer, question)
        web_future = self._pool.submit(self._search_web, question) if self._web_search else None

        graph_content = self._graph_content(graph_future)
        web_content = self._web_content(web_future)

        messages = self.build_messages(
            question=question,
            graph_content=graph_content,
            web_content=web_content,
            history=inputs.get("chat_history") or [],
        )
        try:
            response = self._llm.invoke(messages)
        except Exception:
            logger.exception("Answer synthesis failed")
            return MSG_SYNTHESIS_ERROR
        content = getattr(response, "content", response)
        if isinstance(content, list):  # multimodal content blocks
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return str(content).strip() or MSG_SYNTHESIS_ERROR

    def build_messages(
        self,
        *,
        question: str,
        graph_content: str,
        web_content: str,
        history: Sequence[BaseMessage],
    ) -> list[BaseMessage]:
        recent = list(history)[-self._history_max_messages:] if self._history_max_messages else []
        history_text = "\n".join(_format_turn(m) for m in recent) or "(nessuna)"
        human = "\n\n".join(
            [
                fence_untrusted(question, "domanda"),
                fence_untrusted(history_text, "cronologia"),
                fence_untrusted(graph_content, "fonte_db"),
                fence_untrusted(web_content, "fonte_web"),
                "Risposta finale:",
            ]
        )
        return [SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT), HumanMessage(content=human)]

    def _search_web(self, question: str) -> str:
        assert self._web_search is not None
        return str(self._web_search(question) or "")

    def _graph_content(self, future: Future[GraphQAResult]) -> str:
        try:
            result = future.result(timeout=self._timeout_s)
        except FutureTimeout:
            logger.warning("Graph retrieval timed out")
            return GRAPH_ERROR
        except Exception:
            logger.exception("Graph retrieval failed")
            return GRAPH_ERROR
        if result.has_answer:
            return result.answer
        return _GRAPH_STATUS_TEXT.get(result.status, GRAPH_NO_INFO)

    def _web_content(self, future: Future[str] | None) -> str:
        if future is None:
            return WEB_DISABLED
        try:
            text = future.result(timeout=self._timeout_s)
        except Exception:
            logger.warning("Web search failed", exc_info=True)
            return WEB_UNAVAILABLE
        return truncate(text, self._web_max_chars) if text.strip() else WEB_UNAVAILABLE


def _format_turn(message: BaseMessage) -> str:
    role = "Utente" if isinstance(message, HumanMessage) else "Assistente" if isinstance(message, AIMessage) else message.type
    return f"{role}: {truncate(str(message.content), 1000)}"
