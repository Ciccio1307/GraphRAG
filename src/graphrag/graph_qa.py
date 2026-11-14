"""Natural-language question -> Cypher -> records -> answer pipeline.

Replaces ``GraphCypherQAChain(allow_dangerous_requests=True)``: the steps are the
same (generate, extract, correct relationship directions, execute, summarise),
but generated Cypher is validated as read-only and executed through a bounded,
read-only executor, and every failure mode is reported as a typed status
instead of an exception string.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate

from .cypher_guard import CypherSecurityError, validate_read_only_cypher
from .executor import CypherExecutor, Record
from .prompts import CYPHER_PROMPT, QA_PROMPT

logger = logging.getLogger(__name__)

CypherCorrector = Callable[[str], str]

_NO_ANSWER_MARKERS = ("don't know", "do not know", "non lo so", "non so ")


class GraphQAStatus(str, enum.Enum):
    OK = "ok"                 # records found and summarised
    EMPTY = "empty"           # query ran but nothing relevant was found
    REJECTED = "rejected"     # generated Cypher failed the security guard / corrector
    ERROR = "error"           # LLM or database failure


@dataclass(frozen=True)
class GraphQAResult:
    status: GraphQAStatus
    answer: str = ""
    cypher: str | None = None
    records: list[Record] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        return self.status is GraphQAStatus.OK and bool(self.answer)


def extract_cypher(text: str) -> str:
    """Extract Cypher from LLM output (code fences, quoting of multi-word names)."""
    from langchain_neo4j.chains.graph_qa.cypher import extract_cypher as _extract

    return _extract(text).strip()


class GraphQAPipeline:
    """Answer questions over a Neo4j graph using LLM-generated read-only Cypher."""

    def __init__(
        self,
        *,
        cypher_llm: BaseLanguageModel,
        qa_llm: BaseLanguageModel,
        executor: CypherExecutor,
        schema: str,
        corrector: CypherCorrector | None = None,
        cypher_prompt: BasePromptTemplate = CYPHER_PROMPT,
        qa_prompt: BasePromptTemplate = QA_PROMPT,
        max_context_chars: int = 20_000,
    ) -> None:
        self._cypher_chain = cypher_prompt | cypher_llm | StrOutputParser()
        self._qa_chain = qa_prompt | qa_llm | StrOutputParser()
        self._executor = executor
        self._schema = schema
        self._corrector = corrector
        self._max_context_chars = max_context_chars

    @property
    def schema(self) -> str:
        return self._schema

    def generate_cypher(self, question: str) -> str:
        raw = self._cypher_chain.invoke({"question": question, "schema": self._schema})
        cypher = extract_cypher(raw)
        if self._corrector is not None and cypher:
            # CypherQueryCorrector returns "" when the query references
            # relationships that do not exist in the schema.
            cypher = self._corrector(cypher)
        return cypher

    def answer(self, question: str) -> GraphQAResult:
        question = (question or "").strip()
        if not question:
            return GraphQAResult(GraphQAStatus.EMPTY)

        try:
            cypher = self.generate_cypher(question)
        except Exception:  # LLM/network errors must not reach the user verbatim
            logger.exception("Cypher generation failed")
            return GraphQAResult(GraphQAStatus.ERROR)

        if not cypher:
            return GraphQAResult(GraphQAStatus.REJECTED, cypher=cypher)

        try:
            cypher = validate_read_only_cypher(cypher)
        except CypherSecurityError as exc:
            logger.warning("Rejected generated Cypher (%s): %r", exc, cypher)
            return GraphQAResult(GraphQAStatus.REJECTED, cypher=cypher)

        logger.debug("Generated Cypher: %s", cypher)

        try:
            records = self._executor.run(cypher)
        except CypherSecurityError as exc:
            logger.warning("Executor rejected Cypher (%s): %r", exc, cypher)
            return GraphQAResult(GraphQAStatus.REJECTED, cypher=cypher)
        except Exception:
            logger.exception("Cypher execution failed")
            return GraphQAResult(GraphQAStatus.ERROR, cypher=cypher)

        if not records:
            # Skip the QA LLM round-trip entirely: nothing to summarise.
            return GraphQAResult(GraphQAStatus.EMPTY, cypher=cypher)

        context = _format_context(records, self._max_context_chars)
        try:
            answer = self._qa_chain.invoke({"question": question, "context": context})
        except Exception:
            logger.exception("QA generation failed")
            return GraphQAResult(GraphQAStatus.ERROR, cypher=cypher, records=records)

        answer = (answer or "").strip()
        if not answer or _is_no_answer(answer):
            return GraphQAResult(GraphQAStatus.EMPTY, cypher=cypher, records=records)
        return GraphQAResult(GraphQAStatus.OK, answer=answer, cypher=cypher, records=records)

    # ------------------------------------------------------------------
    # Backwards compatibility with the old GraphCypherQAChain interface:
    # ``chain.invoke({"query": q})["result"]`` and ``chain.invoke(q)``.
    # ------------------------------------------------------------------
    def invoke(self, inputs: str | Mapping[str, Any], *_: Any, **__: Any) -> dict[str, Any]:
        if isinstance(inputs, str):
            question = inputs
        else:
            question = str(inputs.get("query") or inputs.get("question") or "")
        result = self.answer(question)
        return {"query": question, "result": result.answer, "status": result.status.value}


def _is_no_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _NO_ANSWER_MARKERS)


def _format_context(records: list[Record], max_chars: int) -> str:
    text = str(records)
    if len(text) > max_chars:
        text = text[:max_chars] + " …"
    return text
