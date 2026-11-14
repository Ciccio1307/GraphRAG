from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from graphrag.config import Settings
from graphrag.cypher_guard import validate_read_only_cypher

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).resolve().parents[1] / "data"


def load_fixture(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def question_input(record: Mapping[str, Any]) -> Any:
    if "input_repeat" in record:
        spec = record["input_repeat"]
        return spec["text"] * spec["times"]
    return record["input"]


class FakeExecutor:
    """In-memory CypherExecutor that validates like the real one."""

    def __init__(self, records: list[dict[str, Any]] | None = None, *, error: Exception | None = None):
        self.records = records if records is not None else []
        self.error = error
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def run(self, cypher: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            self.calls.append(cypher)
        validate_read_only_cypher(cypher)
        if self.error:
            raise self.error
        return list(self.records)


class RecordingLLM(FakeListChatModel):
    """Fake chat model that also records every prompt it receives."""

    prompts: list[Any] = []

    def invoke(self, input: Any, *args: Any, **kwargs: Any):  # noqa: A002
        self.prompts.append(input)
        return super().invoke(input, *args, **kwargs)


def make_llm(*responses: str) -> RecordingLLM:
    return RecordingLLM(responses=list(responses), prompts=[])


class FailingLLM(FakeListChatModel):
    def invoke(self, *args: Any, **kwargs: Any):
        raise RuntimeError("API key invalid: AIza-SECRET-KEY-should-not-leak")


@pytest.fixture
def settings() -> Settings:
    return Settings(neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="s3cr3t!")


@pytest.fixture
def fake_executor_factory() -> Callable[..., FakeExecutor]:
    return FakeExecutor
