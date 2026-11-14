from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graphrag.memory import BoundedChatMessageHistory, SessionHistoryStore


def test_history_is_bounded():
    h = BoundedChatMessageHistory(max_messages=3)
    for i in range(10):
        h.add_messages([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])
    assert [m.content for m in h.messages] == ["a8", "q9", "a9"]


def test_history_invalid_bound():
    with pytest.raises(Exception):
        BoundedChatMessageHistory(max_messages=0)


def test_sessions_are_isolated():
    store = SessionHistoryStore()
    store.get("alice").add_messages([HumanMessage(content="private")])
    assert store.get("bob").messages == []
    assert store.get("alice").messages[0].content == "private"


def test_lru_eviction():
    store = SessionHistoryStore(max_sessions=2)
    store.get("a")
    store.get("b")
    store.get("a")  # refresh a
    store.get("c")  # evicts b
    assert "a" in store and "c" in store and "b" not in store
    assert len(store) == 2


def test_clear():
    store = SessionHistoryStore()
    store.get("x")
    store.clear("x")
    store.clear("missing")
    assert "x" not in store


@pytest.mark.parametrize("bad", ["", None, 5])
def test_invalid_session_id(bad):
    with pytest.raises(ValueError):
        SessionHistoryStore().get(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"max_sessions": 0}, {"max_messages": -1}])
def test_invalid_store_limits(kwargs):
    with pytest.raises(ValueError):
        SessionHistoryStore(**kwargs)


def test_concurrent_access_returns_single_history_per_session():
    store = SessionHistoryStore(max_sessions=10_000)
    with ThreadPoolExecutor(max_workers=16) as pool:
        histories = list(pool.map(lambda i: store.get(f"s{i % 50}"), range(2000)))
    assert len(store) == 50
    by_id = {}
    for i, h in enumerate(histories):
        by_id.setdefault(i % 50, set()).add(id(h))
    assert all(len(ids) == 1 for ids in by_id.values())


def test_store_is_callable_as_factory():
    store = SessionHistoryStore()
    assert store("abc") is store.get("abc")
