"""Bounded, thread-safe in-memory chat history store."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Sequence

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import Field


class BoundedChatMessageHistory(InMemoryChatMessageHistory):
    """Chat history that keeps only the most recent ``max_messages`` messages."""

    max_messages: int = Field(default=20, gt=0)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        super().add_messages(messages)
        overflow = len(self.messages) - self.max_messages
        if overflow > 0:
            del self.messages[:overflow]


class SessionHistoryStore:
    """Maps session ids to histories with an LRU cap on the number of sessions.

    The original implementation used a plain module-level dict, shared by all
    users through a single hard-coded session id and never evicted, which
    leaked conversations between users and grew without bound.
    """

    def __init__(self, *, max_sessions: int = 1000, max_messages: int = 20) -> None:
        if max_sessions <= 0 or max_messages <= 0:
            raise ValueError("max_sessions and max_messages must be positive")
        self._max_sessions = max_sessions
        self._max_messages = max_messages
        self._sessions: OrderedDict[str, BoundedChatMessageHistory] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> BoundedChatMessageHistory:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                history = BoundedChatMessageHistory(max_messages=self._max_messages)
                self._sessions[session_id] = history
                while len(self._sessions) > self._max_sessions:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(session_id)
            return history

    __call__ = get  # usable directly as RunnableWithMessageHistory's factory

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, session_id: object) -> bool:
        with self._lock:
            return session_id in self._sessions
