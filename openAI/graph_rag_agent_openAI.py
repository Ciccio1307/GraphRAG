"""Legacy OpenAI functions-agent variant (kept for reference).

Changes from the original:
* imports the OpenAI chain module (it previously imported the Gemini one);
* the agent prompt is defined locally instead of ``hub.pull(...)``, which
  downloaded an unpinned prompt from a remote hub at import time;
* the session store is bounded and thread-safe;
* everything is built lazily.

Requires ``pip install langchain-openai langchain-classic``.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from graphRag_cypher_chain_cd_openAI import get_graph_cypher_chain, openai_llm_factory
from graphrag.memory import SessionHistoryStore

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

_lock = threading.Lock()
_agent = None
store = SessionHistoryStore()


def get_agent_with_chat_history():
    global _agent
    with _lock:
        if _agent is None:
            from langchain_classic.agents import AgentExecutor, Tool, create_openai_functions_agent

            load_dotenv()
            graph_chain = get_graph_cypher_chain()
            tools = [
                Tool(
                    name="Graph",
                    func=lambda q: graph_chain.invoke(q)["result"],
                    description=(
                        "Useful for answering questions about a Neo4j biomedical graph "
                        "(genes, diseases, proteins, drugs). Input should be the full question."
                    ),
                )
            ]
            chat_model = openai_llm_factory(os.getenv("GRAPHRAG_AGENT_MODEL", "gpt-4o-mini"))
            agent = create_openai_functions_agent(llm=chat_model, prompt=AGENT_PROMPT, tools=tools)
            executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
            _agent = RunnableWithMessageHistory(
                executor,
                store,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="output",
            )
        return _agent


def __getattr__(name: str) -> Any:
    if name == "agent_with_chat_history":
        return get_agent_with_chat_history()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
