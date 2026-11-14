import logging
import uuid

import streamlit as st

from graph_rag_agent import get_agent_bundle
from graphrag.config import ConfigError
from graphrag.sanitize import sanitize_markdown_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("graphrag.ui")

FALLBACK_ANSWER = "I couldn't generate an answer from the graph."
MAX_INPUT_CHARS = 2000


@st.cache_resource(show_spinner="Connecting to the knowledge graph...")
def load_agent():
    return get_agent_bundle().agent


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Chatbot")
    st.markdown(
        """
        This chatbot interfaces with a **Graph-RAG system** based on:
        - LangChain
        - Neo4j
        - Gemini LLM

        It is designed to answer questions about the **BSG gene**
        and its biological relationships.
        """
    )

    st.header("Example Questions")
    st.markdown("- What diseases are related to the BSG gene?")
    st.markdown("- Find genes associated with diabetes")
    st.markdown("- How many connections does a disease node have?")
    st.markdown("- What is the average distance from BSG?")
    st.markdown("- Which proteins interact with BSG?")

# =========================================================
# Main UI
# =========================================================
st.title("🧬 GraphRAG Neo4j Chatbot")
st.info(
    "Ask questions about BSG genes and their interactions "
    "stored in a Neo4j knowledge graph."
)

# =========================================================
# Session state: messages + a per-browser-session id, so chat
# memory is never shared between different users.
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# Chat input
# =========================================================
if prompt := st.chat_input("What do you want to know?", max_chars=MAX_INPUT_CHARS):

    st.chat_message("user").markdown(sanitize_markdown_output(prompt))
    st.session_state.messages.append(
        {"role": "user", "content": sanitize_markdown_output(prompt)}
    )

    with st.spinner("Querying the knowledge graph..."):
        try:
            agent = load_agent()
            response = agent.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            output = response.get("output") or FALLBACK_ANSWER
        except ConfigError as exc:
            logger.error("Configuration error: %s", exc)
            output = "The service is not configured correctly. Please contact the administrator."
        except Exception:
            logger.exception("Agent invocation failed")
            output = "An unexpected error occurred. Please try again later."

    output = sanitize_markdown_output(output)
    with st.chat_message("assistant"):
        st.markdown(output)

    st.session_state.messages.append(
        {"role": "assistant", "content": output}
    )
