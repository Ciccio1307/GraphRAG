import streamlit as st
from graph_rag_agent import agent_with_chat_history

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
# Session state for chat messages
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# Display previous messages
# =========================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# Chat input
# =========================================================
if prompt := st.chat_input("What do you want to know?"):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.spinner("Querying the knowledge graph..."):
        response = agent_with_chat_history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": "streamlit-session"}}
        )

        output = response.get(
            "output",
            "I couldn't generate an answer from the graph."
        )

    with st.chat_message("assistant"):
        st.markdown(output)

    st.session_state.messages.append(
        {"role": "assistant", "content": output}
    )
