import os
from typing import Dict
from dotenv import load_dotenv

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

from graphRag_cypher_chain_cd import graph_cypher_chain

load_dotenv()

# ======================================================
# Configurazione Strumenti
# ======================================================
# 1. Ricerca Web (DuckDuckGo - gratuito e veloce)
web_search = DuckDuckGoSearchRun()

synthesizer_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    temperature=0
)

# ======================================================
# IBRIDO Graph-RAG + Web Search Runner
# ======================================================
def hybrid_rag_runner(input_dict: Dict) -> Dict:
    question = input_dict.get("input", "").strip()

    if not question:
        return {"output": "Please enter a question."}

    graph_content = ""
    try:
        graph_result = graph_cypher_chain.invoke({"query": question})
        raw_output = graph_result.get("result", "")
        if "don't know" in str(raw_output).lower() or not raw_output:
            graph_content = "Nessuna informazione trovata nel database interno."
        else:
            graph_content = str(raw_output)
    except Exception as e:
        graph_content = f"Errore nel database: {str(e)}"

    web_content = ""
    try:
        web_content = web_search.invoke(question)
    except Exception as e:
        web_content = "Ricerca web non disponibile."

    prompt = f"""
    Sei un assistente scientifico esperto in genetica.
    Hai ricevuto una domanda dall'utente: "{question}"

    Hai a disposizione due fonti di informazione:
    
    [FONTE 1: DATABASE INTERNO NEO4J (Alta Affidabilità)]
    {graph_content}

    [FONTE 2: RICERCA WEB (Contesto Generale)]
    {web_content}

    ISTRUZIONI:
    1. Dai assoluta priorità alla FONTE 1. Se il database contiene la risposta, usala come base principale.
    2. Usa la FONTE 2 per arricchire la risposta o spiegare termini complessi, o se la Fonte 1 è vuota.
    3. Specifica chiaramente se un'informazione viene dal database interno ("Secondo il nostro database...") o dal web.
    4. Sii conciso e professionale.

    Risposta Finale:
    """
    
    try:
        final_response = synthesizer_llm.invoke(prompt)
        return {"output": final_response.content}
    except Exception as e:
        return {"output": f"Error synthesizing response: {str(e)}"}

graph_rag_executor = RunnableLambda(hybrid_rag_runner)

# ======================================================
# Gestione Memoria Chat
# ======================================================
_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    graph_rag_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)