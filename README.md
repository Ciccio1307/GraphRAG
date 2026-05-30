# 🧬 GraphRAG Medical Chatbot  
### Graph-Based Retrieval-Augmented Generation with Neo4j, LangChain & Google Gemini

![System Architecture](solution.png)

---

## 📖 Overview

**GraphRAG Medical Chatbot** is a Retrieval-Augmented Generation (RAG) system designed to answer complex biological questions related to the **BSG gene**, associated diseases, proteins, and molecular interactions.

Unlike traditional RAG pipelines that rely exclusively on vector similarity search, this project leverages a **Knowledge Graph (Neo4j)** to retrieve structured biomedical relationships through dynamically generated **Cypher queries**.

The system combines:

- 🧠 Large Language Models (Google Gemini)
- 🗄 Graph database reasoning (Neo4j)
- 🔎 Optional hybrid fallback search
- 💬 Conversational memory via LangChain
- 🌐 Interactive UI built with Streamlit

This architecture enables precise, explainable, and relationship-aware answers to biomedical queries.

---

## 🚀 Key Features

### 🔗 Graph-Based Retrieval
- Converts natural language questions into **Cypher queries**
- Retrieves structured biomedical data from Neo4j
- Grounds responses in graph data to reduce hallucinations

### 🧠 LLM-Powered Answer Generation
- Uses **Google Gemini (2.x Flash / Pro models)** via `langchain-google-genai`
- Synthesizes structured graph results into human-readable explanations
- Deterministic Cypher generation (temperature = 0)

### 💬 Conversational Memory
- Maintains chat history across interactions
- Supports contextual follow-up questions

### 🌐 Interactive Interface
- Built with **Streamlit**
- Clean chat-based UI
- Sidebar with example biomedical queries

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python 3.10+ |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Graph DB | Neo4j |
| Orchestration | LangChain Core + Community |
| Frontend | Streamlit |
| Environment | python-dotenv |

---

## ⚙️ Installation & Setup

### 1 Clone the Repository

```bash
git clone https://github.com/Ciccio1307/GraphRAG.git
cd GraphRAG
```
### 2 Create a Virtual Environment
```bash

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

```

### 3 Install Dependencies
```bash

pip install -r requirements.txt


```

### 4 Setup Environment Variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder values with your real credentials:

| Variable | Where to get it |
|---|---|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Your local Neo4j instance or [AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) |
| `GRAPHRAG_*_MODEL` | Leave as-is or swap for another Gemini model |

> **Never commit your `.env` file.** It is listed in `.gitignore` and should stay local only.

### ▶️ Running the Application
```bash
Option A — Quick Start (Mac/Linux)
./start_demo.sh
Option B — Manual Start
streamlit run src/main_3.py
```