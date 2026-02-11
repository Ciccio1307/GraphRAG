import os
from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================================================
# Load environment variables
# =========================================================
load_dotenv()

GRAPHRAG_QA_MODEL = os.getenv("GRAPHRAG_QA_MODEL", "models/gemini-2.5-flash-lite")
GRAPHRAG_CYPHER_MODEL = os.getenv("GRAPHRAG_CYPHER_MODEL", "models/gemini-2.5-flash-lite")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# =========================================================
# Labels consentite nel grafo
# =========================================================
labels = [
    "biological_pathway",
    "cell",
    "cell_line",
    "cell_type",
    "chemical_entity",
    "compound",
    "disease",
    "drug",
    "enzyme",
    "gene",
    "gene_variant",
    "protein",
    "species",
    "symptom",
    "tissue",
    "REL",
]

# =========================================================
# Connessione Neo4j
# =========================================================
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
)

# =========================================================
# Filtraggio e normalizzazione schema
# =========================================================
schema_lines = graph.schema.split("\n")
new_schema = [schema_lines[0]]  
section = 0
for line in schema_lines[1:]:
    if "Relationship properties:" in line or "The relationships:" in line:
        section += 1
        new_schema.append(line)
        continue

    if section == 0:  
        label = line.split(" ")[0]
        if label not in labels:
            continue
    elif section == 2:  
        parts = line.split("(:")
        if len(parts) < 3:
            continue
        src = parts[1].split(")")[0]
        dst = parts[2].split(")")[0]
        if src not in labels or dst not in labels:
            continue

    new_schema.append(line)

graph.schema = "\n".join(new_schema)
graph.refresh_schema()

# =========================================================
# Prompt Cypher con esempi concreti
# =========================================================
cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
- Use only the provided relationship types and properties in the schema.
- Do not invent schema elements.
- Return ONLY the Cypher query, no explanations.

Schema:
{schema}

Examples:

# What five diseases are related to the BSG gene?
WITH ["gene","gene_variant"] as LBS
MATCH p=(n:disease)-[:REL]->(m)
WHERE ANY(label IN labels(m) WHERE label IN LBS)
RETURN p limit 5;

# Which proteins interact with BSG?
WITH "BSG" as s_gene
MATCH (g:gene)-[:REL]->(p:protein)
WHERE lower(g.word) = lower(s_gene)
RETURN p.word AS Protein;

# Which drugs or compounds are associated with the disease Melanoma?
WITH "melanoma" as s_disease, ["drug", "compound", "chemical_entity"] as DRUG_LABELS
MATCH (d:disease)-[:REL]-(c)
WHERE lower(d.word) CONTAINS lower(s_disease) 
AND ANY(label IN labels(c) WHERE label IN DRUG_LABELS)
RETURN c.word as Therapeutic_Agent, labels(c) as Type limit 10;

# In which biological pathways is the gene AKT1 involved?
WITH "AKT1" as s_gene
MATCH (g:gene)-[:REL]->(bp:biological_pathway)
WHERE lower(g.word) = lower(s_gene)
RETURN bp.word as Pathway;

# In which tissues is the gene BSG expressed?
WITH "BSG" as s_gene
MATCH (g:gene)-[:REL]->(t:tissue)
WHERE lower(g.word) = lower(s_gene)
RETURN t.word as Tissue;

# Find all genes containing the word "insulin"
WITH "insulin" as search_term
MATCH (g:gene)
WHERE lower(g.word) CONTAINS lower(search_term)
RETURN g.word as Gene_Name limit 10;

# Find drugs that target genes associated with Diabetes
WITH "diabetes" as s_disease
MATCH (d:disease)<-[:REL]-(g:gene)<-[:REL]-(dr:drug)
WHERE lower(d.word) CONTAINS lower(s_disease)
RETURN distinct dr.word as Drug, g.word as Targeted_Gene limit 5;

Question:
{question}
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=cypher_generation_template,
)

# =========================================================
# Prompt QA
# =========================================================
qa_generation_template = """
You are an assistant that converts Neo4j query results into a human-readable answer.

Query Results:
{context}

Question:
{question}

If results are empty, say you don't know the answer.
Answer:
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=qa_generation_template,
)

# =========================================================
# LLM Gemini 2.5 Flash Lite
# =========================================================
llm = ChatGoogleGenerativeAI(
    model=GRAPHRAG_QA_MODEL,
    temperature=0,
)

# =========================================================
# GraphCypherQAChain
# =========================================================
graph_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatGoogleGenerativeAI(model=GRAPHRAG_CYPHER_MODEL, temperature=0),
    qa_llm=llm,
    graph=graph,
    include_types=labels,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    validate_cypher=True,
    allow_dangerous_requests=True,
    verbose=True,
    top_k=100,
)
