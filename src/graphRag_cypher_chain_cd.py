import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain,create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,MessagesPlaceholder,ChatPromptTemplate
labels = ["biological_pathway", "cell", "cell_line", "cell_type", "chemical_entity", "compound", "disease", "drug", "enzyme", "gene", "gene_variant", "protein", "species", "symptom", "tissue","REL"]


GRAPHRAG_QA_MODEL = os.getenv("GRAPHRAG_QA_MODEL")
GRAPHRAG_CYPHER_MODEL = os.getenv("GRAPHRAG_CYPHER_MODEL")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
)

schema  = graph.schema.split("\n")
new_schema = [schema[0]]
schema  = schema[1:]
section = 0
for query in schema:
    if "Relationship properties:" in query or  "The relationships:" in query :
        section += 1
        new_schema.append(query)
        continue
    
    if section == 0:
        label = query.split(" ")[0]
        if label not in labels: continue
    elif section == 2:
        query_component=query.split("(:")
        src = query_component[1].split(")")[0]
        if src not in labels: continue
        dst = query_component[2].split(")")[0]
        if dst not in labels: continue

    new_schema.append(query)
new_schema = "\n".join(new_schema)



graph.schema= new_schema
graph.refresh_schema()


cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note: 
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Examples:

# What five diseases are related to the BSG gene? 
WITH ["gene","gene_variant"] as LBS 
MATCH p=(n:disease)-[:REL]->(m) 
WHERE ANY(label IN labels(m) WHERE label IN LBS) 
RETURN p limit 5;

# Let me know the diseases related to BSG ?
WITH "BSG" as s_gene
MATCH (g:gene)-[:REL]->(d:disease)
WHERE lower(g.word) = lower(s_gene)
RETURN d.word AS Disease



String category values:
Edge type are 'REL'
Node labels are one of :'biological pathway', 'cell', 'cell_line', 'cell_type', 'chemical_entity', 'compound', 'disease', 'drug', 'enzyme', 'gene', 'gene_variant', 'protein', 'species', 'symptom','tissue'


The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.
Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

"""
# Wrapper per validare e riassumere input
class SafeChatOpenAI(ChatOpenAI):
    def __init__(self, model, temperature, max_tokens=16385):
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens

    def generate_prompt(self, messages, run_manager=None, **kwargs):
        # Verifica e riassumi i messaggi se necessario
        valid_messages = []
        for message in messages:
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                if conta_token(content) > self.max_tokens:
                    content = riassumi_contenuto(content, self.max_tokens)
                valid_messages.append({'content': content})
        return super().generate_prompt(valid_messages, run_manager=run_manager, **kwargs)

# Utilizzo del wrapper per evitare superamento dei limiti di token

"""

graph_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=GRAPHRAG_CYPHER_MODEL, temperature=0,verbose=True),
    qa_llm=ChatOpenAI(model=GRAPHRAG_QA_MODEL, temperature=0,verbose=True),
    graph=graph,
    include_types=labels,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)

