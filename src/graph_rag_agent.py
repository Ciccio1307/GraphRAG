import os
from langchain_openai import ChatOpenAI
from langchain.agents import(
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)

from langchain import hub
from graphRag_cypher_chain_cd import graph_cypher_chain

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder

GRAPHRAG_AGENT_MODEL = os.getenv("GRAPHRAG_AGENT_MODEL")

graphRag_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
tools = [
   # Tool(                                                                  ########(was removed, use only Graph tool)
      #  name="Retriever",
      #  func=reviews_vector_chain.invoke,
      #  description="""Useful when need to know something about BSG disease , all nodes and 
     #   his edges.
      #  """,
   # )
    #,
   Tool(
    name="Graph",
    func=graph_cypher_chain.invoke,
    description="""Useful for answering questions about a Neo4j graph. 
    It converts natural language questions into Cypher queries, the query language of Neo4j. 
    It can answer questions about nodes, relationships, types, labels, and words in the graph. 
    Use the entire prompt as input to the tool. This specialized tool offers streamlined search capabilities
    to help you find the movie information you need with ease. Input should be full question
    """,
   )

]

chat_model = ChatOpenAI(
    model=GRAPHRAG_AGENT_MODEL,
    temperature=0,
)
message_history = ChatMessageHistory()

graphRag_agent_prompt = create_openai_functions_agent(
    llm=chat_model,
    prompt=graphRag_agent_prompt,
    tools=tools,
)
graphRag_agent_executor = AgentExecutor(
    agent=graphRag_agent_prompt,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)

store ={}
def get_session_history(session_id : str) ->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    graphRag_agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


