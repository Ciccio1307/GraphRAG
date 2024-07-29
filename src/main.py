import streamlit as st
from graph_rag_agent import agent_with_chat_history
#from graph_rag_agent import graphRag_agent_executor

with st.sidebar:
    st.header("Chatbot")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about BSG gene
        """
    )
    st.header("Example Questions")
    st.markdown("- Let me know something about disease?")
    st.markdown("- What is the distance between from BSG to tumor colon?")
    st.markdown("- How many arcs are connected to disease ?")
    st.markdown("- Can you send me a gene from BSG?")
    st.markdown(
        "- What is the avarange distance from BSG?"
    )
  
    st.markdown("- BSG 'reduced'?")
    st.markdown("- Find Genes Associated with Diabetes?")
   

st.title("GraphRAG Neo4j Chatbot")
st.info(
    "Ask me questions BSG genes and their interactions in the human genome. "
)
# Initialize session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

# Create a text input for the user's message
if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    # Store the user's message
    st.session_state.messages.append({"role": "user", "output": prompt})



    with st.spinner("Searching for an answer..."):
    #UPDATE CHAT_HISTORY 16_07_2024 
        response = agent_with_chat_history.invoke({"input": prompt}, config={"configurable": {"session_id": "<foo>"}})
    output = response['output']
    result = response['intermediate_steps'][0][1]['result']

    # Store the bot's message
    st.session_state.messages.append({"role": "bot", "output": output})

    # Display the bot's message
    with st.chat_message("bot"):
        st.markdown(output)