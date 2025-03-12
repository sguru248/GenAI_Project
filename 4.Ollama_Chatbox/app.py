import streamlit as st

from langchain_ollama import OllamaLLM  ## Need to download model in our system

#from langchain_community.llms import ollama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with Ollama"

# Define the PromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are the helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

# Generate response function
def generate_response(question, llm_model):
    
    llm = OllamaLLM(model=llm_model)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    
    return answer

# Streamlit UI
st.title("Enhanced Q&A Chatbox with Ollama")



# Model selection dropdown
llm_model = st.sidebar.selectbox("Select an Open Source Model", ["gemma2:2b", "llama3.2:1b", ])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
     response = generate_response(user_input, llm_model)
     st.write(response)