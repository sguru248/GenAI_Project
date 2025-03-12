import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with OPENAI"

# Define the PromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are the helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

# Generate response function
def generate_response(question, api_key, llm_model, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    
    return answer

# Streamlit UI
st.title("Enhanced Q&A Chatbox with OpenAI")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key: ", type="password")

# Model selection dropdown
llm_model = st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4.5", "gpt-4o", "gpt-4o-mini", "openai o3-mini"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
    if not api_key:
        st.write("⚠️ Please enter your OpenAI API key in the sidebar.")
    else:
        response = generate_response(user_input, api_key, llm_model, temperature, max_tokens)
        st.write("🤖:", response)
