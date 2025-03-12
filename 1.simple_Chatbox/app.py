import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import  ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import os
from dotenv import load_dotenv
load_dotenv()

##langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')



##Streamlit Framework
st.title("Langchain Demo with llama3.2:1b  Model Chatbot")

input_text = st.text_input('Enter your question here...')

##Selecting model
llm = OllamaLLM(model="llama3.2:1b")

#Prompt template
prompt = ChatPromptTemplate(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}"),
    ]
)


output_parser = StrOutputParser()

#Creating Chain
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))



