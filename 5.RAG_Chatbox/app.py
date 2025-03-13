import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

##Load the Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

##Creating LLM
llm = ChatGroq(groq_api_key=groq_api_key,model_name ="gemma2-9b-it")

##Creating Prompts
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate

prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        """
        Answer the question based on the provided context only. 
        if Question related to context but not in the context then 
        give short description about your own thought.        
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Question: {input}
        """
    )
])


def create_vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        st.session_state.loader = PyPDFLoader(file_path)  # Data Injection
        st.session_state.docs = st.session_state.loader.load()   # Data loading
        st.write(f"Number of documents loaded: {len(st.session_state.docs)}")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents =  st.session_state.text_splitter.split_documents(st.session_state.docs) # Doing TextSplitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("RAG Document Q&A With GROQ and Llama")

uploaded_file = st.file_uploader("Choose a PDF File",type="pdf") 

if uploaded_file:
     temppdf = f"./temp.pdf"
     with open(temppdf,"wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name
     if st.button("Document Embedding"):
          create_vector_embedding(temppdf)
          st.write("Vector Database Ready")
          
user_prompt = st.text_input("Enter your query from the Document you upload.")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
              
    response = retrieval_chain.invoke({'input':user_prompt})
              
    st.write(response['answer'])
              
    ##With a Streamlit expander
    with st.expander("Document Similarity search"):
        for i, doc in enumerate(response['context']):
             st.write(doc.page_content)
             st.write('-----------------------------')
    

