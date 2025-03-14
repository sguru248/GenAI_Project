## RAG Q&A Conversation with pdf Including chat history
import streamlit as st
import re

from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os

from dotenv import load_dotenv
load_dotenv()

#Input Groq API key for LLM
#groq_api_key = os.getenv("GROQ_API_KEY")


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") ## for embedding

## embedding
embeddings = HuggingFaceEmbeddings(model_name ="all-MiniLM-L6-v2")

##Setup Streamlit APP
st.title("Coversational RAG with PDF Upload and Chat History")
st.write("Upload Pdf's and chat with the content")

#Input the Groq Api Key
api_key = st.text_input("Enter your Groq API Key:",type="password")

#Check if Groq API key Provided

if api_key:
    llm = ChatGroq(api_key=api_key,model="qwen-qwq-32b")
    #Chat interface
    session_id = st.text_input("Session ID", value="default_session")

    #Statefully Manage Chat History
    if 'store' not in st.session_state:
        st.session_state.store ={}

    uploaded_files = st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=True)

    #Process the Uploaded file

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()

            documents.extend(docs)
    #splits and create embedding for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits,embedding=embeddings)
        retriever = vectorstore.as_retriever()


        contextualize_q_system_prompt = (
            "Given a chat history and  the latest user question"
            "Which might reference context in the chat history"
            "formulate a standalone question which can be understand"
            "Without the chat history.Do not answer the question"
            "just reformulate it if needed and otherwise return it as it"
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ##Answer question

        system_prompt = (
            "You are an assistant for question-answering task."
            "Use the following pieces of retrieved context to answer"
            "the question. if you don't know answer,say that you"
            "don't know. Use three Sentence maxium and keep the "
            "answer concise"
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")

            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if  session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input = st.text_input("Your question:")
        
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable" :{"session_id":session_id}
                },
            )
            
            
            # Check if response contains <think> blocks
            answer_text = response['answer']
            
            # Extract thinking content and main answer
            think_pattern = r'<think>(.*?)</think>'
            think_blocks = re.findall(think_pattern, answer_text, re.DOTALL)
            
            # Remove <think> blocks from main answer
            clean_answer = re.sub(think_pattern, '', answer_text, flags=re.DOTALL).strip()
            
            
            # Add expander for thinking blocks if they exist
            if think_blocks:
                with st.expander("Show Thinking Process"):
                    for i, think_block in enumerate(think_blocks):
                        st.write(f"Thinking Block {i+1}:")
                        st.write(think_block.strip())
                
                        
            # Display only the clean answer openly
            st.write("Assistant:", clean_answer)
  
            
            # With these expanded sections:
            with st.expander("🧠 Session State Details"):
                st.write(st.session_state.store)
                 
        
             # With these expanded sections:
            with st.expander("📜 Full Chat History"):
                st.write("Chat History:", session_history.messages)
else:
    st.warning("Please Enter the Groq API Key")
            
        

    









