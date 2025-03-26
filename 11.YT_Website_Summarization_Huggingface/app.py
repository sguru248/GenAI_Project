import validators
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain

from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

from langchain_huggingface import HuggingFaceEndpoint


##

import os
from dotenv import load_dotenv
load_dotenv()

#hf_api_key = os.getenv("HF_TOKEN")

##Streamlit App

st.set_page_config(page_title="Langchain: Summarize Text From YT or Website", page_icon="🐧")
st.title("🐧 Langchain: Summarize Text From YT or Website")
st.subheader("Summarize URL")



## Get the Groq API Key  and URL (YT or website) to be summarized

with st.sidebar:
    hf_api_key = st.text_input("Enter your Huggingface API Key", type="password") 
    
    
##LLM  - Gemma model using GroHuggingfaceq API Key

repo_id = "google/gemma-3-27b-it"

llm = HuggingFaceEndpoint(repo_id=repo_id,max_new_tokens=150,temperature=0.7,huggingfacehub_api_token=hf_api_key) 


## Prompt Template
promt_template = """
Provide a summary of the following content in 300 words:

Content : {text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=promt_template 
   )
    
    
generic_url = st.text_input("Enter the URL to be summarized", label_visibility="collapsed")


if st.button("Summarize the Content from YT or Websiite"):
    ## validate the all the input
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please enter the Groq API Key and URL to be summarized")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can may be a Youtube video or a website")
    else:
        try:
            with st.spinner("Waiting...!"):
                #Loading the Website or YT video
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    
                docs= loader.load()
                
                ##Chain for Summarization
                
                chain = load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                
                output_summary = chain.run(docs)
                
                st.success(output_summary)
                
        except Exception as e:
            st.exception(f"Exception: {e}")
                 
                
                 
                    
                    


