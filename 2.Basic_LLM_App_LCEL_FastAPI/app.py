from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


from langserve import add_routes

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


#Model 
model = ChatGroq(api_key=groq_api_key,model="gemma2-9b-it")

#Creating Prompt

system_template = "Translate the following into {language}"

prompt = ChatPromptTemplate(
    [
        ('system',system_template),
        ('user','{text}')
    ]
)

#Create outputparser
parser = StrOutputParser()

#Creating Chain

chain = prompt|model|parser

##App Defintion

app = FastAPI(title="Langchain server",
              version="1.0",
              description="A simple API server using langchain runnable interface")

##Adding chain to route

add_routes(
    app,
    chain,
    path='/chain'
)


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


##uvicorn app:app --reload

