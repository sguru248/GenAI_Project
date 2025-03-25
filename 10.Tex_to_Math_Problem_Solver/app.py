import streamlit as st
from langchain.chains import LLMChain,LLMMathChain
from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent


from langchain.callbacks import StreamlitCallbackHandler


from dotenv import load_dotenv

## Set UPI the sreamlit app
st.set_page_config(page_title="Text to Math Problem Solverer and Data search",page_icon="🧮")
st.title("Text to Math Problem Solverer and Data search Using Gemma 2")


groq_api_key = st.sidebar.text_input(label="Enter your GROQ API key",type="password")

if not groq_api_key:
    st.info("Please enter your GROQ API key")
    st.stop()

llm = ChatGroq(api_key=groq_api_key,model="gemma2-9b-it")

##intialize the tools

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find the various information on the Wikipedia",    
    )

##Intialize Math chain

match_chain = LLMMathChain.from_llm(llm)

calculator_tool= Tool(
    name="Calculator",
    func=match_chain.run,
    description="A tool for answering math related question. only input mathematical expression need to be provided", 
    )



##Prompt
prompt = """
your a agent tasked for solving user mathematical question.
logically arrive at the solution and provide a detailed explaination
and display it point wise for the the question below.

Question:{question}

"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain

chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="A tool for answering logic based and reasoning question.", 
    )

##Intialize the agent

assistant_agent = initialize_agent(
    tools=[wikipedia_tool,calculator_tool,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verpose=False,
    handling_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi,I am a Chatbot who can solve math problems and search the web. How can i help you?"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
## 


##Lets start the interacton

question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")


if st.button("find the answer"):
    if question :
        with st.spinner("Generate Response"):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant","content":response})
            
            st.write("### Response:")
            st.success(response)         
    else:
        st.warning("Please enter the question")
            







    
