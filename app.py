import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text to Math Probem Solver and Assistant")
st.title("Text to Math Problem Solver")

groq_api_key = st.sidebar.text_input(label="Groq Api Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq Api Key to Continue")
    st.stop()

llm = ChatGroq(model='Gemma2-9b-It', groq_api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "Wikipedia",
    func = wikipedia_wrapper.run,
    description="A tool for searching the internet"
)

math_chain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name='Calculator',
    func=math_chain.run,
    description="A tool for solving and answering math related questions. Only input in the form of mathematical expression to be provided."
)

prompt="""
You are agent tasked for solving users's mathematical questions. Logically arrive at the solution and provide the detailed explanation and display it point wise for the question below
Question = {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=['question'],
    template = prompt
)

chain = LLMChain(llm = llm, prompt = prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description = "A tool for answering logic based and reasoning questions."
)

assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parcing_error = True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {'role': "assistant", 'content': 'Hi! I am a Maths Chatbot which can answer mathematical doubts.'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
question = st.text_area("Enter your question:","A Car costs me 25 lakhs. I am getting a discount of 10 percent on cash payment. How much money am i saving if I py by cash?")

if st.button("Find my Answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant", "content":response})
            st.write("### Response:")
            st.success(response)
    
    else:
        st.warning("Please enter the question")