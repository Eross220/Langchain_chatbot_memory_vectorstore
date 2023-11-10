from ast import Set
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"]=[]

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"]=[]

st.header("Langchain Documentation Chatbot")

prompt = st.text_input("Answer",placeholder='Enter your answer here...')

if prompt:
    with st.spinner("Generating the answer..."):
        generated_response =run_llm(query=prompt)
        answer=generated_response["result"]
        print(generated_response["result"])
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(answer)

if st.session_state["chat_answer_history"]:
    for  user_query,generated_response in zip(st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]):
        message(user_query,is_user=True)
        message(generated_response)