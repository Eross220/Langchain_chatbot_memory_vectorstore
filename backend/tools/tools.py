import os
from langchain.tools import BaseTool, StructuredTool, tool
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, List

from langchain.chains import RetrievalQA
from langchain.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeLangchain
from pinecone import Pinecone
from langchain.tools.retriever import create_retriever_tool
from ..agents.car_answer_agent import car_answer_agent

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMEN_REGION=os.getenv("PINECONE_ENVIRONMEN_REGION")
INDEX_NAME=os.getenv("INDEX_NAME")



pc = Pinecone(api_key=PINECONE_API_KEY)


def run_llm (query: str, chat_history: List[Dict[str, Any]] = []):
    print("query:",query)
    embeddings= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch= PineconeLangchain.from_existing_index(index_name=INDEX_NAME ,embedding=embeddings)

    chat= ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY)
    qa= RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    answer= qa.invoke({"query":query, "chat_history": chat_history})
    return answer['result']



def cars_tool():
    
    structured_tool = StructuredTool.from_function(
        name="cars_answer",
        func=run_llm,
        description="Search for information about Cars. For any questions about Cars, you must use this tool!",
        return_direct=True
    )

    return structured_tool

def medical_tool():
    
    structured_tool = StructuredTool.from_function(
        name="medical_answer",
        func=run_llm,
        description="Search for information about Medical. For any questions about medical, you must use this tool!"
    )

    return structured_tool

def general_tool():
    structured_tool = StructuredTool.from_function(
        name="genera_answer",
        func=run_llm,
        description="Use this tool if you want to answer questions that don't involve about certail fields such as cars, medicals and so on."
    )

    return structured_tool

def cars_parameter_confirm_tool():
    
    structured_tool = StructuredTool.from_function(
        name="Car parameter confiirm",
        func=run_llm,
        description="Use this tool when you confirmed all parameter of cars"
    )


    return structured_tool
