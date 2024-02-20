import os
from langchain.tools import BaseTool, StructuredTool, tool
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, List

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeLangchain
from pinecone import Pinecone
from langchain.tools.retriever import create_retriever_tool
from backend.agents.car_answer_agent import car_answer_agent

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from backend.tools.car_tools import run_llm_rag_with_media_link, run_llm
from langchain.tools import BaseTool, StructuredTool, tool
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, List

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeLangchain
from pinecone import Pinecone
from langchain.tools.retriever import create_retriever_tool
from backend.agents.car_answer_agent import car_answer_agent

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain



def cars_tool():
    
    structured_tool = StructuredTool.from_function(
        name="car_answer",
        func=car_answer_agent,
        description="Search for information about Cars. For any questions related Cars, you must use this tool!",
        return_direct=True
    )

    return structured_tool

def medical_tool():
    
    structured_tool = StructuredTool.from_function(
        name="medical_answer",
        func=run_llm_rag_with_media_link,
        description="Search for information about Health & Medicine. For any questions about Health & Medicine, you must use this tool!",
        return_direct=True
    )

    return structured_tool

def general_tool():
    structured_tool = StructuredTool.from_function(
        name="general_answer",
        func=run_llm_rag_with_media_link,
        description="If you can't find certain tools to answer question, you must use this tool. ",
        return_direct=True
    )

    return structured_tool

def cars_parameter_confirm_tool():
    
    structured_tool = StructuredTool.from_function(
        name="car_parameter_confirmed",
        func=run_llm,
        description="Use this tool when you confirmed all parameter of cars. If you didn't confirmed the all parameters about cars, you  don't must use this tool."
    )


    return structured_tool
