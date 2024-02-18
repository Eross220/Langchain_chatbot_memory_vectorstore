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
    qa= RetrievalQA.from_chain_type(
        
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    answer= qa.invoke({"query":query, "chat_history": chat_history})


    return answer['result']

def run_llm1(query: str, chat_history: List[Dict[str, Any]] = []):


    prompt_template="""Use the following pieces of context to answer the question at the end.
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.
                        Please follow the following rules:
                        1 In provided context, if you have the source links( website links) of products which you mentioned in final answer, you must add that in answer.
                        2.If you don't have online links , don't try to make that and just say that you don't have that.
                        {context}

                        Question: {question}
                        Crafted Response:"""


    embeddings= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch= PineconeLangchain.from_existing_index(index_name=INDEX_NAME ,embedding=embeddings)

    chat= ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY)


    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template) # prompt_template defined above
    llm_chain = LLMChain(
        llm=ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY), 
        prompt=QA_CHAIN_PROMPT,
        callbacks=None,
        verbose=True
    )
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
    )
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        callbacks=None,
        verbose=True,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    answer= qa.invoke({"query":query, "chat_history": chat_history})


    return answer['result']



def cars_tool():
    
    structured_tool = StructuredTool.from_function(
        name="car_answer",
        func=car_answer_agent,
        description="Search for information about Cars. For any questions about Cars, you must use this tool!",
        return_direct=True
    )

    return structured_tool

def medical_tool():
    
    structured_tool = StructuredTool.from_function(
        name="medical_answer",
        func=run_llm1,
        description="Search for information about Health & Medicine. For any questions about Health & Medicine, you must use this tool!",
        return_direct=True
    )

    return structured_tool

def general_tool():
    structured_tool = StructuredTool.from_function(
        name="genera_answer",
        func=run_llm,
        description="Use this tool if you want to answer questions that don't involve about certail fields such as cars, medicals and so on.",
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
