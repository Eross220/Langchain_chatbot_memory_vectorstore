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

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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

def run_llm_rag_with_media_link(query: str, chat_history: List[Dict[str, Any]] = []):


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


def general_answer(query:str):
    template= """
        Your name is  Bob, helpful assistant for Car and Medical & Health.
        Your purpose is to answer the question about Car and Medical&Health.
        If the question is not related car and medical, avoid to answer as possible.
        If you don't know the answer, you just say I don't know. and require the question about Car and Medical.
        Begin. 
        Question: {question}
        """


    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    general_prompt_template= PromptTemplate(
        input_variables=["question"],
        template=template,
    )

    general_answer_chain=LLMChain(llm=llm, prompt=general_prompt_template)


    answer=general_answer_chain.invoke({"question":query})

    return answer