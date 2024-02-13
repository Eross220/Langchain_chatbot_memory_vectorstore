import os
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, List

from langchain.chains import RetrievalQA
from langchain.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeLangchain
from pinecone import Pinecone

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMEN_REGION=os.getenv("PINECONE_ENVIRONMEN_REGION")
INDEX_NAME=os.getenv("INDEX_NAME")

print(INDEX_NAME)


pc = Pinecone(api_key=PINECONE_API_KEY)


def run_llm (query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch= PineconeLangchain.from_existing_index(index_name=INDEX_NAME ,embedding=embeddings)

    chat= ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY)
    qa= RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

    return qa.invoke({"query":"You are helpful assistant. You should answer as much as possible based on this concept."+query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="what is Langchain?"))
