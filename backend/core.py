import os
from dotenv import load_dotenv
load_dotenv()
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores.pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
import pinecone


OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMEN_REGION=os.getenv("PINECONE_ENVIRONMEN_REGION")
INDEX_NAME=os.getenv("INDEX_NAME")

print(INDEX_NAME)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMEN_REGION)

def run_llm (query:str)->any:
    embeddings= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch= Pinecone.from_existing_index(index_name=INDEX_NAME ,embedding=embeddings)

    chat= ChatOpenAI(verbose=True,temperature=0,openai_api_key=OPENAI_API_KEY)
    qa= RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

    return qa({"query":query})

if __name__ == "__main__":
    print(run_llm(query="What is Langchain?"))