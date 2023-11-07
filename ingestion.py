import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import pinecone

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMEN_REGION=os.getenv("PINECONE_ENVIRONMEN_REGION")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMEN_REGION)

INDEX_NAME=os.getenv("INDEX_NAME")


def ingest_docs()->None:
    loader=ReadTheDocsLoader(path='langchain-docs\langchain.readthedocs.io\en\latest',encoding='cp437')
    raw_documents= loader.load()
    print(f"loaded{len(raw_documents)}documents")
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50,separators=[ "\n\n", "\n", " ",""])
    documents=text_splitter.split_documents(documents=raw_documents)
    print("lengh",len(documents)) 

    for doc in documents:
        old_path=doc.metadata["source"]
        new_url= old_path.replace("langchain-docs","https:")
        doc.metadata.update({"source":new_url})
    
    print(f"gint to insert {len(documents)} to pinecone")
    
    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    Pinecone.from_documents(documents,embeddings,index_name=INDEX_NAME)

    print("aaaa")
if __name__ == "__main__":
    ingest_docs()
