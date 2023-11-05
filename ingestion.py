from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import pinecone
pinecone.init(api_key="4d743ef1-9f63-45dd-9555-3486fb1d1d60", environment="gcp-starter")

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
    
    embeddings=OpenAIEmbeddings(openai_api_key="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")
    Pinecone.from_documents(documents,embeddings,index_name="langchain-docs-index")

    print("aaaa")
if __name__ == "__main__":
    ingest_docs()
