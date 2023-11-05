from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


if __name__ =='__main__':
    pdf_path="2210.03629.pdf"
    loader= PyPDFLoader(file_path=pdf_path)
    documents=loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator='\n')
    docs = text_splitter.split_documents(documents)

    embeddings= OpenAIEmbeddings()
    vectorstore= FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index_react")

    new_vectore = FAISS.load_local("faiss_index_react", embeddings)

    qa=RetrievalQA.from_chain_type(llm=OpenAI(),chain_type='stuff', retriever=new_vectore.as_retriever())

    result = qa.run("Give me the gist of ReAct in 3 sentences.")

    print(result)



  