import os
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

def answer_question(question: str, index_folder: str = None) -> str:
    if index_folder is None:
        index_folder = os.path.join(os.getcwd(), "faiss_index")  # default location
    
    if not os.path.exists(os.path.join(index_folder, "index.faiss")):
        raise FileNotFoundError(f"FAISS index not found in {index_folder}. Please run ingestion first.")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(index_folder, embedding_model, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    result = qa_chain.invoke(question)
    return result
