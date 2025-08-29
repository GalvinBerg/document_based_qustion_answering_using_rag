from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings




app = FastAPI()

file_path = r"/Users/manimarans/Documents/galvin/DOC-20240322-WA0000.pdf"

@app.post(file_path)
def upload_pdf():
    # Step 1: Save the uploaded file locally
    # temp_file_path = f"temp_{uuid.uuid4().hex}.pdf"
    # with open(temp_file_path, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    temp_file_path=file_path
    try:
        # Step 2: Load and split the PDF
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        # Optional: Add a manual chunk
        extra_doc = Document(
            page_content="This is an additional chunk of text manually added.",
            metadata={"source": "Manual", "section": "Extra"}
        )
        chunks.append(extra_doc)

        # Step 3: Embed and save to FAISS
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)

        # Step 4: Save FAISS index
        index_folder = "faiss_index"
        vectorstore.save_local(index_folder)

        return JSONResponse(content={
            "status": "success",
            "total_chunks": len(chunks),
            "index_saved_to": index_folder
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        print(f"Temporary file {temp_file_path} processed and deleted.")
        

if __name__ == "__main__":
    upload_pdf()