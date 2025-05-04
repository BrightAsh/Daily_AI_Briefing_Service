import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import faiss
import numpy as np

# Load environment variables
load_dotenv()
openai_api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

# Step 1: Load PDF and extract text
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Split text into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Step 3: Embed text chunks
def embed_texts(text_chunks):
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.embed_documents(text_chunks)
    return embeddings

# Step 4: Save embeddings to FAISS index
def save_to_faiss(text_chunks, embeddings, index_path="faiss_index.index"):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)
    print(f"Saved {len(text_chunks)} chunks to FAISS index at {index_path}")

# Step 5: Build LangChain VectorStore from text
def build_langchain_vectorstore(text_chunks):
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents, embedding=embedding_model)
    return vectordb

# Step 6: Run a retrieval QA pipeline
def run_retrieval_qa(vectordb, query):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
        retriever=vectordb.as_retriever()
    )
    return qa.run(query)

# Full pipeline runner
def main(pdf_path, query):
    print("[1] PDF 로딩 중...")
    text = load_pdf(pdf_path)

    print("[2] 텍스트 청킹 중...")
    chunks = chunk_text(text)

    print(f"[3] {len(chunks)}개의 청크 임베딩 중...")
    embeddings = embed_texts(chunks)

    print("[4] 벡터 DB 저장 중...")
    save_to_faiss(chunks, embeddings)

    print("[5] LangChain VectorStore 생성 중...")
    vectordb = build_langchain_vectorstore(chunks)

    print("[6] 질문 실행 중...")
    result = run_retrieval_qa(vectordb, query)
    print("\n[질문 결과]")
    print(result)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    sample_pdf_path = os.path.join(base_path, "..", "sample_papers", "LLM_as_GPM_Paper.pdf")
    sample_query = "이 논문의 주요 기여는 뭐야?"
    if os.path.exists(sample_pdf_path):
        main(sample_pdf_path, sample_query)
    else:
        print(f"[오류] PDF 파일 '{sample_pdf_path}'이(가) 존재하지 않습니다.")