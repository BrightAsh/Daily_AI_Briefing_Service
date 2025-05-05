# 여러 개의 논문 PDF를 처리하여 FAISS + doc_chunks.npy로 저장하는 파이프라인
# -----------------------------------------------------------

import os
import glob
import fitz  # PyMuPDF
import numpy as np
import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# --- 설정 ---
PDF_FOLDER = "../sample_papers"  # 논문 PDF들이 저장된 폴더
FAISS_INDEX_PATH = "faiss_index.index"
CHUNK_SAVE_PATH = "doc_chunks.npy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Step 1: PDF 로딩 및 텍스트 추출
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: 텍스트 청킹
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Step 3~4: 전체 PDF 처리

def load_all_pdfs(pdf_folder):
    all_chunks = []
    all_embeddings = []
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    for file_path in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        print(f"📄 Processing: {file_path}")
        text = load_pdf(file_path)
        chunks = chunk_text(text)
        embeddings = model.embed_documents(chunks)

        named_chunks = [
            {"text": chunk, "source": os.path.basename(file_path)}
            for chunk in chunks
        ]

        all_chunks.extend(named_chunks)
        all_embeddings.extend(embeddings)

    return all_chunks, all_embeddings

# Step 5: FAISS + 청크 저장
def save_to_faiss_with_chunks(named_chunks, embeddings, index_path=FAISS_INDEX_PATH):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)

    # 텍스트만 추출하여 저장
    np.save(CHUNK_SAVE_PATH, np.array([chunk["text"] for chunk in named_chunks]))
    print(f"✅ 총 {len(named_chunks)} chunks 저장 완료!")

# 실행 엔트리포인트
def main_all(pdf_folder):
    print("[1] 전체 PDF 로드 및 처리 중...")
    all_chunks, all_embeddings = load_all_pdfs(pdf_folder)

    print("[2] FAISS + chunk 저장 중...")
    save_to_faiss_with_chunks(all_chunks, all_embeddings)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_pdf_path = os.path.join(base_path, PDF_FOLDER)

    if os.path.exists(full_pdf_path):
        main_all(full_pdf_path)
    else:
        print(f"❌ PDF 폴더가 존재하지 않습니다: {full_pdf_path}")
