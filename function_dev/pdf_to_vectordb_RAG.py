# 뉴스 및 논문 요약용 벡터 저장 파이프라인 (에이전트 분리)
# -----------------------------------------------------------
# 목적: 뉴스 데이터(JSON) 및 논문 PDF를 읽어 벡터 DB와 chunk 데이터 저장

import os
import json
import glob
import fitz  # PyMuPDF
import numpy as np
import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# 설정값
PDF_FOLDER = "../sample_papers"
NEWS_JSON_PATH = "news_data_full.json"
FAISS_INDEX_PATH = "faiss_index.index"
CHUNK_SAVE_PATH = "doc_chunks.npy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# PDF 로딩
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 텍스트 청크
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# 뉴스 JSON 로딩 및 청크화
def load_news_chunks():
    if not os.path.exists(NEWS_JSON_PATH):
        return []
    with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
        news_items = json.load(f)
    chunks = []
    for item in news_items:
        title = item.get("title", "")
        content = item.get("full_text", "")
        if content:
            chunks.append({
                "text": f"[뉴스] {title}\n{content.strip()}",
                "source": "news"
            })
    return chunks

# 논문 PDF 로딩 및 청크화
def load_all_pdfs(pdf_folder):
    all_chunks = []
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    for file_path in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        print(f"📄 Processing: {file_path}")
        text = load_pdf(file_path)
        chunks = chunk_text(text)
        named_chunks = [
            {"text": chunk, "source": "paper"} for chunk in chunks
        ]
        all_chunks.extend(named_chunks)
    return all_chunks

# FAISS + 청크 저장
def save_to_faiss_with_chunks(named_chunks):
    texts = [chunk["text"] for chunk in named_chunks]
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings = model.embed_documents(texts)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(CHUNK_SAVE_PATH, np.array(named_chunks))
    print(f"✅ 총 {len(named_chunks)} chunks 저장 완료! (뉴스 + 논문 포함)")

# 실행 엔트리포인트
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_pdf_path = os.path.join(base_path, PDF_FOLDER)

    print("📥 뉴스 + 논문 처리 시작...")
    news_chunks = load_news_chunks()
    print(f"📰 뉴스 청크 수: {len(news_chunks)}")
    if news_chunks:
        print(f"예시 뉴스 제목: {news_chunks[0]['text'][:100]}...")

    if os.path.exists(full_pdf_path):
        paper_chunks = load_all_pdfs(full_pdf_path)
    else:
        print(f"❌ PDF 폴더가 존재하지 않습니다: {full_pdf_path}")
        paper_chunks = []

    all_chunks = news_chunks + paper_chunks
    if all_chunks:
        save_to_faiss_with_chunks(all_chunks)
    else:
        print("⚠️ 저장할 chunk가 없습니다.")
