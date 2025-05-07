# 뉴스/블로그/논문 JSON 기반 벡터 저장 파이프라인 (에이전트 분리)
# -----------------------------------------------------------
# 목적: 뉴스, 블로그, 논문 JSON 파일을 읽어 벡터 DB와 chunk 데이터 저장

import os
import json
import numpy as np
import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS

# 설정값
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILES = [
    (os.path.join(BASE_DIR, "..", "sample_news", "news_data_summaries.json"), "news"),
    (os.path.join(BASE_DIR, "..", "sample_blogs", "blogs_data_summaries.json"), "blog"),
    (os.path.join(BASE_DIR, "..", "sample_papers", "arxiv_papers_summaries.json"), "paper")
]
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SAVE_PATH = "doc_chunks.npy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 텍스트 청크화 함수
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# JSON 로딩 및 청크화
def load_chunks_from_json(file_path, source_label):
    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for item in data:
        title = item.get("title", "")
        content = item.get("summary", "")
        if content:
            text = f"{title}\n{content.strip()}"
            chunked = chunk_text(text)
            chunks.extend([Document(page_content=c, metadata={"source": source_label}) for c in chunked])
    print(f"✅ {source_label} 청크 수: {len(chunks)}")
    return chunks

# FAISS + 청크 저장
def save_to_faiss_with_chunks(documents):
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(documents, embedding=model)
    vectordb.save_local(FAISS_INDEX_PATH)
    np.save(CHUNK_SAVE_PATH, np.array([{"text": d.page_content, "source": d.metadata["source"]} for d in documents]))
    print(f"📦 총 {len(documents)} chunks 저장 완료!")

# 실행 엔트리포인트
if __name__ == "__main__":
    print("📥 JSON 데이터 처리 시작...")
    all_documents = []
    for file_path, label in JSON_FILES:
        chunks = load_chunks_from_json(file_path, label)
        all_documents.extend(chunks)

    if all_documents:
        save_to_faiss_with_chunks(all_documents)
    else:
        print("⚠️ 저장할 chunk가 없습니다.")
