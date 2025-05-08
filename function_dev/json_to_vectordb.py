import os
import json
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def run_vector_pipeline(database_dir: str = "database"):
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "faiss_index"

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    documents = []

    for file_name in os.listdir(database_dir):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(database_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)

                for item in data:
                    # source 추론: item 내부에 명시된 source가 우선
                    source_label = item.get("source", None)
                    if not source_label:
                        # fallback: file_name에 기반한 추론
                        source_label = (
                            "news" if "news" in file_name else
                            "blog" if "blog" in file_name else
                            "paper" if "paper" in file_name else
                            "unknown"
                        )

                    text = f"{item.get('title', '')}\n{item.get('summary', '')}".strip()
                    if text:
                        chunks = splitter.split_text(text)
                        for chunk in chunks:
                            documents.append(Document(page_content=chunk, metadata={"source": source_label}))
            except Exception as e:
                print(f"❌ {file_name} 로딩 실패: {e}")

    if not documents:
        print("⚠️ 벡터화할 문서가 없습니다.")
        return

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(documents, embedding=embedding_model)
    vectordb.save_local(FAISS_INDEX_PATH)

    chunk_save_path = os.path.join(FAISS_INDEX_PATH, "doc_chunks.npy")
    np.save(chunk_save_path, np.array([{"text": d.page_content, "source": d.metadata["source"]} for d in documents]))
    print(f"✅ 총 {len(documents)}개 문서를 벡터 DB로 저장 완료!")

