# OpenAI + FAISS 기반 Q&A 챗봇 (Gradio + MMR + OpenAI 최신 API)
# -----------------------------------------------------------

import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --- 설정 ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_PATH = "./faiss_index.index"
DOCUMENTS_PATH = "./doc_chunks.npy"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# --- 모델 로드 ---
embedder = SentenceTransformer(EMBEDDING_MODEL)
index = faiss.read_index(FAISS_INDEX_PATH)
chunks = np.load(DOCUMENTS_PATH, allow_pickle=True)

# --- FAISS 전체 벡터 추출 (MMR용) ---
def get_all_vectors():
    return np.array([index.reconstruct(i) for i in range(index.ntotal)])

# --- MMR 알고리즘 ---
def mmr(query_vec, doc_vecs, lambda_param=0.5, top_k=5):
    sim_to_query = cosine_similarity(query_vec, doc_vecs)[0]
    selected = []
    selected_set = set()
    while len(selected) < top_k:
        mmr_score = []
        for i in range(len(doc_vecs)):
            if i in selected_set:
                mmr_score.append(-np.inf)
                continue
            if not selected:
                diversity_penalty = 0
            else:
                sim_to_selected = max(cosine_similarity([doc_vecs[i]], [doc_vecs[j] for j in selected])[0])
                diversity_penalty = sim_to_selected
            score = lambda_param * sim_to_query[i] - (1 - lambda_param) * diversity_penalty
            mmr_score.append(score)
        idx = np.argmax(mmr_score)
        selected.append(idx)
        selected_set.add(idx)
    return selected

# --- Q&A 함수 ---
def answer_question(query, top_k=5):
    query_vec = embedder.encode([query])
    doc_vecs = get_all_vectors()
    mmr_indices = mmr(query_vec, doc_vecs, lambda_param=0.5, top_k=top_k)
    context = "\n\n".join([chunks[i] for i in mmr_indices])

    prompt = f"""
다음 문서 내용을 기반으로 정확하고 간결하게 답변해줘. 문서에 없는 내용은 추측하지 마.

문서:
{context}

질문: {query}
답변:
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "당신은 문서를 기반으로 정확한 답변을 제공하는 AI 비서입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

# --- Gradio UI 구성 ---
def gradio_interface(query):
    return answer_question(query)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="질문을 입력하세요..."),
    outputs="text",
    title="📚 OpenAI RAG 챗봇 (MMR 기반)",
    description="문서를 기반으로 GPT가 답변해주는 챗봇입니다. MMR 방식으로 chunk 검색을 수행합니다."
)

if __name__ == "__main__":
    iface.launch()
