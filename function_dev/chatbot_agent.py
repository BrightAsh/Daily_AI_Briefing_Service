# RAG 기반 에이전트: 뉴스/논문 요약 + 질의응답 + 키워드/출처 추출 + 비교기
# ------------------------------------------------

import os
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS as FAISS_LC

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SAVE_PATH = "doc_chunks.npy"

# 임베딩 모델 & 벡터 DB 로딩
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
chunks = np.load(CHUNK_SAVE_PATH, allow_pickle=True)
texts = [chunk["text"] for chunk in chunks if isinstance(chunk, dict)]
documents = [Document(page_content=t) for t in texts]
vectordb = FAISS_LC.from_documents(documents, embedding=embedding_model)

# 청크 필터링 함수
def get_chunks_by_type(source_type):
    return [c for c in chunks if isinstance(c, dict) and c.get("source") == source_type]

# 뉴스 요약
def summarize_news(*args, **kwargs):
    news_chunks = get_chunks_by_type("news")
    if not news_chunks:
        return "오늘 뉴스 데이터가 없습니다."
    texts = [c["text"] for c in news_chunks[:5]]
    joined = "\n\n".join(texts)
    prompt = f"다음은 최근 뉴스입니다. 주요 내용을 요약해 주세요.\n\n{joined}"
    llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)
    return llm.predict(prompt)

# 논문 요약
def summarize_papers(*args, **kwargs):
    paper_chunks = get_chunks_by_type("paper")
    if not paper_chunks:
        return "논문 데이터가 없습니다."
    texts = [c["text"] for c in paper_chunks[:5]]
    joined = "\n\n".join(texts)
    prompt = f"다음은 최근 논문 내용입니다. 핵심 내용을 요약해 주세요.\n\n{joined}"
    llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)
    return llm.predict(prompt)

# 뉴스 키워드 추출
def extract_news_keywords(*args, **kwargs):
    news_chunks = get_chunks_by_type("news")
    if not news_chunks:
        return "키워드 추출할 뉴스가 없습니다."
    texts = [c["text"] for c in news_chunks[:5]]
    joined = "\n\n".join(texts)
    prompt = f"다음 뉴스에서 핵심 키워드를 5~10개 추출해줘. 키워드만 나열해줘.\n\n{joined}"
    llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)
    return llm.predict(prompt)

# 뉴스 출처 요약
def list_news_sources(*args, **kwargs):
    news_chunks = get_chunks_by_type("news")
    if not news_chunks:
        return "뉴스 출처가 없습니다."
    titles = [f"- {c.get('title', '[제목 없음]')}" for c in news_chunks if c.get("title")]
    return "오늘 뉴스 목록:\n" + "\n".join(titles)

# 뉴스/논문 비교 요약
def compare_news_papers(*args, **kwargs):
    news_chunks = get_chunks_by_type("news")
    paper_chunks = get_chunks_by_type("paper")
    if not news_chunks or not paper_chunks:
        return "뉴스 또는 논문 데이터가 부족합니다."
    news_text = "\n\n".join([c["text"] for c in news_chunks[:3]])
    paper_text = "\n\n".join([c["text"] for c in paper_chunks[:3]])
    prompt = f"다음은 뉴스와 논문 내용입니다. 공통점과 차이점을 비교 요약해 주세요.\n\n[뉴스]\n{news_text}\n\n[논문]\n{paper_text}"
    llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)
    return llm.predict(prompt)

# RAG 기반 질의응답
def rag_qa(query):
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        retriever=vectordb.as_retriever()
    )
    return qa.run(query)

# 에이전트 툴 정의
tools = [
    Tool(name="news_yok", func=summarize_news, description="오늘 뉴스 요약을 수행합니다."),
    Tool(name="paper_yok", func=summarize_papers, description="최근 논문 요약을 수행합니다."),
    Tool(name="qa_tool", func=rag_qa, description="뉴스/논문을 기반으로 자유롭게 질문에 답변합니다."),
    Tool(name="news_keywords", func=extract_news_keywords, description="뉴스 본문에서 핵심 키워드를 추출합니다."),
    Tool(name="news_list", func=list_news_sources, description="오늘의 뉴스 제목 목록을 출력합니다."),
    Tool(name="news_paper_compare", func=compare_news_papers, description="뉴스와 논문 내용을 비교 요약합니다.")
]

# 에이전트 초기화
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 테스트 실행
if __name__ == "__main__":
    print("🤖 에이전트 테스트 시작!")
    print("--- 뉴스 요약 ---")
    print(agent.run("오늘 뉴스 요약해줘"))
    print("\n--- 논문 요약 ---")
    print(agent.run("최근 논문 내용 요약해줘"))
    print("\n--- 자유 질의응답 ---")
    print(agent.run("AI 관련 정책이 포함된 뉴스가 있었어?"))
    print("\n--- 뉴스 키워드 ---")
    print(agent.run("뉴스에서 키워드만 뽑아줘"))
    print("\n--- 뉴스 출처 목록 ---")
    print(agent.run("오늘 뉴스 제목 알려줘"))
    print("\n--- 뉴스와 논문 비교 ---")
    print(agent.run("뉴스와 논문에서 다루는 주제를 비교해줘"))
