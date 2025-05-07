# Agent 기반 리팩터링: LangChain + FAISS + Tool 호출 자동화 + MCP Tool 추가
# ---------------------------------------------------------------------

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 벡터 DB 로딩
CHUNK_PATH = "doc_chunks.npy"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
chunks = np.load(CHUNK_PATH, allow_pickle=True)
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
documents = [
    Document(page_content=c["text"], metadata={"source": c["source"]})
    for c in chunks if isinstance(c, dict)
]
vectordb = FAISS.from_documents(documents, embedding=embedding_model)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# 소스 필터링 기반 질의응답 실행 함수
def filtered_rag_run(query: str, source_filter: str) -> str:
    filtered_docs = [doc for doc in documents if doc.metadata.get("source") == source_filter]
    if not filtered_docs:
        return f"⚠️ '{source_filter}' 소스에서 문서를 찾을 수 없습니다."

    local_vectordb = FAISS.from_documents(filtered_docs, embedding=embedding_model)
    local_retriever = local_vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    rag_chain = RetrievalQA.from_chain_type(
        retriever=local_retriever,
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.3),
        return_source_documents=False
    )
    docs = local_retriever.get_relevant_documents(query)
    print(f"🔍 관련 문서 수 ({source_filter}): {len(docs)}")
    for i, d in enumerate(docs[:2]):
        print(f"[{i}] {d.page_content[:100]}...")

    if not docs or all(len(doc.page_content.strip()) < 50 for doc in docs):
        return f"🔎 '{source_filter}' 문서에서 유효한 정보를 찾지 못했습니다."
    return rag_chain.run(query)

# 요약기 함수 정의
def summary_tool(text: str) -> str:
    messages = [
        {"role": "system", "content": (
            "너는 AI 전문가로서 사용자의 질문에 정확하고 친절하게 답변하는 도우미야. "
            "벡터 DB의 검색 결과가 없거나 문서가 부족하면 직접 답하거나 적절한 요약을 제공해줘. "
            "절대로 '인터넷 검색이 불가능합니다'라고 말하지 마. 답은 항상 한국어로 해줘."
        )},
        {"role": "user", "content": f"{text}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content

# 각 소스 전용 QA 도구 생성
qa_news_tool = Tool(
    name="news_query_tool",
    func=lambda q: filtered_rag_run(q, "news"),
    description="뉴스 관련 정보가 필요할 때 사용합니다."
)

qa_blog_tool = Tool(
    name="blog_query_tool",
    func=lambda q: filtered_rag_run(q, "blog"),
    description="블로그 관련 정보가 필요할 때 사용합니다."
)

qa_paper_tool = Tool(
    name="paper_query_tool",
    func=lambda q: filtered_rag_run(q, "paper"),
    description="논문 관련 정보가 필요할 때 사용합니다."
)

# 일반 요약/처리용 도구
summary_tool = Tool(
    name="text_summarizer",
    func=summary_tool,
    description="일반적인 질문이나 요약 요청에 사용됩니다."
)

# Agent 초기화
agent = initialize_agent(
    tools=[qa_news_tool, qa_blog_tool, qa_paper_tool, summary_tool],
    llm=ChatOpenAI(model_name="gpt-4", temperature=0.3),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 사용자 입력
query = input("🤖 질문을 입력하세요: ")

# 실행
response = agent.run(query)
print("\n🧠 답변:")
print(response)
