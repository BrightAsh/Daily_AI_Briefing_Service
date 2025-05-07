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

# 질의응답 체인 생성 (문서 기반)
rag_chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=ChatOpenAI(model_name="gpt-4", temperature=0.3),
    return_source_documents=False
)

# 안전한 RAG 실행 함수 정의
def safe_rag_run(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    print(f"🔍 관련 문서 수: {len(docs)}")
    for i, d in enumerate(docs[:2]):
        print(f"[{i}] {d.page_content[:100]}...")

    if not docs or all(len(doc.page_content.strip()) < 50 for doc in docs):
        return "🔎 관련 문서를 찾지 못했습니다."
    return rag_chain.run(query)

# MCP 기반 요약기 함수 정의
def mcp_summarize_tool(text: str) -> str:
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

# 툴 정의
qa_tool = Tool(
    name="document_query_tool",
    func=safe_rag_run,
    description=(
        "뉴스, 블로그, 논문 등 벡터 DB에서 정보를 검색해야 할 때 사용하세요. "
        "만약 관련 문서가 없다면 다른 도구를 사용하세요."
    )
)

summary_tool = Tool(
    name="text_summarizer",
    func=mcp_summarize_tool,
    description=(
        "일반적인 질문이나 요약 요청을 처리하는 데 사용하세요. "
        "정보 검색이 불필요하거나 실패한 경우에도 사용됩니다."
    )
)

# Agent 초기화
agent = initialize_agent(
    tools=[qa_tool, summary_tool],
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
