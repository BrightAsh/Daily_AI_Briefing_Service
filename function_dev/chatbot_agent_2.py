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
documents = [Document(page_content=c["text"]) for c in chunks if isinstance(c, dict)]
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
    joined = " ".join([doc.page_content for doc in docs])
    if not joined or len(joined.strip()) < 50:
        return "죄송합니다. 이 질문에 대해 참고할 수 있는 문서가 없습니다."
    return rag_chain.run(query)

# MCP 기반 요약기 함수 정의
def mcp_summarize_tool(text: str) -> str:
    messages = [
        {"role": "system", "content": "너는 입력된 내용을 간결하게 요약하거나 간단한 질문에 답하는 도우미야."},
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
    description="처음 들어보는 개념에 대한 질문에만 사용하세요. 관련 정보가 없으면 검색을 중단합니다."
)

mcp_tool = Tool(
    name="text_summarizer",
    func=mcp_summarize_tool,
    description="일반적인 질문이나 요약 요청을 처리하는 데 적합한 도구입니다. 질문이 단순하거나 문맥 검색이 필요하지 않은 경우 사용하세요."
)

# Agent 초기화
agent = initialize_agent(
    tools=[qa_tool, mcp_tool],
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
