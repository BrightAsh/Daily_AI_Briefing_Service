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
import requests
import gradio as gr

# 🔍 SerpAPI 대체 GoogleSearch 클래스
class GoogleSearch:
    def __init__(self, params):
        self.params = params
        self.api_key = params.get("api_key")

    def get_dict(self):
        response = requests.get("https://serpapi.com/search", params=self.params)
        response.raise_for_status()
        return response.json()

# 🔐 환경 변수
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 📚 벡터DB 로딩
base_path = os.path.dirname(__file__)
CHUNK_PATH = os.path.join(base_path, "faiss_index", "doc_chunks.npy")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
chunks = np.load(CHUNK_PATH, allow_pickle=True)
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
documents = [
    Document(page_content=c["text"], metadata={"source": c["source"]})
    for c in chunks if isinstance(c, dict)
]
vectordb = FAISS.from_documents(documents, embedding=embedding_model)

# 🔎 RAG 함수
def filtered_rag_run(query: str, source_filter: str) -> str:
    filtered_docs = [doc for doc in documents if doc.metadata.get("source") == source_filter]
    if not filtered_docs:
        return f"⚠️ '{source_filter}' 소스에서 문서를 찾을 수 없습니다."

    local_vectordb = FAISS.from_documents(filtered_docs, embedding=embedding_model)
    local_retriever = local_vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    rag_chain = RetrievalQA.from_chain_type(
        retriever=local_retriever,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3),
        return_source_documents=False
    )
    docs = local_retriever.get_relevant_documents(query)
    if not docs or all(len(doc.page_content.strip()) < 50 for doc in docs):
        return f"🔎 '{source_filter}' 문서에서 유효한 정보를 찾지 못했습니다."
    return rag_chain.run(query)

# ✨ 요약 툴
def run_summary_tool(text: str) -> str:
    messages = [
        {"role": "system", "content": (
            "너는 AI 전문가로서 사용자의 질문에 정확하고 친절하게 답변하는 도우미야. Final Answer는 반드시 한국어로 작성해. "
            "벡터 DB의 검색 결과가 없거나 문서가 부족하면 직접 답하거나 적절한 요약을 제공해줘. "
        )},
        {"role": "user", "content": f"{text}"}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1
    )
    return response.choices[0].message.content

# 🌍 웹 검색 툴
def search_web_tool(query: str) -> str:
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 3
    })
    results = search.get_dict()

    if "organic_results" not in results:
        return "❌ 검색 결과를 불러오는 데 실패했습니다."

    output = "🌐 아래 정보는 Google 검색(SerpAPI) 결과를 기반으로 합니다:\n"
    for item in results["organic_results"][:3]:
        title = item.get("title", "제목 없음")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        output += f"\n🔗 [{title}]({link})\n{snippet}\n"
    return output

# 🛠️ Tool 정의
tools = [
    Tool("news_query_tool", lambda q: filtered_rag_run(q, "news"), "뉴스 관련 질의"),
    Tool("blog_query_tool", lambda q: filtered_rag_run(q, "blog"), "블로그 관련 질의"),
    Tool("paper_query_tool", lambda q: filtered_rag_run(q, "paper"), "논문 관련 질의"),
    Tool("text_summarizer", run_summary_tool, "문맥 요약"),
    Tool("web_search", search_web_tool, "실시간 웹 검색")
]

# 🧠 에이전트 초기화
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# 💬 Gradio 챗 UI 정의
def chat_with_agent(message, history):
    try:
        response = agent.run(message)
    except Exception as e:
        response = f"❌ 오류 발생: {str(e)}"
    history.append((message, response))
    return "", history

# 🖼️ Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Daily AI Briefing Chatbot")

    chatbot = gr.Chatbot(label="🧠 AI 챗봇")
    msg = gr.Textbox(placeholder="질문을 입력하세요!", label="💬 입력")
    clear = gr.Button("🧹 대화 초기화")

    msg.submit(chat_with_agent, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch(share=True)
