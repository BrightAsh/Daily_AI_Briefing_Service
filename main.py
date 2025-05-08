import gradio as gr
from agent import agent  # LangChain agent
from function_dev.synonym_finder import find_synonyms  # ✅ 연결된 함수
from function_dev.pdf_creator import export_json_to_pdf  # ✅ 연결된 함수
from function_dev.email_sender import send_email_with_pdf  # ✅ 연결된 함수

from module.News import News_pipeline
from module.Paper import Paper_pipeline
from module.Blogs import Blogs_pipeline

from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from dotenv import load_dotenv

import os
import re
import uuid
import json

def parse_news_output(llm_output: str):
    """
    LLM이 생성한 뉴스 요약 텍스트를 구조화된 JSON 리스트로 변환.
    """
    pattern = r'\*\*\[(.*?)\]\((.*?)\)\*\*\n\s*[-–]\s*(.+?)(?=\n\d+\.|\Z)'
    matches = re.findall(pattern, llm_output.strip(), flags=re.DOTALL)

    parsed_items = []
    for title, url, summary in matches:
        parsed_items.append({
            "title": title.strip(),
            "url": url.strip(),
            "summary": summary.replace("\n", " ").strip()
        })

    return parsed_items

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 툴 정의 (동의어 & 범위 명시)
tools = [
    StructuredTool.from_function(
        News_pipeline,
        name="crawl_news",
        description=(
            "뉴스를 크롤링하고 요약하는 파이프라인. "
            "사용자가 '뉴스', '기사', '보도', 'news' 등의 키워드로 요청할 때 사용합니다."
        ),
    ),
    StructuredTool.from_function(
        Blogs_pipeline,
        name="crawl_blog",
        description=(
            "블로그 글을 크롤링하고 요약하는 파이프라인. "
            "사용자가 '블로그', 'blog', '블로그 글', '블로그 요약' 요청 시 사용합니다."
        ),
    ),
    StructuredTool.from_function(
        Paper_pipeline,
        name="crawl_papers",
        description=(
            "논문을 크롤링하고 요약하는 파이프라인. "
            "사용자가 '논문', '학술자료', 'paper', '논문 요약' 요청 시 사용합니다."
        ),
    ),
]


# ✅ LLM 설정 (OpenAI 모델 사용)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

# ✅ 시스템 프롬프트 설정 (agent_kwargs로 전달)
system_message = """
너는 뉴스, 블로그, 논문 자료를 크롤링하고 요약하는 AI 에이전트야.
사용자의 요청이 뉴스, 블로그, 논문 중 하나라도 포함되면 적절한 툴을 실행해야 해.
하지만 요청이 이 세 가지와 관련이 없다면 절대 툴을 실행하지 말고 이렇게 답해야 해:

"죄송합니다. 현재 뉴스, 블로그, 논문에 대한 요청만 처리할 수 있습니다."
"""

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={
        "system_message": system_message.strip()
    }
)

def run_pipeline(prompt, country, synonym_range, email):
    result = agent.invoke(prompt)
    result_json = parse_news_output(result['output'])
    
    # database 폴더에 JSON 저장, 매번 다른 이름이 필요하니 UUID 사용
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "database")
    os.makedirs(save_dir, exist_ok=True)

    unique_filename = f"summary_{uuid.uuid4()}.json"
    file_path = os.path.join(save_dir, unique_filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)
        
    # 이메일 전송
    if email and email.strip():
        pdf_path = export_json_to_pdf(result_json)
        send_email_with_pdf(email, pdf_path)
    
    return result_json

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Daily AI Briefing Assistant")

    with gr.Row():
        prompt = gr.Textbox(label="📝 프롬프트", placeholder="예: 최근 일본의 AI 뉴스 요약해줘")
        country = gr.Dropdown(choices=["Korea", "Japan", "China", "USA", "Europe"], value="Korea", label="🌍 국가 선택")
        synonym_range = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="🔁 검색 범위")
        email = gr.Textbox(label="📧 이메일 (선택)", placeholder="결과를 이메일로 받고 싶다면 입력해주세요")

    submit = gr.Button("🚀 실행")
    output = gr.Textbox(label="📄 결과", lines=15)

    submit.click(fn=run_pipeline, inputs=[prompt, country, synonym_range, email], outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
