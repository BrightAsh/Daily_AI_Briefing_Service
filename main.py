import gradio as gr
import os
import re
import json
import time
import threading
from dotenv import load_dotenv
from agent import agent
from datetime import datetime

from function_dev.pdf_creator import export_json_to_pdf
from function_dev.email_sender import send_email_with_pdf
from function_dev.json_to_vectordb import run_vector_pipeline
from module.wrapper import (
    News_pipeline_wrapped,
    Blogs_pipeline_wrapped,
    Paper_pipeline_wrapped,
    set_web_params
)
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool


# 소스 추론
def infer_source_type(prompt: str) -> str:
    prompt = prompt.lower()
    if any(k in prompt for k in ["뉴스", "기사", "보도", "news"]):
        return "news"
    elif any(k in prompt for k in ["블로그", "blog"]):
        return "blog"
    elif any(k in prompt for k in ["논문", "paper", "학술자료"]):
        return "paper"
    return "unknown"

# 요약 결과 파싱
def parse_news_output(output: str):
    blocks = output.strip().split("\n\n")
    parsed_items = []

    for block in blocks:
        title_match = re.search(r"\*\*(.+?)\*\*", block) or re.search(r"\[(.+?)\]\(", block)
        url_match = re.search(r"\((https?://[^\s]+)\)", block)
        summary_match = re.findall(r"-\s(.+)", block)

        if title_match and url_match:
            parsed_items.append({
                "title": title_match.group(1).strip(),
                "url": url_match.group(1).strip(),
                "summary": " ".join(summary_match).strip() if summary_match else ""
            })

    return parsed_items

# 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tools = [
    StructuredTool.from_function(News_pipeline_wrapped, name="crawl_news",
        description="뉴스를 크롤링하고 요약하는 파이프라인. '뉴스', '기사', '보도', 'news' 요청 시 사용"),
    StructuredTool.from_function(Blogs_pipeline_wrapped, name="crawl_blog",
        description="블로그 글을 크롤링하고 요약하는 파이프라인. '블로그', 'blog' 요청 시 사용"),
    StructuredTool.from_function(Paper_pipeline_wrapped, name="crawl_papers",
        description="논문을 크롤링하고 요약하는 파이프라인. '논문', 'paper', '학술자료' 요청 시 사용")
]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

system_message = """
너는 뉴스, 블로그, 논문 자료를 크롤링하고 요약하는 AI 에이전트야.
사용자의 요청이 뉴스, 블로그, 논문 중 하나라도 포함되면 적절한 툴을 실행해야 해.

만약 요청이 이 세 가지와 관련이 없다면 이렇게 답해야 해:
"죄송합니다. 현재 뉴스, 블로그, 논문에 대한 요청만 처리할 수 있습니다."
"""

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_message.strip()}
)


# 파이프라인 실행 함수
def run_pipeline(prompt, country, synonym_range, email):
    start_time = time.time()
    result_holder = {"done": False, "result": None}

    def task():
        set_web_params(synonym_range, country)
        result = agent.invoke(prompt, return_only_outputs=True)
        print(result)
        parsed = parse_news_output(result["output"])
        print(parsed)
        result_holder["result"] = parsed
        result_holder["done"] = True

        # JSON 저장
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "database")
        os.makedirs(save_dir, exist_ok=True)
        
        source_type = infer_source_type(prompt)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(save_dir, f"{source_type}_summary_{timestamp}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=4)

        # PDF + 이메일 전송
        if email.strip():
            pdf_path = export_json_to_pdf(parsed)
            send_email_with_pdf(email, pdf_path)

        # 벡터화
        run_vector_pipeline(database_dir=save_dir)

    thread = threading.Thread(target=task)
    thread.start()

    # 실시간 상태 표시
    while not result_holder["done"]:
        elapsed = int(time.time() - start_time)
        yield f"<div style='text-align:center; margin-top: 20px;'>⏳ 에이전트 실행 중입니다...<br><small>경과 시간: {elapsed}초</small></div>"
        time.sleep(1)

    # 최종 결과 표시
    elapsed = int(time.time() - start_time)
    output_md = f"<div style='text-align:center; margin-top: 20px;'>✅ <b>작업이 완료되었습니다!</b><br><small>총 경과 시간: {elapsed}초</small></div><br><br>"
    for i, item in enumerate(result_holder["result"], 1):
        output_md += f"**{i}. [{item['title']}]({item['url']})**\n\n- {item['summary']}\n\n"
    yield output_md


# Gradio UI 정의
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Daily AI Briefing Assistant")

    with gr.Row():
        prompt = gr.Textbox(label="📝 프롬프트", placeholder="예: 최근 2일간 AI 관련 뉴스 정리해줘")
        country = gr.Dropdown(["Korea", "Japan", "China", "USA", "Europe"], value="Korea", label="🌍 국가 선택")
        synonym_range = gr.Slider(1, 5, step=1, value=3, label="🔁 검색 범위")
        email = gr.Textbox(label="📧 이메일 (선택)", placeholder="결과를 이메일로 받고 싶다면 입력해주세요.")

    submit = gr.Button("🚀 실행")
    output = gr.Markdown(label="📄 결과")

    submit.click(fn=run_pipeline, inputs=[prompt, country, synonym_range, email], outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
