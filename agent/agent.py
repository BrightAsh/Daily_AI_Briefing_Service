from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from dotenv import load_dotenv
import os
import sys

sys.path.append("E:\Daily_AI_Briefing_Service")

# ✅ wrapping 함수 import
from module.wrapper import News_pipeline_wrapped, Blogs_pipeline_wrapped, Paper_pipeline_wrapped, set_web_params
from function_dev.pdf_creator import export_json_to_pdf
from function_dev.email_sender import send_email_with_pdf

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# web에서 받을 값
to_email = 'busisi7776@gmail.com'
set_web_params(1,'korea')

# ✅ 툴 정의 (wrapping 함수 사용)
tools = [
    StructuredTool.from_function(
        News_pipeline_wrapped,
        name="news",
        description=(
            "뉴스를 크롤링하고 요약하는 파이프라인. "
            "다음과 같은 키워드가 포함된 사용자의 요청에 대응합니다: "
            "'뉴스', '기사', '보도', '속보', 'news', '언론 보도', '뉴스 요약', '뉴스 내용 알려줘', "
            "'최근 뉴스', '오늘 뉴스', '뉴스 크롤링'."
        ),
    ),
    StructuredTool.from_function(
        Blogs_pipeline_wrapped,
        name="blog",
        description=(
            "블로그 글을 크롤링하고 요약하는 파이프라인. "
            "다음과 같은 키워드가 포함된 사용자의 요청에 대응합니다: "
            "'블로그', 'blog', '블로그 글', '블로그 요약', '블로그 내용', "
            "'블로그 분석', '블로그에서 찾은 글', '블로그 참고', '블로그 정보 알려줘'."
        ),
    ),
    StructuredTool.from_function(
        Paper_pipeline_wrapped,
        name="paper",
        description=(
            "논문을 크롤링하고 요약하는 파이프라인. "
            "다음과 같은 키워드가 포함된 사용자의 요청에 대응합니다: "
            "'논문', '학술자료', '논문 요약', 'paper', 'academic paper', 'research article', "
            "'학술 문서', '연구 자료', '논문 내용 알려줘', '논문 크롤링'."
        ),
    ),
]

# ✅ LLM 설정
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

# ✅ 시스템 프롬프트
system_message = """
너는 뉴스, 블로그, 논문 자료를 크롤링하고 요약하는 AI 에이전트야.

사용자의 요청이 다음 중 하나라도 포함되면 반드시 적절한 툴을 사용해야 해:
- '뉴스', '기사', '보도', 'news'
- '블로그', 'blog', '블로그 글', '블로그 요약'
- '논문', '학술자료', '논문 요약', 'paper', 'academic', 'research article'

예를 들어, 사용자가 '이 논문 좀 요약해줘', '블로그에서 찾은 내용 정리해줘' 라고 해도 툴을 반드시 사용해야 해.

하지만 위 키워드가 포함되지 않은 일반적인 요청이면 툴을 실행하지 말고 이렇게 답해야 해:
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

if __name__ == "__main__":
    user_input = input("요청을 입력하세요: ")
    result = agent.invoke(user_input)
    #pdf_path = export_json_to_pdf(result)
    #send_email_with_pdf(to_email, pdf_path)
    print("\n[최종 결과]\n", result)
