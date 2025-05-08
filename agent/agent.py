from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from dotenv import load_dotenv
import os

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
        name="crawl_news",
        description=(
            "뉴스를 크롤링하고 요약하는 파이프라인. "
            "사용자가 '뉴스', '기사', '보도', 'news' 등의 키워드로 요청할 때 사용합니다."
        ),
    ),
    StructuredTool.from_function(
        Blogs_pipeline_wrapped,
        name="crawl_blog",
        description=(
            "블로그 글을 크롤링하고 요약하는 파이프라인. "
            "사용자가 '블로그', 'blog', '블로그 글', '블로그 요약' 요청 시 사용합니다."
        ),
    ),
    StructuredTool.from_function(
        Paper_pipeline_wrapped,
        name="crawl_papers",
        description=(
            "논문을 크롤링하고 요약하는 파이프라인. "
            "사용자가 '논문', '학술자료', 'paper', '논문 요약' 요청 시 사용합니다."
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

if __name__ == "__main__":
    user_input = input("요청을 입력하세요: ")
    result = agent.invoke(user_input)
    #pdf_path = export_json_to_pdf(result)
    #send_email_with_pdf(to_email, pdf_path)
    print("\n[최종 결과]\n", result)
