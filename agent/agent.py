from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from dotenv import load_dotenv
import os

from module.News import News_pipeline
from module.Paper import Paper_pipeline
from module.Blogs import Blogs_pipeline

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

if __name__ == "__main__":
    user_input = input("요청을 입력하세요: ")
    result = agent.invoke(user_input)
    print("\n[최종 결과]\n", result)
