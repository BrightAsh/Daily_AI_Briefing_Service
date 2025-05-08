# 🧠 Daily_AI_Briefing_Service: 프롬프트 기반 자동 리서치 & 요약 AI 에이전트

자연어로 된 한 줄의 프롬프트로 뉴스, 블로그, 논문 등 다양한 출처에서 관련 정보를 수집하고, 자동 요약을 통해 인사이트를 제공하는 **RAG(Retrieval-Augmented Generation)** 기반 AI 리서치 도우미입니다.

---

## 🚀 주요 기능

- ✍️ **프롬프트 분석**  
  사용자의 입력 문장에서 키워드, 문서 유형, 기간 등을 자동 추출합니다.  
  예: `"최근 3일간 인공지능 논문 요약해줘"` → `{"day": 3, "keyword": "인공지능", "type": "논문"}`

- 🔍 **데이터 수집**  
  - 뉴스: News API 기반 실시간 기사 수집  
  - 블로그: Google API 활용 블로그 본문 크롤링  
  - 논문: arXiv API 활용 최신 논문 수집 및 전처리

- 🧹 **문서 전처리 및 요약**  
  각 문서의 불필요한 요소(광고, 저자 정보, 태그 등)를 제거하고 LLM 기반 요약 수행

- 🧠 **벡터화 및 RAG 적용**  
  요약된 내용을 임베딩하여 벡터DB에 저장하고, LLM이 검색된 문서 기반으로 응답을 생성

- 📧 **PDF 메일 발송**  
  특정 키워드에 대한 PDF 형식의 요약 리포트를 이메일로 전송
  
- 🧾 **하나의 인사이트 문서로 통합 (향후 기능)**  
  여러 문서를 분석하여 주제별 최종 인사이트 보고서 자동 생성 (개발 예정)

---

## 🗂️ 프로젝트 구조
```
Daily_AI_Briefing_Service/
├── agent/                 # 에이전트 관련 모듈 (키워드 확장, 요약 등)
├── docs/                  # 문서 자료 및 참고 파일
├── faiss_index/           # 벡터 DB 인덱스 파일 저장 폴더
├── fonts/                 # PDF 생성 시 사용할 폰트 파일
├── function_dev/          # 기능 단위 개발 스크립트 모음
├── json_data/             # 수집된 원시 데이터 (JSON)
├── module/                # 기능 모듈화된 코드들(뉴스, 논문, 블로그)
├── sample_blogs/          # 테스트용 블로그 데이터
├── sample_news/           # 테스트용 뉴스 데이터
├── sample_papers/         # 테스트용 논문 데이터
├── .gitignore             # Git 추적 제외 설정
├── README.md              # 깃허브 소개 문서
├── chat.py                # RAG 기반 챗봇 인터페이스 (Gradio UI)
├── faiss_index.index      # 저장된 벡터 인덱스 파일
├── main.py                # 메인 실행 파일 (Gradio UI)
└── requirements.txt       # 필요 패키지 목록
```

---

## 🛠️ 기술 스택

| 영역             | 사용 기술 및 도구                     |
|------------------|----------------------------------------|
| LLM              | OpenAI GPT-3.5 Turbo, LangChain        |
| AI 모델           | KoBart, t5                            |
| 키워드 확장       | OpenAI GPT-3.5 Turbo                  |
| 데이터 수집       | News API, Google API, arXiv API          |
| 전처리            | BeautifulSoup, Regex                  |
| 요약              | HuggingFace Transformers              |
| 벡터 임베딩       | OpenAI Embedding API, Sentence-BERT    |
| 벡터DB            | FAISS                                |
| 응답 생성(RAG)    | LangChain + Agent                    |
| UI               | Gradio                               |

---

## 👨‍👨‍👦 팀 소개

| Web / SW | AI / Data | SW / Data |
|:--------:|:---------:|:---------:|
| [<img src="https://github.com/gaeul-3041.png" width="100px">](https://github.com/gaeul-3041) | [<img src="https://github.com/BrightAsh.png" width="100px">](https://github.com/BrightAsh) | [<img src="https://github.com/JReal2.png" width="100px">](https://github.com/JReal2) |
| [진현](https://github.com/gaeul-3041) | [송명재](https://github.com/BrightAsh) | [이정렬](https://github.com/JReal2) |
| Gradio 기반 웹 UI 구현<br>질의응답 챗봇 Agent 모델 개발 | 자료 요약 Agent 모델 개발<br>뉴스 및 논문 요약 Tool 구현 | 웹 크롤링 기반 데이터 수집 및 전처리<br>블로그 및 논문 요약 Tool 구현 |


