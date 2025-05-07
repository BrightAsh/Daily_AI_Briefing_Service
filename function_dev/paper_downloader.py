import feedparser
from datetime import datetime, timedelta
import pytz
import requests
import io
import re
from pdfminer.high_level import extract_text

def download_paper(keyword,day):
    papers = get_recent_arxiv_pdfs(keyword,day, max_results=10)

    paper_body = []

    for paper in papers:
        print(f"\n⏳ Processing: {paper['title']}")
        body_text = extract_body_from_pdf_url(paper['pdf_url'])
        if body_text:
            body_text = body_text.replace('\n', ' ')
            paper["title"] = paper["title"].replace('\n', ' ')
            print(f"✅ Extracted body length: {len(body_text)} chars")
            paper_body.append({
            "title": paper["title"],
            "body": body_text,
            "url": paper["pdf_url"]
        })
        else:
            print("❌ Failed to extract body text.")

    return paper_body

def get_recent_arxiv_pdfs(keyword,day, max_results=100):
    query = f"search_query=all:{keyword}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    url = f"http://export.arxiv.org/api/query?{query}"
    feed = feedparser.parse(url)

    now = datetime.utcnow().replace(tzinfo=pytz.utc)
    one_day_ago = now - timedelta(days=day)

    papers = []
    for entry in feed.entries:
        published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
        if published >= one_day_ago:
            arxiv_id = entry.id.split('/abs/')[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            papers.append({
                "id": arxiv_id,
                "title": entry.title.strip(),
                "pdf_url": pdf_url,
                "published": published.isoformat()
            })
    return papers


def extract_body_from_pdf_url(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=15)
        response.raise_for_status()
        pdf_bytes = io.BytesIO(response.content)
        full_text = extract_text(pdf_bytes)

        # Abstract와 References 제거
        return extract_abstract(full_text)

    except Exception as e:
        print(f"Error extracting PDF from {pdf_url}: {e}")
        return None


def extract_abstract(text: str) -> str:
    """
    논문 텍스트에서 Abstract만 정확히 추출합니다.
    Abstract ~ Introduction 사이를 기준으로 하되, 중간에 삽입된 실험/QA 등도 필터링합니다.
    """

    # 1. Abstract 시작 위치
    match = re.search(r'(?i)(^|\n)\s*abstract\s*[\n:]', text)
    if not match:
        return ""

    start = match.end()

    # 2. Introduction 시작 위치 찾기
    intro = re.search(r'(?i)(\n|^)\s*(?:[0-9]+|[IVXivx]+)?[\.\)]?\s*introduction', text[start:])
    end = start + intro.start() if intro else len(text)

    # 3. Abstract 본문만 추출
    abstract = text[start:end].strip()

    # 4. 불필요한 내용 필터링 (시스템 응답, QA 예시, 태그 등)
    lines = abstract.split('\n')
    filtered = []
    for line in lines:
        # 너무 짧은 문장 / 숫자만 있는 줄 제거
        if len(line.strip()) < 15:
            continue
        # 시스템 응답, 영상, 위키 등 제거
        if any(kw in line for kw in [
            "SoccerAgent", "Video", "This video shows", "Answer>", "<Call>", "Tool>", "Wiki", "Player", "MatchVision"
        ]):
            continue
        if re.match(r'[A-D]\)', line.strip()):  # 선택지
            continue
        filtered.append(line.strip())

    return ' '.join(filtered)
