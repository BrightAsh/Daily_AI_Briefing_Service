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
        print(f" Extracted url: {paper['pdf_url']}")
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

        return extract_abstract(full_text)

    except Exception as e:
        print(f"Error extracting PDF from {pdf_url}: {e}")
        return None


def extract_abstract(text: str) -> str:
    """
    논문 텍스트에서 Abstract만 정확히 추출합니다.
    다양한 형식의 Abstract ~ Introduction 사이를 포괄적으로 인식합니다.
    """

    # 1. Abstract 시작 위치: 다양한 구분자 지원 (줄바꿈, em dash 포함)
    abstract_pattern = re.compile(
        r'(?i)(^|\n)\s*abstract\s*(—|–|:|\.|\n)', re.IGNORECASE)
    match = abstract_pattern.search(text)
    if not match:
        return ""

    start = match.end()

    # 2. Introduction 시작 위치 찾기: 다양한 형식 포괄
    intro_pattern = re.compile(
        r'(?i)(^|\n)\s*(?:[0-9]+|[IVXivx]+)?[\.\)]?\s*introduction', re.IGNORECASE)
    intro_match = intro_pattern.search(text[start:])
    end = start + intro_match.start() if intro_match else len(text)

    # 3. Abstract 본문 추출
    abstract = text[start:end].strip()

    abstract = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', abstract)

    # 4. 불필요한 줄 필터링
    lines = abstract.split('\n')
    filtered = []
    for line in lines:
        line = line.strip()
        if len(line) < 15:
            continue
        if any(kw in line for kw in [
            "SoccerAgent", "Video", "This video shows", "Answer>", "<Call>", "Tool>", "Wiki", "Player", "MatchVision"
        ]):
            continue
        if re.match(r'[A-D]\)', line):
            continue
        filtered.append(line)

    return ' '.join(filtered)
