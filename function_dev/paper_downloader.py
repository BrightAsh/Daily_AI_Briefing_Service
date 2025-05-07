import feedparser
from datetime import datetime, timedelta
import pytz
import requests
import io
from pdfminer.high_level import extract_text

def download_paper(keyword,day):
    papers = get_recent_arxiv_pdfs(keyword,day, max_results=1)

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
        return remove_abstract_and_references(full_text)

    except Exception as e:
        print(f"Error extracting PDF from {pdf_url}: {e}")
        return None


def remove_abstract_and_references(text: str) -> str:
    lines = text.splitlines()
    body_lines = []
    capturing = False

    for line in lines:
        line_lower = line.strip().lower()

        # 시작 조건: Introduction 등장 시부터 시작
        if not capturing and "introduction" in line_lower and len(line_lower) < 30:
            capturing = True
            continue  # 'Introduction' 라인은 건너뜀

        # 종료 조건: References 등장 시 종료
        if capturing and "references" in line_lower and len(line_lower) < 30:
            break

        if capturing:
            body_lines.append(line)

    return "\n".join(body_lines).strip()
