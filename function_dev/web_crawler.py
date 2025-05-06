import requests
from bs4 import BeautifulSoup
import re
import time
import os
from dotenv import load_dotenv

# Google CSE 정보 입력
API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")

def crawl_tistory(keyword):
    results = crawl_tistory_blogs_google(max_results=10)
    for r in results:
        print("\n📌 URL:", r['url'])
        print("내용 미리보기:", r['content'][:300])

# Step 1: Google CSE로 티스토리 글 URL 검색
def search_tistory_google(keyword, max_results=10):
    links = []
    start_index = 1

    while len(links) < max_results:
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={API_KEY}&cx={CSE_ID}&q={keyword}&start={start_index}&sort=date"
        )
        res = requests.get(url)
        data = res.json()

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            link = item.get("link", "")
            if "tistory.com" in link:
                title = item.get("title", "").strip()
                links.append({
                    "url": link,
                    "title": title
                })
                if len(links) >= max_results:
                    break

        start_index += 10
        time.sleep(1)  # API 쿼터 보호

    return links

# Step 2: 티스토리 본문 추출
def extract_tistory_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')

        content_div = (
            soup.find("div", class_=re.compile("tt_article_useless_p_margin|entry-content|article-view")) or
            soup.find("div", id="content") or
            soup.find("div", id="article") or
            soup.find("article")
        )

        if content_div:
            return content_div.get_text(separator="\n").strip()
        else:
            return "❌ 본문을 찾을 수 없습니다."
    except Exception as e:
        return f"⚠️ 오류 발생: {e}"

# Step 3: 실행 예시
def crawl_tistory_blogs_google(keyword, max_results=10):
    blog_links = search_tistory_google(keyword, max_results)
    extracted = []

    for item in blog_links:
        url = item["url"]
        title = item["title"]
        print(f"📘 크롤링 중: {title} ({url})")
        content = extract_tistory_content(url)
        extracted.append({
            "title": title,
            "url": url,
            "content": content
        })
        time.sleep(1)

    return extracted