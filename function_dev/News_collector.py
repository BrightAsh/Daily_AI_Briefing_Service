# agents/data_fetcher.py

import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from urllib.parse import quote

# .env 로드
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_data_newsapi(keywords, day=1):
    day = int(day)
    all_articles = []
    seen_titles = set()  # 중복 방지용

    # 현재 시간 및 하루 전 계산
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=day)

    # 날짜 포맷: YYYY-MM-DDTHH:MM:SSZ (NewsAPI는 ISO 8601 지원)
    from_date_str = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    to_date_str = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    print(f"\n📅 수집 기간: {from_date_str} ~ {to_date_str}")

    for keyword in keywords:
        print(f"\n🔎 '{keyword}'로 뉴스 검색 중...")
        encoded_keyword = quote(keyword)
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={encoded_keyword}&"
            f"from={from_date_str}&"
            f"to={to_date_str}&"
            f"language=ko&"
            f"sortBy=publishedAt&"
            f"apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            for article in articles:
                title = article.get('title', '').strip()
                description = article.get('description', '').strip()
                url = article.get('url', '').strip()
                source = article.get('source', {}).get('name', '').strip()

                # 중복 기사 방지 (title 기준)
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    all_articles.append({
                        "title": title,
                        "url": url,
                        "source": source,
                        "publishedAt": article.get('publishedAt', '')
                    })
            print(f"✅ '{keyword}' 결과: {len(articles)}건")
        else:
            print(f"❌ '{keyword}' 검색 실패: {response.status_code}")
            try:
                print(f"↪️ 오류 내용: {response.json()}")
            except:
                pass

    print(f"\n📰 총 수집된 고유 기사 수: {len(all_articles)}개")
    return all_articles


import json

if __name__ == "__main__":
    news = fetch_data_newsapi(['인공지능', 'AI', '인공신경망', '머신러닝', '딥러닝'], 2)

    for idx, article in enumerate(news, 1):
        print(f"\n[{idx}] {article['title']}")
    with open("news_data.json", "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 뉴스 데이터 {len(news)}개를 'news_data.json'에 저장 완료!")
