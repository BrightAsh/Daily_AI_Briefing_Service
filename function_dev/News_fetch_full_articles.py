import json
from newspaper import Article
import time

def fetch_full_article_auto(url):
    try:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"❌ {url} 본문 추출 실패: {e}")
        return None


def process_articles(articles):
    full_articles = []
    for idx, article in enumerate(articles, 1):
        url = article.get('url')
        title = article.get('title')
        print(f"\n[{idx}] {title}")
        print(f"🌐 {url}")

        full_text = fetch_full_article_auto(url)
        if full_text:
            print(f"✅ 본문 추출 성공 (길이: {len(full_text)}자)")
        else:
            print(f"⚠️ 본문 없음")

        full_articles.append({
            "title": title,
            "url": url,
            "source": article.get('source'),
            "publishedAt": article.get('publishedAt'),
            "full_text": full_text or ""
        })

        time.sleep(1)  # 너무 빠른 요청 방지

    print(f"\n✅ 본문 포함 뉴스 {len(full_articles)}개 완료!")
    return full_articles