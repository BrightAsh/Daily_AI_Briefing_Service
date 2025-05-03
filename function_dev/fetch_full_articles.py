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


def process_articles(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

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

    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_articles, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 본문 포함 뉴스 {len(full_articles)}개를 '{output_path}'에 저장 완료!")


if __name__ == "__main__":
    process_articles("news_data.json", "news_data_full.json")
