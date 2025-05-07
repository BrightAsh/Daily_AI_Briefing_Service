from function_dev.synonym_finder import find_synonyms
from function_dev.News_collector import fetch_data_newsapi
from function_dev.News_fetch_full_articles import process_articles
from function_dev.News_summarizer import hierarchical_summary

def News_pipeline(keyword, days, n=1, country='Korea'):
    keywords = find_synonyms(keyword, n, country)
    all_articles = fetch_data_newsapi(keywords, days)
    full_articles = process_articles(all_articles)

    summarized_articles = []
    for idx, article in enumerate(full_articles, 1):
        title = article.get("title", "")
        full_text = article.get("full_text", "").strip()

        if not full_text:
            print(f"\n[{idx}] ⚠️ {title}: 본문 없음 (스킵)")
            continue

        print(f"\n[{idx}] 📰 {title}")
        print(f"📄 본문 길이: {len(full_text)}자")

        try:
            summary = hierarchical_summary(full_text,keyword)
            print(f"✅ 최종 요약 완료:\n{summary}")

            summarized_articles.append({
                "title": title,
                "url": article.get("url"),
                "summary": summary
            })
        except Exception as e:
            print(f"❌ 요약 실패: {e}")
    return summarized_articles


import json

result = News_pipeline('인공지능', 2, 1)  # 기존 실행 코드

# ✅ JSON 파일 저장
with open('news_summary_result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("🎉 결과가 news_summary_result.json 파일에 저장되었습니다!")
