import sys
sys.path.append("E:\Daily_AI_Briefing_Service")

from function_dev.synonym_finder import find_synonyms
from function_dev.web_crawler import crawl_tistory_blogs_google
from function_dev.blog_summarizer import hierarchical_summary, is_relevant

def Blogs_pipeline(keyword, days, n=1, country='Korea'):
    keywords = find_synonyms(keyword, n, country)
    summarized_blogs = []
    seen_urls = set()   # ✅ 중복 방지용

    for keyword in keywords:
        blogs = crawl_tistory_blogs_google(keyword, days, 20)
        for idx, blog in enumerate(blogs, 1):
            title = blog.get("title", "")
            full_text = blog.get("full_text", "").strip()
            url = blog.get("url")

            if not full_text:
                print(f"\n[{idx}] ⚠️ {title}: 본문 없음 (스킵)")
                continue

            # ✅ URL 기준 중복 제거
            if url in seen_urls:
                print(f"\n[{idx}] 🚫 {title}: 이미 처리된 블로그 (중복 스킵)")
                continue
            seen_urls.add(url)

            print(f"\n[{idx}] 📰 {title}")
            print(f"📄 본문 길이: {len(full_text)}자")

            try:
                summary = hierarchical_summary(full_text, keyword)
                if summary is None:  # ✅ 요약 결과가 None인 경우 스킵
                    print(f"⚠️ {title}: 키워드와 무관한 블로그")
                    continue

                print(f"✅ 최종 요약 완료:\n{summary[:500]}...")

                summarized_blogs.append({
                    "title": title,
                    "url": url,
                    "summary": summary
                })
            except Exception as e:
                print(f"❌ 요약 실패: {e}")
    return summarized_blogs

if __name__ == "__main__":
    Blogs_pipeline("ai", 3)