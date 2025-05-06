from function_dev.synonym_finder import find_synonyms
from function_dev.paper_downloader import download_paper
from function_dev.papaer_summarizer_connector import hierarchical_summary_led

def Paper_pipeline(keyword, days, n=1, country='Korea'):
    keywords = find_synonyms(keyword, n, country)
    summarized_papers = []
    seen_urls = set()   # ✅ 중복 방지용

    for keyword in keywords:
        papers = download_paper(keyword, days)
        for idx, paper in enumerate(papers, 1):
            title = paper.get("title", "")
            full_text = paper.get("body", "").strip()
            url = paper.get("url")

            if not full_text:
                print(f"\n[{idx}] ⚠️ {title}: 본문 없음 (스킵)")
                continue

            # ✅ URL 기준 중복 제거
            if url in seen_urls:
                print(f"\n[{idx}] 🚫 {title}: 이미 처리된 논문 (중복 스킵)")
                continue
            seen_urls.add(url)

            print(f"\n[{idx}] 📰 {title}")
            print(f"📄 본문 길이: {len(full_text)}자")

            try:
                summary = hierarchical_summary_led(full_text)
                print(f"✅ 최종 요약 완료:\n{summary[:500]}...")

                summarized_papers.append({
                    "title": title,
                    "url": url,
                    "summary": summary
                })
            except Exception as e:
                print(f"❌ 요약 실패: {e}")
    return summarized_papers
