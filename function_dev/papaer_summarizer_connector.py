import json
from transformers import pipeline

# 1️⃣ 모델 초기화
summarizer = pipeline("summarization", model="allenai/led-base-16384")

# 2️⃣ 요약할 파일 경로
INPUT_FILE = "news_data_full.json"
OUTPUT_FILE = "news_data_summaries.json"

# 3️⃣ 데이터 불러오기
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

summarized_articles = []

# 4️⃣ 요약 실행
for idx, article in enumerate(articles, 1):
    title = article.get("title", "")
    full_text = article.get("full_text", "").strip()

    if not full_text:
        print(f"\n[{idx}] ⚠️ {title}: 본문 없음 (스킵)")
        continue

    print(f"\n[{idx}] 📰 {title}")
    print(f"📄 본문 길이: {len(full_text)}자")

    # 요약 실행
    try:
        summary = summarizer(full_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        print(f"✅ 요약 완료:\n{summary}")

        summarized_articles.append({
            "title": title,
            "url": article.get("url"),
            "source": article.get("source"),
            "publishedAt": article.get("publishedAt"),
            "summary": summary
        })
    except Exception as e:
        print(f"❌ 요약 실패: {e}")

# 5️⃣ 요약 결과 저장
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summarized_articles, f, ensure_ascii=False, indent=2)

print(f"\n✅ 총 {len(summarized_articles)}건 요약 완료 → '{OUTPUT_FILE}' 저장됨")
