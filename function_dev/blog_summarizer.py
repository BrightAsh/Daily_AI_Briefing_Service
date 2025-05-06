import json
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# 1️⃣ KoBART 요약 모델 로드
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')

# 2️⃣ 파일 경로
INPUT_FILE = "blogs_data.json"
OUTPUT_FILE = "blogs_data_summaries.json"

# 3️⃣ 요약 함수 (반복 억제 파라미터 적용)
def summarize_kobart(text, max_input_length=1024):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_input_length, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=200,
        min_length=30,
        num_beams=4,
        no_repeat_ngram_size=3,   # 🔥 3그램 반복 방지
        repetition_penalty=2.0,   # 🔥 반복 억제 강화
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 4️⃣ 키워드 포함 문장 추출 함수
def extract_sentences_with_keywords(text, keywords):
    sentences = text.split('.')
    selected = [s for s in sentences if any(kw in s for kw in keywords)]
    return '. '.join(selected)

# 5️⃣ 중복 문장 제거 함수
def remove_duplicate_sentences(text):
    seen = set()
    result = []
    sentences = text.split('.')
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            result.append(s_clean)
    return '. '.join(result)

# 6️⃣ 계층적 요약 함수 (분할 + 부분 요약 + 최종 요약 + 중복 제거)
def hierarchical_summary(full_text, keywords=None, chunk_size=1000):
    # 키워드 필터링 (선택)
    if keywords:
        print(f"🔎 키워드 중심 문장 추출 중... 키워드: {keywords}")
        filtered_text = extract_sentences_with_keywords(full_text, keywords)
        if filtered_text.strip():
            full_text = filtered_text
        else:
            print("⚠️ 키워드 포함 문장이 없어 전체 본문으로 진행합니다.")

    # 분할 요약
    text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    chunk_summaries = []
    for i, chunk in enumerate(text_chunks, 1):
        print(f"    🧩 부분 {i}/{len(text_chunks)} 요약 중...")
        summary = summarize_kobart(chunk)
        chunk_summaries.append(summary)

    # 부분 요약 합쳐서 최종 요약
    combined_summary = " ".join(chunk_summaries)
    print("    🔄 최종 요약 생성 중...")
    final_summary = summarize_kobart(combined_summary)

    # 최종 요약 후 중복 제거
    cleaned_summary = remove_duplicate_sentences(final_summary)
    return cleaned_summary

