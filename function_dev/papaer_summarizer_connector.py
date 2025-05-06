import json
from transformers import LEDTokenizer, LEDForConditionalGeneration

# 1️⃣ LED 요약 모델 로드
model_name = 'allenai/led-base-16384'
tokenizer = LEDTokenizer.from_pretrained(model_name)
model = LEDForConditionalGeneration.from_pretrained(model_name)

# 2️⃣ 요약 함수 (LED)
def summarize_led(text, max_input_length=16000):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_input_length, truncation=True)
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()  # LED는 attention_mask 필요
    summary_ids = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=512,      # 출력 길이
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 3️⃣ 중복 문장 제거 함수 (동일)
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

# 4️⃣ 계층적 요약 함수 (영어 논문 전용)
def hierarchical_summary_led(full_text, chunk_size=12000):
    # 분할 요약
    text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    chunk_summaries = []
    for i, chunk in enumerate(text_chunks, 1):
        print(f"    🧩 부분 {i}/{len(text_chunks)} 요약 중...")
        summary = summarize_led(chunk)
        chunk_summaries.append(summary)

    # 부분 요약 합쳐서 최종 요약
    combined_summary = " ".join(chunk_summaries)
    print("    🔄 최종 요약 생성 중...")
    final_summary = summarize_led(combined_summary)

    # 최종 요약 후 중복 제거
    cleaned_summary = remove_duplicate_sentences(final_summary)
    return cleaned_summary