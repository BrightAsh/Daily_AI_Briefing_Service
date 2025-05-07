import json
# from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration

# 1️⃣ LED 요약 모델 로드
# model_name = 'allenai/led-base-16384'
# tokenizer = LEDTokenizer.from_pretrained(model_name)
# model = LEDForConditionalGeneration.from_pretrained(model_name)

# # 2️⃣ 요약 함수 (LED)
# def summarize_led(text, max_input_length=16000):
#     inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_input_length, truncation=True)
#     attention_mask = inputs.ne(tokenizer.pad_token_id).long()  # LED는 attention_mask 필요
#     summary_ids = model.generate(
#         inputs,
#         attention_mask=attention_mask,
#         max_length=512,      # 출력 길이
#         num_beams=4,
#         early_stopping=True
#     )
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# # 3️⃣ 중복 문장 제거 함수 (동일)
# def remove_duplicate_sentences(text):
#     seen = set()
#     result = []
#     sentences = text.split('.')
#     for s in sentences:
#         s_clean = s.strip()
#         if s_clean and s_clean not in seen:
#             seen.add(s_clean)
#             result.append(s_clean)
#     return '. '.join(result)

# 4️⃣ 블로그 번역 모델
translation_model_name = "seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# 5️⃣ 블로그 번역 함수
def translate_summaries(text):
    print(text)
    inputs = translation_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    outputs = translation_model.generate(**inputs, max_length=2048)
    translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# # 6️⃣ 계층적 요약 함수 (영어 논문 전용)
# def hierarchical_summary_led(full_text, chunk_size=12000):
#     # 분할 요약
#     text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
#     chunk_summaries = []
#     for i, chunk in enumerate(text_chunks, 1):
#         print(f"    🧩 부분 {i}/{len(text_chunks)} 요약 중...")
#         summary = summarize_led(chunk)
#         chunk_summaries.append(summary)

#     # 부분 요약 합쳐서 최종 요약
#     combined_summary = " ".join(chunk_summaries)
#     print("    🔄 최종 요약 생성 중...")
#     final_summary = summarize_led(combined_summary)

#     # 최종 요약 후 중복 제거
#     cleaned_summary = remove_duplicate_sentences(final_summary)

#     translated_summary = translate_summaries(cleaned_summary)

#     return translated_summary

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def summarize_bart(text):
    inputs = tokenizer([text], return_tensors="pt", max_length=1536, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # # 최종 요약 후 중복 제거
    # cleaned_summary = remove_duplicate_sentences(final_summary)

    translated_summary = translate_summaries(final_summary)

    return translated_summary