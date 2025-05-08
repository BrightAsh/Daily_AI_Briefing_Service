import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# 블로그 번역 모델
translation_model_name = "seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# 블로그 번역 함수
def translate_summaries(text):
    print(text)
    inputs = translation_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    outputs = translation_model.generate(**inputs, max_length=2048)
    translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# ✅ 임베딩 모델 로드
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ✅ 유사도 필터 함수
def is_relevant(summary: str, keyword: str, threshold=0.1) -> bool:
    embeddings = embedder.encode([summary, keyword])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    print(f"    🔍 유사도 점수: {similarity:.4f} (키워드: {keyword})")
    return similarity >= threshold

# BART 요약 모델
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# ✅ 문장 단위 분할
def split_text_into_sentences(text):
    import re
    return re.split(r'(?<=[.!?])\s+', text)

# ✅ 문장 토큰 기준 그룹화
def group_sentences_by_token_limit(sentences, tokenizer, max_tokens):
    groups = []
    current_group = ""
    for sentence in sentences:
        tentative_group = current_group + " " + sentence if current_group else sentence
        tokenized = tokenizer.encode(tentative_group, return_tensors="pt")
        if tokenized.size(1) <= max_tokens:
            current_group = tentative_group
        else:
            if current_group:
                groups.append(current_group.strip())
            current_group = sentence
    if current_group:
        groups.append(current_group.strip())
    return groups

# 요약 함수
def summarize_bart(text, keyword):
    inputs = tokenizer([text], return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=1024, num_beams=4, early_stopping=True)
    final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    if keyword and is_relevant:
        if is_relevant(final_summary, keyword):
            final_summary = final_summary.replace('\n', ' ')
        else:
            print(f"⛔️ 유사도 기준 미달 → 요약 제외")
            return None
    else:
        final_summary = final_summary.replace('\n', ' ')

    # 요약본 문장 단위 분할
    sentences = split_text_into_sentences(final_summary)
    
    # 분할된 문장 각각 번역
    translated_sentences = [translate_summaries(sentence) for sentence in sentences]

    # 최종 요약본 생성
    final_summary = " ".join(translated_sentences)

    return final_summary