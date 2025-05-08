import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer, util

# ✅ 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ KoBART 모델 로드
kobart_model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization').to(device)
kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')

# ✅ 임베딩 모델 로드
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

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

# ✅ 요약 함수
def summarize_kobart(text, max_input_length=1024, max_output_length=700):
    inputs = kobart_tokenizer.encode(text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    summary_ids = kobart_model.generate(
        inputs,
        max_length=max_output_length,
        min_length=100,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        length_penalty=1.0,
        early_stopping=False
    )
    return kobart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ 유사도 필터 함수
def is_relevant(summary: str, keyword: str, threshold=0.1) -> bool:
    embeddings = embedder.encode([summary, keyword])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    print(f"    🔍 유사도 점수: {similarity:.4f} (키워드: {keyword})")
    return similarity >= threshold

# ✅ 계층적 요약 함수 (코사인 유사도 포함, 중복 제거 X)
def hierarchical_summary(full_text, keyword=None, max_input_length=1024):
    sentences = split_text_into_sentences(full_text)
    text_chunks = group_sentences_by_token_limit(sentences, kobart_tokenizer, max_input_length)
    chunk_summaries = []

    for i, chunk in enumerate(text_chunks, 1):
        print(f"🧩 부분 {i}/{len(text_chunks)} 요약 중...")
        summary = summarize_kobart(chunk)
        chunk_summaries.append(summary)
        print(f"🧩 부분 {i}/{len(text_chunks)} summary: {summary}")

    if len(chunk_summaries) == 1:
        # ✅ 청크가 하나라면 → 요약 한 번만 반환
        final_summary = chunk_summaries[0]
        print("✅ 청크 1개 → 추가 요약 없이 반환")
    else:
        # ✅ 청크가 여러개 → 합치기
        combined_summary = " ".join(chunk_summaries)
        combined_token_count = len(kobart_tokenizer.encode(combined_summary))
        print(f"🔍 combined summary token count: {combined_token_count}")

        if combined_token_count <= max_input_length:
            print("✅ 최종 요약 입력 길이 가능 → 추가 요약")
            final_summary = summarize_kobart(combined_summary)
        else:
            print("⚠️ combined summary 길이 초과 → 다시 나누기")
            new_sentences = split_text_into_sentences(combined_summary)
            new_chunks = group_sentences_by_token_limit(new_sentences, kobart_tokenizer, max_input_length)

            new_summaries = []
            for i, chunk in enumerate(new_chunks, 1):
                print(f"🔄 재분할 {i}/{len(new_chunks)} 요약 중...")
                summary = summarize_kobart(chunk)
                new_summaries.append(summary)
                print(f"🔄 재분할 {i}/{len(new_chunks)} summary: {summary}")

            final_combined = " ".join(new_summaries)
            final_combined_token_count = len(kobart_tokenizer.encode(final_combined))
            print(f"🔍 재분할 combined summary token count: {final_combined_token_count}")

            if final_combined_token_count <= max_input_length:
                print("✅ 재분할된 combined summary 입력 가능 → 최종 요약")
                final_summary = summarize_kobart(final_combined)
            else:
                print("⚠️ 재분할된 combined summary도 입력 초과 → 더 이상 나누지 않고 그대로 사용")
                final_summary = final_combined
    if keyword and is_relevant:
        if is_relevant(final_summary, keyword):
            return final_summary.replace('\n', ' ')
        else:
            print(f"⛔️ 유사도 기준 미달 → 요약 제외")
            return None
    else:
        return final_summary.replace('\n', ' ')