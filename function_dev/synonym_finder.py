import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def find_synonyms(keyword, n=5, country='Korea'):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3,
    )

    # Step 1️⃣ 동의어 후보군을 충분히 받는다 (ex: 최대 20개 요청)
    gen_prompt = PromptTemplate(
        input_variables=["keyword", "country"],
        template="""
    Please provide a comprehensive list of at least 20 synonyms for the keyword "{keyword}".

    Important instructions:
    - The synonyms must be commonly used in {country}.
    - Use both the primary local language of {country} and English if English terms are also widely used in {country}.
    - You should automatically detect the local language based on {country}.
    - Provide **strict synonyms only** (no related terms, no broader/narrower concepts).
    - Do not include the keyword itself.
    - No synonym should be a substring of another synonym.
    - List only synonyms, separated by commas.

    Example:
    If the country is Korea and the keyword is "인공지능", include synonyms like "AI", "Artificial Intelligence" along with local synonyms like "인공신경망", "기계지능".

    Keyword: {keyword}
    """
    )

    chain = gen_prompt | llm
    response = chain.invoke({
        "keyword": keyword,
        "country": country
    })
    response_text = response.content.strip()
    print(f"🛠️ Raw synonyms response: {response_text}")

    # 후보 리스트 정리
    candidates = [word.strip() for word in response_text.split(",") if word.strip()]
    print(f"📋 Initial candidates: {candidates}")

    # Step 2️⃣ 포함관계 제거 + 중복 제거
    filtered = []
    for word in candidates:
        if keyword in word:  # 자기 자신 제거
            continue
        if not any(word in other and word != other for other in candidates):
            if word not in filtered:
                filtered.append(word)
    print(f"✅ After filtering substrings & duplicates: {filtered}")

    # Step 3️⃣ 검수: 잘못된 항목 제거
    review_prompt = PromptTemplate(
        input_variables=["final_list", "keyword", "country"],
        template="""
        Review the following list of synonyms for the keyword "{keyword}" used in {country}:

        {final_list}

        Please return a **clean list** of only the correct synonyms (no related/broader/narrower terms),
        with no substring overlaps, and that are commonly used in {country}. List them separated by commas.
        """
    )

    review_chain = review_prompt | llm
    review_response = review_chain.invoke({
        "final_list": ', '.join(filtered),
        "keyword": keyword,
        "country": country
    })
    review_cleaned = [word.strip() for word in review_response.content.strip().split(",") if word.strip()]
    print(f"🔍 After review: {review_cleaned}")

    # Step 4️⃣ 빈도 기반 정렬 요청
    sort_prompt = PromptTemplate(
        input_variables=["word_list", "country"],
        template="""
        Rank the following list of synonyms by how frequently they are used in {country}, from most common to least common.
        Return only the sorted list, separated by commas.

        Synonyms: {word_list}
        """
    )

    sort_chain = sort_prompt | llm
    sort_response = sort_chain.invoke({
        "word_list": ', '.join(review_cleaned),
        "country": country
    })
    sorted_final = [word.strip() for word in sort_response.content.strip().split(",") if word.strip()]
    print(f"🏆 Final sorted list: {sorted_final}")

    # 최종 n개만 반환
    result = [keyword] + sorted_final
    result = result[:n]

    if len(result) < n:
        print(f"⚠️ Only found {len(result)} valid synonyms (less than requested {n}).")
    return result


# Example usage:
if __name__ == "__main__":
    result = find_synonyms("인공지능", n=5)
    print(f"🔍 Final synonym list: {result}")

    # ['인공지능', 'AI', '인공신경망', '머신러닝', '딥러닝']
