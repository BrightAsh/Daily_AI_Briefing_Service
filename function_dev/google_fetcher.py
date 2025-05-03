# agents/google_fetcher.py

import requests
from dotenv import load_dotenv
import os

# .env에서 불러오기
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

def fetch_google_search(query, num=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": num
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = []
        for item in data.get('items', []):
            results.append({
                'title': item.get('title'),
                'link': item.get('link'),
                'snippet': item.get('snippet')
            })
        print(f"✅ {len(results)}건 수집 완료.")
        return results
    else:
        print(f"❌ 요청 실패: {response.status_code}, {response.text}")
        return []

if __name__ == "__main__":
    search_results = fetch_google_search("인공지능", num=5)
    for idx, result in enumerate(search_results, 1):
        print(f"\n[{idx}] {result['title']}")
        print(f"- {result['snippet']}")
        print(f"- {result['link']}")
