"""
Google Search Tool - 웹 검색 기능

Google Custom Search API를 사용하여 구글 검색 결과를 가져옵니다.
"""

import os
import requests
import json
from typing import Dict, Any
from dotenv import load_dotenv

# 스크립트 실행 시 .env 파일에서 환경 변수를 로드
load_dotenv()

def google_search(query: str, num_results: int = 5) -> str:
    """
    Google 검색을 수행하여 결과를 반환합니다.
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수 (기본 5개, 최대 10개)
    
    Returns:
        JSON 문자열 형태의 검색 결과
    """
    
    # Google API 키와 Search Engine ID 확인
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")  # CX
    
    if not api_key or not search_engine_id:
        # API 키가 없을 때 Mock 데이터 반환
        mock_results = {
            "success": True,
            "query": query,
            "results": [
                {
                    "title": f"Search result for '{query}' - Example 1",
                    "link": "https://example.com/1",
                    "snippet": "This is a mock search result. Set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env to use real Google Search."
                },
                {
                    "title": f"Search result for '{query}' - Example 2",
                    "link": "https://example.com/2",
                    "snippet": "Mock data is being used. Real search requires Google API key."
                }
            ],
            "note": "⚠️ Mock data - Set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env for real search"
        }
        return json.dumps(mock_results, ensure_ascii=False, indent=2)
    
    # Google Custom Search API 호출
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "num": min(num_results, 10),  # 최대 10개
            "hl": "ko",  # 한국어
            "gl": "kr"   # 한국 지역
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # 결과 파싱
        items = data.get("items", [])
        
        if not items:
            return json.dumps({
                "success": False,
                "message": "검색 결과가 없습니다.",
                "query": query
            }, ensure_ascii=False, indent=2)
        
        # 결과 포맷팅
        formatted_results = []
        for item in items[:num_results]:
            formatted_results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }, ensure_ascii=False, indent=2)
    
    except requests.exceptions.RequestException as e:
        return json.dumps({
            "success": False,
            "error": f"Search API error: {str(e)}",
            "query": query
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "query": query
        }, ensure_ascii=False, indent=2)
    
    # =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("🧪 Google Search Tool 테스트\n")
    
    # 테스트 1: 기본 검색
    print("1️⃣ 기본 검색:")
    result = google_search("LangGraph tutorial", num_results=3)
    print(result)
    
    print("\n" + "="*60)
    
    # 테스트 2: 한글 검색
    print("\n2️⃣ 한글 검색:")
    result = google_search("생성형AI 최신 뉴스", num_results=3)
    print(result)
    
    print("\n✅ 테스트 완료!")
    print("\n💡 Tip: .env 파일에 다음을 추가하세요:")
    print("   GOOGLE_API_KEY=your-google-api-key")
    print("   GOOGLE_SEARCH_ENGINE_ID=your-search-engine-id")
    print("\n   API 키 받기: https://developers.google.com/")
    print("   Search Engine 생성: https://programmablesearchengine.google.com/")