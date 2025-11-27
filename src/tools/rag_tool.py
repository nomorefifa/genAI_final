"""
RAG Tool - search_documents
수업 자료 PDF에서 관련 내용을 검색하는 Tool
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import json

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from src.rag.utils import embed_texts

# 설정
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

def search_documents(query: str, n_results: int = 5) -> str:
    """
    수업 자료 PDF에서 관련 내용을 검색합니다.
    
    Args:
        query: 검색 질문
        n_results: 반환할 결과 수 (기본 5개)
    
    Returns:
        JSON 문자열 형태의 검색 결과
    """
    try:
        # ChromaDB 연결
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # 쿼리 임베딩
        query_embedding = embed_texts([query])[0]
        
        # 검색
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # 결과 포맷팅
        if not results['documents'] or not results['documents'][0]:
            return json.dumps({
                "success": False,
                "message": "검색 결과가 없습니다.",
                "results": []
            }, ensure_ascii=False)
        
        formatted_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            formatted_results.append({
                "content": doc,
                "source": os.path.basename(metadata.get('source', 'unknown')),
                "chunk_id": metadata.get('chunk_id', 0),
                "similarity": round(1 - distance, 3)  # 거리를 유사도로 변환
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "count": len(formatted_results),
            "results": formatted_results
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": "검색 중 오류가 발생했습니다."
        }, ensure_ascii=False)


# OpenAI Tool Spec 정의
SEARCH_DOCUMENTS_SPEC = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "수업 자료 PDF 파일에서 관련 내용을 검색합니다. LangGraph, ReAct, RAG, Memory, Tool Calling 등 수업 관련 질문에 사용하세요.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 질문 또는 키워드"
                },
                "n_results": {
                    "type": "integer",
                    "description": "반환할 결과 개수 (기본값: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}


# 테스트 코드
if __name__ == "__main__":
    print("="*50)
    print("search_documents Tool 테스트")
    print("="*50)
    
    # 테스트 1
    print("\n[테스트 1] LangGraph State 설계")
    result = search_documents("LangGraph State 설계 방법", n_results=3)
    print(result)
    
    # 테스트 2
    print("\n[테스트 2] ReAct 패턴")
    result = search_documents("ReAct 패턴이란?", n_results=3)
    print(result)