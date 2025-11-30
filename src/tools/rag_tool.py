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
from sentence_transformers import CrossEncoder

# 설정
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

# Reranker 초기화 (Cross-Encoder for MS MARCO)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def search_documents(query: str, n_results: int = 5) -> str:
    """
    수업 자료 PDF에서 관련 내용을 검색합니다.
    Cross-Encoder 기반 Reranking을 사용하여 검색 품질을 향상시킵니다.

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

        # 1. 초기 검색 (Reranking을 위해 더 많은 결과 가져오기)
        initial_n = min(n_results * 2, 20)  # 최대 20개
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_n
        )

        # 결과 포맷팅
        if not results['documents'] or not results['documents'][0]:
            return json.dumps({
                "success": False,
                "message": "검색 결과가 없습니다.",
                "results": []
            }, ensure_ascii=False)

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # 2. Reranking (Cross-Encoder)
        # 쿼리와 각 문서를 쌍으로 만들어 점수 계산
        pairs = [[query, doc] for doc in documents]
        rerank_scores = reranker.predict(pairs)

        # 3. Reranking 점수로 정렬 후 상위 n_results개만 선택
        ranked = sorted(
            zip(documents, metadatas, distances, rerank_scores),
            key=lambda x: x[3],  # rerank_score 기준
            reverse=True
        )[:n_results]

        # 4. 결과 포맷팅
        formatted_results = []
        for i, (doc, metadata, distance, rerank_score) in enumerate(ranked):
            formatted_results.append({
                "rank": i + 1,
                "content": doc,
                "source": os.path.basename(metadata.get('source', 'unknown')),
                "chunk_id": metadata.get('chunk_id', 0),
                "embedding_similarity": round(1 - distance, 3),  # 임베딩 유사도
                "rerank_score": round(float(rerank_score), 4)  # Reranking 점수
            })

        return json.dumps({
            "success": True,
            "query": query,
            "count": len(formatted_results),
            "results": formatted_results,
            "reranked": True
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