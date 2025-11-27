"""
Memory Tool - read_memory, write_memory
과거 대화 내용을 저장하고 검색하는 Tool
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from src.rag.utils import embed_texts

# 설정
CHROMA_DIR = "chroma_db"
MEMORY_COLLECTION = "memory_collection"

def _get_memory_collection():
    """Memory 컬렉션 가져오기 (없으면 생성)"""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(name=MEMORY_COLLECTION)
    except:
        collection = client.create_collection(name=MEMORY_COLLECTION)
    return collection


def read_memory(query: str, memory_type: str = "all", top_k: int = 5) -> str:
    """
    과거 대화 내용에서 관련 기억을 검색합니다.
    
    Args:
        query: 검색할 내용
        memory_type: 메모리 타입 필터 ("all", "profile", "episodic", "knowledge")
        top_k: 반환할 결과 수
    
    Returns:
        JSON 문자열 형태의 검색 결과
    """
    try:
        collection = _get_memory_collection()
        
        # 쿼리 임베딩
        query_embedding = embed_texts([query])[0]
        
        # 검색 (필터링 옵션)
        where_filter = None if memory_type == "all" else {"memory_type": memory_type}
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        # 결과 포맷팅
        if not results['documents'] or not results['documents'][0]:
            return json.dumps({
                "success": True,
                "message": "저장된 기억이 없습니다.",
                "memories": []
            }, ensure_ascii=False)
        
        memories = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # tags를 문자열에서 리스트로 변환
            tags_str = metadata.get('tags', '')
            tags = tags_str.split(',') if tags_str else []
            
            memories.append({
                "content": doc,
                "memory_type": metadata.get('memory_type', 'unknown'),
                "importance": metadata.get('importance', 0),
                "timestamp": metadata.get('timestamp', ''),
                "tags": tags,
                "similarity": round(1 - distance, 3)
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "count": len(memories),
            "memories": memories
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


def write_memory(
    content: str,
    memory_type: str = "episodic",
    importance: int = 3,
    tags: list = None
) -> str:
    """
    중요한 정보를 장기 메모리에 저장합니다.
    
    Args:
        content: 저장할 내용
        memory_type: "profile" (사용자 정보), "episodic" (대화 내용), "knowledge" (학습 내용)
        importance: 중요도 (1-5, 5가 가장 중요)
        tags: 태그 리스트
    
    Returns:
        JSON 문자열 형태의 저장 결과
    """
    try:
        collection = _get_memory_collection()
        
        if tags is None:
            tags = []
        
        # 임베딩
        embedding = embed_texts([content])[0]
        
        # 고유 ID 생성
        timestamp = datetime.now().isoformat()
        memory_id = f"memory_{timestamp}_{hash(content) % 10000}"
        
        # 메타데이터 (tags를 쉼표로 구분된 문자열로 변환)
        metadata = {
            "memory_type": memory_type,
            "importance": importance,
            "timestamp": timestamp,
            "tags": ','.join(tags)  # 리스트 → 문자열 변환
        }
        
        # 저장
        collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
        
        return json.dumps({
            "success": True,
            "message": "메모리 저장 완료",
            "memory_id": memory_id,
            "memory_type": memory_type,
            "importance": importance,
            "tags": tags
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# OpenAI Tool Spec 정의
READ_MEMORY_SPEC = {
    "type": "function",
    "function": {
        "name": "read_memory",
        "description": "과거 대화 내용이나 저장된 정보를 검색합니다. 사용자가 '지난번에', '이전에' 등 과거를 언급하면 사용하세요.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 내용"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["all", "profile", "episodic", "knowledge"],
                    "description": "메모리 타입 (all: 전체, profile: 사용자정보, episodic: 대화내용, knowledge: 학습내용)",
                    "default": "all"
                },
                "top_k": {
                    "type": "integer",
                    "description": "반환할 결과 수",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

WRITE_MEMORY_SPEC = {
    "type": "function",
    "function": {
        "name": "write_memory",
        "description": "중요한 정보를 장기 메모리에 저장합니다. 사용자 선호도, 프로젝트 정보, 중요한 결정사항 등을 저장할 때 사용하세요.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "저장할 내용"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["profile", "episodic", "knowledge"],
                    "description": "메모리 타입",
                    "default": "episodic"
                },
                "importance": {
                    "type": "integer",
                    "description": "중요도 (1-5)",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 3
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "태그 리스트",
                    "default": []
                }
            },
            "required": ["content"]
        }
    }
}


# 테스트 코드
if __name__ == "__main__":
    print("="*50)
    print("Memory Tool 테스트")
    print("="*50)
    
    # 테스트 1: 메모리 저장
    print("\n[테스트 1] 메모리 저장")
    result = write_memory(
        content="사용자는 LangGraph 프로젝트를 진행 중이며, ReAct 패턴에 관심이 많다.",
        memory_type="profile",
        importance=4,
        tags=["LangGraph", "ReAct", "프로젝트"]
    )
    print(result)
    
    # 테스트 2: 또 다른 메모리 저장
    print("\n[테스트 2] 학습 내용 저장")
    result = write_memory(
        content="HNSW 알고리즘은 계층적 그래프 구조로 빠른 벡터 검색을 가능하게 한다.",
        memory_type="knowledge",
        importance=3,
        tags=["HNSW", "RAG", "벡터검색"]
    )
    print(result)
    
    # 테스트 3: 메모리 검색
    print("\n[테스트 3] 메모리 검색 - LangGraph 관련")
    result = read_memory("LangGraph 프로젝트", memory_type="all", top_k=3)
    print(result)
    
    # 테스트 4: 메모리 검색 - HNSW 관련
    print("\n[테스트 4] 메모리 검색 - 벡터 검색")
    result = read_memory("벡터 검색 알고리즘", memory_type="knowledge", top_k=2)
    print(result)