import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from src.rag.utils import embed_texts, build_prompt, chat_with_openai

# 설정
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

def search_documents(query: str, n_results: int = 5):
    """ChromaDB에서 문서 검색"""
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
    
    return results

def main():
    print("\n" + "="*50)
    print("RAG 검색 테스트")
    print("="*50 + "\n")
    
    # 테스트 쿼리들
    test_queries = [
        "LangGraph State 설계 방법",
        "ReAct 패턴이란?",
        "HNSW 알고리즘 설명",
        "Memory 시스템 구성요소"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[테스트 {i}] 질문: {query}")
        print("-" * 50)
        
        # 검색
        results = search_documents(query, n_results=3)
        
        # 결과 출력
        if results['documents'] and results['documents'][0]:
            print(f"✅ 검색 성공! {len(results['documents'][0])}개 결과 발견\n")
            
            for j, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                source = os.path.basename(metadata.get('source', 'unknown'))
                chunk_id = metadata.get('chunk_id', '?')
                print(f"  [{j}] {source} (청크 {chunk_id}) - 유사도: {1-distance:.3f}")
                print(f"      내용 미리보기: {doc[:100]}...")
                print()
        else:
            print("❌ 검색 결과 없음")
    
    print("\n" + "="*50)
    print("검색 테스트 완료!")
    print("="*50)
    
    # 대화형 검색 모드
    print("\n💬 대화형 검색 모드 (종료: 'quit' 또는 'exit')")
    print("-" * 50)
    
    while True:
        user_query = input("\n질문을 입력하세요: ").strip()
        
        if user_query.lower() in ['quit', 'exit', '종료']:
            print("👋 검색을 종료합니다.")
            break
        
        if not user_query:
            continue
        
        # 검색
        results = search_documents(user_query, n_results=5)
        
        if results['documents'] and results['documents'][0]:
            # 컨텍스트 구성
            contexts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                contexts.append({
                    'text': doc,
                    'source': metadata.get('source', 'unknown'),
                    'chunk_id': metadata.get('chunk_id', 0)
                })
            
            # LLM에게 질문
            messages = build_prompt(user_query, contexts)
            answer = chat_with_openai(messages)
            
            print("\n🤖 답변:")
            print("-" * 50)
            print(answer)
        else:
            print("❌ 관련 문서를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()