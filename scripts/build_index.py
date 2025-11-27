import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import PyPDF2
from src.rag.utils import chunk_document, embed_texts

# 설정
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

def load_pdf(filepath: str) -> str:
    """PDF 파일에서 텍스트 추출"""
    text = ""
    with open(filepath, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def main():
    print("=" * 50)
    print("벡터 DB 구축 시작")
    print("=" * 50)
    
    # ChromaDB PersistentClient 초기화
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # 기존 컬렉션이 있으면 삭제하고 새로 생성
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제 완료")
    except:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"새 컬렉션 '{COLLECTION_NAME}' 생성 완료\n")
    
    # data 폴더에서 모든 PDF 파일 찾기
    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ {DATA_DIR} 폴더에 PDF 파일이 없습니다.")
        return
    
    print(f"📁 {len(pdf_files)}개의 PDF 파일 발견\n")
    
    # 각 PDF 처리
    all_ids = []
    all_texts = []
    all_metadatas = []
    
    for pdf_path in pdf_files:
        print(f"처리 중: {pdf_path.name}")
        
        # 1. PDF 텍스트 추출
        doc_text = load_pdf(str(pdf_path))
        print(f"  - 텍스트 추출 완료 ({len(doc_text)} 글자)")
        
        # 2. 청킹
        chunks = chunk_document(doc_text, str(pdf_path))
        print(f"  - 청킹 완료 ({len(chunks)}개 청크)")
        
        # 3. 청크 정보 수집
        for chunk in chunks:
            all_ids.append(chunk.id)
            all_texts.append(chunk.text)
            all_metadatas.append(chunk.metadata)
        
        print(f"  ✅ {pdf_path.name} 처리 완료\n")
    
    # 4. 일괄 임베딩
    print(f"🔄 전체 {len(all_texts)}개 청크 임베딩 중...")
    embeddings = embed_texts(all_texts)
    print(f"  ✅ 임베딩 완료\n")
    
    # 5. ChromaDB에 저장
    print("💾 ChromaDB에 저장 중...")
    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas
    )
    print(f"  ✅ 저장 완료\n")
    
    # 통계 출력
    print("=" * 50)
    print("벡터 DB 구축 완료!")
    print("=" * 50)
    print(f"📊 통계:")
    print(f"  - 처리된 PDF: {len(pdf_files)}개")
    print(f"  - 생성된 청크: {len(all_ids)}개")
    print(f"  - 저장 위치: {CHROMA_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()