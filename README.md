# 생성형AI응용 기말 프로젝트 - ReAct Agent

LangGraph + RAG + Memory를 활용한 수업 자료 검색 Agent

## 프로젝트 목표
생성형AI응용 수업의 이해를 돕는 AI Agent 구현
- 수업 자료 PDF를 RAG로 검색
- Memory로 학습 진도와 개념 관리
- ReAct 패턴으로 Reasoning + Acting 구현

## 기술 스택
- **LLM**: OpenAI GPT-4o-mini
- **Framework**: LangGraph
- **Vector DB**: ChromaDB
- **Tools**: RAG(검색), Memory(기억), Google Search
- **UI**: Gradio

## 프로젝트 구조
```
finalPJ/
├── src/
│   ├── graph/              # LangGraph Agent 엔진
│   │   ├── state.py        # AgentState 정의
│   │   ├── nodes.py        # LLM, Tool 노드
│   │   └── agent.py        # StateGraph 생성
│   │
│   ├── tools/              # Tool Layer
│   │   ├── tool_definitions.py  # Pydantic Input Models + ToolSpec
│   │   ├── tool_registry.py     # ToolRegistry 클래스
│   │   ├── rag_tool.py          # RAG 검색 (with Reranking)
│   │   ├── memory_tool.py       # read_memory, write_memory
│   │   └── google_search_tool.py
│   │
│   ├── memory/             # Memory Layer
│   │   └── reflection.py   # Memory Extractor (자동 저장)
│   │
│   ├── rag/                # RAG Layer
│   │   └── utils.py        # 임베딩 유틸리티
│   │
│   └── ui/                 # UI Layer
│       └── gradio_app.py   # Gradio 채팅 인터페이스
│
├── chroma_db/              # ChromaDB Persistent Storage
│   ├── documents/          # RAG 문서 컬렉션
│   └── memory_collection/  # Memory 컬렉션
│
├── test/                   # ChromaDB Persistent Storage
    ├── test_integration.py # 통합 테스트
    └── test_reranking.py   # Reranking 성능 테스트
```

## 설치 방법

### 1. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정
`.env` 파일 생성:
```
OPENAI_API_KEY=your-api-key-here
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
GOOGLE_API_KEY = your-google-search-api
GOOGLE_SEARCH_ENGINE_ID = your-google-search-engine-id
```

## 요구사항
- Python 3.8+
- OpenAI API Key

## 라이센스
MIT License
