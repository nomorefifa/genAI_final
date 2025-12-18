# 생성형AI응용 기말 프로젝트 - ReAct Agent

OpenAI GPT-4o-mini와 LangGraph를 활용하여 생성형 AI 수업의 개념을 대화형으로 학습할 수 있도록 지원하는 에이전트입니다. 19개의 강의 자료 PDF(Tokenization, Transformer, RAG, Function Calling, Memory 등)를 ChromaDB 벡터 데이터베이스에 인덱싱하여 RAG 기반 문서 검색을 제공하며, Cross-Encoder 리랭킹을 통해 검색 정확도를 향상시킵니다.

핵심 기능으로는 강의 자료 검색, 대화 기억 관리(profile/episodic/knowledge 메모리) 구글 검색 등의 도구를 제공하며, 대화 중 중요한 정보를 자동으로 추출하여 저장하는 기능을 갖추고 있습니다. ReAct 패턴을 통해 에이전트가 질문을 받으면 단계별로 추론하고 필요한 도구를 선택하여 실행한 후 답변을 생성하는 방식으로 작동합니다.

FastAPI 서버와 Gradio 웹 인터페이스를 통해 사용자 친화적인 채팅 환경을 제공하며, 세션 체크포인팅으로 대화 컨텍스트를 지속적으로 유지합니다. 이 시스템은 단순한 질의응답을 넘어 진행한 대화간 선호도를 기억하면서 점진적으로 맥락을 축적하는 개인화된 AI 학습 도우미로 설계하였습니다.

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
genAI_final/
├── chroma_db/              # ChromaDB Persistent Storage
│
├── data/                   # Lab-01-spm-kenlm.pdf 등 강의 pdf 자료
│
├── scripts/
│   └── build_index.py
│   └── query.py
│
└── src/
    ├── graph/              # LangGraph Agent 엔진
    │   ├── state.py        # AgentState 정의
    │   ├── nodes.py        # LLM, Tool 노드
    │   └── agent.py        # StateGraph 생성
    │
    ├── tools/                   # Tool Layer
    │   ├── tool_definitions.py  # Pydantic Input Models + ToolSpec
    │   ├── tool_registry.py     # ToolRegistry 클래스
    │   ├── rag_tool.py          # RAG 검색 (with Reranking)
    │   ├── memory_tool.py       # read_memory, write_memory
    │   └── google_search_tool.py
    │
    ├── memory/             # Memory Layer
    │   └── reflection.py   # Memory Extractor (자동 저장)
    │
    ├── rag/                # RAG Layer
    │   └── utils.py        # 임베딩 유틸리티
    │
    └── ui/                 # UI Layer
        └── gradio_app.py   # Gradio 채팅 인터페이스
        └── server.py       # FastAPI 서버 정의

(chroma_db의 하위 파일, __pycache__ 폴더, __init__.py는 생략함)
```

**전체 시스템 흐름 (ReAct Loop):**
```
사용자 입력
↓
[START] → llm_node (Thought + Action 결정)
↓
should_continue (도구 필요 여부 판단) → (tool_calls 없음) → [END] (최종 답변)
↓ (tool_calls 있음)          
tool_node (도구 실행)              
↓
llm_node (결과 반영 후 재사고)
↓
should_continue
↓
(반복 최대 10회)
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
`.env` 파일 생성: (팀원들과 공유한 key가 있지만 README.md에는 작성하지 않았습니다.)
```
OPENAI_API_KEY=your-api-key-here
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
GOOGLE_API_KEY = your-google-search-api
GOOGLE_SEARCH_ENGINE_ID = your-google-search-engine-id
```

### 4. Gradio 실행
```
가상환경 터미널에:
uvicorn src.ui.server:app --host 0.0.0.0 --port 7860 --reload

브라우저 주소창에:
http://localhost:7860/
```

## 요구사항
- Python 3.8+
- OpenAI API Key

## 라이센스
MIT License
