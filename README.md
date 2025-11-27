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
project_root/
├── data/                # PDF 원본
├── src/
│   ├── rag/            # RAG 시스템
│   ├── tools/          # Tool 구현
│   ├── memory/         # Memory 시스템
│   ├── graph/          # LangGraph Agent
│   └── ui/             # Gradio UI
└── scripts/            # 유틸리티 스크립트
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
```

### 4. 환경 테스트
```bash
python test_setup.py
```

## 요구사항
- Python 3.8+
- OpenAI API Key

## 라이센스
MIT License