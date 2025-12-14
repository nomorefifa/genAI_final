"""
Tool Definitions - Pydantic 기반 Tool Input Models 및 ToolSpec 정의

이 모듈은:
1. 각 Tool의 Input을 Pydantic BaseModel로 정의하여 자동 검증
2. ToolSpec을 정의하여 Tool의 메타데이터와 핸들러를 관리
3. JSON Schema를 자동 생성하여 OpenAI Function Calling 형식으로 변환
"""

from __future__ import annotations
from typing import Any, Dict, Callable
from pydantic import BaseModel, Field
from datetime import datetime
from dateutil import tz
import json
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 기존 Tool 임포트
from src.tools.rag_tool import search_documents as _search_documents
from src.tools.memory_tool import read_memory as _read_memory, write_memory as _write_memory
from src.tools.google_search_tool import google_search as _google_search


# =============================================================================
# Tool Input Models (Pydantic BaseModel)
# =============================================================================

class SearchDocumentsInput(BaseModel):
    """수업 자료 검색 Tool의 입력 모델"""
    query: str = Field(..., description="검색 질문 또는 키워드")
    n_results: int = Field(default=5, description="반환할 결과 수")


class ReadMemoryInput(BaseModel):
    """과거 대화 내용 검색 Tool의 입력 모델"""
    query: str = Field(..., description="검색할 기억 내용")
    memory_type: str = Field(
        default="all",
        description="메모리 타입 ('all', 'profile', 'episodic', 'knowledge')",
        pattern="^(all|profile|episodic|knowledge)$"
    )
    top_k: int = Field(default=5, description="반환할 결과 수")


class WriteMemoryInput(BaseModel):
    """장기 기억 저장 Tool의 입력 모델"""
    content: str = Field(..., description="저장할 내용")
    memory_type: str = Field(
        default="episodic",
        description="메모리 타입 ('profile': 개인정보, 'episodic': 대화/사건, 'knowledge': 학습한 지식)",
        pattern="^(profile|episodic|knowledge)$"
    )
    importance: int = Field(
        default=3,
        description="중요도 (1: 낮음, 2: 약간 중요, 3: 보통, 4: 중요, 5: 매우 중요)",
        ge=1,
        le=5
    )


class CalculatorInput(BaseModel):
    """계산기 Tool의 입력 모델"""
    a: float = Field(..., description="첫 번째 숫자")
    op: str = Field(..., description="연산자 (+, -, *, /)", pattern=r"^[+\-*/]$")
    b: float = Field(..., description="두 번째 숫자")


class GetTimeInput(BaseModel):
    """시간 조회 Tool의 입력 모델"""
    timezone: str = Field(
        default="Asia/Seoul",
        description="IANA 타임존 이름 (예: 'Asia/Seoul', 'America/New_York', 'Europe/London')"
    )


class GoogleSearchInput(BaseModel):
    """Google 검색 Tool의 입력 모델"""
    query: str = Field(..., description="검색 쿼리")
    num_results: int = Field(default=5, description="반환할 결과 수 (기본 5개)")


# =============================================================================
# Tool Handler Functions
# =============================================================================

def search_documents(input: SearchDocumentsInput) -> str:
    """
    수업 자료 PDF에서 관련 내용을 검색

    Args:
        input: SearchDocumentsInput 모델

    Returns:
        JSON 문자열 형태의 검색 결과
    """
    return _search_documents(query=input.query, n_results=input.n_results)


def read_memory(input: ReadMemoryInput) -> str:
    """
    과거 대화 내용에서 관련 기억을 검색

    Args:
        input: ReadMemoryInput 모델

    Returns:
        JSON 문자열 형태의 검색 결과
    """
    return _read_memory(
        query=input.query,
        memory_type=input.memory_type,
        top_k=input.top_k
    )


def write_memory(input: WriteMemoryInput) -> str:
    """
    중요한 정보를 장기 기억에 저장

    Args:
        input: WriteMemoryInput 모델

    Returns:
        JSON 문자열 형태의 저장 결과
    """
    return _write_memory(
        content=input.content,
        memory_type=input.memory_type,
        importance=input.importance,
        tags=[]  # 기본 빈 리스트
    )


def calculator(input: CalculatorInput) -> Dict[str, Any]:
    """
    간단한 사칙연산을 수행

    Args:
        input: CalculatorInput 모델

    Returns:
        계산 결과 딕셔너리
    """
    if input.op == '+':
        val = input.a + input.b
    elif input.op == '-':
        val = input.a - input.b
    elif input.op == '*':
        val = input.a * input.b
    elif input.op == '/':
        if input.b == 0:
            raise RuntimeError("Division by zero")
        val = input.a / input.b
    else:
        raise RuntimeError(f"Unsupported operator: {input.op}")

    return {
        "expression": f"{input.a} {input.op} {input.b}",
        "result": val
    }


def get_time(input: GetTimeInput) -> Dict[str, Any]:
    """
    특정 타임존의 현재 시간을 반환

    Args:
        input: GetTimeInput 모델

    Returns:
        현재 시간 정보 딕셔너리
    """
    try:
        target_tz = tz.gettz(input.timezone)
        if target_tz is None:
            raise ValueError(f"Unknown timezone: {input.timezone}")

        now = datetime.now(target_tz)
        return {
            "timezone": input.timezone,
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A")
        }
    except Exception as e:
        raise RuntimeError(str(e))


def google_search(input: GoogleSearchInput) -> str:
    """
    Google 검색을 통해 최신 정보를 검색

    Args:
        input: GoogleSearchInput 모델

    Returns:
        JSON 문자열 형태의 검색 결과
    """
    return _google_search(query=input.query, num_results=input.num_results)


# =============================================================================
# ToolSpec 클래스 정의 (강의 자료와 동일)
# =============================================================================

class ToolSpec(BaseModel):
    """
    Tool의 메타데이터와 핸들러를 관리하는 클래스

    Attributes:
        name: Tool 이름
        description: Tool 설명
        input_model: Pydantic Input Model (자동 검증 및 JSON Schema 생성)
        handler: Tool 실행 함수
    """
    name: str
    description: str
    input_model: Any
    handler: Callable[[Any], Dict[str, Any] | str]

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Helper: ToolSpec → OpenAI Tools 변환 (강의 자료와 동일)
# =============================================================================

def as_openai_tool_spec(spec: ToolSpec) -> Dict[str, Any]:
    """
    ToolSpec을 OpenAI Function Calling 형식으로 변환

    Args:
        spec: ToolSpec 객체

    Returns:
        OpenAI tools[] 형식의 딕셔너리
    """
    # Pydantic 모델의 JSON Schema 자동 생성
    schema = spec.input_model.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": schema,
        },
    }


# =============================================================================
# Default Tool Specs 정의 (강의 자료와 동일한 패턴)
# =============================================================================

def get_default_tool_specs() -> list[ToolSpec]:
    """
    모든 Tool의 ToolSpec을 반환

    Returns:
        ToolSpec 리스트
    """
    return [
        ToolSpec(
            name="search_documents",
            description="수업 자료 PDF에서 관련 내용을 검색합니다. Function Calling, RAG, LangGraph, ReAct 등 강의 내용에 대한 질문에 사용하세요.",
            input_model=SearchDocumentsInput,
            handler=lambda args: search_documents(SearchDocumentsInput(**args)),
        ),
        ToolSpec(
            name="read_memory",
            description="과거 대화 내용에서 관련 기억을 검색합니다. 사용자의 이전 발언, 선호사항, 과거 대화 내용을 찾을 때 사용하세요.",
            input_model=ReadMemoryInput,
            handler=lambda args: read_memory(ReadMemoryInput(**args)),
        ),
        ToolSpec(
            name="write_memory",
            description="중요한 정보를 장기 기억에 저장합니다. 사용자의 개인정보, 선호사항, 중요한 대화 내용을 기록할 때 사용하세요.",
            input_model=WriteMemoryInput,
            handler=lambda args: write_memory(WriteMemoryInput(**args)),
        ),
        ToolSpec(
            name="calculator",
            description="간단한 사칙연산을 수행합니다. 덧셈, 뺄셈, 곱셈, 나눗셈을 지원합니다.",
            input_model=CalculatorInput,
            handler=lambda args: calculator(CalculatorInput(**args)),
        ),
        ToolSpec(
            name="get_time",
            description="특정 타임존의 현재 시간을 조회합니다.",
            input_model=GetTimeInput,
            handler=lambda args: get_time(GetTimeInput(**args)),
        ),
        ToolSpec(
            name="google_search",
            description="Google 검색을 통해 최신 정보를 검색합니다. 실시간 뉴스, 최신 기술 동향, 현재 사건 등을 찾을 때 사용하세요.",
            input_model=GoogleSearchInput,
            handler=lambda args: google_search(GoogleSearchInput(**args)),
        ),
    ]

