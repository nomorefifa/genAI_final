"""
Tool Registry - 모든 Tool을 중앙에서 관리하고 실행

이 모듈은:
1. 모든 Tool Spec을 OpenAI Function Calling 형식으로 제공
2. Tool 이름으로 적절한 핸들러를 실행하는 Dispatcher 제공
3. RAG Tool, Memory Tool, 기본 Tool들을 통합
"""

from typing import Any, Dict, List, Callable
from datetime import datetime
from dateutil import tz
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 기존 Tool 임포트
from src.tools.rag_tool import search_documents
from src.tools.memory_tool import read_memory, write_memory
from src.tools.google_search_tool import google_search


# =============================================================================
# 기본 Tool 구현 (calculator, get_time)
# =============================================================================

def calculator(a: float, op: str, b: float) -> Dict[str, Any]:
    """
    간단한 계산기 Tool
    
    Args:
        a: 첫 번째 피연산자
        op: 연산자 (+, -, *, /)
        b: 두 번째 피연산자
    
    Returns:
        계산 결과 딕셔너리
    """
    if op == '+':
        val = a + b
    elif op == '-':
        val = a - b
    elif op == '*':
        val = a * b
    elif op == '/':
        if b == 0:
            return {"error": "Division by zero"}
        val = a / b
    else:
        return {"error": f"Unsupported operator: {op}"}
    
    return {
        "expression": f"{a} {op} {b}",
        "result": val
    }


def get_time(timezone: str = "Asia/Seoul") -> Dict[str, Any]:
    """
    특정 타임존의 현재 시간을 반환하는 Tool
    
    Args:
        timezone: IANA 타임존 이름 (예: 'Asia/Seoul', 'America/New_York')
    
    Returns:
        현재 시간 정보 딕셔너리
    """
    try:
        target_tz = tz.gettz(timezone)
        if target_tz is None:
            return {"error": f"Unknown timezone: {timezone}"}
        
        now = datetime.now(target_tz)
        return {
            "timezone": timezone,
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A")
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Tool Specifications (OpenAI Function Calling 형식)
# =============================================================================

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "수업 자료 PDF에서 관련 내용을 검색합니다. Function Calling, RAG, LangGraph, ReAct 등 강의 내용에 대한 질문에 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 질문 또는 키워드"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "반환할 결과 수 (기본 5개)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "과거 대화 내용에서 관련 기억을 검색합니다. 사용자의 이전 발언, 선호사항, 과거 대화 내용을 찾을 때 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 기억 내용"
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "메모리 타입 ('all', 'profile', 'episodic', 'knowledge')",
                        "enum": ["all", "profile", "episodic", "knowledge"],
                        "default": "all"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환할 결과 수 (기본 5개)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": "중요한 정보를 장기 기억에 저장합니다. 사용자의 개인정보, 선호사항, 중요한 대화 내용을 기록할 때 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "저장할 내용"
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "메모리 타입 ('profile': 개인정보, 'episodic': 대화/사건, 'knowledge': 학습한 지식)",
                        "enum": ["profile", "episodic", "knowledge"],
                        "default": "episodic"
                    },
                    "importance": {
                        "type": "string",
                        "description": "중요도 ('low', 'medium', 'high')",
                        "enum": ["low", "medium", "high"],
                        "default": "medium"
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "간단한 사칙연산을 수행합니다. 덧셈, 뺄셈, 곱셈, 나눗셈을 지원합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "첫 번째 숫자"
                    },
                    "op": {
                        "type": "string",
                        "description": "연산자",
                        "enum": ["+", "-", "*", "/"]
                    },
                    "b": {
                        "type": "number",
                        "description": "두 번째 숫자"
                    }
                },
                "required": ["a", "op", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "특정 타임존의 현재 시간을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA 타임존 이름 (예: 'Asia/Seoul', 'America/New_York', 'Europe/London')",
                        "default": "Asia/Seoul"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Google 검색을 통해 최신 정보를 검색합니다. 실시간 뉴스, 최신 기술 동향, 현재 사건 등을 찾을 때 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "반환할 결과 수 (기본 5개)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# =============================================================================
# Tool Dispatcher - Tool 이름으로 실제 함수 호출
# =============================================================================

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Tool 이름과 인자를 받아서 적절한 Tool을 실행하고 결과를 반환
    
    Args:
        tool_name: 실행할 Tool 이름
        arguments: Tool에 전달할 인자 (딕셔너리)
    
    Returns:
        Tool 실행 결과 (JSON 문자열)
    """
    import json
    
    try:
        if tool_name == "search_documents":
            result = search_documents(**arguments)
            return result  # 이미 JSON 문자열
        
        elif tool_name == "read_memory":
            result = read_memory(**arguments)
            return result  # 이미 JSON 문자열
        
        elif tool_name == "write_memory":
            result = write_memory(**arguments)
            return result  # 이미 JSON 문자열
        
        elif tool_name == "calculator":
            result = calculator(**arguments)
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        elif tool_name == "get_time":
            result = get_time(**arguments)
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        elif tool_name == "google_search":
            result = google_search(**arguments)
            return result  # 이미 JSON 문자열
        
        else:
            error_result = {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": [spec["function"]["name"] for spec in TOOL_SPECS]
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        error_result = {
            "error": f"Tool execution failed: {str(e)}",
            "tool_name": tool_name,
            "arguments": arguments
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


# =============================================================================
# Helper Functions
# =============================================================================

def get_tool_specs() -> List[Dict[str, Any]]:
    """
    모든 Tool Spec을 OpenAI Function Calling 형식으로 반환
    
    Returns:
        Tool Spec 리스트
    """
    return TOOL_SPECS


def get_tool_names() -> List[str]:
    """
    사용 가능한 모든 Tool 이름 리스트 반환
    
    Returns:
        Tool 이름 리스트
    """
    return [spec["function"]["name"] for spec in TOOL_SPECS]


def print_available_tools():
    """사용 가능한 Tool 목록을 출력"""
    print("\n📦 사용 가능한 Tool 목록:")
    print("=" * 60)
    for spec in TOOL_SPECS:
        func = spec["function"]
        print(f"\n🔧 {func['name']}")
        print(f"   {func['description']}")
    print("=" * 60)


# =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("🧪 Tool Registry 테스트\n")
    
    # 1. 사용 가능한 Tool 목록 출력
    print_available_tools()
    
    # 2. 각 Tool 테스트
    print("\n\n🔬 Tool 실행 테스트:")
    print("=" * 60)
    
    # Calculator 테스트
    print("\n1️⃣ Calculator:")
    result = execute_tool("calculator", {"a": 10, "op": "+", "b": 5})
    print(result)
    
    # Get Time 테스트
    print("\n2️⃣ Get Time:")
    result = execute_tool("get_time", {"timezone": "Asia/Seoul"})
    print(result)
    
    # Google Search 테스트
    print("\n3️⃣ Google Search:")
    result = execute_tool("google_search", {
        "query": "LangGraph tutorial",
        "num_results": 3
    })
    print(result[:500] + "..." if len(result) > 500 else result)
    
    # Search Documents 테스트 (RAG DB가 있다면)
    print("\n4️⃣ Search Documents:")
    result = execute_tool("search_documents", {
        "query": "ReAct 패턴이 뭐야?",
        "n_results": 3
    })
    print(result[:500] + "..." if len(result) > 500 else result)
    
    print("\n" + "=" * 60)
    print("✅ Tool Registry 테스트 완료!")