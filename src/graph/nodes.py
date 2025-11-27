"""
Nodes - LangGraph의 Node 함수들

- llm_node: LLM에게 Thought + Action을 결정하도록 요청
- tool_node: Action(Tool 호출)을 실행하고 Observation 반환
"""

import os
import json
from typing import Dict, Any, List
from openai import OpenAI
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.state import AgentState
from src.tools.tool_registry import get_tool_specs, execute_tool

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# =============================================================================
# System Prompt - ReAct 패턴 가이드
# =============================================================================

SYSTEM_PROMPT = """\
You are a helpful AI assistant that uses tools with a ReAct-style loop.

당신은 다음과 같은 Tool들을 사용할 수 있습니다:
- search_documents: 수업 자료 PDF에서 정보 검색 (Function Calling, RAG, LangGraph 등)
- read_memory: 과거 대화 내용에서 기억 검색
- write_memory: 중요한 정보를 장기 기억에 저장
- calculator: 사칙연산 수행
- get_time: 현재 시간 조회
- google_search: Google 검색으로 최신 정보 검색

**ReAct 패턴 가이드:**

1. **Thought (생각)**: 질문을 분석하고 어떤 도구가 필요한지 생각합니다.
2. **Action (행동)**: 필요한 도구를 호출합니다 (tool_calls).
3. **Observation (관찰)**: 도구 실행 결과를 확인합니다.
4. **Final Answer (최종 답변)**: 관찰 결과를 바탕으로 사용자에게 친절하게 답변합니다.

**중요 규칙:**
- 강의 내용에 대한 질문은 반드시 `search_documents` 도구를 사용하세요.
- 최신 뉴스, 실시간 정보는 `google_search` 도구를 사용하세요.
- 계산이 필요하면 `calculator` 도구를 사용하세요 (추측하지 마세요).
- 사용자의 개인정보나 선호사항은 `write_memory`로 저장하세요.
- 도구 결과를 직접 인용할 때는 출처를 명확히 밝히세요.
- 답변은 항상 한국어로 친절하게 작성하세요.

**메모리 저장 가이드:**
다음과 같은 정보는 자동으로 저장해야 합니다:
- 사용자의 이름, 전공, 관심사 등 개인정보 (memory_type: "profile")
- 중요한 대화 내용, 사건, 경험 (memory_type: "episodic")
- 사용자가 학습한 개념, 이해한 내용 (memory_type: "knowledge")
"""


# =============================================================================
# Helper: tool_calls를 딕셔너리로 변환
# =============================================================================

def convert_tool_calls_to_dict(tool_calls) -> List[Dict]:
    """OpenAI tool_calls를 딕셔너리로 변환"""
    if not tool_calls:
        return None
    
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            # 이미 딕셔너리면 그대로
            result.append(tc)
        else:
            # 객체면 딕셔너리로 변환
            result.append({
                "id": getattr(tc, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(tc.function, "name", ""),
                    "arguments": getattr(tc.function, "arguments", "")
                }
            })
    return result


# =============================================================================
# Helper: 메시지를 OpenAI 형식으로 변환
# =============================================================================

def convert_messages_to_openai_format(messages: List) -> List[Dict[str, Any]]:
    """
    LangGraph 메시지를 OpenAI API 형식으로 변환

    Args:
        messages: LangGraph의 메시지 리스트 (딕셔너리 또는 객체)

    Returns:
        OpenAI API 형식의 메시지 리스트
    """
    converted = []

    for idx, msg in enumerate(messages):
        # 이미 딕셔너리면 검증 후 사용
        if isinstance(msg, dict):
            # 딕셔너리가 이미 올바른 OpenAI 형식인지 확인
            # role이 있고, tool인 경우 tool_call_id가 있어야 함
            if msg.get("role") == "tool":
                # tool 메시지는 tool_call_id와 content가 필수
                if "tool_call_id" not in msg or "content" not in msg:
                    print(f"⚠️ Warning: Incomplete tool message at index {idx}: {msg.get('name', 'unknown')}")
                    # 스킵하지 않고 경고만 출력
                # tool 메시지는 무조건 추가 (스킵하면 OpenAI API 에러 발생)
                converted.append(msg)
            else:
                # 다른 메시지는 그대로 추가
                converted.append(msg)

        # 객체면 딕셔너리로 변환
        else:
            # LangGraph 메시지 타입 매핑
            role_map = {
                "human": "user",
                "ai": "assistant",
                "tool": "tool",
                "system": "system"
            }

            msg_type = getattr(msg, "type", "human")
            role = role_map.get(msg_type, "user")

            msg_dict = {
                "role": role,
                "content": getattr(msg, "content", "") or ""
            }

            # tool_calls가 있으면 추가 (OpenAI 형식으로 정규화)
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                openai_tool_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        # LangGraph 형식: {name, args, id, type: "tool_call"}
                        if "name" in tc and "args" in tc:
                            openai_tool_calls.append({
                                "id": tc.get("id", ""),
                                "type": "function",  # ← OpenAI는 "function"만 받음!
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["args"], ensure_ascii=False)  # 다시 JSON 문자열로
                                }
                            })
                        # 이미 OpenAI 형식
                        elif "function" in tc:
                            openai_tool_calls.append(tc)
                    else:
                        # 객체 형태
                        openai_tool_calls.append({
                            "id": getattr(tc, "id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(tc.function, "name", ""),
                                "arguments": getattr(tc.function, "arguments", "")
                            }
                        })

                if openai_tool_calls:
                    msg_dict["tool_calls"] = openai_tool_calls

            # tool 메시지인 경우 tool_call_id 추가
            if role == "tool":
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    msg_dict["tool_call_id"] = msg.tool_call_id

                # name도 추가
                if hasattr(msg, "name") and msg.name:
                    msg_dict["name"] = msg.name

            converted.append(msg_dict)

    return converted


# =============================================================================
# LLM Node - Thought + Action 결정
# =============================================================================

def llm_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM에게 현재 상태를 전달하고 다음 행동을 결정하도록 요청
    
    Returns:
        - messages: LLM의 응답 메시지 (tool_calls 포함 가능)
        - loop_count: 현재 루프 카운트 +1
    """
    messages = state["messages"]
    loop_count = state.get("loop_count", 0)
    
    # 메시지를 OpenAI 형식으로 변환
    openai_messages = convert_messages_to_openai_format(messages)
    
    # OpenAI API 호출
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *openai_messages
        ],
        tools=get_tool_specs(),
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    # 디버깅 출력
    print(f"\n{'='*60}")
    print(f"🤖 LLM Node (루프 {loop_count + 1})")
    print(f"{'='*60}")
    
    if msg.content:
        print(f"💭 Thought: {msg.content[:200]}...")
    
    if msg.tool_calls:
        print(f"🔧 Actions:")
        for tc in msg.tool_calls:
            print(f"   - {tc.function.name}({tc.function.arguments})")
    
    # 메시지 구성
    new_message = {
        "role": "assistant",
        "content": msg.content or "",
    }
    
    if msg.tool_calls:
        # OpenAI tool_calls를 딕셔너리로 변환
        new_message["tool_calls"] = convert_tool_calls_to_dict(msg.tool_calls)
    
    return {
        "messages": [new_message],
        "loop_count": loop_count + 1
    }


# =============================================================================
# Tool Node - Action 실행 + Observation
# =============================================================================

def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM이 요청한 Tool을 실행하고 결과를 반환
    
    Returns:
        - messages: Tool 실행 결과 메시지들
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    print(f"\n{'='*60}")
    print(f"🔨 Tool Node - Executing Actions")
    print(f"{'='*60}")
    
    # last_message를 딕셔너리로 변환
    if not isinstance(last_message, dict):
        role_map = {
            "human": "user",
            "ai": "assistant",
            "tool": "tool",
            "system": "system"
        }
        msg_type = getattr(last_message, "type", "ai")
        role = role_map.get(msg_type, "assistant")
        
        last_message = {
            "role": role,
            "content": getattr(last_message, "content", "") or "",
            "tool_calls": convert_tool_calls_to_dict(getattr(last_message, "tool_calls", None))
        }
    
    # tool_calls가 없으면 에러
    tool_calls = last_message.get("tool_calls")
    if not tool_calls:
        print("⚠️ Warning: tool_node called but no tool_calls found")
        return {"messages": []}
    
    # 각 Tool 실행
    tool_messages = []
    
    for tool_call in tool_calls:
        # LangGraph 형식인지 OpenAI 형식인지 확인
        if "name" in tool_call and "args" in tool_call:
            # LangGraph 형식: {name, args, id, type}
            tool_name = tool_call["name"]
            arguments = tool_call["args"]  # 이미 딕셔너리
            tool_call_id = tool_call["id"]
        elif "function" in tool_call:
            # OpenAI 형식: {id, type, function: {name, arguments}}
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"] or "{}")
            tool_call_id = tool_call["id"]
        else:
            # 알 수 없는 형식 - 객체로 강제 변환 시도
            print(f"⚠️ Warning: Unknown tool_call format, attempting conversion")
            tool_name = getattr(tool_call.function, "name", "") if hasattr(tool_call, "function") else ""
            arguments = json.loads(getattr(tool_call.function, "arguments", "{}") if hasattr(tool_call, "function") else "{}")
            tool_call_id = getattr(tool_call, "id", "")
        
        print(f"\n🔍 Executing: {tool_name}")
        print(f"   Arguments: {arguments}")
        
        try:
            # Tool 실행
            tool_output = execute_tool(tool_name, arguments)
            print(f"   ✅ Success")
            
            # 결과 미리보기 (너무 길면 잘라서 표시)
            preview = tool_output[:200] + "..." if len(tool_output) > 200 else tool_output
            print(f"   📊 Result Preview: {preview}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            tool_output = json.dumps({
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments
            }, ensure_ascii=False)
        
        # Tool 메시지 추가
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_output
        })
    
    return {"messages": tool_messages}


# =============================================================================
# Helper: Should Continue 결정 함수
# =============================================================================

def should_continue(state: AgentState) -> str:
    """
    다음에 어디로 갈지 결정하는 함수
    
    Returns:
        - "tools": Tool 실행이 필요함 (tool_calls가 있음)
        - "end": 최종 답변 완료 (tool_calls가 없음)
    """
    messages = state["messages"]
    last_message = messages[-1]
    loop_count = state.get("loop_count", 0)
    
    # 무한 루프 방지 (최대 10번)
    MAX_LOOPS = 10
    if loop_count >= MAX_LOOPS:
        print(f"\n⚠️ Max loops ({MAX_LOOPS}) reached. Forcing end.")
        return "end"
    
    # last_message를 딕셔너리로 변환
    if not isinstance(last_message, dict):
        role_map = {
            "human": "user",
            "ai": "assistant",
            "tool": "tool",
            "system": "system"
        }
        msg_type = getattr(last_message, "type", "ai")
        role = role_map.get(msg_type, "assistant")
        
        last_message = {
            "role": role,
            "content": getattr(last_message, "content", "") or "",
            "tool_calls": convert_tool_calls_to_dict(getattr(last_message, "tool_calls", None))
        }
    
    # tool_calls가 있으면 Tool Node로
    if last_message.get("tool_calls"):
        return "tools"
    
    # 없으면 종료
    return "end"