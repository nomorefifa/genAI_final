"""
Nodes - LangGraph의 Node 함수들

- llm_node: LLM에게 Thought + Action 결정 요청
- tool_node: LLM이 선택한 도구 실행 및 결과 반환
- should_continue: 다음 흐름(도구 실행 vs 종료) 판단
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
from src.tools.tool_registry import get_tool_specs, execute_tool, register_default_tools

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# ToolRegistry 싱글톤
_tool_registry = None

def get_tool_registry():
    """ToolRegistry 싱글톤 인스턴스 반환"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = register_default_tools()
    return _tool_registry

# =============================================================================
# System Prompt - ReAct 패턴 인스트럭션
# =============================================================================

SYSTEM_PROMPT = """\
You are an AI assistant that uses tools (functions), RAG, and memory.

# High-level behavior
- Be helpful, honest, and concise.
- Answer primarily in Korean unless the user clearly wants another language.
- Think step by step internally, but do NOT expose chain-of-thought.
- When tools are available and helpful, call them instead of guessing.

# Tools and ReAct-style behavior
- You may call tools such as:
  - read_memory: to recall important past information about the user or past sessions.
  - write_memory: to store new, useful information about the user or this conversation.
  - search_documents: to search course materials (RAG with Reranking for LangGraph, ReAct, Function Calling, etc.).
  - google_search: to search the web for latest information.
  - calculator: for arithmetic operations.
  - get_time: to check current time in a specific timezone.

- Use tools when:
  - You lack required factual details.
  - You need to recall prior user preferences, past discussions, or long-term context.
  - You need domain knowledge stored in a vector database or document store.
- After receiving a tool result, incorporate it into your reasoning and produce a final answer.

# Memory usage guidelines
- Memory is not magic; you must explicitly call `read_memory` or `write_memory` to use it.
- Call `read_memory` when:
  - The user refers to "지난 번", "이전에 말했듯이", "저번에 만들던 코드" 등 과거 내용.
  - The answer clearly depends on the user's preferences, profile, or long-term history.
- Call `write_memory` when:
  - The user shares stable personal preferences (e.g., 좋아하는 스타일, 선호 옵션).
  - The user states long-term goals, ongoing projects, or recurring topics.
  - The user corrects you or provides important facts that will be useful later.
- Do NOT write memory for:
  - Short-lived, one-off facts (예: 오늘 점심 메뉴).
  - Extremely detailed logs that are unlikely to be reused.
  - Sensitive personal data, unless the user explicitly requests you to remember it.

# RAG usage guidelines
- Call search_documents when:
  - The user asks for factual information from course materials.
  - You need detailed or authoritative content about LangGraph, ReAct, RAG, Memory, Function Calling, etc.
- When you get retrieved documents, read them and synthesize a clear, concise answer.

# Answer style
- Default: Korean, 친절하지만 군더더기 없이.
- Provide structure (번호, 소제목) for teaching/explaining technical concepts.
- If the user is building a system or code, show step-by-step reasoning in high level,
  but do NOT output low-level hidden chain-of-thought or internal scratch work.

# Safety
- If a user asks you to perform unsafe, illegal, or harmful actions, politely refuse.
- If you're unsure, say so and explain what additional information would be needed.
"""


# =============================================================================
# Helper: tool_calls를 딕셔너리로 변환
# =============================================================================

def convert_tool_calls_to_dict(tool_calls) -> List[Dict]:
    """OpenAI ToolCall 객체를 딕셔너리로 변환"""
    if not tool_calls:
        return None
    
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            result.append(tc)
        else:
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
        messages: LangGraph 메시지 리스트 (딕셔너리 또는 객체)
    
    Returns:
        OpenAI API 호환 메시지 리스트
    """
    converted = []

    for idx, msg in enumerate(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "tool":
                if "tool_call_id" not in msg or "content" not in msg:
                    print(f"⚠️ 경고: 불완전한 tool 메시지 (인덱스 {idx}): {msg.get('name', 'unknown')}")
                converted.append(msg)
            else:
                converted.append(msg)
        else:
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

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                openai_tool_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        if "name" in tc and "args" in tc:
                            openai_tool_calls.append({
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["args"], ensure_ascii=False)
                                }
                            })
                        elif "function" in tc:
                            openai_tool_calls.append(tc)
                    else:
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

            if role == "tool":
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if hasattr(msg, "name") and msg.name:
                    msg_dict["name"] = msg.name

            converted.append(msg_dict)

    return converted


# =============================================================================
# Memory Read 파이프라인 - 자동 메모리 검색
# =============================================================================

def execute_memory_read_pipeline(openai_messages: List[Dict[str, Any]]) -> str:
    """
    사용자 질문에서 과거 참조를 감지하고 자동으로 관련 메모리를 검색
    
    Args:
        openai_messages: OpenAI 형식 메시지 리스트
    
    Returns:
        메모리 컨텍스트 문자열 (없으면 빈 문자열)
    """
    # 마지막 사용자 메시지 추출
    last_user_msg = None
    for msg in reversed(openai_messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if not last_user_msg:
        return ""

    # 과거 참조 키워드 감지
    past_keywords = [
        "지난번", "지난 번", "저번", "이전", "전에",
        "아까", "방금", "전에 말했듯", "말했던", "얘기했던"
    ]

    if not any(keyword in last_user_msg for keyword in past_keywords):
        return ""

    try:
        registry = get_tool_registry()
        memory_result = registry.call("read_memory", {
            "query": last_user_msg,
            "memory_type": "all",
            "top_k": 3
        })

        memory_data = json.loads(memory_result)
        if not memory_data.get("success"):
            return ""

        memories = memory_data.get("memories", [])
        if not memories:
            return ""

        # 메모리 컨텍스트 구성
        memory_context = "\n\n" + "="*60 + "\n"
        memory_context += "📚 관련 기억 (자동 검색)\n"
        memory_context += "="*60 + "\n\n"

        for i, mem in enumerate(memories, 1):
            memory_context += f"{i}. [{mem.get('memory_type', 'unknown')}] "
            memory_context += f"(중요도: {mem.get('importance', 0)}/5)\n"
            memory_context += f"   {mem.get('content', '')}\n"
            memory_context += f"   (유사도: {mem.get('similarity', 0):.3f})\n\n"

        return memory_context

    except Exception as e:
        print(f"⚠️ Memory Read 파이프라인 에러: {e}")
        return ""


# =============================================================================
# LLM Node - Thought + Action 결정
# =============================================================================

def llm_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM 노드: 현재 상태를 기반으로 Thought와 Action 결정
    
    Returns:
        messages: LLM 응답 (도구 호출 정보 포함 가능)
        loop_count: 루프 카운트 증가
    """
    messages = state["messages"]
    loop_count = state.get("loop_count", 0)

    # LangGraph 메시지를 OpenAI 형식으로 변환
    openai_messages = convert_messages_to_openai_format(messages)

    # Memory Read 파이프라인 실행
    memory_context = execute_memory_read_pipeline(openai_messages)
    system_prompt = SYSTEM_PROMPT + (memory_context if memory_context else "")

    # OpenAI API 호출
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *openai_messages
        ],
        tools=get_tool_specs(),
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    # 메시지 구성
    new_message = {
        "role": "assistant",
        "content": msg.content or "",
    }
    
    if msg.tool_calls:
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
    도구 노드: LLM이 요청한 도구 실행 및 결과 반환
    
    Returns:
        messages: 도구 실행 결과 메시지들
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 메시지를 딕셔너리로 변환
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
    
    tool_calls = last_message.get("tool_calls")
    if not tool_calls:
        return {"messages": []}
    
    # 도구 실행
    tool_messages = []
    
    for tool_call in tool_calls:
        # LangGraph vs OpenAI 형식 파싱
        if "name" in tool_call and "args" in tool_call:
            tool_name = tool_call["name"]
            arguments = tool_call["args"]
            tool_call_id = tool_call["id"]
        elif "function" in tool_call:
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"] or "{}")
            tool_call_id = tool_call["id"]
        else:
            tool_name = getattr(tool_call.function, "name", "") if hasattr(tool_call, "function") else ""
            arguments = json.loads(getattr(tool_call.function, "arguments", "{}") if hasattr(tool_call, "function") else "{}")
            tool_call_id = getattr(tool_call, "id", "")
        
        try:
            tool_output = execute_tool(tool_name, arguments)
        except Exception as e:
            tool_output = json.dumps({
                "error": f"도구 실행 실패: {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments
            }, ensure_ascii=False)
        
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
    다음 흐름 결정: 도구 실행 vs 최종 답변
    
    Returns:
        "tools": 도구 실행 필요
        "end": 최종 답변 완료
    """
    messages = state["messages"]
    last_message = messages[-1]
    loop_count = state.get("loop_count", 0)
    
    # 무한 루프 방지 (최대 10회)
    MAX_LOOPS = 10
    if loop_count >= MAX_LOOPS:
        return "end"
    
    # 메시지를 딕셔너리로 변환
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
    
    # tool_calls 여부로 판단
    return "tools" if last_message.get("tool_calls") else "end"