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
from src.tools.tool_registry import get_tool_specs, execute_tool, register_default_tools

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# ToolRegistry 초기화 (Memory Read 파이프라인에서 사용)
_tool_registry = None

def get_tool_registry():
    """ToolRegistry 싱글톤 가져오기"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = register_default_tools()
    return _tool_registry

# =============================================================================
# System Prompt - ReAct 패턴 가이드
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
# Memory Read 파이프라인 - 자동 메모리 검색
# =============================================================================

def execute_memory_read_pipeline(openai_messages: List[Dict[str, Any]]) -> str:
    """
    전처리 단계에서 자동으로 관련 메모리를 검색하는 파이프라인

    Args:
        openai_messages: OpenAI 형식의 메시지 리스트

    Returns:
        메모리 컨텍스트 문자열 (없으면 빈 문자열)
    """

    # 1. 마지막 사용자 메시지 추출
    last_user_msg = None
    for msg in reversed(openai_messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if not last_user_msg:
        return ""

    # 2. 과거 참조 키워드 감지
    past_keywords = [
        "지난번", "지난 번", "저번", "이전", "전에",
        "아까", "방금", "전에 말했듯", "말했던", "얘기했던"
    ]

    has_past_reference = any(keyword in last_user_msg for keyword in past_keywords)

    if not has_past_reference:
        return ""  # 과거 참조가 없으면 검색 안 함

    # 3. read_memory 자동 호출
    try:
        registry = get_tool_registry()

        print(f"\n💾 Memory Read 파이프라인 실행:")
        print(f"   📝 Query: {last_user_msg[:50]}...")

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
            print(f"   ℹ️  관련 기억을 찾지 못했습니다.")
            return ""

        # 4. 메모리 컨텍스트 생성
        memory_context = "\n\n" + "="*60 + "\n"
        memory_context += "📚 관련 기억 (자동 검색됨)\n"
        memory_context += "="*60 + "\n\n"

        for i, mem in enumerate(memories, 1):
            memory_context += f"{i}. [{mem.get('memory_type', 'unknown')}] "
            memory_context += f"(중요도: {mem.get('importance', 0)}/5)\n"
            memory_context += f"   {mem.get('content', '')}\n"
            memory_context += f"   (유사도: {mem.get('similarity', 0):.3f})\n\n"

        print(f"   ✅ {len(memories)}개의 관련 기억을 찾았습니다.")

        return memory_context

    except Exception as e:
        print(f"   ⚠️ Memory Read 파이프라인 에러: {e}")
        return ""


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

    # ===== Memory Read 파이프라인 실행 (전처리) =====
    memory_context = execute_memory_read_pipeline(openai_messages)

    # System Prompt에 메모리 컨텍스트 추가
    system_prompt_with_memory = SYSTEM_PROMPT
    if memory_context:
        system_prompt_with_memory = SYSTEM_PROMPT + memory_context

    # OpenAI API 호출
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt_with_memory},
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