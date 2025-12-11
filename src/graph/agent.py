"""
Agent - LangGraph 기반 ReAct Agent

StateGraph를 사용하여 ReAct 패턴을 구현합니다:
- START → llm_node → (should_continue 판단) → tool_node or END
- tool_node → llm_node (루프)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AgentState
from src.graph.nodes import llm_node, tool_node, should_continue


def create_react_agent():
    """
    ReAct Agent 그래프를 생성하고 반환
    
    Returns:
        compiled graph (실행 가능한 LangGraph 객체)
    """
    
    # 1. StateGraph 생성
    builder = StateGraph(AgentState)
    
    # 2. Node 추가
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    
    # 3. Edge 설정
    # START → llm (시작은 항상 LLM에서)
    builder.add_edge(START, "llm")
    
    # llm → should_continue 판단
    # - "tools" → tool_node
    # - "end" → END
    builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # tools → llm (Tool 실행 후 다시 LLM으로)
    builder.add_edge("tools", "llm")
    
    # 4. 컴파일 (메모리 저장 포함)
    memory = MemorySaver()

    # Interrupt 기능 - Gradio UI에서는 복잡한 로직이 필요하므로 주석 처리
    # 단순 CLI 테스트에서는 interrupt_before=["tools"] 사용 가능
    graph = builder.compile(
        checkpointer=memory,
        # interrupt_before=["tools"]  # 주석 처리: Gradio UI와 호환 문제
    )

    return graph


# =============================================================================
# Agent 실행 함수
# =============================================================================

def run_agent(user_input: str, thread_id: str = "default") -> str:
    """
    ReAct Agent 단일 실행 (UI가 아닌 스크립트용)
    
    Args:
        user_input: 사용자 질문
        thread_id: 대화 세션 ID
    
    Returns:
        최종 답변 문자열
    """
    graph = create_react_agent()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "loop_count": 0
    }
    
    result = graph.invoke(initial_state, config=config)
    final_message = result["messages"][-1]
    
    if hasattr(final_message, "content"):
        return final_message.content
    else:
        return final_message.get("content", "")


def run_agent_stream(user_input: str, thread_id: str = "default"):
    """
    ReAct Agent 스트리밍 실행 (UI가 아닌 스크립트용)
    
    Args:
        user_input: 사용자 질문
        thread_id: 대화 세션 ID
    
    Yields:
        각 단계의 이벤트 딕셔너리
    """
    graph = create_react_agent()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "loop_count": 0
    }
    
    for event in graph.stream(initial_state, config=config):
        yield event