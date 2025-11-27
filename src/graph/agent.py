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
    ReAct Agent를 실행하고 최종 답변을 반환
    
    Args:
        user_input: 사용자 질문
        thread_id: 대화 세션 ID (메모리 저장용)
    
    Returns:
        최종 답변 문자열
    """
    graph = create_react_agent()
    
    # 설정
    config = {"configurable": {"thread_id": thread_id}}
    
    # 초기 상태
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "loop_count": 0
    }
    
    print("\n" + "="*60)
    print("🚀 ReAct Agent 시작")
    print("="*60)
    print(f"📝 User: {user_input}\n")
    
    # 그래프 실행
    result = graph.invoke(initial_state, config=config)
    
    # 최종 답변 추출
    final_message = result["messages"][-1]
    
    if hasattr(final_message, "content"):
        final_answer = final_message.content
    else:
        final_answer = final_message.get("content", "")
    
    print("\n" + "="*60)
    print("✅ ReAct Agent 완료")
    print("="*60)
    print(f"🤖 Assistant: {final_answer}\n")
    
    return final_answer


def run_agent_stream(user_input: str, thread_id: str = "default"):
    """
    ReAct Agent를 스트리밍 방식으로 실행 (이벤트별 출력)
    
    Args:
        user_input: 사용자 질문
        thread_id: 대화 세션 ID
    
    Yields:
        각 단계의 이벤트 (Node 실행 결과)
    """
    graph = create_react_agent()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "loop_count": 0
    }
    
    print("\n" + "="*60)
    print("🚀 ReAct Agent 시작 (Stream Mode)")
    print("="*60)
    print(f"📝 User: {user_input}\n")
    
    # 스트리밍 실행
    for event in graph.stream(initial_state, config=config):
        yield event
    
    print("\n" + "="*60)
    print("✅ ReAct Agent 완료")
    print("="*60)


# =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("🧪 ReAct Agent 테스트\n")
    
    # 테스트 1: RAG 검색
    print("\n" + "🔬 Test 1: RAG 검색")
    print("-" * 60)
    answer1 = run_agent("ReAct 패턴이 뭔지 설명해줘")
    
    # 테스트 2: 계산기
    print("\n" + "🔬 Test 2: 계산기")
    print("-" * 60)
    answer2 = run_agent("1234 * 5678을 계산해줘")
    
    # 테스트 3: 시간 조회
    print("\n" + "🔬 Test 3: 시간 조회")
    print("-" * 60)
    answer3 = run_agent("지금 서울 시간이 몇 시야?")
    
    # 테스트 4: 복합 질문 (RAG + Memory)
    print("\n" + "🔬 Test 4: 복합 질문")
    print("-" * 60)
    answer4 = run_agent(
        "LangGraph의 StateGraph에 대해 설명해주고, "
        "이 내용을 내 학습 기록으로 저장해줘"
    )
    
    print("\n✅ 모든 테스트 완료!")