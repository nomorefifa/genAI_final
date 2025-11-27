"""
AgentState - LangGraph의 State 정의

ReAct Agent의 상태를 관리하는 TypedDict입니다.
- messages: 대화 기록 (자동으로 메시지 추가)
- loop_count: 무한 루프 방지용 카운터
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    ReAct Agent의 상태 정의
    
    Attributes:
        messages: 대화 기록 (LLM과 Tool의 대화)
                  Annotated[list, add_messages]를 사용하면
                  LangGraph가 자동으로 메시지를 누적 관리합니다.
        
        loop_count: Tool 사용 횟수 (무한 루프 방지)
    """
    messages: Annotated[list, add_messages]
    loop_count: int