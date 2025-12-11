"""
Gradio UI - ReAct Agent 웹 인터페이스

FastAPI를 통해 서빙되는 Gradio 기반 대화형 UI
- ChatInterface로 사용자와 상호작용
- ReAct 패턴 기반 에이전트와 통신
- 자동 메모리 저장 및 세션 관리
"""

import os
import gradio as gr
from typing import List, Dict, Any, Generator
import sys
from pathlib import Path
import uuid

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.agent import create_react_agent
from src.memory.reflection import auto_save_recent_memories


# =============================================================================
# 전역 변수
# =============================================================================

# Agent 그래프 (한 번만 생성)
AGENT = create_react_agent()

# 현재 세션의 전체 메시지 기록 (메모리 저장용)
CONVERSATION_HISTORY: List[Dict[str, Any]] = []


# =============================================================================
# Chat Function
# =============================================================================

def chat_function(message: str, history: List[List[str]]) -> Generator[str, None, None]:
    """
    채팅 함수 - Gradio ChatInterface 콜백
    
    Args:
        message: 사용자 입력
        history: 대화 기록
    
    Yields:
        스트리밍 응답 텍스트
    """
    global CONVERSATION_HISTORY

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [{"role": "user", "content": message}],
        "loop_count": 0
    }
    
    CONVERSATION_HISTORY.append({"role": "user", "content": message})
    full_response = ""
    
    try:
        for event in AGENT.stream(initial_state, config=config):
            for node_name, node_output in event.items():
                
                # LLM Node 처리
                if node_name == "llm":
                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        
                        if isinstance(last_msg, dict) and last_msg.get("content"):
                            thought = last_msg["content"]
                            if thought and not full_response:
                                full_response += f"💭 **생각중...**\n\n"
                                yield full_response
                        
                        tool_calls = last_msg.get("tool_calls")
                        if tool_calls:
                            full_response += f"🔧 **도구 사용:**\n"
                            for tc in tool_calls:
                                if isinstance(tc, dict):
                                    tool_name = tc.get("name") or tc.get("function", {}).get("name", "unknown")
                                else:
                                    tool_name = tc.function.name
                                full_response += f"- {tool_name}\n"
                            full_response += "\n"
                            yield full_response
                
                # Tool Node 처리
                elif node_name == "tools":
                    full_response += f"📊 **결과 확인중...**\n\n"
                    yield full_response
        
        # 최종 상태에서 답변 추출
        final_state = AGENT.get_state(config)
        final_messages = final_state.values.get("messages", [])
        
        if final_messages:
            last_message = final_messages[-1]
            
            if hasattr(last_message, "content"):
                final_answer = last_message.content
            else:
                final_answer = last_message.get("content", "")
            
            CONVERSATION_HISTORY.append({"role": "assistant", "content": final_answer})
            full_response = f"✅ **답변:**\n\n{final_answer}"
            yield full_response
        else:
            yield "⚠️ 답변을 생성할 수 없습니다."
    
    except Exception as e:
        yield f"❌ 에러 발생: {str(e)}"
    
    # 메모리 자동 저장
    try:
        saved_count = auto_save_recent_memories(
            messages=CONVERSATION_HISTORY,
            recent_n=6,
            min_importance=3,
            verbose=False
        )
    except Exception as e:
        pass


# =============================================================================
# Gradio Interface
# =============================================================================

def create_gradio_interface():
    """
    Gradio 인터페이스 생성
    
    Returns:
        gr.Blocks 데모 객체
    """
    
    css = """
    .chat-container {
        height: 600px;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(title="ReAct Agent Chat") as demo:
        style_html = f"<style>{css}</style>"
        
        gr.HTML(f"""
        {style_html}
        <div class="header">
            <h1>🤖 ReAct Agent Chat</h1>
            <p>LangGraph 기반 ReAct 패턴 AI Assistant</p>
            <p><em>강의 자료 검색 | 메모리 저장 | 계산 | 시간 조회</em></p>
        </div>
        """)
        
        chatbot = gr.ChatInterface(
            fn=chat_function,
            chatbot=gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=(None, "🤖")
            ),
            textbox=gr.Textbox(
                placeholder="메시지를 입력하세요... (예: LangGraph가 뭐야?)",
                container=False,
                scale=7
            ),
            title=None, 
            description=None,
            examples=[
                "ReAct 패턴이 뭔지 설명해줘",
                "LangGraph의 StateGraph에 대해 알려줘",
                "1234 * 5678을 계산해줘",
                "지금 서울 시간이 몇 시야?",
                "최신 AI 뉴스를 검색해줘",
                "Function Calling과 Tool Calling의 차이는?",
                "내 이름은 김철수이고, 컴퓨터공학과 3학년이야",
            ],
            cache_examples=False,
        )
        
        gr.Markdown("""
        ---
        ### 💡 사용 가능한 기능
        
        - **📚 강의 자료 검색**: Function Calling, RAG, LangGraph 등 수업 내용 질문
        - **💾 메모리 관리**: 개인 학습 정보 자동 저장 및 조회
        - **🧮 계산**: 사칙연산 및 수학 문제
        - **⏰ 시간 조회**: 세계 주요 도시 시간 확인
        - **🌐 웹 검색**: 최신 정보 및 뉴스 검색
        """)
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🎓 생성형AI응용 기말 프로젝트</p>
            <p><em>Powered by LangGraph + OpenAI + ChromaDB</em></p>
        </div>
        """)
        
    return demo


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """Gradio 앱 시작"""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다!")
        return
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
