"""
Gradio UI - ReAct Agent 웹 인터페이스

이 모듈은:
1. Gradio ChatInterface로 웹 UI 제공
2. ReAct 과정 실시간 시각화 (Thought, Action, Observation)
3. 스트리밍 방식으로 응답 출력
4. Memory 자동 저장 기능
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
    Gradio ChatInterface용 채팅 함수

    Args:
        message: 사용자 입력
        history: 대화 기록 [[user, assistant], ...]

    Yields:
        Assistant의 응답 (스트리밍)
    """
    global CONVERSATION_HISTORY

    # 설정 - 매번 새로운 thread_id 사용 (대화 기록 충돌 방지)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 초기 상태
    initial_state = {
        "messages": [{"role": "user", "content": message}],
        "loop_count": 0
    }
    
    # 전체 대화 기록에 추가
    CONVERSATION_HISTORY.append({"role": "user", "content": message})
    
    # 응답 누적 변수
    full_response = ""
    
    # Agent 실행 (스트리밍)
    try:
        for event in AGENT.stream(initial_state, config=config):
            # 각 Node의 출력 처리
            for node_name, node_output in event.items():
                
                # LLM Node 출력
                if node_name == "llm":
                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        
                        # Thought 출력
                        if isinstance(last_msg, dict) and last_msg.get("content"):
                            thought = last_msg["content"]
                            if thought and not full_response:  # 첫 Thought만 표시
                                full_response += f"💭 **생각중...**\n\n"
                                yield full_response
                        
                        # Tool Calls 표시
                        tool_calls = last_msg.get("tool_calls")
                        if tool_calls:
                            full_response += f"🔧 **도구 사용:**\n"
                            for tc in tool_calls:
                                # 딕셔너리/객체 양방향 처리
                                if isinstance(tc, dict):
                                    # LangGraph 형식
                                    if "name" in tc:
                                        tool_name = tc["name"]
                                    # OpenAI 형식
                                    elif "function" in tc:
                                        tool_name = tc["function"]["name"]
                                    else:
                                        tool_name = "unknown"
                                else:
                                    # 객체 형식
                                    tool_name = tc.function.name
                                full_response += f"- {tool_name}\n"
                            full_response += "\n"
                            yield full_response
                
                # Tool Node 출력
                elif node_name == "tools":
                    full_response += f"📊 **결과 확인중...**\n\n"
                    yield full_response
        
        # 최종 상태에서 답변 추출
        final_state = AGENT.get_state(config)
        final_messages = final_state.values.get("messages", [])
        
        if final_messages:
            last_message = final_messages[-1]
            
            # 최종 답변
            if hasattr(last_message, "content"):
                final_answer = last_message.content
            else:
                final_answer = last_message.get("content", "")
            
            # 전체 대화 기록에 추가
            CONVERSATION_HISTORY.append({"role": "assistant", "content": final_answer})
            
            # 답변 출력 (Thought/Action 정보 제거하고 깔끔하게)
            full_response = f"✅ **답변:**\n\n{final_answer}"
            yield full_response
        
        else:
            yield "⚠️ 답변을 생성할 수 없습니다."
    
    except Exception as e:
        yield f"❌ 에러 발생: {str(e)}"
    
    # ===== Memory Write 파이프라인 (자동 저장) =====
    try:
        print(f"\n💾 Memory Write 파이프라인 실행:")
        print(f"   📝 최근 {min(6, len(CONVERSATION_HISTORY))}개 메시지 분석 중...")

        saved_count = auto_save_recent_memories(
            messages=CONVERSATION_HISTORY,
            recent_n=6,            # 최근 6개 메시지만 분석
            min_importance=3,      # 중요도 3 이상만 저장
            verbose=False          # 상세 로그 비활성화
        )

        if saved_count > 0:
            print(f"   ✅ {saved_count}개의 중요한 정보를 장기 기억에 저장했습니다.")
        else:
            print(f"   ℹ️  저장할 중요한 정보가 없습니다.")

    except Exception as e:
        print(f"   ⚠️ 메모리 저장 실패: {e}")


# =============================================================================
# Gradio Interface
# =============================================================================

def create_gradio_interface():
    """
    Gradio ChatInterface 생성
    
    Returns:
        gr.ChatInterface 객체 (또는 gr.Blocks를 사용하는 경우 Blocks 객체)
    """
    
    # CSS 스타일 (변경 없음)
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
    
    # [수정 1: HTML 요소들을 gr.Blocks에 넣고, ChatInterface를 반환합니다.]
    # 그러나 복잡한 HTML 구조 때문에,
    # [수정 2: CSS 인수를 지원하는 gr.ChatInterface로 CSS를 직접 전달합니다.]

    # 최상위 컨테이너를 gr.Blocks로 유지하되, css 인수를 제거하고 gr.HTML로 대체합니다.
    with gr.Blocks(title="ReAct Agent Chat") as demo:
        # 경고: 최신 Gradio 버전에서는 CSS를 <style> 태그로 HTML에 넣는 것이 더 안정적입니다.
        
        # 1. CSS를 직접 HTML 스타일 태그에 넣습니다.
        style_html = f"<style>{css}</style>"
        
        # 2. 헤더와 스타일을 HTML로 통합
        gr.HTML(f"""
        {style_html}
        <div class="header">
            <h1>🤖 ReAct Agent Chat</h1>
            <p>LangGraph 기반 ReAct 패턴 AI Assistant</p>
            <p><em>강의 자료 검색 | 메모리 저장 | 계산 | 시간 조회</em></p>
        </div>
        """)
        
        # 채팅 인터페이스 (ChatInterface 자체는 CSS 인수를 지원합니다.)
        chatbot = gr.ChatInterface(
            fn=chat_function,
            chatbot=gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=(None, "🤖")
            ),
            # ... (나머지 ChatInterface 설정은 그대로 둡니다.)
            textbox=gr.Textbox(
                placeholder="메시지를 입력하세요... (예: LangGraph가 뭐야?)",
                container=False,
                scale=7
            ),
            title=None, 
            description=None,
            #theme="soft",
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
        
        # 하단 정보
        gr.Markdown("""
        ---
        ### 💡 사용 가능한 기능
        
        - **📚 강의 자료 검색**: Function Calling, RAG, LangGraph 등 수업 내용 질문
        # ... (나머지 Markdown 내용 유지) ...
        """)
        
        # 푸터
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
    """
    Gradio 앱 실행
    """
    print("\n" + "="*60)
    print("🚀 Gradio UI 시작")
    print("="*60)
    
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다!")
        print("   .env 파일을 확인하세요.")
        return
    
    # Gradio 앱 생성
    demo = create_gradio_interface()
    
    # 실행
    print("\n✅ UI 준비 완료!")
    print("   브라우저에서 http://localhost:7860 접속하세요\n")
    
    demo.launch(
        server_name="0.0.0.0",  # 외부 접속 허용
        server_port=7860,
        share=False,  # True로 설정하면 공개 URL 생성
        inbrowser=True  # 자동으로 브라우저 열기
    )


if __name__ == "__main__":
    main()