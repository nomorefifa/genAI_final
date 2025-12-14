"""
Memory Reflection - 대화 중 중요 정보 자동 저장

이 모듈은:
1. 대화 내용을 분석하여 중요한 정보 추출
2. 자동으로 write_memory를 호출하여 장기 기억에 저장
3. 사용자에게 방해되지 않게 백그라운드에서 작동
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tools.memory_tool import write_memory

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# =============================================================================
# Memory Extractor Prompt
# =============================================================================

MEMORY_EXTRACTOR_PROMPT = """\
당신은 대화에서 중요한 정보를 추출하여 장기 기억에 저장하는 AI입니다.

다음 대화 내용을 분석하여 저장할 가치가 있는 정보를 찾아주세요.

**저장해야 할 정보 유형:**

1. **Profile (개인정보)** - memory_type: "profile"
   - 사용자의 이름, 나이, 직업, 전공
   - 거주지, 가족 관계
   - 연락처, 이메일 등
   예시: "사용자 이름은 김철수, 컴퓨터공학과 3학년"

2. **Episodic (대화/사건)** - memory_type: "episodic"
   - 중요한 대화 내용, 결정사항
   - 특정 날짜의 사건이나 경험
   - 사용자의 의견, 감정 표현
   예시: "사용자가 LangGraph를 처음 학습함 (2024-11-27)"

3. **Knowledge (학습/지식)** - memory_type: "knowledge"
   - 사용자가 배운 개념이나 이해한 내용
   - 학습 진도, 완료한 과제
   - 관심 있는 주제나 기술
   예시: "ReAct 패턴: Thought-Action-Observation 순서로 작동하는 AI Agent 방법론"

**중요도 판단 기준:**
- 5: 매우 중요한 정보 (이름, 전공, 중요한 결정사항, 장기 목표)
- 4: 중요한 정보 (학습 내용, 선호사항, 프로젝트 정보)
- 3: 보통 정보 (일반적인 대화 내용)
- 2: 약간 중요한 정보 (참고용 정보)
- 1: 낮은 중요도 (단순한 대화)

**출력 형식:**
JSON 배열로 반환하세요. 저장할 정보가 없으면 빈 배열 []을 반환하세요.

[
  {
    "content": "저장할 내용",
    "memory_type": "profile | episodic | knowledge",
    "importance": 1~5 (정수)
  },
  ...
]

**중요:** 
- 저장할 만한 정보가 없으면 빈 배열 []을 반환하세요.
- 단순한 인사말이나 질문만 있는 경우 저장하지 마세요.
- 이미 알고 있는 일반 상식은 저장하지 마세요.
"""


# =============================================================================
# Memory Extraction Function
# =============================================================================

def extract_memories_from_conversation(
    messages: List[Dict[str, Any]],
    min_importance: int = 3
) -> List[Dict[str, str]]:
    """
    대화 내용을 분석하여 저장할 정보 추출

    Args:
        messages: 대화 메시지 리스트
        min_importance: 최소 중요도 (1-5, 기본값 3)

    Returns:
        저장할 메모리 정보 리스트
    """
    
    # 대화 내용 포맷팅
    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        
        if role == "user":
            content = msg.get("content", "")
            conversation_text += f"User: {content}\n\n"
        
        elif role == "assistant":
            content = msg.get("content", "")
            if content:  # 빈 content는 스킵
                conversation_text += f"Assistant: {content}\n\n"
    
    if not conversation_text.strip():
        return []
    
    # LLM에게 메모리 추출 요청
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # JSON 파싱
        # 가끔 LLM이 ```json ``` 같은 마크다운을 붙이는 경우 제거
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        memories = json.loads(result_text)

        # 중요도 필터링 (1-5 정수 기준)
        filtered_memories = [
            m for m in memories
            if m.get("importance", 1) >= min_importance
        ]

        return filtered_memories
    
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 파싱 에러: {e}")
        print(f"   Response: {result_text[:200]}")
        return []
    
    except Exception as e:
        print(f"⚠️ Memory 추출 에러: {e}")
        return []


# =============================================================================
# Auto Save Function
# =============================================================================

def auto_save_memories(
    messages: List[Dict[str, Any]],
    min_importance: int = 3,
    verbose: bool = True
) -> int:
    """
    대화 내용을 분석하여 자동으로 메모리 저장

    Args:
        messages: 대화 메시지 리스트
        min_importance: 최소 중요도 (1-5, 기본값 3)
        verbose: 저장 과정 출력 여부

    Returns:
        저장된 메모리 개수
    """
    
    # 1. 메모리 추출
    memories = extract_memories_from_conversation(messages, min_importance)
    
    if not memories:
        if verbose:
            print("\n💾 Auto-save: 저장할 메모리가 없습니다.")
        return 0
    
    # 2. 각 메모리 저장
    saved_count = 0
    
    if verbose:
        print(f"\n💾 Auto-save: {len(memories)}개의 메모리를 저장합니다...")
    
    for mem in memories:
        content = mem.get("content", "")
        memory_type = mem.get("memory_type", "episodic")
        importance = mem.get("importance", "medium")
        
        try:
            result = write_memory(content, memory_type, importance)
            
            if verbose:
                print(f"   ✅ [{memory_type}] {content[:50]}...")
            
            saved_count += 1
        
        except Exception as e:
            if verbose:
                print(f"   ❌ 저장 실패: {e}")
    
    if verbose:
        print(f"💾 총 {saved_count}개 메모리 저장 완료!\n")
    
    return saved_count


# =============================================================================
# Helper: 마지막 N개 메시지만 분석
# =============================================================================

def auto_save_recent_memories(
    messages: List[Dict[str, Any]],
    recent_n: int = 10,
    min_importance: int = 3,
    verbose: bool = True
) -> int:
    """
    최근 N개 메시지만 분석하여 메모리 저장
    (긴 대화에서 매번 전체 분석하면 비효율적)

    Args:
        messages: 전체 대화 메시지
        recent_n: 분석할 최근 메시지 개수
        min_importance: 최소 중요도 (1-5, 기본값 3)
        verbose: 출력 여부

    Returns:
        저장된 메모리 개수
    """
    recent_messages = messages[-recent_n:] if len(messages) > recent_n else messages
    return auto_save_memories(recent_messages, min_importance, verbose)
