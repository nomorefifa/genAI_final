"""
Tool Registry - ToolSpec을 사용하여 Agent와 Interaction하는 클래스 (강의 자료 구조)

이 모듈은:
1. ToolSpec을 Tool 'name'으로 인덱싱하여 등록
2. Tool name으로 해당 handler 함수를 호출
3. 등록된 Tool Spec을 OpenAI Tool 스타일로 반환
"""

from __future__ import annotations
from typing import Dict, Any
from pydantic import ValidationError
import json
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tools.tool_definitions import ToolSpec, get_default_tool_specs, as_openai_tool_spec


# =============================================================================
# ToolRegistry 클래스 (강의 자료와 동일한 구조)
# =============================================================================

class ToolRegistry:
    """
    Tool Spec을 사용하여 실제 Agent와 Interaction하는 클래스

    Methods:
        register_tool(spec): ToolSpec을 Tool 'name'으로 인덱싱하여 등록
        call(name, args): Tool name으로 해당 handler 함수를 호출
        list_openai_tools(): 등록된 Tool Spec을 OpenAI Tool 스타일로 반환
        get(name): Tool name으로 ToolSpec 조회
    """

    def __init__(self):
        """ToolRegistry 초기화"""
        self._tools: Dict[str, ToolSpec] = {}

    def register_tool(self, spec: ToolSpec) -> None:
        """
        ToolSpec을 등록

        Args:
            spec: ToolSpec 객체

        Raises:
            ValueError: 이미 등록된 Tool인 경우
        """
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """
        Tool name으로 ToolSpec 조회

        Args:
            name: Tool 이름

        Returns:
            ToolSpec 객체

        Raises:
            KeyError: 등록되지 않은 Tool인 경우
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list_openai_tools(self) -> list[Dict[str, Any]]:
        """
        등록된 모든 Tool을 OpenAI Function Calling 형식으로 반환

        Returns:
            OpenAI tools[] 형식의 리스트
        """
        return [as_openai_tool_spec(spec) for spec in self._tools.values()]

    def call(self, name: str, args: Dict[str, Any]) -> str:
        """
        Tool name으로 해당 handler 함수를 호출

        Args:
            name: Tool 이름
            args: Tool에 전달할 인자 (딕셔너리)

        Returns:
            Tool 실행 결과 (JSON 문자열)
        """
        spec = self.get(name)
        try:
            # handler 실행
            result = spec.handler(args)

            # 결과가 이미 JSON 문자열이면 그대로 반환
            if isinstance(result, str):
                return result

            # 딕셔너리면 JSON 문자열로 변환
            return json.dumps(result, ensure_ascii=False, indent=2)

        except ValidationError as ve:
            # Pydantic 검증 오류
            error_result = {
                "error": "validation_error",
                "details": ve.errors(),
                "tool_name": name,
                "arguments": args
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

        except Exception as e:
            # 런타임 오류
            error_result = {
                "error": "runtime_error",
                "details": str(e),
                "tool_name": name,
                "arguments": args
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

    def get_tool_names(self) -> list[str]:
        """
        등록된 모든 Tool 이름 반환

        Returns:
            Tool 이름 리스트
        """
        return list(self._tools.keys())

    def print_available_tools(self):
        """등록된 Tool 목록을 출력"""
        print("\n📦 등록된 Tool 목록:")
        print("=" * 60)
        for spec in self._tools.values():
            print(f"\n🔧 {spec.name}")
            print(f"   {spec.description}")
        print("=" * 60)


# =============================================================================
# Helper Function (강의 자료와 동일)
# =============================================================================

def register_default_tools() -> ToolRegistry:
    """
    기본 Tool들을 등록한 ToolRegistry 반환

    Returns:
        Tool들이 등록된 ToolRegistry 객체
    """
    reg = ToolRegistry()
    for spec in get_default_tool_specs():
        reg.register_tool(spec)
    return reg


# =============================================================================
# Backward Compatibility (기존 코드와의 호환성)
# =============================================================================

# 기존 코드에서 사용하던 함수들을 registry 기반으로 재구현
_global_registry = None


def _get_global_registry() -> ToolRegistry:
    """전역 ToolRegistry 싱글톤"""
    global _global_registry
    if _global_registry is None:
        _global_registry = register_default_tools()
    return _global_registry


def get_tool_specs() -> list[Dict[str, Any]]:
    """
    모든 Tool Spec을 OpenAI Function Calling 형식으로 반환
    (기존 코드와의 호환성을 위한 함수)

    Returns:
        OpenAI tools[] 형식의 리스트
    """
    return _get_global_registry().list_openai_tools()


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Tool 이름과 인자를 받아서 적절한 Tool을 실행하고 결과를 반환
    (기존 코드와의 호환성을 위한 함수)

    Args:
        tool_name: 실행할 Tool 이름
        arguments: Tool에 전달할 인자 (딕셔너리)

    Returns:
        Tool 실행 결과 (JSON 문자열)
    """
    return _get_global_registry().call(tool_name, arguments)


def get_tool_names() -> list[str]:
    """
    사용 가능한 모든 Tool 이름 리스트 반환
    (기존 코드와의 호환성을 위한 함수)

    Returns:
        Tool 이름 리스트
    """
    return _get_global_registry().get_tool_names()


def print_available_tools():
    """
    사용 가능한 Tool 목록을 출력
    (기존 코드와의 호환성을 위한 함수)
    """
    _get_global_registry().print_available_tools()

