"""
통합 테스트 - Pydantic + ToolRegistry + LangGraph ReAct 전체 시스템 검증
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# OpenAI API Key 임시 설정 (import를 위해)
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing-only'


def test_tool_definitions():
    """Tool Definitions (Pydantic) 테스트"""
    print("\n" + "="*60)
    print("TEST 1: Tool Definitions (Pydantic)")
    print("="*60)

    from src.tools.tool_definitions import (
        ToolSpec,
        get_default_tool_specs,
        as_openai_tool_spec,
        CalculatorInput,
        calculator
    )

    # ToolSpec 생성 확인
    specs = get_default_tool_specs()
    print(f"OK: {len(specs)} ToolSpecs created")

    # Pydantic 검증 확인
    try:
        calc_input = CalculatorInput(a=10, op="+", b=5)
        result = calculator(calc_input)
        assert result["result"] == 15.0
        print(f"OK: Pydantic validation works (10 + 5 = {result['result']})")
    except Exception as e:
        print(f"ERROR: Pydantic validation failed - {e}")
        return False

    # JSON Schema 자동 생성 확인
    openai_spec = as_openai_tool_spec(specs[0])
    assert "function" in openai_spec
    assert "parameters" in openai_spec["function"]
    print(f"OK: JSON Schema auto-generation works")

    return True


def test_tool_registry():
    """ToolRegistry 클래스 테스트"""
    print("\n" + "="*60)
    print("TEST 2: ToolRegistry Class")
    print("="*60)

    from src.tools.tool_registry import ToolRegistry, register_default_tools

    # Registry 생성
    registry = register_default_tools()
    print(f"OK: ToolRegistry created with {len(registry.get_tool_names())} tools")

    # Tool 호출 테스트
    result = registry.call("calculator", {"a": 100, "op": "*", "b": 2})
    assert "200.0" in result
    print(f"OK: registry.call() works (100 * 2 = 200.0)")

    # OpenAI 형식 변환 테스트
    openai_tools = registry.list_openai_tools()
    assert len(openai_tools) == 6
    print(f"OK: registry.list_openai_tools() returns {len(openai_tools)} tools")

    # Pydantic 검증 오류 처리 테스트
    error_result = registry.call("calculator", {"a": 10, "op": "**", "b": 5})
    assert "validation_error" in error_result
    print(f"OK: Pydantic validation error handling works")

    return True


def test_backward_compatibility():
    """기존 함수 호환성 테스트"""
    print("\n" + "="*60)
    print("TEST 3: Backward Compatibility")
    print("="*60)

    from src.tools.tool_registry import get_tool_specs, execute_tool

    # get_tool_specs() 함수
    tools = get_tool_specs()
    assert len(tools) == 6
    print(f"OK: get_tool_specs() returns {len(tools)} tools")

    # execute_tool() 함수
    result = execute_tool("calculator", {"a": 50, "op": "+", "b": 50})
    assert "100.0" in result
    print(f"OK: execute_tool() works (50 + 50 = 100.0)")

    return True


def test_nodes_integration():
    """nodes.py 통합 테스트"""
    print("\n" + "="*60)
    print("TEST 4: nodes.py Integration")
    print("="*60)

    try:
        from src.graph.nodes import tool_node, should_continue
        from src.tools.tool_registry import get_tool_specs

        # nodes.py가 새로운 get_tool_specs()를 사용하는지 확인
        tools = get_tool_specs()
        print(f"OK: nodes.py can import get_tool_specs()")

        # Mock state로 tool_node 테스트 (이모지 없이)
        mock_state = {
            'messages': [
                {
                    'role': 'assistant',
                    'content': 'Testing',
                    'tool_calls': [
                        {
                            'id': 'test123',
                            'type': 'function',
                            'function': {
                                'name': 'calculator',
                                'arguments': '{"a": 20, "op": "+", "b": 30}'
                            }
                        }
                    ]
                }
            ],
            'loop_count': 1
        }

        # tool_node 실행 (print 출력 억제)
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            result = tool_node(mock_state)

        assert 'messages' in result
        assert len(result['messages']) > 0
        print(f"OK: tool_node executes with new registry")

        return True

    except Exception as e:
        print(f"ERROR: nodes.py integration failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Agent 생성 테스트"""
    print("\n" + "="*60)
    print("TEST 5: Agent Graph Creation")
    print("="*60)

    try:
        from src.graph.agent import create_react_agent

        graph = create_react_agent()
        print(f"OK: Agent graph created successfully")

        # 그래프 구조 확인
        assert hasattr(graph, 'nodes')
        node_names = list(graph.nodes.keys())
        assert 'llm' in node_names
        assert 'tools' in node_names
        print(f"OK: Graph has correct nodes: {node_names}")

        return True

    except Exception as e:
        print(f"ERROR: Agent creation failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """전체 통합 테스트 실행"""
    print("\n" + "="*70)
    print(" "*15 + "INTEGRATION TEST SUITE")
    print("="*70)
    print("Testing: Pydantic + ToolSpec + ToolRegistry + LangGraph ReAct")
    print("="*70)

    tests = [
        ("Tool Definitions (Pydantic)", test_tool_definitions),
        ("ToolRegistry Class", test_tool_registry),
        ("Backward Compatibility", test_backward_compatibility),
        ("nodes.py Integration", test_nodes_integration),
        ("Agent Graph Creation", test_agent_creation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 결과 요약
    print("\n" + "="*70)
    print(" "*25 + "TEST RESULTS")
    print("="*70)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "[OK]" if success else "[FAIL]"
        print(f"  {symbol} {test_name}: {status}")

    total = len(results)
    passed = sum(1 for _, success in results if success)

    print("="*70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\nSUCCESS: All systems operational!")
        print("Pydantic + ToolRegistry + LangGraph ReAct is working correctly.")
    else:
        print("\nWARNING: Some tests failed. Please check the errors above.")

    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
