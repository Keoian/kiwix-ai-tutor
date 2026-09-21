"""Contract tests for tutor/tools/schemas.py (RED: module does not exist yet)."""

import json

from tutor.tools.schemas import (
    CALC_TOOL,
    RESEARCH_TOOL,
    TOOLS,
    ValidationResult,
    validate_tool_call,
)


def test_tools_list_contains_research_and_calc_in_order():
    assert TOOLS == [RESEARCH_TOOL, CALC_TOOL]


def test_research_tool_shape():
    assert RESEARCH_TOOL["type"] == "function"
    fn = RESEARCH_TOOL["function"]
    assert fn["name"] == "research"
    assert isinstance(fn["description"], str) and fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    # Additive (docs/rewrite_on_weak_evidence.md): neither "query" nor
    # "queries" is schema-required on its own -- validate_tool_call
    # enforces "at least one of the two" instead, so the older single
    # ``query`` form keeps working unchanged.
    assert params["required"] == []
    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["keywords"]["type"] == "array"
    assert params["properties"]["keywords"]["items"]["type"] == "string"
    assert params["properties"]["queries"]["type"] == "array"
    assert params["properties"]["queries"]["items"]["type"] == "string"


def test_calc_tool_shape():
    assert CALC_TOOL["type"] == "function"
    fn = CALC_TOOL["function"]
    assert fn["name"] == "calc"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert params["required"] == ["expression"]
    assert params["properties"]["expression"]["type"] == "string"


def test_validation_result_is_frozen_dataclass():
    result = ValidationResult(ok=True, arguments={"query": "x"}, error=None)
    assert result.ok is True
    assert result.arguments == {"query": "x"}
    assert result.error is None
    try:
        result.ok = False
    except Exception as exc:  # dataclasses.FrozenInstanceError is a subclass of AttributeError
        assert "frozen" in str(exc).lower() or isinstance(exc, AttributeError)
    else:
        raise AssertionError("ValidationResult must be frozen")


def test_validate_valid_research_call_query_only():
    result = validate_tool_call("research", json.dumps({"query": "photosynthesis"}))
    assert result.ok is True
    assert result.arguments == {"query": "photosynthesis"}
    assert result.error is None


def test_validate_valid_research_call_with_keywords():
    args = {"query": "mitosis stages", "keywords": ["prophase", "metaphase"]}
    result = validate_tool_call("research", json.dumps(args))
    assert result.ok is True
    assert result.arguments == args


def test_validate_valid_research_call_with_question():
    args = {"queries": ["Titin"], "question": "What is the longest molecule?"}
    result = validate_tool_call("research", json.dumps(args))
    assert result.ok is True
    assert result.arguments == args


def test_validate_research_call_without_question_still_ok():
    # ``question`` is optional -- validation must accept calls without it
    # (existing behaviour, unchanged).
    result = validate_tool_call("research", json.dumps({"queries": ["Titin"]}))
    assert result.ok is True
    assert "question" not in result.arguments


def test_validate_research_call_question_wrong_type_rejected():
    args = {"queries": ["Titin"], "question": 123}
    result = validate_tool_call("research", json.dumps(args))
    assert result.ok is False


def test_research_tool_schema_declares_optional_question():
    params = RESEARCH_TOOL["function"]["parameters"]
    assert "question" not in params.get("required", [])
    assert params["properties"]["question"]["type"] == "string"


def test_validate_valid_calc_call():
    result = validate_tool_call("calc", json.dumps({"expression": "2 + 2"}))
    assert result.ok is True
    assert result.arguments == {"expression": "2 + 2"}


def test_validate_unknown_tool_name():
    result = validate_tool_call("frobnicate", json.dumps({"query": "x"}))
    assert result.ok is False
    assert result.arguments is None
    assert result.error


def test_validate_arguments_not_valid_json():
    result = validate_tool_call("research", "{not json")
    assert result.ok is False
    assert result.arguments is None
    assert result.error


def test_validate_arguments_not_a_json_object():
    result = validate_tool_call("research", json.dumps(["query", "x"]))
    assert result.ok is False
    assert result.error


def test_validate_missing_required_field_research():
    result = validate_tool_call("research", json.dumps({"keywords": ["a"]}))
    assert result.ok is False
    assert result.error


def test_validate_missing_required_field_calc():
    result = validate_tool_call("calc", json.dumps({}))
    assert result.ok is False
    assert result.error


def test_validate_wrong_type_query_not_string():
    result = validate_tool_call("research", json.dumps({"query": 123}))
    assert result.ok is False


def test_validate_wrong_type_keywords_not_array():
    result = validate_tool_call("research", json.dumps({"query": "x", "keywords": "a,b"}))
    assert result.ok is False


def test_validate_wrong_type_keywords_items_not_strings():
    result = validate_tool_call("research", json.dumps({"query": "x", "keywords": [1, 2]}))
    assert result.ok is False


def test_validate_wrong_type_expression_not_string():
    result = validate_tool_call("calc", json.dumps({"expression": 4}))
    assert result.ok is False


def test_validate_unknown_extra_key_rejected():
    result = validate_tool_call("research", json.dumps({"query": "x", "extra": "nope"}))
    assert result.ok is False


def test_validate_unknown_extra_key_rejected_calc():
    result = validate_tool_call("calc", json.dumps({"expression": "1+1", "unit": "cm"}))
    assert result.ok is False


def test_validate_empty_query_rejected():
    result = validate_tool_call("research", json.dumps({"query": ""}))
    assert result.ok is False


def test_validate_whitespace_only_query_rejected():
    result = validate_tool_call("research", json.dumps({"query": "   "}))
    assert result.ok is False


def test_validate_empty_expression_rejected():
    result = validate_tool_call("calc", json.dumps({"expression": ""}))
    assert result.ok is False


def test_validate_whitespace_only_expression_rejected():
    result = validate_tool_call("calc", json.dumps({"expression": "\t\n"}))
    assert result.ok is False


def test_validate_needs_search_false_allows_empty_queries():
    result = validate_tool_call("research", json.dumps({"needs_search": False, "queries": []}))
    assert result.ok is True


def test_validate_needs_search_true_still_requires_a_query():
    result = validate_tool_call("research", json.dumps({"needs_search": True}))
    assert result.ok is False


def test_validate_needs_search_must_be_boolean():
    result = validate_tool_call("research", json.dumps({"needs_search": "false", "query": "x"}))
    assert result.ok is False


def test_validate_needs_search_false_with_no_query_or_queries_key_at_all():
    result = validate_tool_call("research", json.dumps({"needs_search": False}))
    assert result.ok is True
