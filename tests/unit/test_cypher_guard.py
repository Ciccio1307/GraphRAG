from __future__ import annotations

import pytest

from conftest import load_fixture
from graphrag.cypher_guard import (
    CypherSecurityError,
    is_read_only_cypher,
    mask_literals,
    validate_read_only_cypher,
)

CASES = load_fixture("cypher_cases.json")


def test_dataset_is_large_and_balanced():
    assert len(CASES) >= 50
    assert sum(c["allowed"] for c in CASES) >= 20
    assert sum(not c["allowed"] for c in CASES) >= 30
    assert len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", [c for c in CASES if c["allowed"]], ids=lambda c: f"{c['id']}-{c['category']}")
def test_read_only_queries_are_accepted(case):
    result = validate_read_only_cypher(case["query"])
    assert result
    assert not result.endswith(";")
    assert result == result.strip()


@pytest.mark.parametrize("case", [c for c in CASES if not c["allowed"]], ids=lambda c: f"{c['id']}-{c['category']}")
def test_dangerous_queries_are_rejected(case):
    with pytest.raises(CypherSecurityError):
        validate_read_only_cypher(case["query"])
    assert is_read_only_cypher(case["query"]) is False


def test_trailing_semicolon_and_whitespace_are_removed():
    assert validate_read_only_cypher("  MATCH (n) RETURN n ;  \n") == "MATCH (n) RETURN n"


def test_semicolon_inside_string_is_kept():
    assert validate_read_only_cypher("RETURN 'a;b';") == "RETURN 'a;b'"


@pytest.mark.parametrize("bad", [None, 42, b"MATCH (n) RETURN n", ["MATCH"]])
def test_non_string_input_is_rejected(bad):
    with pytest.raises(CypherSecurityError):
        validate_read_only_cypher(bad)  # type: ignore[arg-type]


def test_max_length_boundary():
    base = "RETURN 1 AS x"
    padded = base + " " * 0
    query = "RETURN '" + "a" * (100 - len("RETURN ''")) + "'"
    assert len(query) == 100
    assert validate_read_only_cypher(query, max_chars=100) == query
    with pytest.raises(CypherSecurityError, match="exceeds"):
        validate_read_only_cypher(query + " ", max_chars=99)
    assert validate_read_only_cypher(padded)


def test_mask_literals_preserves_length_and_blanks_data():
    query = "MATCH (n) WHERE n.w = 'DELETE' // SET\nRETURN n /* DROP */"
    masked = mask_literals(query)
    assert len(masked) == len(query)
    for word in ("DELETE", "SET", "DROP"):
        assert word not in masked
    assert "MATCH" in masked and "RETURN" in masked


def test_mask_literals_handles_escaped_backticks():
    assert mask_literals("MATCH (n:`a``b`) RETURN n") == "MATCH (n:`____`) RETURN n"


@pytest.mark.parametrize("query", ["CALL db.labels", "CALL db.labels() YIELD label RETURN label", "CALL DB.LABELS()"])
def test_allowlisted_procedures(query):
    assert is_read_only_cypher(query)


def test_call_without_target_is_rejected():
    with pytest.raises(CypherSecurityError):
        validate_read_only_cypher("MATCH (n) CALL")


def test_error_message_names_the_clause():
    with pytest.raises(CypherSecurityError, match="'DETACH'"):
        validate_read_only_cypher("MATCH (n) DETACH DELETE n")


def test_guard_is_fast_on_large_inputs():
    import time

    query = "MATCH (n) WHERE n.w IN [" + ",".join(f"'v{i}'" for i in range(600)) + "] RETURN n"
    start = time.perf_counter()
    for _ in range(50):
        validate_read_only_cypher(query)
    assert time.perf_counter() - start < 1.0
