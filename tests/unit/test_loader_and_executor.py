from __future__ import annotations

import io
from unittest import mock

import pytest

from conftest import DATA
from graphrag.cypher_guard import CypherSecurityError
from graphrag.executor import CypherExecutor, Neo4jReadOnlyExecutor
from graphrag.loader import (
    DatasetError,
    parse_edges,
    parse_nodes,
    quote_identifier,
    resolve_csv_source,
    strip_backticks,
)

# ---------------------------------------------------------------- loader


def test_real_dataset_parses():
    nodes = parse_nodes(DATA / "nodes.csv")
    edges = parse_edges(DATA / "edges.csv", node_ids={n.id for n in nodes})
    assert len(nodes) == 830 and len(edges) == 2728
    bsg = next(n for n in nodes if n.id == 0)
    assert bsg.word == "Bsg" and "gene" in bsg.labels and "protein" in bsg.labels
    assert all("`" not in n.word for n in nodes)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("gene", "`gene`"),
        ("immunoglobulin_subtype,_protein_family", "`immunoglobulin_subtype,_protein_family`"),
        ("a`b", "`a``b`"),
        ("x`) DETACH DELETE n //", "`x``) DETACH DELETE n //`"),
        ("ünïcödé", "`ünïcödé`"),
    ],
)
def test_quote_identifier(name, expected):
    assert quote_identifier(name) == expected


@pytest.mark.parametrize("bad", ["", "   ", "a\nb", "a\x00", "x" * 300, None])
def test_quote_identifier_rejects(bad):
    with pytest.raises(DatasetError):
        quote_identifier(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("raw,expected", [("`Bsg`", "Bsg"), ("Bsg", "Bsg"), ("``", ""), ("`", "`"), (" `x` ", "x")])
def test_strip_backticks(raw, expected):
    assert strip_backticks(raw) == expected


NODES_HDR = "id:ID\tlabel:LABEL\tword\n"
EDGES_HDR = "src:START_ID\tdst:END_ID\ttype:TYPE\tverb\n"


def test_parse_nodes_dedupes_labels():
    nodes = parse_nodes(io.StringIO(NODES_HDR + "1\tgene|gene||protein\t`X`\n"))
    assert nodes[0].labels == ("gene", "protein")


@pytest.mark.parametrize(
    "body",
    ["abc\tgene\tx\n", "1\t\tx\n", "1\tgene\tx\n1\tgene\ty\n", "1\tgene\n"],
    ids=["non-int-id", "no-labels", "duplicate", "missing-col"],
)
def test_parse_nodes_errors(body):
    with pytest.raises(DatasetError):
        parse_nodes(io.StringIO(NODES_HDR + body))


def test_parse_edges_dangling():
    with pytest.raises(DatasetError, match="dangling"):
        parse_edges(io.StringIO(EDGES_HDR + "1\t99\tREL\t`v`\n"), node_ids={1})


def test_parse_edges_bad_type():
    with pytest.raises(DatasetError):
        parse_edges(io.StringIO(EDGES_HDR + "1\t2\t\t`v`\n"))


@pytest.mark.parametrize(
    "src,expected",
    [
        ("https://github.com/Ciccio1307/GraphRAG/blob/main/data/nodes.csv", "https://raw.githubusercontent.com/Ciccio1307/GraphRAG/main/data/nodes.csv"),
        ("data/nodes.csv", "data/nodes.csv"),
        ("https://example.com/blob/x.csv", "https://example.com/blob/x.csv"),
    ],
)
def test_resolve_csv_source(src, expected):
    assert resolve_csv_source(src) == expected


# ---------------------------------------------------------------- executor


class _Rec:
    def __init__(self, i):
        self.i = i

    def data(self):
        return {"i": self.i}


def _mock_driver(n_records: int):
    produced = []

    def gen():
        for i in range(n_records):
            produced.append(i)
            yield _Rec(i)

    tx = mock.Mock()
    tx.run.return_value = gen()
    session = mock.MagicMock()
    session.__enter__.return_value = session
    session.execute_read.side_effect = lambda work: work(tx)
    driver = mock.Mock()
    driver.session.return_value = session
    return driver, session, tx, produced


def test_executor_uses_read_session_and_bounds_results():
    import neo4j

    driver, session, tx, produced = _mock_driver(10_000)
    ex = Neo4jReadOnlyExecutor(driver, database="bio", max_records=7, timeout_s=3)
    assert isinstance(ex, CypherExecutor)
    rows = ex.run("MATCH (n) RETURN n;", {"x": 1})
    assert rows == [{"i": i} for i in range(7)]
    assert len(produced) == 7  # streamed, not fully materialised
    driver.session.assert_called_once_with(default_access_mode=neo4j.READ_ACCESS, database="bio")
    tx.run.assert_called_once_with("MATCH (n) RETURN n", {"x": 1})


def test_executor_rejects_before_touching_db():
    driver, session, tx, _ = _mock_driver(1)
    ex = Neo4jReadOnlyExecutor(driver)
    with pytest.raises(CypherSecurityError):
        ex.run("MATCH (n) DETACH DELETE n")
    driver.session.assert_not_called()


def test_executor_without_database_omits_kwarg():
    driver, *_ = _mock_driver(0)
    assert Neo4jReadOnlyExecutor(driver).run("RETURN 1") == []
    assert "database" not in driver.session.call_args.kwargs


@pytest.mark.parametrize("kwargs", [{"max_records": 0}, {"timeout_s": 0}, {"timeout_s": -1}])
def test_executor_invalid_args(kwargs):
    with pytest.raises(ValueError):
        Neo4jReadOnlyExecutor(mock.Mock(), **kwargs)


def test_executor_close():
    driver = mock.Mock()
    Neo4jReadOnlyExecutor(driver).close()
    driver.close.assert_called_once()
