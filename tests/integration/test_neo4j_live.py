"""Integration tests against a real Neo4j.

Uses ``NEO4J_TEST_URI``/``NEO4J_TEST_USER``/``NEO4J_TEST_PASSWORD`` if set,
otherwise starts a throw-away ``neo4j:5.26-community`` Docker container.
Skipped when neither is available. The database is wiped and re-loaded.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from unittest import mock

import pytest

from conftest import DATA, load_fixture
from graphrag.cypher_guard import CypherSecurityError
from graphrag.executor import Neo4jReadOnlyExecutor
from graphrag.loader import load_graph, parse_edges, parse_nodes

neo4j = pytest.importorskip("neo4j")

pytestmark = pytest.mark.neo4j

IMAGE = os.getenv("NEO4J_TEST_IMAGE", "neo4j:5.26-community")
PASSWORD = "test-password-123"


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def neo4j_conn():
    if os.getenv("NEO4J_TEST_URI"):
        yield os.environ["NEO4J_TEST_URI"], os.getenv("NEO4J_TEST_USER", "neo4j"), os.environ["NEO4J_TEST_PASSWORD"]
        return
    if not _docker_ok():
        pytest.skip("No NEO4J_TEST_URI and Docker is not available")

    name = f"graphrag-test-{uuid.uuid4().hex[:8]}"
    run = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-p", "127.0.0.1::7687",
         "-e", f"NEO4J_AUTH=neo4j/{PASSWORD}", "-e", "NEO4J_server_memory_heap_max__size=512m",
         "-e", 'NEO4J_PLUGINS=["apoc"]', IMAGE],
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"Could not start Neo4j container: {run.stderr.strip()[:200]}")
    try:
        port = subprocess.run(["docker", "port", name, "7687"], capture_output=True, text=True, check=True).stdout.split(":")[-1].strip()
        uri = f"bolt://127.0.0.1:{port}"
        deadline = time.time() + 120
        while True:
            try:
                with neo4j.GraphDatabase.driver(uri, auth=("neo4j", PASSWORD)) as d:
                    d.verify_connectivity()
                break
            except Exception:
                if time.time() > deadline:
                    pytest.fail("Neo4j container did not become ready in 120s")
                time.sleep(2)
        yield uri, "neo4j", PASSWORD
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture(scope="module")
def driver(neo4j_conn):
    uri, user, pw = neo4j_conn
    with neo4j.GraphDatabase.driver(uri, auth=(user, pw)) as d:
        nodes = parse_nodes(DATA / "nodes.csv")
        edges = parse_edges(DATA / "edges.csv", node_ids={n.id for n in nodes})
        counts = load_graph(d, nodes, edges, wipe=True)
        assert counts == {"nodes": 830, "edges": 2728}
        yield d


@pytest.fixture
def executor(driver):
    return Neo4jReadOnlyExecutor(driver, max_records=100, timeout_s=10)


def _count(driver) -> tuple[int, int]:
    with driver.session() as s:
        return s.run("MATCH (n) RETURN count(n) AS c").single()["c"], s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]


def test_loader_refuses_non_empty_db_without_wipe(driver):
    with pytest.raises(Exception, match="already contains"):
        load_graph(driver, [], [])


def test_words_are_stored_without_backticks(executor):
    rows = executor.run("MATCH (g:gene) WHERE lower(g.word) = lower('Bsg') RETURN g.word AS w, labels(g) AS l")
    assert rows and rows[0]["w"] == "Bsg" and "protein" in rows[0]["l"]


def test_few_shot_queries_run_on_real_data(executor):
    for case in load_fixture("cypher_cases.json"):
        if case["category"] == "few-shot":
            executor.run(case["query"])  # must not raise


def test_unicode_values_roundtrip(executor):
    rows = executor.run("MATCH (c) WHERE c.word = 'یک ترکیب شیمیایی' RETURN c.word AS w")
    assert rows == [{"w": "یک ترکیب شیمیایی"}]


def test_results_are_bounded(driver):
    ex = Neo4jReadOnlyExecutor(driver, max_records=5)
    assert len(ex.run("MATCH (n) RETURN n.word")) == 5


def test_degree_query(executor):
    rows = executor.run("MATCH (n)-[r:REL]-() WHERE n.id = 1 RETURN n.word AS w, count(r) AS deg")
    assert rows == [{"w": "house mouse", "deg": 778}]  # 802 incident edge-ends, 24 are self-loops


@pytest.mark.parametrize("case", [c for c in load_fixture("cypher_cases.json") if not c["allowed"]], ids=lambda c: c["id"])
def test_guard_blocks_writes_on_real_db(driver, executor, case):
    before = _count(driver)
    with pytest.raises(CypherSecurityError):
        executor.run(case["query"])
    assert _count(driver) == before


@pytest.mark.parametrize(
    "write",
    ["MATCH (n) DETACH DELETE n", "CREATE (:gene {word: 'pwned'})", "MATCH (g:gene) SET g.word = 'x'"],
)
def test_database_rejects_writes_even_if_guard_is_bypassed(driver, executor, write):
    """Defence in depth: the READ transaction alone must stop writes."""
    before = _count(driver)
    with mock.patch("graphrag.executor.validate_read_only_cypher", lambda q: q):
        with pytest.raises(neo4j.exceptions.ClientError) as err:
            executor.run(write)
    assert "AccessMode" in (err.value.code or "") or "read" in str(err.value).lower()
    assert _count(driver) == before


def test_server_side_timeout(driver):
    ex = Neo4jReadOnlyExecutor(driver, timeout_s=0.5)
    with pytest.raises(neo4j.exceptions.Neo4jError):
        ex.run("UNWIND range(1, 100000000) AS x WITH x WHERE x % 7 = 123456789 RETURN count(x)")


# ------------------------------------------------------------ real factory wiring


@pytest.fixture
def live_settings(neo4j_conn, driver):
    from graphrag.config import Settings

    uri, user, pw = neo4j_conn
    return Settings(neo4j_uri=uri, neo4j_user=user, neo4j_password=pw, top_k=3, enable_web_search=False)


def test_factory_builds_real_executor(live_settings):
    from graphrag.factory import build_executor

    ex = build_executor(live_settings)
    try:
        assert len(ex.run("MATCH (n:gene) RETURN n.word")) == 3
    finally:
        ex.close()


def test_factory_full_pipeline_on_real_graph(live_settings):
    from conftest import make_llm
    from graphrag.factory import build_agent, build_graph_qa_pipeline, build_web_search, fetch_structured_schema
    from graphrag.graph_qa import GraphQAStatus

    try:
        structured = fetch_structured_schema(live_settings)
    except Exception as exc:  # APOC plugin could not be installed (offline)
        pytest.skip(f"APOC not available: {exc}")
    assert {"start": "disease", "type": "REL", "end": "gene"} in structured["relationships"]
    assert build_web_search(live_settings) is None

    llm = make_llm(
        "WITH 'Bsg' AS s MATCH (g:gene)-[:REL]-(d:disease) WHERE lower(g.word) = lower(s) RETURN d.word AS Disease",
        "BSG is linked to several diseases.",
    )
    pipeline = build_graph_qa_pipeline(live_settings, llm_factory=lambda m: llm)
    try:
        assert "disease" in pipeline.schema and "membrane_proteins" not in pipeline.schema
        result = pipeline.answer("What diseases are related to BSG?")
        assert result.status is GraphQAStatus.OK and 0 < len(result.records) <= 3
        bundle = build_agent(live_settings, llm_factory=lambda m: make_llm("final"), graph_pipeline=pipeline)
        assert bundle.agent.invoke({"input": "hi"}, config={"configurable": {"session_id": "s"}}) == {"output": "final"}
    finally:
        pipeline._executor.close()
