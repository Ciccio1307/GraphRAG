from __future__ import annotations

import pytest

from graphrag.config import DEFAULT_MODEL, ConfigError, Settings

BASE = {"NEO4J_URI": "bolt://localhost:7687", "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "pw"}


def test_defaults():
    s = Settings.from_env(BASE)
    assert s.qa_model == s.cypher_model == s.agent_model == DEFAULT_MODEL
    assert s.top_k == 100 and s.enable_web_search is True and s.neo4j_database is None


def test_password_not_in_repr():
    s = Settings.from_env({**BASE, "NEO4J_PASSWORD": "super-secret-value"})
    assert "super-secret-value" not in repr(s)


@pytest.mark.parametrize("missing", ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"])
def test_missing_required(missing):
    env = {k: v for k, v in BASE.items() if k != missing}
    with pytest.raises(ConfigError, match=missing):
        Settings.from_env(env)


def test_blank_required_counts_as_missing():
    with pytest.raises(ConfigError, match="NEO4J_PASSWORD"):
        Settings.from_env({**BASE, "NEO4J_PASSWORD": "   "})


def test_quoted_values_from_dotenv_style_are_unwrapped():
    s = Settings.from_env({"NEO4J_URI": ' "neo4j+s://abc.databases.neo4j.io" ', "NEO4J_USER": "'u'", "NEO4J_PASSWORD": '"p w"'})
    assert s.neo4j_uri == "neo4j+s://abc.databases.neo4j.io"
    assert s.neo4j_user == "u" and s.neo4j_password == "p w"


@pytest.mark.parametrize("uri", ["http://localhost:7474", "file:///etc/passwd", "localhost:7687", "javascript:alert(1)", ""])
def test_invalid_uri_scheme(uri):
    with pytest.raises(ConfigError):
        Settings.from_env({**BASE, "NEO4J_URI": uri})


@pytest.mark.parametrize("uri", ["bolt://h:1", "bolt+s://h", "bolt+ssc://h", "neo4j://h", "neo4j+s://h", "NEO4J+S://h"])
def test_valid_uri_schemes(uri):
    assert Settings.from_env({**BASE, "NEO4J_URI": uri}).neo4j_uri == uri


@pytest.mark.parametrize(
    "name,value",
    [
        ("GRAPHRAG_TOP_K", "abc"),
        ("GRAPHRAG_TOP_K", "0"),
        ("GRAPHRAG_TOP_K", "-5"),
        ("GRAPHRAG_TOP_K", "99999999"),
        ("GRAPHRAG_QUERY_TIMEOUT_S", "nan"),
        ("GRAPHRAG_QUERY_TIMEOUT_S", "inf"),
        ("GRAPHRAG_QUERY_TIMEOUT_S", "0"),
        ("GRAPHRAG_ENABLE_WEB_SEARCH", "maybe"),
        ("GRAPHRAG_MAX_SESSIONS", "1.5"),
    ],
)
def test_invalid_numeric_and_bool(name, value):
    with pytest.raises(ConfigError, match=name):
        Settings.from_env({**BASE, name: value})


@pytest.mark.parametrize("raw,expected", [("true", True), ("1", True), ("YES", True), ("off", False), ("0", False), ("False", False)])
def test_bool_parsing(raw, expected):
    assert Settings.from_env({**BASE, "GRAPHRAG_ENABLE_WEB_SEARCH": raw}).enable_web_search is expected


def test_overrides():
    s = Settings.from_env(
        {**BASE, "GRAPHRAG_TOP_K": "5", "GRAPHRAG_QUERY_TIMEOUT_S": "2.5", "GRAPHRAG_AGENT_MODEL": "m", "NEO4J_DATABASE": "bio"}
    )
    assert (s.top_k, s.query_timeout_s, s.agent_model, s.neo4j_database) == (5, 2.5, "m", "bio")


def test_settings_are_immutable(settings):
    with pytest.raises(Exception):
        settings.top_k = 1  # type: ignore[misc]


def test_from_env_reads_os_environ(monkeypatch):
    for k, v in BASE.items():
        monkeypatch.setenv(k, v)
    assert Settings.from_env().neo4j_user == "neo4j"
