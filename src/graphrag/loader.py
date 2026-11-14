"""Load the nodes/edges CSV dataset (neo4j-admin import format) into Neo4j.

* values like ```Bsg``` are stripped of their wrapping backticks, otherwise
  ``lower(g.word) = "bsg"`` (as used by the few-shot prompts) never matches;
* labels and relationship types come from the data file and are therefore
  untrusted: they are always backtick-quoted with :func:`quote_identifier`,
  never interpolated raw (Cypher injection via a crafted label);
* rows are written in batches with ``UNWIND`` (one round-trip per batch rather
  than one per row), and edges are linked through ``elementId`` look-ups.
"""

from __future__ import annotations

import csv
import io
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

MAX_IDENTIFIER_CHARS = 256


@dataclass(frozen=True)
class NodeRow:
    id: int
    labels: tuple[str, ...]
    word: str


@dataclass(frozen=True)
class EdgeRow:
    src: int
    dst: int
    type: str
    verb: str


class DatasetError(ValueError):
    """Raised for malformed dataset rows."""


def quote_identifier(name: str) -> str:
    """Return ``name`` as a safely backtick-quoted Cypher identifier."""
    if not isinstance(name, str) or not name.strip():
        raise DatasetError("Identifier must be a non-empty string")
    if len(name) > MAX_IDENTIFIER_CHARS:
        raise DatasetError(f"Identifier longer than {MAX_IDENTIFIER_CHARS} chars")
    if any(ord(c) < 32 for c in name) or "\u0000" in name:
        raise DatasetError("Identifier contains control characters")
    return "`" + name.replace("`", "``") + "`"


def strip_backticks(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == "`":
        value = value[1:-1]
    return value


def _reader(source: str | Path | io.TextIOBase) -> Iterator[dict[str, str]]:
    if isinstance(source, io.TextIOBase):
        yield from csv.DictReader(source, delimiter="\t")
        return
    path = Path(source)
    with path.open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh, delimiter="\t")


def parse_nodes(source: str | Path | io.TextIOBase) -> list[NodeRow]:
    rows: list[NodeRow] = []
    seen: set[int] = set()
    for lineno, row in enumerate(_reader(source), start=2):
        if None in row.values():
            raise DatasetError(f"nodes line {lineno}: missing columns")
        try:
            node_id = int(row["id:ID"])
            raw_labels = row["label:LABEL"] or ""
            word = strip_backticks(row["word"] or "")
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"nodes line {lineno}: malformed row") from exc
        labels = tuple(dict.fromkeys(lbl.strip() for lbl in raw_labels.split("|") if lbl.strip()))
        if not labels:
            raise DatasetError(f"nodes line {lineno}: node {node_id} has no labels")
        for label in labels:
            quote_identifier(label)  # validates
        if node_id in seen:
            raise DatasetError(f"nodes line {lineno}: duplicate id {node_id}")
        seen.add(node_id)
        rows.append(NodeRow(node_id, labels, word))
    return rows


def parse_edges(source: str | Path | io.TextIOBase, *, node_ids: set[int] | None = None) -> list[EdgeRow]:
    rows: list[EdgeRow] = []
    for lineno, row in enumerate(_reader(source), start=2):
        if None in row.values():
            raise DatasetError(f"edges line {lineno}: missing columns")
        try:
            edge = EdgeRow(
                src=int(row["src:START_ID"]),
                dst=int(row["dst:END_ID"]),
                type=(row["type:TYPE"] or "").strip(),
                verb=strip_backticks(row.get("verb") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"edges line {lineno}: malformed row") from exc
        quote_identifier(edge.type)
        if node_ids is not None and (edge.src not in node_ids or edge.dst not in node_ids):
            raise DatasetError(f"edges line {lineno}: dangling edge {edge.src}->{edge.dst}")
        rows.append(edge)
    return rows


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_graph(
    driver: Any,
    nodes: Sequence[NodeRow],
    edges: Sequence[EdgeRow],
    *,
    database: str | None = None,
    batch_size: int = 500,
    wipe: bool = False,
) -> dict[str, int]:
    """Write ``nodes`` and ``edges``; returns counts. Refuses a non-empty DB unless ``wipe``."""
    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        existing = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        if existing and not wipe:
            raise DatasetError(f"Database already contains {existing} nodes; use wipe=True")
        if wipe:
            session.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS").consume()

        by_labels: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            by_labels[node.labels].append({"id": node.id, "word": node.word})

        element_ids: dict[int, str] = {}
        for labels, rows in by_labels.items():
            label_expr = ":".join(quote_identifier(lbl) for lbl in labels)
            query = (
                f"UNWIND $rows AS r CREATE (n:{label_expr}) SET n.id = r.id, n.word = r.word "
                "RETURN r.id AS id, elementId(n) AS eid"
            )
            for batch in _batches(rows, batch_size):
                for rec in session.run(query, rows=list(batch)):
                    element_ids[rec["id"]] = rec["eid"]

        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            by_type[edge.type].append(
                {"src": element_ids[edge.src], "dst": element_ids[edge.dst], "verb": edge.verb}
            )
        for rel_type, rows in by_type.items():
            query = (
                "UNWIND $rows AS r "
                "MATCH (a) WHERE elementId(a) = r.src "
                "MATCH (b) WHERE elementId(b) = r.dst "
                f"CREATE (a)-[:{quote_identifier(rel_type)} {{verb: r.verb}}]->(b)"
            )
            for batch in _batches(rows, batch_size):
                session.run(query, rows=list(batch)).consume()

    return {"nodes": len(element_ids), "edges": sum(len(v) for v in by_type.values())}


def resolve_csv_source(value: str) -> str:
    """Turn a GitHub ``/blob/`` URL into its raw-content URL; leave paths untouched."""
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme in ("http", "https") and parsed.netloc == "github.com" and "/blob/" in parsed.path:
        owner_repo, _, rest = parsed.path.lstrip("/").partition("/blob/")
        return f"https://raw.githubusercontent.com/{owner_repo}/{rest}"
    return value
