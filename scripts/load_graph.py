#!/usr/bin/env python3
"""Load data/nodes.csv and data/edges.csv into the Neo4j instance configured in .env.

Usage:
    python scripts/load_graph.py [--nodes data/nodes.csv] [--edges data/edges.csv] [--wipe]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from graphrag.config import Settings  # noqa: E402
from graphrag.loader import load_graph, parse_edges, parse_nodes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=ROOT / "data" / "nodes.csv")
    parser.add_argument("--edges", type=Path, default=ROOT / "data" / "edges.csv")
    parser.add_argument("--wipe", action="store_true", help="delete all existing nodes first")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    settings = Settings.from_env()

    import neo4j

    nodes = parse_nodes(args.nodes)
    edges = parse_edges(args.edges, node_ids={n.id for n in nodes})
    with neo4j.GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    ) as driver:
        counts = load_graph(driver, nodes, edges, database=settings.neo4j_database, wipe=args.wipe)
    print(f"Loaded {counts['nodes']} nodes and {counts['edges']} relationships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
