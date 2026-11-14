"""Read-only, bounded execution of validated Cypher against Neo4j."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .cypher_guard import validate_read_only_cypher

Record = dict[str, Any]


@runtime_checkable
class CypherExecutor(Protocol):
    """Anything that can run a read-only Cypher query and return records."""

    def run(self, cypher: str, params: Mapping[str, Any] | None = None) -> list[Record]:
        ...


class Neo4jReadOnlyExecutor:
    """Executes Cypher with defence in depth.

    1. the statement is statically validated by :func:`validate_read_only_cypher`;
    2. it runs inside a managed **read** transaction (``execute_read``), so the
       server rejects any write with ``Neo.ClientError.Statement.AccessMode``;
    3. a server-side transaction timeout bounds runaway queries;
    4. records are streamed and only the first ``max_records`` are materialised
       (the previous implementation fetched *all* rows and sliced afterwards).
    """

    def __init__(
        self,
        driver: Any,
        *,
        database: str | None = None,
        max_records: int = 100,
        timeout_s: float = 15.0,
    ) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._driver = driver
        self._database = database
        self._max_records = max_records
        self._timeout_s = timeout_s

    @property
    def max_records(self) -> int:
        return self._max_records

    def run(self, cypher: str, params: Mapping[str, Any] | None = None) -> list[Record]:
        import neo4j  # local import keeps the module importable without the driver

        query = validate_read_only_cypher(cypher)
        limit = self._max_records

        @neo4j.unit_of_work(timeout=self._timeout_s)
        def _work(tx: Any) -> list[Record]:
            result = tx.run(query, dict(params or {}))
            records: list[Record] = []
            for record in result:
                records.append(record.data())
                if len(records) >= limit:
                    break
            return records

        session_kwargs: dict[str, Any] = {"default_access_mode": neo4j.READ_ACCESS}
        if self._database:
            session_kwargs["database"] = self._database
        with self._driver.session(**session_kwargs) as session:
            return session.execute_read(_work)

    def close(self) -> None:
        self._driver.close()
