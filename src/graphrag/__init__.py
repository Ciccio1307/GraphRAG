"""GraphRAG core package.

The package is split into small, side-effect-free modules so every piece can be
unit tested without a live Neo4j instance or an LLM API key:

* :mod:`graphrag.config`        – typed, validated settings loaded from the environment
* :mod:`graphrag.cypher_guard`  – static read-only validation of LLM-generated Cypher
* :mod:`graphrag.executor`      – bounded, read-only Cypher execution on Neo4j
* :mod:`graphrag.graph_qa`      – NL question -> Cypher -> records -> answer pipeline
* :mod:`graphrag.hybrid`        – graph + web retrieval and final answer synthesis
* :mod:`graphrag.memory`        – bounded, thread-safe chat history store
* :mod:`graphrag.sanitize`      – input normalisation and output sanitisation
* :mod:`graphrag.factory`       – wiring of real external services (the only I/O-bound module)
"""

__all__ = ["__version__"]

__version__ = "2.0.0"
