"""End-to-end check with the real local models and an in-memory Qdrant.

Downloads the embedding and reranker models (~200 MB) on first run.
Skip with: pytest -m "not integration"
"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from rag.ingest import ingest_file, iter_files
from rag.retrieve import MODES, search

pytestmark = pytest.mark.integration
SAMPLE = Path(__file__).parent.parent / "sample_data"


@pytest.fixture(scope="module")
def client():
    c = QdrantClient(":memory:")
    for f in iter_files([SAMPLE]):
        assert ingest_file(f, client=c) > 0
    yield c
    c.close()


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_results(client, mode):
    hits = search("How much does the Pro plan cost?", mode=mode, top_k=3, client=client)
    assert len(hits) == 3


def test_error_code_found_by_hybrid_rerank(client):
    hits = search("What does error NB-4012 mean?", top_k=3, client=client)
    assert "NB-4012" in hits[0].text
    assert {"rrf", "rerank"} <= set(hits[0].stages)


def test_source_filter(client):
    hits = search("price", source="nimbus_user_guide.md", client=client)
    assert hits and all(h.source == "nimbus_user_guide.md" for h in hits)


def test_reingest_unchanged_is_skipped(client):
    assert ingest_file(SAMPLE / "nimbus_user_guide.md", client=client) == 0
