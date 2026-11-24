from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from rag.ingest import ingest_file, iter_files

SAMPLE = Path(__file__).parent.parent / "sample_data"


@pytest.fixture(scope="session")
def sample_index():
    """The sample documents indexed into an in-memory Qdrant (built once per test run)."""
    client = QdrantClient(":memory:")
    for f in iter_files([SAMPLE]):
        assert ingest_file(f, client=client) > 0
    yield client
    client.close()
