"""Qdrant storage and the local embedding models shared by ingestion and retrieval."""

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from rag.config import get_settings

DENSE = "dense"
SPARSE = "bm25"


@lru_cache
def get_client() -> QdrantClient:
    """Qdrant in local mode: data lives on disk, no server needed."""
    path = get_settings().qdrant_path
    path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(path))
    except RuntimeError as e:
        if "already accessed" not in str(e):
            raise
        raise SystemExit(
            f"The index in {path} is in use by another process (probably the Streamlit app). "
            "Local-mode Qdrant allows one process at a time; stop the other one and retry."
        ) from e


@lru_cache
def dense_model() -> TextEmbedding:
    s = get_settings()
    return TextEmbedding(s.dense_model, cache_dir=str(s.model_cache_dir))


@lru_cache
def sparse_model() -> SparseTextEmbedding:
    s = get_settings()
    return SparseTextEmbedding(s.sparse_model, cache_dir=str(s.model_cache_dir))


@lru_cache
def rerank_model() -> TextCrossEncoder:
    s = get_settings()
    return TextCrossEncoder(s.rerank_model, cache_dir=str(s.model_cache_dir))


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection with one dense and one BM25 sparse vector per point."""
    name = get_settings().collection_name
    if client.collection_exists(name):
        return
    dim = len(next(iter(dense_model().embed(["dimension probe"]))))
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        # IDF modifier makes Qdrant compute proper BM25 scores across the collection.
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def reset_collection(client: QdrantClient) -> None:
    """Remove every stored chunk.

    Deletes points rather than the collection: in local mode on Windows, dropping and
    recreating a collection can reopen the old on-disk data.
    """
    ensure_collection(client)
    client.delete(
        get_settings().collection_name,
        points_selector=models.FilterSelector(filter=models.Filter()),
    )
