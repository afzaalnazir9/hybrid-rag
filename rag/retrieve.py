"""Hybrid retrieval: semantic + keyword search, Reciprocal Rank Fusion, cross-encoder rerank.

Usage:
    python -m rag.retrieve "How do I fix error NB-4012?"
    python -m rag.retrieve "How do I fix error NB-4012?" --mode keyword
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from qdrant_client import QdrantClient, models

from rag.config import get_settings
from rag.store import DENSE, SPARSE, dense_model, get_client, rerank_model, sparse_model

Mode = Literal["semantic", "keyword", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("semantic", "keyword", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int
    score: float
    # Score from each stage the hit went through, e.g. {"semantic": 0.71, "rrf": 0.032}.
    stages: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.source} (p. {self.page})" if self.page else self.source


# ---------- individual searches ----------


def _to_hits(points: list[models.ScoredPoint], stage: str) -> list[Hit]:
    hits = []
    for p in points:
        payload = p.payload or {}
        hits.append(
            Hit(
                id=str(p.id),
                text=payload["text"],
                source=payload["source"],
                page=payload.get("page"),
                chunk_index=payload["chunk_index"],
                score=p.score,
                stages={stage: p.score},
            )
        )
    return hits


def semantic_search(
    client: QdrantClient, query: str, limit: int, flt: models.Filter | None = None
) -> list[Hit]:
    vector = next(iter(dense_model().query_embed(query))).tolist()
    res = client.query_points(
        get_settings().collection_name, query=vector, using=DENSE, limit=limit, query_filter=flt
    )
    return _to_hits(res.points, "semantic")


def keyword_search(
    client: QdrantClient, query: str, limit: int, flt: models.Filter | None = None
) -> list[Hit]:
    sp = next(iter(sparse_model().query_embed(query)))
    vector = models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
    res = client.query_points(
        get_settings().collection_name, query=vector, using=SPARSE, limit=limit, query_filter=flt
    )
    return _to_hits(res.points, "keyword")


# ---------- fusion and rerank ----------


def rrf_fuse(result_lists: Sequence[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d)).

    Uses only ranks, so the very different score scales of cosine similarity and
    BM25 don't need to be normalised. Stage scores from every list are kept.
    """
    fused: dict[str, Hit] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            if hit.id not in fused:
                fused[hit.id] = Hit(**{**hit.__dict__, "score": 0.0, "stages": {}})
            entry = fused[hit.id]
            entry.score += 1.0 / (k + rank)
            entry.stages.update(hit.stages)
    out = sorted(fused.values(), key=lambda h: h.score, reverse=True)
    for h in out:
        h.stages["rrf"] = h.score
    return out


def rerank(
    query: str,
    hits: list[Hit],
    scorer: Callable[[str, list[str]], list[float]] | None = None,
) -> list[Hit]:
    """Re-score (query, chunk) pairs with a cross-encoder and sort by that score."""
    if not hits:
        return []
    if scorer is None:

        def scorer(q: str, docs: list[str]) -> list[float]:
            return [float(s) for s in rerank_model().rerank(q, docs)]

    scores = scorer(query, [h.text for h in hits])
    for h, s in zip(hits, scores, strict=True):
        h.score = s
        h.stages["rerank"] = s
    return sorted(hits, key=lambda h: h.score, reverse=True)


# ---------- public entry point ----------


def search(
    query: str,
    mode: Mode = "hybrid_rerank",
    top_k: int | None = None,
    source: str | None = None,
    client: QdrantClient | None = None,
) -> list[Hit]:
    """Return the best chunks for `query`. `source` limits results to one document."""
    s = get_settings()
    client = client or get_client()
    top_k = top_k or s.top_k
    if not client.collection_exists(s.collection_name):
        return []

    flt = None
    if source:
        flt = models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
        )

    n = s.candidates_per_search
    if mode == "semantic":
        return semantic_search(client, query, top_k, flt)
    if mode == "keyword":
        return keyword_search(client, query, top_k, flt)

    fused = rrf_fuse(
        [semantic_search(client, query, n, flt), keyword_search(client, query, n, flt)],
        k=s.rrf_k,
    )
    if mode == "hybrid":
        return fused[:top_k]
    if mode == "hybrid_rerank":
        return rerank(query, fused[: s.rerank_candidates])[:top_k]
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles

    client = get_client()
    try:
        hits = search(args.query, mode=args.mode, top_k=args.top_k, client=client)
    finally:
        client.close()
    if not hits:
        print("No results. Did you run `python -m rag.ingest sample_data` first?")
    for i, h in enumerate(hits, 1):
        stages = ", ".join(f"{k}={v:.3f}" for k, v in h.stages.items())
        preview = h.text[:160].replace("\n", " ")
        print(f"{i}. {h.label} #{h.chunk_index}  [{stages}]\n   {preview}...\n")


if __name__ == "__main__":
    main()
