# Hybrid RAG

Ask questions about your own documents and get answers with citations.

Retrieval is **hybrid**: semantic search (embeddings) and keyword search (BM25) run together, their results are merged with Reciprocal Rank Fusion, and a cross-encoder reranks the best candidates before Claude writes the answer.

> 🚧 Work in progress. See [PLAN.md](PLAN.md) for the roadmap.

## Quick start

```bash
uv sync
cp .env.example .env   # then add your ANTHROPIC_API_KEY
uv run pytest
```

## License

MIT
