# Hybrid RAG

Ask questions about your own documents and get answers with citations.

Retrieval is **hybrid**: semantic search (embeddings) and keyword search (BM25) run together, their results are merged with Reciprocal Rank Fusion, and a cross-encoder reranks the best candidates before Claude writes the answer.

> 🚧 Work in progress. See [PLAN.md](PLAN.md) for the roadmap.

## Quick start

```bash
uv sync
cp .env.example .env   # then add your ANTHROPIC_API_KEY
uv run pytest

# Index the demo documents (downloads small local models on first run)
uv run python -m rag.ingest sample_data
```

Search from the command line (modes: `semantic`, `keyword`, `hybrid`, `hybrid_rerank`):

```bash
uv run python -m rag.retrieve "What does error NB-4012 mean?"
uv run python -m rag.retrieve "What does error NB-4012 mean?" --mode keyword
```

Each result shows its score from every stage (semantic, keyword, RRF, rerank).

Ask a question and get a cited answer from Claude (needs `ANTHROPIC_API_KEY` in `.env`):

```bash
uv run python -m rag.generate "Can I get a refund on a yearly plan?"
```

Each retrieved chunk is sent to Claude as a separate document with citations enabled, so the `[n]` markers in the answer point at the exact passages used.

Supported files: `.pdf`, `.docx`, `.txt`, `.md`. Re-running skips unchanged files, and a changed file replaces its old chunks. Use `--reset` to start over.

The `sample_data/` documents describe **Nimbus Notes**, a fictional app written for this demo. They include error codes like `NB-4012` that show where keyword search beats semantic search.

## License

MIT
