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

Ask a question and get a cited answer:

```bash
uv run python -m rag.generate "Can I get a refund on a yearly plan?"
```

The answer model is chosen automatically:

- **Claude** if `ANTHROPIC_API_KEY` is set. Each retrieved chunk is sent as a separate document with citations enabled, so the `[n]` markers point at the exact passages used.
- **A local model via [Ollama](https://ollama.com)** otherwise, so no API key is needed. Install Ollama and run `ollama pull llama3.2:3b` once. Chunks are numbered in the prompt and the model cites them as `[n]`.

Force one with `LLM_PROVIDER=claude` or `LLM_PROVIDER=ollama` in `.env`, or `--provider` on the command line.

Supported files: `.pdf`, `.docx`, `.txt`, `.md`. Re-running skips unchanged files, and a changed file replaces its old chunks. Use `--reset` to start over.

The `sample_data/` documents describe **Nimbus Notes**, a fictional app written for this demo. They include error codes like `NB-4012` that show where keyword search beats semantic search.

## License

MIT
