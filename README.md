# Hybrid RAG

Ask questions about your own documents and get answers with citations.

Retrieval is **hybrid**: semantic search (embeddings) and keyword search (BM25) run together, their results are merged with Reciprocal Rank Fusion, and a cross-encoder reranks the best candidates before Claude writes the answer.

> 🚧 Work in progress. See [PLAN.md](PLAN.md) for the roadmap.

## Web app

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501. From the sidebar you can upload files or load the sample documents, remove documents, and switch retrieval mode. Answers stream in, with expandable **Sources** (the passages the answer cites) and **Retrieved passages and scores** (every chunk the search returned, with its semantic, keyword, RRF and rerank scores).

## Does hybrid search help? (evaluation)

[eval.py](eval.py) runs 22 labelled questions ([eval_questions.json](eval_questions.json)) against the sample documents in every retrieval mode. Each question names a phrase the correct passage contains. No language model is involved, so results are the same on every run.

```bash
uv run python eval.py
```

| Mode | Hit@1 | Hit@3 | MRR | Avg search time |
|---|---|---|---|---|
| semantic | 64% | 91% | 0.78 | 9 ms |
| keyword | 68% | 82% | 0.75 | 1 ms |
| hybrid | 77% | 95% | 0.87 | 9 ms |
| **hybrid_rerank** | **82%** | **100%** | **0.91** | 1063 ms |

Split by question type:

| Mode | Keyword questions (10) Hit@1 | Paraphrased questions (12) Hit@1 |
|---|---|---|
| semantic | 80% | 50% |
| keyword | 100% | 42% |
| hybrid | 100% | 58% |
| hybrid_rerank | 100% | 67% |

- **Keyword search** is perfect on exact terms like error codes (`NB-4012`) but weak when the question uses different words than the document.
- **Semantic search** handles paraphrases better but misses some exact codes.
- **Hybrid** gets the best of both, and **reranking** puts the right passage first more often, at the cost of about one second per query on a CPU.

The sample set is small (3 documents, 14 chunks), so treat these numbers as a demonstration rather than a benchmark. Full tables: [eval_results.md](eval_results.md).

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
