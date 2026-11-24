# Hybrid RAG System — Project Plan (v2, reduced scope)

> Status: **built**. All seven steps are done; see [README.md](README.md). Changes from this plan:
> - Reranker is `ms-marco-MiniLM-L-12-v2` (~120 MB) instead of `bge-reranker-base` (~1 GB).
> - Answers fall back to a local Ollama model (`llama3.2:3b`) when no Claude API key is set.

## 1. What we're building

A small, clean app that answers questions about your documents (PDF, DOCX, TXT, Markdown) with **cited answers**.

It's "hybrid" because it searches in two ways and combines them:

1. **Semantic search** (embeddings): finds passages with the same *meaning*.
2. **Keyword search** (BM25): finds passages with the exact *words*, names, IDs.
3. **Fusion** (Reciprocal Rank Fusion): merges the two result lists.
4. **Rerank**: a small cross-encoder model reorders the top results.
5. **Answer**: Claude writes the answer using only those passages, with citations.

## 2. How it works

```
Upload docs → split into chunks → embed (semantic + keyword) → store in Qdrant

Question → semantic search ┐
                           ├─► RRF fusion → rerank top 20 → top 5
         → keyword search  ┘                                  │
                                                              ▼
                                    Claude answer + citations → UI
```

## 3. Tech stack

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.12+ | Standard for RAG |
| Vector store | **Qdrant in local mode** (on-disk, no Docker or server needed) | Supports semantic + keyword vectors and fusion together |
| Embeddings + BM25 | **FastEmbed** (`bge-small-en-v1.5` + `Qdrant/bm25`) | Free, runs locally on CPU |
| Reranker | FastEmbed cross-encoder (`bge-reranker-base`) | Clear quality boost, a few lines of code |
| LLM | Claude via official `anthropic` SDK, built-in citations | Reliable source citations |
| UI | **Streamlit** | Chat + upload in one simple file |
| Tooling | `uv`, `pytest`, `ruff`, GitHub Actions | Clean repo |

Not included: no LangChain, no separate frontend app, no database server, no auth, no agents.

## 4. Repository layout

```
rag-app/
├── rag/
│   ├── config.py      # settings from .env
│   ├── ingest.py      # load files + chunk + embed + store
│   ├── retrieve.py    # semantic, keyword, fusion, rerank
│   └── generate.py    # Claude call with citations
├── app.py             # Streamlit UI (upload, chat, show sources)
├── eval.py            # small comparison: semantic vs keyword vs hybrid
├── sample_data/       # a few demo documents
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

## 5. Steps

1. **Setup**: git, `uv` project, `.env.example`, lint/test config.
2. **Ingestion**: read PDF/DOCX/TXT/MD, split into overlapping chunks, store with semantic + keyword vectors.
3. **Hybrid retrieval**: both searches, RRF fusion, reranking.
4. **Answer generation**: Claude with citations and an "I couldn't find this in the documents" fallback.
5. **Streamlit UI**: upload files, chat, expandable sources under each answer.
6. **Mini evaluation**: ~20 test questions; report hit rate for semantic vs keyword vs hybrid vs hybrid + rerank.
7. **README + CI**: diagram, screenshot, results table, quick start.

Estimated time: **about 3–4 days**.

## 6. Open choices (defaults in bold)

- LLM model: **Claude Opus 5.5** (highest quality) or Claude Sonnet 5.5 (cheaper). You need an Anthropic API key either way, and the model can be changed later in `.env`.
- Demo documents: **a few public-domain docs I choose**, or your own.
