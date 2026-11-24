"""Streamlit chat UI for the hybrid RAG app.

Run with:  uv run streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from rag.config import get_settings
from rag.generate import LLMError, answer, resolve_provider
from rag.ingest import SUPPORTED, delete_document, ingest_file, iter_files, list_documents
from rag.retrieve import MODES, search
from rag.store import dense_model, get_client, rerank_model, reset_collection, sparse_model

SAMPLE_DIR = Path(__file__).parent / "sample_data"
MODE_LABELS = {
    "hybrid_rerank": "Hybrid + rerank (best)",
    "hybrid": "Hybrid (semantic + keyword)",
    "semantic": "Semantic only",
    "keyword": "Keyword (BM25) only",
}
HISTORY_TURNS = 3  # earlier Q&A pairs sent along for follow-up questions

st.set_page_config(page_title="Hybrid RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading search models (first run downloads them)...")
def load_models() -> bool:
    # Assigned so Streamlit's "magic" doesn't render the model objects on the page.
    _ = (dense_model(), sparse_model(), rerank_model())
    return True


@st.cache_resource
def qdrant():
    # Local-mode Qdrant allows one client per folder, so the whole app shares this one.
    return get_client()


load_models()
client = qdrant()
settings = get_settings()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- sidebar: model, documents, search settings ----------

with st.sidebar:
    st.title("🔎 Hybrid RAG")

    try:
        provider = resolve_provider()
    except LLMError as e:
        st.error(str(e))
        st.stop()
    if provider == "claude":
        st.success(f"Answers by **Claude** (`{settings.claude_model}`)")
    else:
        st.info(
            f"Answers by local model **`{settings.ollama_model}`** via Ollama. "
            "Add `ANTHROPIC_API_KEY` to `.env` to use Claude."
        )

    st.subheader("Documents")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader(
            "Add files",
            type=[s.lstrip(".") for s in sorted(SUPPORTED)],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Index files", width="stretch")
    if submitted and files:
        with st.spinner("Indexing..."), tempfile.TemporaryDirectory() as tmp:
            for f in files:
                path = Path(tmp) / f.name
                path.write_bytes(f.getvalue())
                try:
                    n = ingest_file(path, source=f.name, client=client)
                    st.toast(f"{f.name}: {n} chunks" if n else f"{f.name}: unchanged")
                except Exception as e:  # show bad files instead of crashing the app
                    st.error(f"Could not index {f.name}: {e}")

    docs = list_documents(client)
    if not docs:
        st.caption("No documents indexed yet.")
        if st.button("Load sample documents", width="stretch"):
            with st.spinner("Indexing sample documents..."):
                for f in iter_files([SAMPLE_DIR]):
                    ingest_file(f, client=client)
            st.rerun()
    for source, count in docs.items():
        col1, col2 = st.columns([5, 1])
        col1.markdown(f"📄 {source}  \n<small>{count} chunks</small>", unsafe_allow_html=True)
        if col2.button("✕", key=f"del-{source}", help=f"Remove {source}"):
            delete_document(source, client=client)
            st.rerun()
    if docs and st.button("Remove all documents", width="stretch"):
        reset_collection(client)
        st.rerun()

    st.subheader("Search")
    mode = st.selectbox("Retrieval mode", MODES, format_func=MODE_LABELS.get)
    top_k = st.slider("Passages sent to the model", 1, 10, settings.top_k)
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# ---------- chat ----------


def show_sources(msg: dict) -> None:
    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for n, label, text in msg["sources"]:
                st.markdown(f"**[{n}] {label}**")
                st.caption(text)
    if msg.get("retrieved"):
        with st.expander(f"Retrieved passages and scores ({len(msg['retrieved'])})"):
            for i, (label, stages, preview) in enumerate(msg["retrieved"], 1):
                scores = " · ".join(f"{k} {v:.3f}" for k, v in stages.items())
                st.markdown(f"**{i}. {label}**  \n`{scores}`")
                st.caption(preview)
    if msg.get("model"):
        st.caption(f"Answered by {msg['model']}")


st.header("Ask your documents")
if not docs:
    st.info("Add files or load the sample documents from the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg)

if question := st.chat_input("Ask a question...", disabled=not docs):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Earlier turns as plain (question, answer) pairs for follow-up questions.
    past = st.session_state.messages[:-1]
    history = [
        (past[i]["content"], past[i + 1]["content"])
        for i in range(0, len(past) - 1, 2)
        if past[i]["role"] == "user" and past[i + 1]["role"] == "assistant"
    ][-HISTORY_TURNS:]

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            hits = search(question, mode=mode, top_k=top_k, client=client)

        placeholder = st.empty()
        streamed: list[str] = []

        def on_text(piece: str) -> None:
            streamed.append(piece)
            placeholder.markdown("".join(streamed) + " ▌")

        try:
            with st.spinner("Writing answer..."):
                result = answer(question, hits, history=history, on_text=on_text)
        except LLMError as e:
            placeholder.empty()
            st.error(str(e))
            st.session_state.messages.pop()  # let the user retry the same question
            st.stop()

        placeholder.markdown(result.text)
        msg = {
            "role": "assistant",
            "content": result.text,
            "sources": [(n, h.label, h.text) for n, h in result.cited_hits],
            "retrieved": [(h.label, h.stages, h.text[:300]) for h in hits],
            "model": result.model,
        }
        show_sources(msg)
    st.session_state.messages.append(msg)
