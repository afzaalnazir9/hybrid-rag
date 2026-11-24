"""Load documents, split them into chunks, embed them, and store them in Qdrant.

Usage:
    python -m rag.ingest sample_data/          # ingest a folder (or single files)
    python -m rag.ingest sample_data/ --reset  # wipe the collection first
"""

import argparse
import hashlib
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from rag.config import get_settings
from rag.store import DENSE, SPARSE, dense_model, ensure_collection, get_client, sparse_model

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Section:
    """A piece of a document before chunking (a PDF page, or a whole text file)."""

    text: str
    page: int | None = None


# ---------- loading ----------


def load_document(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return [Section(p.extract_text() or "", page=i) for i, p in enumerate(reader.pages, 1)]
    if suffix == ".docx":
        import docx

        doc = docx.Document(str(path))
        return [Section("\n\n".join(p.text for p in doc.paragraphs))]
    if suffix in {".txt", ".md"}:
        return [Section(path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type: {path.suffix}")


# ---------- chunking ----------


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~`size`-character chunks that overlap by ~`overlap` characters.

    Chunks end at the nicest boundary found in the second half of the window:
    paragraph break, then line break, then sentence end, then space.
    """
    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep, size // 2)
                if cut != -1:
                    end = start + cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # Step back for overlap, then move forward to the next word start.
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


# ---------- storing ----------


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_filter(source: str, doc_hash: str | None = None) -> models.Filter:
    must = [models.FieldCondition(key="source", match=models.MatchValue(value=source))]
    if doc_hash:
        must.append(models.FieldCondition(key="doc_hash", match=models.MatchValue(value=doc_hash)))
    return models.Filter(must=must)


def ingest_file(path: Path, source: str | None = None, client: QdrantClient | None = None) -> int:
    """Ingest one file. Returns the number of chunks stored (0 if unchanged since last time).

    `source` is the name shown in citations; it defaults to the file name. Re-ingesting
    a changed file with the same source name replaces its old chunks.
    """
    s = get_settings()
    client = client or get_client()
    ensure_collection(client)
    source = source or path.name
    doc_hash = file_hash(path)

    already = client.count(s.collection_name, count_filter=_source_filter(source, doc_hash))
    if already.count > 0:
        return 0
    client.delete(s.collection_name, points_selector=_source_filter(source))

    records = []  # (text, page)
    for section in load_document(path):
        for chunk in chunk_text(section.text, s.chunk_size, s.chunk_overlap):
            records.append((chunk, section.page))
    if not records:
        return 0

    texts = [t for t, _ in records]
    dense = list(dense_model().passage_embed(texts))
    sparse = list(sparse_model().passage_embed(texts))

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_hash}:{i}")),
            vector={
                DENSE: d.tolist(),
                SPARSE: models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
            },
            payload={
                "text": text,
                "source": source,
                "page": page,
                "chunk_index": i,
                "doc_hash": doc_hash,
            },
        )
        for i, ((text, page), d, sp) in enumerate(zip(records, dense, sparse, strict=True))
    ]
    client.upsert(s.collection_name, points=points)
    return len(points)


def list_documents(client: QdrantClient | None = None) -> dict[str, int]:
    """Map each stored source name to its number of chunks."""
    s = get_settings()
    client = client or get_client()
    if not client.collection_exists(s.collection_name):
        return {}
    counts: dict[str, int] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            s.collection_name, limit=256, offset=offset, with_payload=["source"]
        )
        for p in points:
            src = (p.payload or {}).get("source", "?")
            counts[src] = counts.get(src, 0) + 1
        if offset is None:
            return dict(sorted(counts.items()))


def delete_document(source: str, client: QdrantClient | None = None) -> None:
    client = client or get_client()
    client.delete(get_settings().collection_name, points_selector=_source_filter(source))


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED)
        elif p.suffix.lower() in SUPPORTED:
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the hybrid RAG index.")
    parser.add_argument("paths", nargs="+", type=Path, help="files or folders to ingest")
    parser.add_argument("--reset", action="store_true", help="delete all stored chunks first")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles

    files = iter_files(args.paths)
    if not files:
        print("No supported files found (.pdf, .docx, .txt, .md).")
        return

    client = get_client()
    try:
        if args.reset:
            from rag.store import reset_collection

            reset_collection(client)
        for f in files:
            n = ingest_file(f, client=client)
            print(f"{f.name}: {n} chunks" if n else f"{f.name}: unchanged, skipped")
        total = client.count(get_settings().collection_name).count
        print(f"Done. Collection now holds {total} chunks.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
