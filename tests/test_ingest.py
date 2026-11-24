from pathlib import Path

import pytest

from rag.ingest import chunk_text, iter_files, load_document, normalize


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("  \n\n ", size=100, overlap=10) == []


def test_chunks_respect_size_and_cover_text():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Every word appears in some chunk, and no chunk starts mid-word.
    joined = " ".join(chunks).split()
    assert set(words) <= set(joined)
    assert all(c.split()[0] in words for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    a, b = chunk_text(text, size=200, overlap=50)[:2]
    assert a.split()[-1] in b.split()


def test_prefers_paragraph_breaks():
    para = "Sentence one. " * 10
    text = f"{para.strip()}\n\n{para.strip()}"
    chunks = chunk_text(text, size=len(text) - 20, overlap=0)
    assert chunks[0] == para.strip()


def test_normalize_collapses_whitespace():
    assert normalize("a  \t b\r\n\n\n\nc ") == "a b\n\nc"


def test_load_text_and_docx(tmp_path: Path):
    import docx

    md = tmp_path / "a.md"
    md.write_text("# Title\n\nBody", encoding="utf-8")
    assert load_document(md)[0].text == "# Title\n\nBody"

    d = docx.Document()
    d.add_paragraph("First paragraph")
    d.add_paragraph("Second paragraph")
    path = tmp_path / "b.docx"
    d.save(str(path))
    assert load_document(path)[0].text == "First paragraph\n\nSecond paragraph"


def test_unsupported_type(tmp_path: Path):
    f = tmp_path / "x.csv"
    f.write_text("a,b")
    with pytest.raises(ValueError):
        load_document(f)


def test_iter_files_filters_types(tmp_path: Path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.csv").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("x")
    assert sorted(f.name for f in iter_files([tmp_path])) == ["a.md", "c.txt"]
