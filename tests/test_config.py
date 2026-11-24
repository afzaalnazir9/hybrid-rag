from rag.config import Settings


def test_defaults_are_sane():
    s = Settings(_env_file=None)
    assert s.chunk_overlap < s.chunk_size
    assert s.top_k <= s.candidates_per_search


def test_env_override(monkeypatch):
    monkeypatch.setenv("TOP_K", "3")
    assert Settings(_env_file=None).top_k == 3
