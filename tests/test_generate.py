from types import SimpleNamespace as NS

import httpx
import ollama
import pytest

from rag import generate
from rag.config import Settings
from rag.generate import (
    NOT_FOUND,
    REFUSED,
    LLMError,
    answer,
    build_local_messages,
    build_messages,
    parse_marked_text,
    parse_response,
    resolve_provider,
)
from rag.retrieve import Hit


def hit(n: int, page: int | None = None) -> Hit:
    return Hit(
        id=str(n), text=f"chunk text {n}", source=f"doc{n}.md", page=page,
        chunk_index=0, score=1.0,
    )  # fmt: skip


def text_block(text: str, *doc_indexes: int) -> NS:
    cites = [
        NS(type="char_location", document_index=i, cited_text=f"cited {i}") for i in doc_indexes
    ]
    return NS(type="text", text=text, citations=cites or None)


def message(*blocks, stop_reason="end_turn") -> NS:
    return NS(content=list(blocks), stop_reason=stop_reason, model="claude-opus-5-5")


# ---------- request building ----------


def test_build_messages_sends_each_hit_as_cited_document():
    msgs = build_messages("Q?", [hit(1, page=3), hit(2)])
    assert len(msgs) == 1
    content = msgs[0]["content"]
    docs = [b for b in content if b["type"] == "document"]
    assert [d["title"] for d in docs] == ["doc1.md (p. 3)", "doc2.md"]
    assert all(d["citations"] == {"enabled": True} for d in docs)
    assert docs[0]["source"]["data"] == "chunk text 1"
    assert content[-1] == {"type": "text", "text": "Q?"}


def test_build_messages_includes_history_as_plain_text():
    msgs = build_messages("Q2", [hit(1)], history=[("Q1", "A1")])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "A1"


# ---------- response parsing ----------


def test_parse_adds_markers_and_citations():
    hits = [hit(1), hit(2), hit(3)]
    msg = message(text_block("Pro costs $6. "), text_block("Refunds within 14 days.", 2, 0))
    ans = parse_response(msg, hits)
    assert ans.text == "Pro costs $6. Refunds within 14 days. [3] [1]"
    assert [c.number for c in ans.citations] == [3, 1]
    assert [n for n, _ in ans.cited_hits] == [1, 3]


def test_parse_ignores_out_of_range_citation():
    ans = parse_response(message(text_block("x", 7)), [hit(1)])
    assert ans.citations == [] and ans.text == "x"


def test_parse_keeps_only_text_after_fallback():
    msg = message(
        text_block("partial from declined model"), NS(type="fallback"), text_block("final")
    )
    assert parse_response(msg, [hit(1)]).text == "final"


def test_parse_refusal():
    ans = parse_response(message(text_block("half", 0), stop_reason="refusal"), [hit(1)])
    assert ans.text == REFUSED and ans.citations == []


def test_parse_max_tokens_adds_note():
    ans = parse_response(message(text_block("long"), stop_reason="max_tokens"), [hit(1)])
    assert ans.text.startswith("long") and "cut off" in ans.text


# ---------- answer() with a fake client ----------


class FakeStream:
    def __init__(self, final, chunks):
        self.final, self.text_stream = final, iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, final, chunks=()):
        self.calls = []
        outer = self

        def stream(**kwargs):
            outer.calls.append(kwargs)
            return FakeStream(final, chunks)

        self.beta = NS(messages=NS(stream=stream))


def test_answer_streams_and_sets_request_options():
    client = FakeClient(message(text_block("Hi", 0)), chunks=["H", "i"])
    seen = []
    ans = answer("Q?", [hit(1)], on_text=seen.append, provider="claude", client=client)
    assert seen == ["H", "i"]
    assert ans.provider == "claude"
    assert ans.text == "Hi [1]"
    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in kwargs["betas"]
    assert "thinking" not in kwargs and "temperature" not in kwargs


def test_answer_without_hits_skips_api_call():
    client = FakeClient(message())
    assert answer("Q?", [], client=client).text == NOT_FOUND
    assert client.calls == []


# ---------- provider choice ----------


@pytest.fixture
def settings(monkeypatch):
    """Fresh settings for each test, with no Claude credentials from the environment."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    monkeypatch.setattr(generate, "get_settings", lambda: s)
    return s


def test_auto_uses_ollama_without_key(settings):
    assert resolve_provider() == "ollama"


def test_auto_uses_claude_with_key(settings):
    settings.anthropic_api_key = "sk-test"
    assert resolve_provider() == "claude"


def test_auto_uses_claude_with_env_key(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_provider() == "claude"


def test_forced_provider_and_bad_value(settings):
    settings.anthropic_api_key = "sk-test"
    settings.llm_provider = "ollama"
    assert resolve_provider() == "ollama"
    settings.llm_provider = "gpt"
    with pytest.raises(LLMError):
        resolve_provider()


# ---------- local model (Ollama) ----------


def test_local_messages_number_documents():
    msgs = build_local_messages("Q?", [hit(1, page=2), hit(2)], history=[("Q0", "A0")])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "[2]" in msgs[0]["content"]  # citation rule is in the system prompt
    assert "[1] doc1.md (p. 2)\nchunk text 1" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("Question: Q?")


def test_parse_marked_text_builds_citations_and_drops_bad_markers():
    hits = [hit(1), hit(2)]
    ans = parse_marked_text("Pro costs $6 [2]. Refunds apply [1, 9]. Other [7] .", hits)
    assert ans.text == "Pro costs $6 [2]. Refunds apply [1]. Other."
    assert [c.number for c in ans.citations] == [2, 1]
    assert ans.provider == "ollama"


class FakeOllama:
    def __init__(self, pieces=(), error=None):
        self.pieces, self.error, self.calls = pieces, error, []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return (NS(message=NS(content=p)) for p in self.pieces)


def test_answer_with_ollama_streams(settings):
    client = FakeOllama(["Refunds within ", "14 days [1]."])
    seen = []
    ans = answer("Q?", [hit(1)], on_text=seen.append, client=client)
    assert seen == ["Refunds within ", "14 days [1]."]
    assert ans.text == "Refunds within 14 days [1]."
    assert ans.model == "llama3.2:3b" and [c.number for c in ans.citations] == [1]
    assert client.calls[0]["stream"] is True


@pytest.mark.parametrize("error", [ConnectionError("refused"), httpx.ConnectError("refused")])
def test_ollama_not_running(settings, error):
    with pytest.raises(LLMError, match="Ollama isn't running"):
        answer("Q?", [hit(1)], client=FakeOllama(error=error))


def test_ollama_model_missing(settings):
    err = ollama.ResponseError("model not found", 404)
    with pytest.raises(LLMError, match="ollama pull llama3.2:3b"):
        answer("Q?", [hit(1)], client=FakeOllama(error=err))
