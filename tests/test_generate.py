from types import SimpleNamespace as NS

from rag.generate import NOT_FOUND, REFUSED, answer, build_messages, parse_response
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
    ans = answer("Q?", [hit(1)], on_text=seen.append, client=client)
    assert seen == ["H", "i"]
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
