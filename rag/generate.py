"""Answer questions with Claude, using retrieved chunks as cited sources.

Each retrieved chunk is sent as its own `document` block with citations enabled, so
Claude's answer comes back with citations pointing at exact passages. The CLI runs
the full pipeline (search + answer):

    python -m rag.generate "Can I get a refund on a yearly plan?"
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic

from rag.config import get_settings
from rag.retrieve import Hit

NOT_FOUND = "I couldn't find this in the documents."
REFUSED = "Sorry, I can't help with that request."

SYSTEM_PROMPT = f"""You answer questions using only the documents provided in the user's message.

- Base every statement on the documents and cite the passages you used.
- If the documents don't contain the answer, reply exactly: "{NOT_FOUND}" \
You may add one sentence saying what related information the documents do contain.
- If the documents only partly answer the question, answer that part and say what is missing.
- Be concise: a short paragraph or a few bullet points. Don't mention "the documents" \
or "the context"; just answer."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Citation:
    number: int  # 1-based index into Answer.hits, matches the [n] markers in the text
    source: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    stop_reason: str | None = None
    model: str | None = None

    @property
    def cited_hits(self) -> list[tuple[int, Hit]]:
        """(number, hit) for each source the answer actually cites, in order."""
        numbers = sorted({c.number for c in self.citations})
        return [(n, self.hits[n - 1]) for n in numbers]


def build_messages(
    question: str, hits: list[Hit], history: list[tuple[str, str]] | None = None
) -> list[dict[str, Any]]:
    """Earlier turns are plain text; the new question carries the retrieved documents."""
    messages: list[dict[str, Any]] = []
    for q, a in history or []:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.text},
            "title": h.label,
            "citations": {"enabled": True},
        }
        for h in hits
    ]
    messages.append({"role": "user", "content": [*documents, {"type": "text", "text": question}]})
    return messages


def parse_response(message: Any, hits: list[Hit]) -> Answer:
    """Turn Claude's content blocks into answer text with [n] markers plus a citation list."""
    blocks = list(message.content)
    # With server-side fallback, a declined attempt's partial text comes before a
    # `fallback` block; only the text after the last one belongs to the final answer.
    last_fallback = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)

    parts: list[str] = []
    citations: list[Citation] = []
    for block in blocks[last_fallback + 1 :]:
        if block.type != "text":
            continue
        parts.append(block.text)
        numbers: list[int] = []
        for c in getattr(block, "citations", None) or []:
            idx = getattr(c, "document_index", None)
            if idx is None or not 0 <= idx < len(hits):
                continue
            n = idx + 1
            citations.append(Citation(n, hits[idx].label, c.cited_text))
            if n not in numbers:
                numbers.append(n)
        if numbers:
            parts.append("".join(f" [{n}]" for n in numbers))

    text = "".join(parts).strip()
    if message.stop_reason == "refusal":
        text, citations = REFUSED, []
    elif message.stop_reason == "max_tokens":
        text += "\n\n(The answer was cut off because it reached the length limit.)"
    return Answer(text, citations, hits, message.stop_reason, getattr(message, "model", None))


def make_client() -> anthropic.Anthropic:
    # With no key in .env, the SDK falls back to ANTHROPIC_API_KEY and other standard sources.
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key or None)


def answer(
    question: str,
    hits: list[Hit],
    history: list[tuple[str, str]] | None = None,
    on_text: Callable[[str], None] | None = None,
    client: anthropic.Anthropic | None = None,
) -> Answer:
    """Stream an answer from Claude. `on_text` receives text as it arrives (for live UIs);
    the returned Answer holds the final text with citation markers."""
    if not hits:
        return Answer(NOT_FOUND)

    s = get_settings()
    client = client or make_client()
    with client.beta.messages.stream(
        model=s.claude_model,
        max_tokens=s.max_tokens,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, hits, history),
        output_config={"effort": s.claude_effort},
        # If a safety classifier declines, retry server-side on Anthropic's recommended model.
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        if on_text:
            for text in stream.text_stream:
                on_text(text)
        final = stream.get_final_message()
    return parse_response(final, hits)


def main() -> None:
    from rag.retrieve import MODES, search
    from rag.store import get_client

    parser = argparse.ArgumentParser(description="Ask a question about the indexed documents.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles

    qdrant = get_client()
    try:
        hits = search(args.question, mode=args.mode, client=qdrant)
    finally:
        qdrant.close()

    try:
        result = answer(args.question, hits)
    except anthropic.AuthenticationError:
        sys.exit("Claude API key missing or invalid. Set ANTHROPIC_API_KEY in .env.")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Claude API. Check your internet connection.")
    except anthropic.APIStatusError as e:
        sys.exit(f"Claude API error ({e.status_code}): {e.message}")
    except anthropic.AnthropicError as e:
        sys.exit(f"Claude client error: {e}")
    except TypeError as e:  # the SDK raises TypeError when no credentials are configured
        if "authentication" not in str(e):
            raise
        sys.exit("No Claude API key found. Copy .env.example to .env and set ANTHROPIC_API_KEY.")

    print(result.text)
    if result.cited_hits:
        print("\nSources:")
        for n, hit in result.cited_hits:
            print(f"  [{n}] {hit.label}")


if __name__ == "__main__":
    main()
