"""Answer questions from retrieved chunks, with citations.

Two ways to write the answer:
- **Claude** (when an API key is available): each chunk is sent as its own `document`
  block with citations enabled, so citations come back from the API itself.
- **Local model via Ollama** (no key needed): chunks are numbered in the prompt and the
  model is asked to cite them as [n]; those markers are parsed into citations.

With LLM_PROVIDER=auto (the default), Claude is used if a key is set, otherwise Ollama.
The CLI runs the full pipeline (search + answer):

    python -m rag.generate "Can I get a refund on a yearly plan?"
"""

import argparse
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx
import ollama

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

LOCAL_CITATION_RULE = """- Cite by putting the document number in square brackets right after \
the sentence that uses it, like [2]. Only use numbers of documents that are listed."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(Exception):
    """A problem reaching or using the answer model, with a message fit for users."""


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
    provider: str | None = None

    @property
    def cited_hits(self) -> list[tuple[int, Hit]]:
        """(number, hit) for each source the answer actually cites, in order."""
        numbers = sorted({c.number for c in self.citations})
        return [(n, self.hits[n - 1]) for n in numbers]


# ---------- choosing the model ----------


def has_claude_key() -> bool:
    return bool(
        get_settings().anthropic_api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )


def resolve_provider() -> str:
    provider = get_settings().llm_provider.lower()
    if provider == "auto":
        return "claude" if has_claude_key() else "ollama"
    if provider not in {"claude", "ollama"}:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use auto, claude or ollama.")
    return provider


# ---------- Claude ----------


def build_messages(
    question: str, hits: list[Hit], history: list[tuple[str, str]] | None = None
) -> list[dict[str, Any]]:
    """Claude request: earlier turns are plain text; the new question carries the documents."""
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
    return Answer(
        text, citations, hits, message.stop_reason, getattr(message, "model", None), "claude"
    )


def _answer_claude(question, hits, history, on_text, client) -> Answer:
    s = get_settings()
    try:
        client = client or anthropic.Anthropic(api_key=s.anthropic_api_key or None)
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
    except anthropic.AuthenticationError as e:
        raise LLMError("Claude API key is invalid. Check ANTHROPIC_API_KEY in .env.") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Claude API. Check your internet connection.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude API error ({e.status_code}): {e.message}") from e
    except TypeError as e:  # the SDK raises TypeError when no credentials are configured
        if "authentication" not in str(e):
            raise
        raise LLMError("No Claude API key found. Set ANTHROPIC_API_KEY in .env.") from e
    return parse_response(final, hits)


# ---------- local model (Ollama) ----------

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def build_local_messages(
    question: str, hits: list[Hit], history: list[tuple[str, str]] | None = None
) -> list[dict[str, str]]:
    """Ollama request: documents are numbered inline so the model can cite them as [n]."""
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n{LOCAL_CITATION_RULE}"}]
    for q, a in history or []:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    docs = "\n\n".join(f"[{i}] {h.label}\n{h.text}" for i, h in enumerate(hits, 1))
    messages.append({"role": "user", "content": f"Documents:\n\n{docs}\n\nQuestion: {question}"})
    return messages


def parse_marked_text(text: str, hits: list[Hit], model: str | None = None) -> Answer:
    """Build citations from [n] / [n, m] markers; drop markers that point at no document."""
    seen: list[int] = []

    def keep_valid(match: re.Match) -> str:
        nums = [int(x) for x in match.group(1).split(",")]
        valid = [n for n in nums if 1 <= n <= len(hits)]
        seen.extend(n for n in valid if n not in seen)
        return "".join(f"[{n}]" for n in valid)

    cleaned = re.sub(r" +([.,;:!?])", r"\1", _MARKER.sub(keep_valid, text)).strip()
    citations = [Citation(n, hits[n - 1].label, hits[n - 1].text[:300]) for n in seen]
    return Answer(cleaned, citations, hits, "end_turn", model, "ollama")


def _answer_ollama(question, hits, history, on_text, client) -> Answer:
    s = get_settings()
    client = client or ollama.Client(host=s.ollama_host)
    parts: list[str] = []
    try:
        for chunk in client.chat(
            model=s.ollama_model,
            messages=build_local_messages(question, hits, history),
            stream=True,
            options={"temperature": 0.1},
        ):
            piece = chunk.message.content or ""
            parts.append(piece)
            if on_text and piece:
                on_text(piece)
    except (ConnectionError, httpx.TransportError) as e:
        raise LLMError(
            f"No Claude API key, and Ollama isn't running at {s.ollama_host}. "
            "Start the Ollama app, or set ANTHROPIC_API_KEY in .env to use Claude."
        ) from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise LLMError(
                f"The local model '{s.ollama_model}' isn't downloaded yet. "
                f"Run: ollama pull {s.ollama_model}"
            ) from e
        raise LLMError(f"Ollama error: {e.error}") from e
    return parse_marked_text("".join(parts), hits, s.ollama_model)


# ---------- public entry point ----------


def answer(
    question: str,
    hits: list[Hit],
    history: list[tuple[str, str]] | None = None,
    on_text: Callable[[str], None] | None = None,
    provider: str | None = None,
    client: Any = None,
) -> Answer:
    """Write an answer from `hits`, streaming text to `on_text` as it arrives.

    `provider` overrides the configured one ("claude" or "ollama"); `client` lets tests
    pass a fake Anthropic or Ollama client. Raises LLMError with a readable message.
    """
    if not hits:
        return Answer(NOT_FOUND)
    provider = provider or resolve_provider()
    if provider == "claude":
        return _answer_claude(question, hits, history, on_text, client)
    return _answer_ollama(question, hits, history, on_text, client)


def main() -> None:
    from rag.retrieve import MODES, search
    from rag.store import get_client

    parser = argparse.ArgumentParser(description="Ask a question about the indexed documents.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--provider", choices=["claude", "ollama"], default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles

    qdrant = get_client()
    try:
        hits = search(args.question, mode=args.mode, client=qdrant)
    finally:
        qdrant.close()

    try:
        result = answer(args.question, hits, provider=args.provider)
    except LLMError as e:
        sys.exit(str(e))

    print(result.text)
    if result.cited_hits:
        print("\nSources:")
        for n, hit in result.cited_hits:
            print(f"  [{n}] {hit.label}")
    if result.model:
        print(f"\n(answered by {result.model})")


if __name__ == "__main__":
    main()
