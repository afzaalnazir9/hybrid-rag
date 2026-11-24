"""Example questions for the sample documents, with their expected answers.

Two kinds of tests use the same examples:

1. **Search tests** (`integration`): the passage holding the answer must be in the top 3
   results. They use only the local search models, so they run in CI.

2. **Live answer tests** (`live`): a real model (Claude if a key is set, otherwise Ollama)
   answers each question and the answer must mention the expected facts. They are slow
   (~30-40 s per question with a local CPU model) and are skipped unless you ask for them:

       uv run pytest -m live -v

   They skip automatically if no answer model is available.

Small local models phrase things differently from run to run, so answer checks look for
key facts (e.g. "14 days"), not exact wording.
"""

from dataclasses import dataclass, field

import httpx
import ollama
import pytest

from rag.config import get_settings
from rag.generate import answer, resolve_provider
from rag.retrieve import search


@dataclass
class Example:
    question: str
    passage: str | None  # phrase the retrieved passage must contain (None: not in the docs)
    answer_any: list[str] = field(default_factory=list)  # answer must mention one of these
    answer_all: list[str] = field(default_factory=list)  # ...and all of these

    def __str__(self) -> str:  # readable test ids
        return self.question


NOT_FOUND_PHRASES = ["couldn't find", "could not find", "not mentioned", "no information"]

EXAMPLES = [
    # Exact terms and error codes (keyword search's strength)
    Example(
        "What does error NB-4012 mean and how do I fix it?",
        "Rebuild local cache",
        answer_all=["rebuild local cache"],
    ),
    Example("What is NB-1004?", "Workspace limit reached", answer_any=["workspace"]),
    Example(
        "What is the shortcut for focus mode on macOS?",
        "Cmd+Ctrl+F",
        answer_all=["cmd+ctrl+f"],
    ),
    Example("Is SAML 2.0 supported?", "SAML 2.0", answer_any=["saml"]),
    # Reworded questions (meaning-based search's strength)
    Example(
        "Can I get my money back after buying a yearly subscription?",
        "refunded in full within 14 days",
        answer_any=["14 days", "14-day", "fourteen"],
    ),
    Example(
        "My notes disappeared, where could they be?",
        "Check the Trash",
        answer_any=["trash"],
    ),
    Example(
        "What happens if my card payment doesn't go through?",
        "retries three times",
        answer_any=["three", "3 times", "retr"],
    ),
    Example(
        "Will my writing be used to train AI?",
        "train machine-learning models",
        answer_any=["not", "no", "doesn't", "never"],
    ),
    Example(
        "Can my coworkers edit a notebook I share?",
        "Viewer, Editor or Owner",
        answer_any=["editor"],
    ),
    # Facts from tables
    Example(
        "How much does the Team plan cost?",
        "$10 per user per month",
        answer_any=["$10", "10 per user", "10 dollars"],
    ),
    Example("How much storage does Pro include?", "100 GB", answer_any=["100 gb", "100gb"]),
    Example(
        "How long is version history kept on the Free plan?",
        "7 days",
        answer_any=["7 days", "seven days"],
    ),
    Example(
        "Compare the attachment size limits of Free and Pro.",
        "25 MB",
        answer_all=["25", "250"],
    ),
    # Not in the documents: the model should say so instead of inventing an answer
    Example("Does Nimbus Notes support video calls?", None, answer_any=NOT_FOUND_PHRASES),
    Example("Does the app have a dark mode?", None, answer_any=NOT_FOUND_PHRASES),
    Example("Who is the CEO of Nimbus?", None, answer_any=NOT_FOUND_PHRASES),
]

ANSWERABLE = [e for e in EXAMPLES if e.passage]


# ---------- 1. search tests (run in CI) ----------


@pytest.mark.integration
@pytest.mark.parametrize("example", ANSWERABLE, ids=str)
def test_search_finds_the_answer_passage(sample_index, example):
    hits = search(example.question, mode="hybrid_rerank", top_k=3, client=sample_index)
    texts = [" ".join(h.text.split()) for h in hits]
    assert any(example.passage in t for t in texts), (
        f"'{example.passage}' not in top 3 for: {example.question}"
    )


@pytest.mark.integration
def test_keyword_mode_ranks_exact_code_first(sample_index):
    hits = search("What is NB-1004?", mode="keyword", top_k=1, client=sample_index)
    assert "NB-1004" in hits[0].text


# ---------- 2. live answer tests (opt in with: pytest -m live) ----------


@pytest.fixture(scope="session")
def answer_model():
    """Skip live tests when no answer model is usable; otherwise return the provider name."""
    provider = resolve_provider()
    if provider == "ollama":
        s = get_settings()
        try:
            ollama.Client(host=s.ollama_host).show(s.ollama_model)
        except (ConnectionError, httpx.TransportError):
            pytest.skip(f"Ollama is not running at {s.ollama_host}")
        except ollama.ResponseError:
            pytest.skip(f"Ollama model {s.ollama_model} is not pulled")
    return provider


def ask(question: str, index, history=None):
    hits = search(question, client=index)
    return answer(question, hits, history=history)


def check(text: str, example: Example) -> None:
    low = text.lower()
    if example.answer_any:
        assert any(p.lower() in low for p in example.answer_any), (
            f"expected one of {example.answer_any} in answer: {text}"
        )
    for p in example.answer_all:
        assert p.lower() in low, f"expected '{p}' in answer: {text}"


@pytest.mark.live
@pytest.mark.parametrize("example", EXAMPLES, ids=str)
def test_live_answer(sample_index, answer_model, example):
    result = ask(example.question, sample_index)
    check(result.text, example)
    if answer_model == "claude" and example.passage:
        assert result.citations, "Claude answers should carry citations"


@pytest.mark.live
def test_live_follow_up_uses_chat_history(sample_index, answer_model):
    first_q = "How much is the Pro plan?"
    first = ask(first_q, sample_index)
    assert "$6" in first.text or "6 per month" in first.text, first.text

    follow_up = ask("Is there a cheaper price for students?", sample_index, [(first_q, first.text)])
    assert "$3" in follow_up.text or "3 per month" in follow_up.text, follow_up.text
