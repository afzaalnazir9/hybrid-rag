"""Compare retrieval modes on a small labelled question set.

Each question in eval_questions.json names a phrase that the right passage contains.
A retrieved chunk counts as correct if it contains that phrase. For every mode we report:

- Hit@1: the first result is correct
- Hit@3: a correct chunk is in the top 3
- MRR:   mean reciprocal rank of the first correct chunk (top 5; 0 if none)

The sample documents are indexed into a fresh in-memory Qdrant, so results don't depend
on what you've uploaded and are the same on every run. No language model is used.

    uv run python eval.py                       # print the tables
    uv run python eval.py --out eval_results.md # also save them
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from rag.ingest import ingest_file, iter_files
from rag.retrieve import MODES, search

ROOT = Path(__file__).parent
K = 5


@dataclass
class Result:
    hit1: float
    hit3: float
    mrr: float
    ms: float  # average search time per question


def norm(text: str) -> str:
    return " ".join(text.split())


def first_correct_rank(hits, phrase: str) -> int | None:
    for rank, h in enumerate(hits, 1):
        if norm(phrase) in norm(h.text):
            return rank
    return None


def evaluate(client: QdrantClient, questions: list[dict], mode: str) -> Result:
    ranks, start = [], time.perf_counter()
    for q in questions:
        hits = search(q["question"], mode=mode, top_k=K, client=client)
        ranks.append(first_correct_rank(hits, q["answer_contains"]))
    elapsed = (time.perf_counter() - start) * 1000 / len(questions)
    n = len(questions)
    return Result(
        hit1=sum(r == 1 for r in ranks) / n,
        hit3=sum(r is not None and r <= 3 for r in ranks) / n,
        mrr=sum(1 / r for r in ranks if r) / n,
        ms=elapsed,
    )


def table(results: dict[str, Result], with_time: bool) -> str:
    head = "| Mode | Hit@1 | Hit@3 | MRR |" + (" Avg search time |" if with_time else "")
    sep = "|---|---|---|---|" + ("---|" if with_time else "")
    rows = [head, sep]
    for mode, r in results.items():
        row = f"| {mode} | {r.hit1:.0%} | {r.hit3:.0%} | {r.mrr:.2f} |"
        rows.append(row + (f" {r.ms:.0f} ms |" if with_time else ""))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", type=Path, default=ROOT / "eval_questions.json")
    parser.add_argument("--out", type=Path, help="also write the tables to this Markdown file")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    client = QdrantClient(":memory:")
    for f in iter_files([ROOT / "sample_data"]):
        ingest_file(f, client=client)

    # A phrase split across two chunks can't be found by any mode; flag it.
    chunks = [norm(p.payload["text"]) for p in client.scroll("documents", limit=10_000)[0]]
    for q in questions:
        if not any(norm(q["answer_contains"]) in c for c in chunks):
            print(f"warning: no single chunk contains {q['answer_contains']!r}")

    search("warm up", client=client)  # load models before timing
    sections = []
    groups = [("All questions", questions)] + [
        (f"{t.capitalize()} questions", [q for q in questions if q["type"] == t])
        for t in sorted({q["type"] for q in questions})
    ]
    for title, qs in groups:
        results = {m: evaluate(client, qs, m) for m in MODES}
        sections.append(f"### {title} ({len(qs)})\n\n{table(results, title == 'All questions')}")
    client.close()

    report = "\n\n".join(sections)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
