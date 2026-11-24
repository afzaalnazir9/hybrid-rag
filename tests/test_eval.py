import json
from pathlib import Path

from eval import first_correct_rank, norm
from rag.retrieve import Hit

ROOT = Path(__file__).parent.parent


def hit(text: str) -> Hit:
    return Hit(id=text, text=text, source="d.md", page=None, chunk_index=0, score=0.0)


def test_first_correct_rank():
    hits = [hit("nothing here"), hit("refunded in\nfull  within 14 days"), hit("refunded in full")]
    assert first_correct_rank(hits, "refunded in full within 14 days") == 2
    assert first_correct_rank(hits, "absent phrase") is None


def test_norm_collapses_whitespace():
    assert norm(" a \n b\t c ") == "a b c"


def test_question_set_is_valid():
    questions = json.loads((ROOT / "eval_questions.json").read_text(encoding="utf-8"))
    docs = norm(" ".join(p.read_text(encoding="utf-8") for p in (ROOT / "sample_data").glob("*")))
    assert len(questions) >= 20
    for q in questions:
        assert q["type"] in {"keyword", "paraphrase"}
        assert norm(q["answer_contains"]) in docs, q
