from rag.retrieve import Hit, rerank, rrf_fuse


def hit(id_: str, stage: str = "semantic", score: float = 1.0) -> Hit:
    return Hit(
        id=id_, text=f"text {id_}", source="doc.md", page=None, chunk_index=0,
        score=score, stages={stage: score},
    )  # fmt: skip


def test_rrf_rewards_agreement():
    semantic = [hit("a"), hit("b"), hit("c")]
    keyword = [hit("c", "keyword"), hit("b", "keyword"), hit("d", "keyword")]
    fused = rrf_fuse([semantic, keyword], k=60)
    ids = [h.id for h in fused]
    # b is 2nd in both lists, so it beats a (1st in one list only).
    assert ids[0] in {"b", "c"}
    assert ids.index("b") < ids.index("a")
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_score_formula_and_stages():
    fused = rrf_fuse([[hit("a")], [hit("x", "keyword"), hit("a", "keyword")]], k=60)
    a = next(h for h in fused if h.id == "a")
    assert abs(a.score - (1 / 61 + 1 / 62)) < 1e-12
    assert {"semantic", "keyword", "rrf"} <= set(a.stages)


def test_rrf_does_not_mutate_inputs():
    original = hit("a")
    rrf_fuse([[original]])
    assert original.score == 1.0 and original.stages == {"semantic": 1.0}


def test_rerank_sorts_by_scorer():
    hits = [hit("a"), hit("b"), hit("c")]
    scores = {"text a": 0.1, "text b": 0.9, "text c": 0.5}
    out = rerank("q", hits, scorer=lambda q, docs: [scores[d] for d in docs])
    assert [h.id for h in out] == ["b", "c", "a"]
    assert out[0].stages["rerank"] == 0.9


def test_rerank_empty():
    assert rerank("q", [], scorer=lambda q, d: []) == []
