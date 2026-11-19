"""Tests for the six milestones in app/core.py."""
from app import core


def test_milestone_1():
    baskets = core.milestone_1(n_baskets=100, seed=1)
    assert len(baskets) == 100
    assert all(len(b) >= 2 for b in baskets)
    assert all(item in core.ITEMS for b in baskets for item in b)
    assert baskets == core.milestone_1(n_baskets=100, seed=1)   # reproducible


def test_milestone_2():
    baskets = core.milestone_1(n_baskets=200, seed=2)
    model = core.milestone_2(baskets, seed=2, warmup_epochs=20, continue_epochs=60)
    probs = core.next_probs(model, core.IDX["bread"])
    assert abs(sum(probs) - 1.0) < 1e-9
    untrained = core._new_model(seed=2)
    pairs = core.to_pairs(baskets)
    assert core.evaluate(model, pairs)["top1"] > core.evaluate(untrained, pairs)["top1"]


def test_milestone_3():
    out = core.milestone_3()
    assert out["suggestions_after_bread"][0][0] == "butter"
    assert out["together_score"] > out["unrelated_score"]
    assert len(core.predict_next(core._get_trained()[2], ["bread", "butter"], k=2)) == 2


def test_milestone_4():
    base = core.milestone_4()
    assert base["size_bytes"] == (core.V * core.H + core.H * core.V + core.V) * 4
    assert 0.5 < base["top1"] <= 1.0


def test_milestone_5():
    dynamic, static = core.milestone_5()
    assert dynamic["kind"] == "dynamic" and static["kind"] == "static"
    for quant in (dynamic, static):
        weights = [q for row in quant["Eq"] + quant["Wq"] for q in row]
        assert all(-127 <= q <= 127 for q in weights)
        assert abs(sum(core.next_probs(quant, 0)) - 1.0) < 1e-9


def test_milestone_6():
    report = core.milestone_6()
    for name in ("dynamic", "static"):
        assert report[name]["size_reduction"] > 3
        assert report[name]["top1_change"] > -0.05   # accuracy barely moves
