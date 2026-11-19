"""Quantized Retail Recommender: core logic.

A tiny next-item model for shopping baskets (embedding -> tanh -> linear -> softmax),
trained in pure Python, then shrunk with int8 post-training quantization.
Only the standard library is used, so it runs anywhere and the tests stay fast.

Run `python -m app.core` to print the full results table.
"""
import math
import random
from array import array
from functools import lru_cache

ITEMS = ["bread", "butter", "jam", "milk", "cereal", "coffee", "sugar", "pasta",
         "sauce", "cheese", "chips", "soda", "salsa", "eggs", "bacon", "tea"]
V = len(ITEMS)          # vocabulary size
H = 12                  # hidden size
IDX = {name: i for i, name in enumerate(ITEMS)}

# Ordered "what sells together" patterns used to generate fake baskets.
PATTERNS = [
    ["bread", "butter", "jam"],
    ["cereal", "milk", "sugar"],
    ["coffee", "sugar", "milk"],
    ["pasta", "sauce", "cheese"],
    ["chips", "salsa", "soda"],
    ["eggs", "bacon", "bread"],
    ["tea", "milk", "sugar"],
]


# --- Milestone 1: Prepare synthetic shopping basket data for training
def milestone_1(n_baskets=600, seed=0):
    """Return a list of baskets (lists of item names) built from PATTERNS plus noise."""
    rng = random.Random(seed)
    baskets = []
    for _ in range(n_baskets):
        basket = list(rng.choice(PATTERNS))
        if rng.random() < 0.2:                       # shopper forgot one item
            basket.pop(rng.randrange(len(basket)))
        if rng.random() < 0.3:                       # impulse buy
            basket.insert(rng.randrange(len(basket) + 1), rng.choice(ITEMS))
        baskets.append(basket)
    return baskets


def split_baskets(baskets, test_fraction=0.2):
    cut = int(len(baskets) * (1 - test_fraction))
    return baskets[:cut], baskets[cut:]


def to_pairs(baskets):
    """(item, next item) index pairs from consecutive items in each basket."""
    return [(IDX[a], IDX[b]) for basket in baskets for a, b in zip(basket, basket[1:])]


def _count_table(pairs):
    counts = [[0] * V for _ in range(V)]
    for a, b in pairs:
        counts[a][b] += 1
    return counts


# --- the small model -------------------------------------------------------
def _new_model(seed=0):
    rng = random.Random(seed)
    return {
        "kind": "float",
        "E": [[rng.uniform(-0.5, 0.5) for _ in range(H)] for _ in range(V)],
        "W": [[rng.uniform(-0.5, 0.5) for _ in range(V)] for _ in range(H)],
        "b": [0.0] * V,
    }


def _softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _hidden(model, idx):
    return [math.tanh(x) for x in model["E"][idx]]


def _train(model, counts, epochs, lr):
    """Full-batch gradient descent on the next-item cross-entropy loss."""
    total = sum(sum(row) for row in counts) or 1
    E, W, b = model["E"], model["W"], model["b"]
    for _ in range(epochs):
        gE = [[0.0] * H for _ in range(V)]
        gW = [[0.0] * V for _ in range(H)]
        gb = [0.0] * V
        for a in range(V):
            n_a = sum(counts[a])
            if n_a == 0:
                continue
            h = _hidden(model, a)
            logits = [sum(h[j] * W[j][k] for j in range(H)) + b[k] for k in range(V)]
            p = _softmax(logits)
            weight = n_a / total
            d = [weight * (p[k] - counts[a][k] / n_a) for k in range(V)]
            for k in range(V):
                gb[k] += d[k]
            for j in range(H):
                dh = 0.0
                for k in range(V):
                    gW[j][k] += h[j] * d[k]
                    dh += W[j][k] * d[k]
                gE[a][j] += dh * (1 - h[j] ** 2)
        for a in range(V):
            for j in range(H):
                E[a][j] -= lr * gE[a][j]
        for j in range(H):
            for k in range(V):
                W[j][k] -= lr * gW[j][k]
        for k in range(V):
            b[k] -= lr * gb[k]
    return model


# --- Milestone 2: Continue pretraining a small LM on the basket data
def milestone_2(baskets=None, seed=0, warmup_epochs=100, continue_epochs=400, lr=3.0):
    """Stage 1: train on a quarter of the data. Stage 2: continue from those weights on all of it."""
    baskets = baskets if baskets is not None else milestone_1(seed=seed)
    model = _new_model(seed)
    first = baskets[: max(1, len(baskets) // 4)]
    _train(model, _count_table(to_pairs(first)), warmup_epochs, lr)
    _train(model, _count_table(to_pairs(baskets)), continue_epochs, lr)
    return model


# --- Milestone 3: Implement next-item prediction and itemset probing
def next_probs(model, idx):
    """Probability of every item following item `idx`, for float or quantized models."""
    kind = model["kind"]
    if kind == "float":
        h = _hidden(model, idx)
        return _softmax([sum(h[j] * model["W"][j][k] for j in range(H)) + model["b"][k]
                         for k in range(V)])
    row = [q * model["sE"] for q in model["Eq"][idx]]
    h = [math.tanh(x) for x in row]
    s_h = _scale(h) if kind == "dynamic" else model["s_h"]
    hq = _quantize(h, s_h)
    logits = [sum(hq[j] * model["Wq"][j][k] for j in range(H)) * s_h * model["sW"] + model["b"][k]
              for k in range(V)]
    return _softmax(logits)


def predict_next(model, basket, k=3):
    """Top-k items to recommend after the last item in `basket` (skips items already in it)."""
    probs = next_probs(model, IDX[basket[-1]])
    ranked = sorted(range(V), key=lambda i: -probs[i])
    return [(ITEMS[i], probs[i]) for i in ranked if ITEMS[i] not in basket][:k]


def itemset_score(model, items):
    """How strongly the model thinks `items` go together (0 to 1, higher = more together)."""
    total = 0.0
    for a in items:
        probs = next_probs(model, IDX[a])
        total += sum(probs[IDX[b]] for b in items if b != a)
    return total / len(items)


def milestone_3(model=None):
    model = model if model is not None else _get_trained()[2]
    return {
        "suggestions_after_bread": predict_next(model, ["bread"]),
        "together_score": itemset_score(model, ["bread", "butter", "jam"]),
        "unrelated_score": itemset_score(model, ["tea", "salsa", "bacon"]),
    }


# --- Milestone 4: Measure the baseline size and accuracy of the model
def size_bytes(model):
    """Bytes needed to store the model's numbers (float32 or int8 as appropriate)."""
    flat = lambda m: [x for row in m for x in row]
    if model["kind"] == "float":
        return sum(len(array("f", part)) * 4 for part in (flat(model["E"]), flat(model["W"]), model["b"]))
    ints = len(array("b", flat(model["Eq"]))) + len(array("b", flat(model["Wq"])))
    floats = len(array("f", model["b"])) * 4 + 3 * 4   # biases + a few scale factors
    return ints + floats


def evaluate(model, pairs):
    """Top-1 and top-3 next-item accuracy over (item, next item) pairs."""
    hit1 = hit3 = 0
    for a, b in pairs:
        probs = next_probs(model, a)
        ranked = sorted(range(V), key=lambda i: -probs[i])
        hit1 += ranked[0] == b
        hit3 += b in ranked[:3]
    n = len(pairs) or 1
    return {"top1": hit1 / n, "top3": hit3 / n}


def milestone_4(model=None):
    model, test_pairs = _model_and_test(model)
    return {"size_bytes": size_bytes(model), **evaluate(model, test_pairs)}


# --- Milestone 5: Apply dynamic and static post-training quantization to the model
def _scale(values):
    m = max(abs(v) for v in values)
    return m / 127 if m > 0 else 1.0


def _quantize(values, scale):
    return [max(-127, min(127, round(v / scale))) for v in values]


def _quantize_weights(model, kind):
    flat = lambda m: [x for row in m for x in row]
    sE, sW = _scale(flat(model["E"])), _scale(flat(model["W"]))
    return {
        "kind": kind, "b": list(model["b"]), "sE": sE, "sW": sW,
        "Eq": [_quantize(row, sE) for row in model["E"]],
        "Wq": [_quantize(row, sW) for row in model["W"]],
    }


def quantize_dynamic(model):
    """int8 weights; the activation scale is worked out fresh for every input."""
    return _quantize_weights(model, "dynamic")


def quantize_static(model, calibration_items):
    """int8 weights; the activation scale is fixed in advance from calibration data."""
    quant = _quantize_weights(model, "static")
    quant["s_h"] = max(_scale(_hidden(model, i)) for i in calibration_items)
    return quant


def milestone_5(model=None):
    model = model if model is not None else _get_trained()[2]
    train_pairs = _get_trained()[0]
    calibration_items = sorted({a for a, _ in train_pairs})
    return quantize_dynamic(model), quantize_static(model, calibration_items)


# --- Milestone 6: Demonstrate size reduction and evaluate accuracy impact
def milestone_6(model=None):
    model, test_pairs = _model_and_test(model)
    dynamic, static = milestone_5(model)
    base = {"size_bytes": size_bytes(model), **evaluate(model, test_pairs)}
    report = {"baseline": base}
    for name, quant in (("dynamic", dynamic), ("static", static)):
        row = {"size_bytes": size_bytes(quant), **evaluate(quant, test_pairs)}
        row["size_reduction"] = base["size_bytes"] / row["size_bytes"]
        row["top1_change"] = row["top1"] - base["top1"]
        report[name] = row
    return report


# --- helpers ---------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_trained(seed=0):
    """(train_pairs, test_pairs, trained model) for the default synthetic data."""
    train, test = split_baskets(milestone_1(seed=seed))
    return to_pairs(train), to_pairs(test), milestone_2(train, seed=seed)


def _model_and_test(model):
    train_pairs, test_pairs, default_model = _get_trained()
    return (model if model is not None else default_model), test_pairs


if __name__ == "__main__":
    for name, row in milestone_6().items():
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    print(milestone_3())
