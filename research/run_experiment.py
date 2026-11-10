"""Compare query strategies under the same annotation budget, over several seeds.

For every seed the data are split into a stratified held-out test set and a
pool; each strategy starts from the SAME seed labels and pool, labels `batch`
examples per cycle (using the pool's true labels as a simulated annotator),
retrains and is evaluated on the test set.

    python research/run_experiment.py --data data/demo_pool.csv --seeds 5 --cycles 15
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sklearn.model_selection import train_test_split  # noqa: E402

from app.active_learning import STRATEGIES, ActiveLearner  # noqa: E402


def seed_set(labels, size, rng):
    """Seed labels with every class present (stratified as far as possible)."""
    classes = sorted(set(labels))
    idx = []
    for c in classes:
        idx.append(rng.choice([i for i, y in enumerate(labels) if y == c]))
    rest = [i for i in rng.permutation(len(labels)) if i not in idx]
    return idx + rest[:max(0, size - len(idx))]


def run(path, seeds=5, cycles=15, seed_size=12, batch=10, test_size=0.3):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts, labels = [r["text"] for r in rows], [r["label"] for r in rows]
    out = []
    for seed in range(seeds):
        tr_x, te_x, tr_y, te_y = train_test_split(texts, labels, test_size=test_size,
                                                  random_state=seed, stratify=labels)
        start = seed_set(tr_y, seed_size, np.random.default_rng(seed))
        for strategy in STRATEGIES:
            learner = ActiveLearner(seed)
            labeled = set(start)
            for cycle in range(cycles + 1):
                idx = sorted(labeled)
                learner.train([tr_x[i] for i in idx], [tr_y[i] for i in idx])
                out.append({"seed": seed, "strategy": strategy, "cycle": cycle,
                            "labeled_samples": len(labeled), **learner.evaluate(te_x, te_y)})
                pool = [i for i in range(len(tr_x)) if i not in labeled]
                if cycle == cycles or not pool:
                    break
                chosen, _, _ = learner.select([tr_x[i] for i in pool], batch, strategy)
                labeled.update(pool[j] for j in chosen)
    return out


def summarise(rows):
    """Mean and std per strategy and budget, plus area under the F1 learning curve."""
    import pandas as pd

    d = pd.DataFrame(rows)
    curve = d.groupby(["strategy", "labeled_samples"])[["accuracy", "f1_macro"]].agg(["mean", "std"]).reset_index()
    curve.columns = ["_".join(c).strip("_") for c in curve.columns]
    alc = (d.groupby(["strategy", "seed"])["f1_macro"].mean().groupby("strategy").agg(["mean", "std"])
           .rename(columns={"mean": "alc_f1_mean", "std": "alc_f1_std"}).reset_index())
    final = d[d.cycle == d.cycle.max()].groupby("strategy")["f1_macro"].agg(["mean", "std"])
    alc = alc.merge(final.rename(columns={"mean": "final_f1_mean", "std": "final_f1_std"}).reset_index())
    return curve, alc.sort_values("alc_f1_mean", ascending=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/demo_pool.csv")
    p.add_argument("--output", default="research/results.csv")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--cycles", type=int, default=15)
    p.add_argument("--seed-size", type=int, default=12)
    p.add_argument("--batch", type=int, default=10)
    a = p.parse_args()
    rows = run(a.data, a.seeds, a.cycles, a.seed_size, a.batch)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    curve, alc = summarise(rows)
    base = os.path.splitext(a.output)[0]
    curve.to_csv(base + "_curve.csv", index=False)
    alc.to_csv(base + "_summary.csv", index=False)
    print(f"Saved {len(rows)} rows to {a.output}\n")
    print(alc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
